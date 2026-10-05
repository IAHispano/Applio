"""Shared V3 file/batch/live conversion with strict representation checks.

Load a target acoustic package and a separate universal vocoder. Check mel
semantics and frontend identity before producing audio. Bounded causal packages
use the same retained-state path for file and live conversion; offline profiles
are a separate contract. This module does not route classic v1/v2 generators.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rvc.configs.v3 import MelConfig, require_contract
from rvc.realtime.v3_streaming import (
    AcousticStream,
    FeatureStream,
    VocoderStream,
    coordinate_noise,
)
from rvc.train.extract.v3 import FeatureExtractor, read_audio
from rvc.train.process.v3_checkpoints import construct, load_payload
from rvc.train.v3.trainer import resolve_device


class Converter:
    """Validate package/frontend compatibility and own frozen inference models."""

    def __init__(
        self, acoustic, vocoder, encoder, pitch_path=None, device="auto", extractor=None
    ):
        self.device = resolve_device(device)
        self.acoustic_package, self.vocoder_package = (
            load_payload(acoustic),
            load_payload(vocoder),
        )
        a, v = self.acoustic_package, self.vocoder_package
        if a["kind"] != "acoustic" or v["kind"] != "vocoder":
            raise ValueError(
                "Choose an acoustic package and a separate universal vocoder package"
            )
        require_contract(v["mel"], a["mel"], "acoustic/vocoder mel")
        self.mel_config = MelConfig(**a["mel"])
        self.acoustic = construct(a, self.device).eval().requires_grad_(False)
        self.vocoder = construct(v, self.device).eval().requires_grad_(False)
        f = a["features"]
        self.extractor = extractor or FeatureExtractor(
            encoder,
            self.mel_config,
            self.device,
            layer=f["layer"],
            pitch=f["pitch_method"],
            pitch_path=pitch_path,
            threshold=f["pitch_threshold"],
            profile=f["profile"],
            context_seconds=f["context_seconds"],
            lookahead_seconds=f["lookahead_seconds"],
            packet_seconds=f["packet_seconds"],
        )
        require_contract(asdict(self.extractor.config), f, "runtime features")
        require_contract(asdict(self.extractor.mel_config), a["mel"], "runtime mel")
        c = self.vocoder.config
        if (c.sample_rate, c.hop_length, c.mel_dim) != (
            self.mel_config.sample_rate,
            self.mel_config.hop_length,
            self.mel_config.n_mels,
        ):
            raise ValueError("Vocoder dimensions disagree with its mel contract")
        if (self.acoustic.config.mel_dim, self.acoustic.config.content_dim) != (
            self.mel_config.n_mels,
            f["content_dim"],
        ):
            raise ValueError("Acoustic dimensions disagree with its feature contract")

    def controls(self, features, speaker=0, semitones=0):
        """Move aligned features to device, select target voice and shift F0 in semitones."""

        if not np.isfinite(semitones) or not -48 <= semitones <= 48:
            raise ValueError("Pitch shift must be finite and within four octaves")
        if not 0 <= speaker < len(self.acoustic_package["speakers"]):
            raise ValueError("Unknown target speaker")
        values = {
            key: torch.from_numpy(np.ascontiguousarray(value)).to(self.device)[None]
            for key, value in features.items()
            if key != "mel"
        }
        values["f0"] = values["f0"] * 2 ** (semitones / 12)
        values["speaker"] = torch.tensor([speaker], device=self.device)
        return values

    @torch.inference_mode()
    def convert(self, audio, speaker=0, semitones=0, steps=4, seed=0, ordinary=False):
        audio = np.asarray(audio, dtype=np.float32)
        if (
            self.extractor.config.profile == "bounded"
            and self.acoustic.config.causal
            and self.vocoder.config.causal
        ):
            live = LiveConverter(self, speaker, semitones, steps, seed, ordinary)
            pieces = []
            # Bounded chunks keep frontend/synthesis memory independent of file duration.
            size = round(
                self.mel_config.sample_rate * self.extractor.config.packet_seconds
            )
            for offset in range(0, len(audio), size):
                pieces.append(live.push(audio[offset : offset + size]))
            pieces.append(live.flush())
            return np.concatenate(pieces)
        features = self.extractor.extract(audio)
        x = self.controls(features, speaker, semitones)
        condition = self.acoustic.condition(
            x["content"],
            x["f0"],
            x["voiced"],
            x["energy"],
            x["speaker"],
            x["confidence"],
            x["confidence_valid"],
        )
        noise = coordinate_noise(len(features["f0"]), self.mel_config.n_mels, seed=seed)
        mel = self.acoustic.sample(
            condition,
            steps,
            torch.from_numpy(noise.T.copy()).to(self.device)[None],
            ordinary=ordinary,
        )
        excitation = coordinate_noise(
            mel.shape[-1] * self.mel_config.hop_length, seed=seed + 1
        )
        waveform = self.vocoder(
            mel,
            x["f0"],
            x["voiced"],
            noise=torch.from_numpy(excitation.T.copy()).to(self.device),
        )
        output = waveform[0, : len(audio)].float().cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite waveform; check the trained packages")
        return output

    def convert_file(self, source, destination, **options):
        import soundfile as sf

        source, destination = Path(source).resolve(), Path(destination).resolve()
        if source == destination:
            raise ValueError("Output must differ from input audio")
        audio = read_audio(source, self.mel_config.sample_rate)
        result = self.convert(audio, **options)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), result, self.mel_config.sample_rate, subtype="FLOAT")
        return str(destination)


class LiveConverter:
    """Keep one recording/session state across PCM packets and final flush."""

    def __init__(
        self, converter, speaker=0, semitones=0, steps=4, seed=0, ordinary=False
    ):
        if not converter.vocoder.config.causal:
            raise ValueError(
                "This vocoder supports file conversion only; live conversion requires a streaming-compatible vocoder"
            )
        if not 0 <= speaker < len(converter.acoustic_package["speakers"]):
            raise ValueError("Unknown target speaker")
        if not np.isfinite(semitones) or not -48 <= semitones <= 48 or seed < 0:
            raise ValueError("Invalid live pitch shift or noise seed")
        self.converter, self.speaker, self.semitones = converter, speaker, semitones
        self.frontend = FeatureStream(converter.extractor)
        self.acoustic = AcousticStream(converter.acoustic, steps, seed, ordinary)
        self.vocoder = VocoderStream(converter.vocoder, seed)
        self.reset()

    def reset(self):
        self.frontend.reset()
        self.acoustic.reset()
        self.vocoder.reset()
        self.input_samples = self.output_samples = 0
        self.closed = False

    @torch.inference_mode()
    def push(self, audio, final=False):
        if self.closed:
            raise ValueError("Session is closed; reset before another recording")
        self.input_samples += len(audio)
        packets = self.frontend.push(audio, final)
        pieces = []
        for packet in packets:
            x = self.converter.controls(packet, self.speaker, self.semitones)
            mel = self.acoustic.push(
                x["content"],
                x["f0"],
                x["voiced"],
                x["energy"],
                x["speaker"],
                x["confidence"],
                x["confidence_valid"],
            )
            pieces.append(
                self.vocoder.push(mel, x["f0"], x["voiced"]).float().cpu().numpy()[0]
            )
        if final and self.vocoder.mel is not None:
            c = self.converter.vocoder.config
            empty_mel = torch.empty(1, c.mel_dim, 0, device=self.converter.device)
            empty_f0 = torch.empty(1, 0, device=self.converter.device)
            pieces.append(
                self.vocoder.push(empty_mel, empty_f0, empty_f0, final=True)
                .float()
                .cpu()
                .numpy()[0]
            )
        output = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
        output = output[: max(0, self.input_samples - self.output_samples)]
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite live waveform")
        self.output_samples += len(output)
        self.closed = final
        return output

    def flush(self):
        return self.push(np.empty(0, dtype=np.float32), final=True)

    @property
    def latency(self):
        c, f = self.converter.mel_config, self.converter.extractor.config
        return {
            "frontend_packet_seconds": f.packet_seconds,
            "frontend_lookahead_seconds": f.lookahead_seconds,
            "vocoder_lookahead_frames": self.vocoder.right,
            "vocoder_lookahead_seconds": self.vocoder.right
            * c.hop_length
            / c.sample_rate,
            "pending_samples": self.input_samples - self.output_samples,
        }
