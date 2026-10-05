"""Frozen V3 frontend: audio -> aligned content, pitch, voicing and energy.

Encoder/pitch observations are aligned to physical mel-frame timestamps, not
stretched to match tensor lengths. File hashes and extractor identities bind
the cache/checkpoint contract. Offline features can use full context; bounded
features use declared packets, past context and lookahead for streaming parity.
Frontend weights stay frozen; cached extraction keeps them out of the training
activation graph and allows acoustic/vocoder training to use GPU memory.
"""

import hashlib
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from rvc.configs.neural import FeatureConfig, MelConfig
from rvc.lib.algorithm.acoustic.spectral import MelExtractor


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hash(path):
    root = Path(path)
    files = sorted(
        p for p in root.iterdir() if p.suffix in {".json", ".bin", ".safetensors"}
    )
    if not files:
        raise FileNotFoundError(f"No encoder weights/configuration in {root}")
    return hashlib.sha256(
        "\n".join(f"{p.name}:{file_hash(p)}" for p in files).encode()
    ).hexdigest()


def read_audio(path, sample_rate):
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite recording: {path}")
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        divisor = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, sr // divisor).astype(
            np.float32
        )
    return np.ascontiguousarray(audio)


class FeatureExtractor:
    """Own the pinned local encoder, pitch extractor and declared observation profile."""

    def __init__(
        self,
        encoder_dir,
        mel=MelConfig(),
        device="cpu",
        layer=12,
        pitch="swift",
        pitch_path=None,
        threshold=None,
        profile="offline",
        context_seconds=1.2,
        lookahead_seconds=0.2,
        packet_seconds=0.25,
    ):
        from transformers import HubertModel

        self.device = torch.device(device)
        root = Path(encoder_dir)
        digest = directory_hash(root)
        self.encoder = (
            HubertModel.from_pretrained(str(root), local_files_only=True)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        if layer > self.encoder.config.num_hidden_layers:
            raise ValueError("Extraction layer exceeds encoder depth")
        if pitch == "swift":
            import swift_f0

            self.pitch = swift_f0.SwiftF0()
            assets = sorted(Path(swift_f0.__file__).parent.rglob("*.onnx"))
            identity = (
                "swift-f0:"
                + version("swift-f0")
                + ":"
                + hashlib.sha256(
                    "".join(file_hash(p) for p in assets).encode()
                ).hexdigest()
            )
            threshold = 0.5 if threshold is None else threshold
        elif pitch == "rmvpe":
            if pitch_path is None:
                raise ValueError("RMVPE requires an explicit checkpoint path")
            from rvc.lib.predictors.RMVPE import RMVPE0Predictor

            self.pitch = RMVPE0Predictor(str(pitch_path), device=self.device)
            identity = "rmvpe:" + file_hash(pitch_path)
            threshold = 0.03 if threshold is None else threshold
        else:
            raise ValueError("Unknown pitch extractor")
        self.mel_config = mel
        self.config = FeatureConfig(
            encoder_id="hubert:" + digest[:16],
            encoder_hash=digest,
            layer=layer,
            content_dim=self.encoder.config.hidden_size,
            pitch_method=pitch,
            pitch_id=identity,
            pitch_threshold=threshold,
            profile=profile,
            context_seconds=context_seconds,
            lookahead_seconds=lookahead_seconds,
            packet_seconds=packet_seconds,
        )
        self.mel = MelExtractor(mel).to(self.device)
        receptive, stride = 1, 1
        for kernel, factor in zip(
            self.encoder.config.conv_kernel,
            self.encoder.config.conv_stride,
            strict=True,
        ):
            receptive += (kernel - 1) * stride
            stride *= factor
        self.encoder_receptive, self.encoder_stride = receptive, stride

    @torch.no_grad()
    def _observe(self, audio):
        padded = np.pad(audio, (0, max(5120 - len(audio), 0)))
        waveform = torch.from_numpy(padded).to(self.device)[None]
        output = self.encoder(waveform, output_hidden_states=True)
        content = output.hidden_states[self.config.layer][0].float().cpu().numpy()
        content_time = (
            np.arange(len(content)) * self.encoder_stride
            + (self.encoder_receptive - 1) / 2
        ) / 16000
        if self.config.pitch_method == "swift":
            result = self.pitch.detect(padded, 16000, fmin=32.7, fmax=1975.5)
            f0, confidence, pitch_time = (
                result.pitch_hz,
                result.confidence,
                result.timestamps,
            )
        else:
            mel = self.pitch.mel_extractor(waveform, center=True)
            salience = self.pitch.mel2hidden(mel)[0].float().cpu().numpy()
            f0 = self.pitch.decode(salience, thred=self.config.pitch_threshold)
            confidence = salience.max(axis=-1)
            pitch_time = np.arange(len(f0)) * 0.01
        return content_time, content, pitch_time, np.asarray(f0), np.asarray(confidence)

    def _align(self, observations, timestamps):
        ct, content, pt, pitch, confidence = observations
        # All content channels share timestamps. Compute interpolation indices
        # once instead of running 768 separate searches. Float64 arithmetic and
        # endpoint clamping preserve the cached np.interp feature convention.
        right = np.searchsorted(ct, timestamps, side="right").clip(0, len(ct) - 1)
        left = (right - 1).clip(0)
        span = ct[right] - ct[left]
        values = content.astype(np.float64, copy=False)
        slope = np.divide(
            values[right] - values[left],
            span[:, None],
            out=np.zeros((len(timestamps), content.shape[1]), dtype=np.float64),
            where=span[:, None] != 0,
        )
        aligned = values[left] + (timestamps - ct[left])[:, None] * slope
        aligned[timestamps <= ct[0]] = values[0]
        aligned[timestamps >= ct[-1]] = values[-1]
        # Nearest raw observations keep voicing decisions distinct from interpolation.
        positions = np.searchsorted(pt, timestamps).clip(0, len(pt) - 1)
        left = (positions - 1).clip(0)
        positions = np.where(
            np.abs(pt[left] - timestamps) <= np.abs(pt[positions] - timestamps),
            left,
            positions,
        )
        voiced = (confidence >= self.config.pitch_threshold) & (pitch > 0)
        observed = np.where(voiced[positions], pitch[positions], 0)
        if voiced.any():
            continuous = np.exp(
                np.interp(timestamps, pt[voiced], np.log(pitch[voiced]))
            )
        else:
            continuous = np.zeros(len(timestamps))
        return {
            "content": aligned,
            "f0": continuous,
            "observed_f0": observed,
            "voiced": voiced[positions].astype(np.float32),
            "confidence": np.interp(timestamps, pt, confidence),
            "confidence_valid": np.ones(len(timestamps), dtype=np.float32),
        }

    def packet(self, audio, origin, total_samples, packet_index, final=False):
        """A finalized bounded-context packet in absolute waveform coordinates."""
        from scipy.signal import resample_poly

        c, f = self.mel_config, self.config
        start = packet_index * f.packet_seconds
        stop = start + f.packet_seconds
        frame_start = max(0, math.ceil(start * c.sample_rate / c.hop_length - 0.5))
        frame_stop = min(
            c.frames(total_samples),
            math.ceil(stop * c.sample_rate / c.hop_length - 0.5),
        )
        if final and stop >= total_samples / c.sample_rate:
            frame_stop = c.frames(total_samples)
        timestamps = (
            (np.arange(frame_start, frame_stop) + 0.5) * c.hop_length / c.sample_rate
        )
        if not len(timestamps):
            return None
        divisor = math.gcd(c.sample_rate, 16000)
        up, down = 16000 // divisor, c.sample_rate // divisor
        left = max(0, int((start - f.context_seconds) * 16000))
        right = min(
            math.ceil(total_samples * 16000 / c.sample_rate),
            int((stop + f.lookahead_seconds) * 16000),
        )
        # Align resampling phases to the rational-rate grid and preserve its FIR halo.
        source_left = max(0, (left // up - 1) * down)
        source_right = min(total_samples, (math.ceil(right / up) + 1) * down)
        if source_left < origin or source_right > origin + len(audio):
            raise ValueError("Insufficient finalized audio/context for feature packet")
        window = audio[source_left - origin : source_right - origin]
        source = resample_poly(window, up, down).astype(np.float32)
        source_offset = source_left // down * up
        observations = self._observe(
            source[left - source_offset : right - source_offset]
        )
        result = self._align(observations, timestamps - left / 16000)
        first_sample, last_sample = (
            frame_start * c.hop_length,
            frame_stop * c.hop_length,
        )
        waveform = audio[
            first_sample - origin : min(last_sample, total_samples) - origin
        ]
        padded = np.pad(waveform, (0, last_sample - first_sample - len(waveform)))
        energy = np.sqrt(
            np.mean(padded.reshape(-1, c.hop_length).astype(np.float64) ** 2, axis=-1)
        )
        result["energy"] = np.log(np.maximum(energy, 1e-5))
        result["voiced"][energy < 1e-5] = 0
        result["observed_f0"][energy < 1e-5] = 0
        return {
            key: np.asarray(value, dtype=np.float32) for key, value in result.items()
        }

    @torch.no_grad()
    def extract(self, audio):
        from scipy.signal import resample_poly

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Feature extraction requires finite, nonempty mono audio")
        c = self.mel_config
        frames = c.frames(len(audio))
        timestamps = np.asarray(c.timestamps(frames))
        if self.config.profile == "offline":
            divisor = math.gcd(c.sample_rate, 16000)
            source = resample_poly(
                audio, 16000 // divisor, c.sample_rate // divisor
            ).astype(np.float32)
            result = self._align(self._observe(source), timestamps)
        else:
            packets = [
                self.packet(audio, 0, len(audio), i, final=True)
                for i in range(
                    math.ceil(len(audio) / c.sample_rate / self.config.packet_seconds)
                )
            ]
            packets = [p for p in packets if p is not None]
            result = {
                key: np.concatenate([p[key] for p in packets]) for key in packets[0]
            }
        padded = np.pad(audio, (0, frames * c.hop_length - len(audio)))
        energy = np.sqrt(
            np.mean(
                padded.reshape(frames, c.hop_length).astype(np.float64) ** 2, axis=-1
            )
        )
        result["energy"] = np.log(np.maximum(energy, 1e-5))
        silence = energy < 1e-5
        result["voiced"][silence], result["observed_f0"][silence] = 0, 0
        result["mel"] = (
            self.mel(torch.from_numpy(audio).to(self.device)[None])[0].cpu().numpy().T
        )
        return {
            key: np.asarray(value, dtype=np.float32) for key, value in result.items()
        }
