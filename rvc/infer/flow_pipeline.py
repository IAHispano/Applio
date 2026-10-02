import os
import sys

import faiss
import numpy as np
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.infer.pipeline import AudioProcessor, Pipeline
from rvc.lib.algorithm.rectified_flow import build_flow
from rvc.lib.algorithm.rectified_flow_features import (
    TENSION_SMOOTH_SECONDS,
    aperiodicity,
    f0_to_mel_rate,
    frame_energy,
    mel_frames,
    smooth_curve,
    tension,
    to_mel_rate,
    upsample_content,
)
from rvc.lib.algorithm.vocoders import find_vocoder, load_vocoder

# Sampling settings
FLOW_STEPS = 16
FLOW_SAMPLER = "euler"
CFG_SCALE = 2.0
CONTENT_GUIDANCE = 0.1
GUIDANCE_RESCALE = 0.7

# The embedder's receptive field and stride, in samples at 16 kHz
EMBEDDER_FIELD = 400
EMBEDDER_STRIDE = 320
# Content frames (20 ms) per embedder pass and the context on each side
EMBEDDER_CHUNK = 1500
EMBEDDER_CONTEXT = 100
# Mel frames per flow pass and the overlap crossfaded between passes
FLOW_CHUNK = 3000
FLOW_OVERLAP = 100
# Mel frames per vocoder pass, the context rendered on each side and the
# crossfade at each join
VOCODER_CHUNK = 6000
VOCODER_CONTEXT = 50
VOCODER_CROSSFADE = 4


class FlowSynthesizer(torch.nn.Module):
    """
    A Rectified Flow model and the vocoder that renders its mel spectrogram.

    Args:
        cpt (dict): The checkpoint of the exported flow model.
    """

    def __init__(self, cpt):
        super().__init__()
        self.data = cpt["config"]["data"]
        self.flow = build_flow(cpt["config"], cpt["speaker_count"])
        self.flow.load_state_dict(cpt["model"])

        vocoder_path = find_vocoder(cpt.get("vocoder", ""))
        if not vocoder_path:
            raise FileNotFoundError(
                "No vocoder found for the Rectified Flow model. Place one in rvc/models/pretraineds/rectified-flow."
            )
        self.vocoder = load_vocoder(vocoder_path, self.data)


class FlowPipeline(Pipeline):
    """
    The pipeline for Rectified Flow models: content, pitch, loudness,
    breathiness and tension of the input go through the flow to a mel
    spectrogram, which the vocoder renders.
    """

    def get_content(self, model, feats):
        """
        Extracts the content features in passes with context on each side, so
        long inputs fit in memory.

        Args:
            model: The feature extractor model.
            feats: The input audio, shape (1, samples).
        """
        frames = (feats.shape[-1] - EMBEDDER_FIELD) // EMBEDDER_STRIDE + 1
        parts = []
        for start in range(0, frames, EMBEDDER_CHUNK):
            stop = min(frames, start + EMBEDDER_CHUNK)
            left = max(0, start - EMBEDDER_CONTEXT)
            right = min(frames, stop + EMBEDDER_CONTEXT)
            window = feats[
                :,
                left * EMBEDDER_STRIDE : (right - 1) * EMBEDDER_STRIDE + EMBEDDER_FIELD,
            ]
            part = model(window)["last_hidden_state"].float()
            parts.append(part[:, start - left : stop - left])
        return torch.cat(parts, 1)

    def get_pitch(
        self,
        audio,
        p_len,
        f0_method,
        pitch,
        f0_autotune,
        f0_autotune_strength,
        proposed_pitch,
        proposed_pitch_threshold,
    ):
        """
        Estimates the F0 of the input, which its breathiness and tension are
        measured against, and the F0 to sing.

        Args:
            audio: The input audio signal as a NumPy array.
            p_len: Desired length of the F0 output.
            f0_method: Method to use for F0 estimation.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_autotune: Whether to apply autotune to the F0 contour.
            f0_autotune_strength: Strength of the autotune.
            proposed_pitch: whether to apply proposed pitch adjustment
            proposed_pitch_threshold: target frequency, 155.0 for male, 255.0 for female
        """
        _, source_f0 = self.get_f0(audio, p_len, f0_method, 0)
        source_f0 = np.asarray(source_f0, dtype=np.float32)[:p_len]
        source_f0 = np.pad(source_f0, (0, p_len - source_f0.shape[0]))

        if f0_autotune is True:
            f0 = self.autotune.autotune_f0(source_f0.copy(), f0_autotune_strength)
        else:
            up_key = 0
            valid_f0 = np.where(source_f0 > 0)[0]
            if proposed_pitch is True and len(valid_f0) >= 2:
                median_f0 = float(
                    np.median(
                        np.interp(np.arange(p_len), valid_f0, source_f0[valid_f0])
                    )
                )
                up_key = 12 * np.log2(proposed_pitch_threshold / median_f0)
                up_key = max(-12, min(12, int(np.round(up_key))))
                print("calculated pitch offset:", up_key)
            f0 = source_f0 * pow(2, (pitch + up_key) / 12)
        return source_f0, np.asarray(f0, dtype=np.float32)

    def sample_mel(self, flow, feats, pitchf, energy, breathiness, strain, sid):
        """
        Samples the normalised mel spectrogram in overlapping passes, crossfaded.

        Args:
            flow: The flow model.
            feats: Content features at the mel frame rate.
            pitchf: F0 contour at the mel frame rate.
            energy: Loudness curve at the mel frame rate.
            breathiness: Aperiodicity curve at the mel frame rate.
            strain: Tension curve at the mel frame rate, None for a model without it.
            sid: Speaker ID for the target voice.
        """
        frames = feats.shape[1]
        noise = torch.randn(1, flow.n_mels, frames, device=self.device)
        mel = torch.zeros_like(noise)
        weight = torch.zeros(1, 1, frames, device=self.device)
        start = 0
        while start < frames:
            stop = min(frames, start + FLOW_CHUNK)
            mask = torch.ones(1, 1, stop - start, device=self.device)
            part = flow.sample(
                feats[:, start:stop],
                pitchf[:, start:stop],
                energy[:, start:stop],
                sid,
                mask,
                steps=FLOW_STEPS,
                method=FLOW_SAMPLER,
                cfg_scale=CFG_SCALE,
                content_guidance=CONTENT_GUIDANCE,
                guidance_rescale=GUIDANCE_RESCALE,
                noise=noise[..., start:stop],
                breathiness=breathiness[:, start:stop],
                tension=None if strain is None else strain[:, start:stop],
            )
            ramp = torch.ones(stop - start, device=self.device)
            fade = min(FLOW_OVERLAP, stop - start)
            if start > 0:
                ramp[:fade] = torch.linspace(0.0, 1.0, fade, device=self.device)
            if stop < frames:
                ramp[-fade:] = torch.minimum(
                    ramp[-fade:], torch.linspace(1.0, 0.0, fade, device=self.device)
                )
            mel[..., start:stop] += part * ramp
            weight[..., start:stop] += ramp
            if stop == frames:
                break
            start = stop - FLOW_OVERLAP
        return mel / weight.clamp_min(1e-4)

    def render(self, vocoder, mel, pitchf, hop_length):
        """
        Renders the waveform of a mel spectrogram in passes. Every pass starts
        its excitation at phase zero, so the joins are crossfaded.

        Args:
            vocoder: The vocoder model.
            mel: Normalised mel spectrogram.
            pitchf: F0 contour at the mel frame rate.
            hop_length: Hop length of the mel spectrogram.
        """
        frames = mel.shape[-1]
        half = VOCODER_CROSSFADE * hop_length // 2
        output = torch.zeros(frames * hop_length, device=self.device)
        for start in range(0, frames, VOCODER_CHUNK):
            stop = min(frames, start + VOCODER_CHUNK)
            left = max(0, start - VOCODER_CONTEXT)
            right = min(frames, stop + VOCODER_CONTEXT)
            audio = vocoder(mel[..., left:right], pitchf[:, left:right])[0, 0].float()
            offset = left * hop_length
            position = torch.arange(
                offset, offset + audio.shape[0], device=self.device
            )
            weight = torch.ones_like(audio)
            if start > 0:
                fade_in = (position - (start * hop_length - half)) / (2 * half)
                weight = weight * fade_in.clamp(0, 1)
            if stop < frames:
                fade_out = ((stop * hop_length + half) - position) / (2 * half)
                weight = weight * fade_out.clamp(0, 1)
            output[offset : offset + audio.shape[0]] += audio * weight
        return output.cpu().numpy()

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        f0_autotune,
        f0_autotune_strength,
        proposed_pitch,
        proposed_pitch_threshold,
    ):
        """
        The main pipeline function for performing voice conversion.

        Args:
            model: The feature extractor model.
            net_g: The flow model and its vocoder.
            sid: Speaker ID for the target voice.
            audio: The input audio signal.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation.
            file_index: Path to the FAISS index file for speaker embedding retrieval.
            index_rate: Blending rate for speaker embedding retrieval.
            pitch_guidance: Unused, a flow model always takes the pitch.
            volume_envelope: Blending rate of the input RMS into the output.
            version: Unused, kept for the signature of the RVC pipeline.
            protect: Protection level for the unvoiced frames against the index.
            f0_autotune: Whether to apply autotune to the F0 contour.
            f0_autotune_strength: Strength of the autotune.
            proposed_pitch: whether to apply proposed pitch adjustment
            proposed_pitch_threshold: target frequency, 155.0 for male, 255.0 for female
        """
        if file_index != "" and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                print(f"An error occurred reading the FAISS index: {error}")
                index = big_npy = None
        else:
            index = big_npy = None

        data = net_g.data
        sample_rate, hop_length = data["sample_rate"], data["hop_length"]

        # The loudness input is absolute, so the input is brought to the peak
        # the training data has and the output scaled back
        peak = float(np.abs(audio).max())
        gain = 0.95 / peak if peak > 0 else 1.0
        restore = 1.0 / max(gain, 1.0)
        audio = (audio * gain).astype(np.float32)

        if audio.shape[0] < EMBEDDER_FIELD + self.window:
            return np.zeros(
                round(audio.shape[0] * sample_rate / self.sample_rate), dtype=np.float32
            )

        with torch.no_grad():
            source = torch.from_numpy(audio).view(1, -1).to(self.device)
            # extract features
            feats = self.get_content(model, source)
            feats0 = feats
            if index:
                feats = self._retrieve_speaker_embeddings(
                    feats, index, big_npy, index_rate
                ).float()
            # feature upsampling
            feats = upsample_content(feats[0], data["content_interpolation"])
            feats0 = upsample_content(feats0[0], data["content_interpolation"])
            p_len = min(audio.shape[0] // self.window, feats.shape[0])
            feats, feats0 = feats[None, :p_len], feats0[None, :p_len]

            source_pitchf, pitchf = self.get_pitch(
                audio,
                p_len,
                f0_method,
                pitch,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
            )
            source_pitchf = torch.from_numpy(source_pitchf).view(1, -1).to(self.device)
            pitchf = torch.from_numpy(pitchf).view(1, -1).to(self.device)

            energy = smooth_curve(frame_energy(source, self.sample_rate, p_len))
            breathiness = smooth_curve(
                aperiodicity(source, self.sample_rate, source_pitchf, p_len)
            )
            strain = None
            if net_g.flow.encoder.tension is not None:
                strain = smooth_curve(
                    tension(source, self.sample_rate, source_pitchf, p_len),
                    TENSION_SMOOTH_SECONDS,
                )

            # everything above is at 100 frames per second, the mel has its own rate
            frames = mel_frames(p_len, sample_rate, hop_length)
            feats = to_mel_rate(feats, frames, sample_rate, hop_length)
            feats0 = to_mel_rate(feats0, frames, sample_rate, hop_length)
            pitchf = f0_to_mel_rate(pitchf, frames, sample_rate, hop_length)
            energy = to_mel_rate(
                energy.unsqueeze(-1), frames, sample_rate, hop_length
            )[..., 0]
            breathiness = to_mel_rate(
                breathiness.unsqueeze(-1), frames, sample_rate, hop_length
            )[..., 0]
            if strain is not None:
                strain = to_mel_rate(
                    strain.unsqueeze(-1), frames, sample_rate, hop_length
                )[..., 0]

            # Pitch protection blending
            if index and protect < 0.5:
                pitchff = torch.where(pitchf > 0, 1.0, float(protect)).unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)

            sid = torch.tensor([sid], device=self.device).long()
            mel = self.sample_mel(
                net_g.flow, feats, pitchf, energy, breathiness, strain, sid
            )
            audio_opt = self.render(net_g.vocoder, mel, pitchf, hop_length) * restore

            # clean up
            del feats, feats0, mel, source
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(
                audio * restore,
                self.sample_rate,
                audio_opt,
                self.tgt_sr,
                volume_envelope,
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max
        return audio_opt
