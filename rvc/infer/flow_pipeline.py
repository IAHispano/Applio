import os
import sys

import faiss
import numpy as np
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.infer.pipeline import AudioProcessor, Pipeline
from rvc.lib.algorithm.rectified_flow import Conditioning, build_flow, match_inputs
from rvc.lib.algorithm.rectified_flow.features import (
    TENSION_SMOOTH_SECONDS,
    aperiodicity,
    curve_to_mel_rate,
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
GUIDANCE_UNTIL = 1.0

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
        self.flow.load_state_dict(match_inputs(cpt["model"], self.flow))

        self.default_vocoder = find_vocoder(cpt.get("vocoder", ""))
        if not self.default_vocoder:
            raise FileNotFoundError(
                "No vocoder found for the Rectified Flow model. Place one in rvc/models/pretraineds/rectified-flow."
            )
        self.vocoder_path = self.default_vocoder
        self.vocoder = load_vocoder(self.vocoder_path, self.data)

    def set_vocoder(self, path=""):
        """
        Switches the vocoder that renders the mel spectrogram.

        Args:
            path (str): Path to the vocoder, empty for the one the model was trained with.
        """
        path = path or self.default_vocoder
        if path != self.vocoder_path:
            device = next(self.flow.parameters()).device
            self.vocoder = load_vocoder(path, self.data).to(device).float()
            self.vocoder_path = path


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

    def get_inputs(
        self,
        model,
        net_g,
        audio,
        sid,
        pitch,
        f0_method,
        index,
        big_npy,
        index_rate,
        protect,
        f0_autotune,
        f0_autotune_strength,
        proposed_pitch,
        proposed_pitch_threshold,
        audio_full=None,
    ):
        """
        Extracts the inputs of the flow from the audio, at the mel frame rate.

        Args:
            model: The feature extractor model.
            net_g: The flow model and its vocoder.
            audio: The input audio signal as a NumPy array.
            sid: Speaker ID for the target voice.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation.
            index: The FAISS index for speaker embedding retrieval, None for no retrieval.
            big_npy: The vectors of the FAISS index.
            index_rate: Blending rate for speaker embedding retrieval.
            protect: Protection level for the unvoiced frames against the index.
            f0_autotune: Whether to apply autotune to the F0 contour.
            f0_autotune_strength: Strength of the autotune.
            proposed_pitch: whether to apply proposed pitch adjustment
            proposed_pitch_threshold: target frequency, 155.0 for male, 255.0 for female
            audio_full: The input audio at the sampling rate of the model, which its loudness, breathiness and tension are measured from.
        """
        data = net_g.data
        sample_rate, hop_length = data["sample_rate"], data["hop_length"]
        source = torch.from_numpy(audio).float().view(1, -1).to(self.device)

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
        source_pitchf = torch.from_numpy(source_pitchf).float().view(1, -1).to(self.device)
        pitchf = torch.from_numpy(pitchf).float().view(1, -1).to(self.device)

        # The curves are measured as in training, which reads them from the
        # audio at the sampling rate of the model: at 16 kHz the loudness of a
        # sibilant has lost what it carries above 8 kHz
        curves, curves_sr = source, self.sample_rate
        if audio_full is not None:
            curves = torch.from_numpy(audio_full).float().view(1, -1).to(self.device)
            curves_sr = sample_rate
        energy = smooth_curve(frame_energy(curves, curves_sr, p_len))
        breathiness = smooth_curve(
            aperiodicity(curves, curves_sr, source_pitchf, p_len)
        )
        strain = None
        if net_g.flow.encoder.tension is not None:
            strain = smooth_curve(
                tension(curves, curves_sr, source_pitchf, p_len),
                TENSION_SMOOTH_SECONDS,
            )

        # everything above is at 100 frames per second, the mel has its own rate
        frames = mel_frames(p_len, sample_rate, hop_length)
        feats = to_mel_rate(feats, frames, sample_rate, hop_length)
        feats0 = to_mel_rate(feats0, frames, sample_rate, hop_length)
        pitchf = f0_to_mel_rate(pitchf, frames, sample_rate, hop_length)
        energy = curve_to_mel_rate(energy, frames, sample_rate, hop_length)
        breathiness = curve_to_mel_rate(breathiness, frames, sample_rate, hop_length)
        if strain is not None:
            strain = curve_to_mel_rate(strain, frames, sample_rate, hop_length)

        # Pitch protection blending
        if index and protect < 0.5:
            pitchff = torch.where(pitchf > 0, 1.0, float(protect)).unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)

        return Conditioning(
            content=feats,
            f0=pitchf,
            energy=energy,
            speaker=torch.tensor([sid], device=self.device).long(),
            mask=torch.ones(1, 1, frames, device=self.device),
            breathiness=breathiness,
            tension=strain,
        )

    def sample_mel(
        self,
        flow,
        inputs,
        steps,
        cfg_scale,
        content_guidance,
        sampler=FLOW_SAMPLER,
        guidance_rescale=GUIDANCE_RESCALE,
        guidance_until=GUIDANCE_UNTIL,
    ):
        """
        Samples the normalised mel spectrogram in overlapping passes, crossfaded.

        Args:
            flow: The flow model.
            inputs: The inputs of the flow, at the mel frame rate.
            steps: Number of sampling steps.
            cfg_scale: Guidance scale towards the speaker.
            content_guidance: Guidance scale towards the content.
            sampler: Sampling method, "euler" or "heun".
            guidance_rescale: Pull of the guided output's level back to the unguided one's.
            guidance_until: Flow time the guidances apply until, 1 for the whole sampling.
        """
        frames = inputs.content.shape[1]
        noise = torch.randn(1, flow.n_mels, frames, device=self.device)
        mel = torch.zeros_like(noise)
        weight = torch.zeros(1, 1, frames, device=self.device)
        start = 0
        while start < frames:
            stop = min(frames, start + FLOW_CHUNK)
            part = flow.sample(
                inputs.crop(start, stop),
                steps=steps,
                method=sampler,
                cfg_scale=cfg_scale,
                content_guidance=content_guidance,
                guidance_rescale=guidance_rescale,
                guidance_interval=(0.0, guidance_until),
                noise=noise[..., start:stop],
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
        audio_full=None,
        steps=FLOW_STEPS,
        cfg_scale=CFG_SCALE,
        content_guidance=CONTENT_GUIDANCE,
        sampler=FLOW_SAMPLER,
        guidance_rescale=GUIDANCE_RESCALE,
        guidance_until=GUIDANCE_UNTIL,
        formant_shift=0.0,
        tension_strength=1.0,
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
            audio_full: The input audio at the sampling rate of the model, None to measure its curves at 16 kHz.
            steps: Number of sampling steps.
            cfg_scale: Guidance scale towards the speaker.
            content_guidance: Guidance scale towards the content.
            sampler: Sampling method, "euler" or "heun".
            guidance_rescale: Pull of the guided output's level back to the unguided one's.
            guidance_until: Flow time the guidances apply until, 1 for the whole sampling.
            formant_shift: Shift of the formants in semitones, apart from the pitch.
            tension_strength: Scale of the tension of the input, 0 leaves the voice at its own.
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

        if audio.shape[0] < EMBEDDER_FIELD + self.window:
            return np.zeros(
                round(audio.shape[0] * sample_rate / self.sample_rate), dtype=np.float32
            )

        with torch.no_grad():
            inputs = self.get_inputs(
                model,
                net_g,
                audio,
                sid,
                pitch,
                f0_method,
                index,
                big_npy,
                index_rate,
                protect,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
                audio_full,
            )
            key_shift = torch.full((1,), float(formant_shift), device=self.device)
            inputs = inputs._replace(key_shift=key_shift)
            if inputs.tension is not None:
                inputs = inputs._replace(tension=inputs.tension * tension_strength)
            mel = self.sample_mel(
                net_g.flow,
                inputs,
                int(steps),
                cfg_scale,
                content_guidance,
                sampler,
                guidance_rescale,
                guidance_until,
            )
            audio_opt = self.render(net_g.vocoder, mel, inputs.f0, hop_length)

            # clean up
            del inputs, mel
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(
                audio,
                self.sample_rate,
                audio_opt,
                self.tgt_sr,
                volume_envelope,
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max
        return audio_opt
