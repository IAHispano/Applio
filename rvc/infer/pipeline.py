import os
import gc
import sys
import torch
import torch.nn.functional as F
import torchcrepe
import faiss
import librosa
import numpy as np
from scipy import signal
from torch import Tensor

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE, SWIFT

import logging

logging.getLogger("faiss").setLevel(logging.WARNING)

FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)


class AudioProcessor:
    """
    A class for processing audio signals, specifically for adjusting RMS levels.
    """

    def change_rms(
        source_audio: np.ndarray,
        source_rate: int,
        target_audio: np.ndarray,
        target_rate: int,
        rate: float,
    ):
        """
        Adjust the RMS level of target_audio to match the RMS of source_audio, with a given blending rate.

        Args:
            source_audio: The source audio signal as a NumPy array.
            source_rate: The sampling rate of the source audio.
            target_audio: The target audio signal to adjust.
            target_rate: The sampling rate of the target audio.
            rate: The blending rate between the source and target RMS levels.
        """
        # Calculate RMS of both audio data
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        # Interpolate RMS to match target audio length
        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        # Adjust target audio RMS based on the source audio RMS
        adjusted_audio = (
            target_audio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )
        return adjusted_audio


class Autotune:
    """
    A class for applying autotune to a given fundamental frequency (F0) contour.
    """

    def __init__(self):
        """
        Initializes the Autotune class with a set of reference frequencies.
        """
        self.note_dict = [
            49.00,  # G1
            51.91,  # G#1 / Ab1
            55.00,  # A1
            58.27,  # A#1 / Bb1
            61.74,  # B1
            65.41,  # C2
            69.30,  # C#2 / Db2
            73.42,  # D2
            77.78,  # D#2 / Eb2
            82.41,  # E2
            87.31,  # F2
            92.50,  # F#2 / Gb2
            98.00,  # G2
            103.83,  # G#2 / Ab2
            110.00,  # A2
            116.54,  # A#2 / Bb2
            123.47,  # B2
            130.81,  # C3
            138.59,  # C#3 / Db3
            146.83,  # D3
            155.56,  # D#3 / Eb3
            164.81,  # E3
            174.61,  # F3
            185.00,  # F#3 / Gb3
            196.00,  # G3
            207.65,  # G#3 / Ab3
            220.00,  # A3
            233.08,  # A#3 / Bb3
            246.94,  # B3
            261.63,  # C4
            277.18,  # C#4 / Db4
            293.66,  # D4
            311.13,  # D#4 / Eb4
            329.63,  # E4
            349.23,  # F4
            369.99,  # F#4 / Gb4
            392.00,  # G4
            415.30,  # G#4 / Ab4
            440.00,  # A4
            466.16,  # A#4 / Bb4
            493.88,  # B4
            523.25,  # C5
            554.37,  # C#5 / Db5
            587.33,  # D5
            622.25,  # D#5 / Eb5
            659.25,  # E5
            698.46,  # F5
            739.99,  # F#5 / Gb5
            783.99,  # G5
            830.61,  # G#5 / Ab5
            880.00,  # A5
            932.33,  # A#5 / Bb5
            987.77,  # B5
            1046.50,  # C6
        ]

    def autotune_f0(self, f0, f0_autotune_strength):
        """
        Autotunes a given F0 contour by snapping each frequency to the closest reference frequency.

        Args:
            f0: The input F0 contour as a NumPy array.
        """
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = freq + (closest_note - freq) * f0_autotune_strength
        return autotuned_f0


class Pipeline:
    """
    The main pipeline class for performing voice conversion, including preprocessing, F0 estimation,
    voice conversion using a model, and post-processing.
    """

    def __init__(self, tgt_sr, config):
        """
        Initializes the Pipeline class with target sampling rate and configuration parameters.

        Args:
            tgt_sr: The target sampling rate for the output audio.
            config: A configuration object containing various parameters for the pipeline.
        """
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.tgt_sr = tgt_sr
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        self.autotune = Autotune()

    def get_f0(
        self,
        x,
        p_len,
        f0_method: str = "rmvpe",
        pitch: int = 0,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using various methods.

        Args:
            x: The input audio signal as a NumPy array.
            p_len: Desired length of the F0 output.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation (e.g., "crepe").
            f0_autotune: Whether to apply autotune to the F0 contour.
            proposed_pitch: whether to apply proposed pitch adjustment
            proposed_pitch_threshold: target frequency, 155.0 for male, 255.0 for female
        """
        if f0_method == "crepe":
            model = CREPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.window
            )
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "full")
            del model
        elif f0_method == "crepe-tiny":
            model = CREPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.window
            )
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "tiny")
            del model
        elif f0_method == "rmvpe":
            model = RMVPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.window
            )
            f0 = model.get_f0(x, filter_radius=0.03)
            del model
        elif f0_method == "fcpe":
            model = FCPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.window
            )
            f0 = model.get_f0(x, p_len, filter_radius=0.006)
            del model
        elif f0_method == "swift":
            model = SWIFT(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.window
            )
            f0 = model.get_f0(
                x, self.f0_min, self.f0_max, p_len, confidence_threshold=0.887
            )
            del model

        # f0 adjustments
        if f0_autotune is True:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        elif proposed_pitch is True:
            limit = 12
            # calculate median f0 of the audio
            valid_f0 = np.where(f0 > 0)[0]
            if len(valid_f0) < 2:
                # no valid f0 detected
                up_key = 0
            else:
                median_f0 = float(
                    np.median(np.interp(np.arange(len(f0)), valid_f0, f0[valid_f0]))
                )
                if median_f0 <= 0 or np.isnan(median_f0):
                    up_key = 0
                else:
                    # calculate proposed shift
                    up_key = max(
                        -limit,
                        min(
                            limit,
                            int(
                                np.round(
                                    12 * np.log2(proposed_pitch_threshold / median_f0)
                                )
                            ),
                        ),
                    )
            print("calculated pitch offset:", up_key)
            f0 *= pow(2, (pitch + up_key) / 12)
        else:
            f0 *= pow(2, pitch / 12)
        # quantizing f0 to 255 buckets to make coarse f0
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak

    def voice_conversion(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        """
        Performs voice conversion on a given audio segment.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio0: The input audio segment.
            pitch: Quantized F0 contour for pitch guidance.
            pitchf: Original F0 contour for pitch guidance.
            index: FAISS index for speaker embedding retrieval.
            big_npy: Speaker embeddings stored in a NumPy array.
            index_rate: Blending rate for speaker embedding retrieval.
            version: Model version (Keep to support old models).
            protect: Protection level for preserving the original pitch.
        """
        with torch.no_grad():
            pitch_guidance = pitch != None and pitchf != None
            # prepare source audio
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            assert feats.dim() == 1, feats.dim()
            feats = feats.view(1, -1).to(self.device)
            # extract features
            feats = model(feats)["last_hidden_state"]
            feats = (
                model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            )
            # make a copy for pitch guidance and protection
            feats0 = feats.clone() if pitch_guidance else None
            if (
                index
            ):  # set by parent function, only true if index is available, loaded, and index rate > 0
                feats = self._retrieve_speaker_embeddings(
                    feats, index, big_npy, index_rate
                )
            # feature upsampling
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            # adjust the length if the audio is short
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
                # Pitch protection blending
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                        1 - pitchff.unsqueeze(-1)
                    )
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None
            p_len = torch.tensor([p_len], device=self.device).long()
            audio1 = (
                (net_g.infer(feats.float(), p_len, pitch, pitchf.float(), sid)[0][0, 0])
                .data.cpu()
                .float()
                .numpy()
            )
            # clean up
            del feats, feats0, p_len
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return audio1

    def _retrieve_speaker_embeddings(self, feats, index, big_npy, index_rate):
        npy = feats[0].cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats = (
            torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
            + (1 - index_rate) * feats
        )
        return feats

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
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio: The input audio signal.
            input_audio_path: Path to the input audio file.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation.
            file_index: Path to the FAISS index file for speaker embedding retrieval.
            index_rate: Blending rate for speaker embedding retrieval.
            pitch_guidance: Whether to use pitch guidance during voice conversion.
            tgt_sr: Target sampling rate for the output audio.
            resample_sr: Resampling rate for the output audio.
            version: Model version.
            protect: Protection level for preserving the original pitch.
            hop_length: Hop length for F0 estimation methods.
            f0_autotune: Whether to apply autotune to the F0 contour.
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
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        if pitch_guidance:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                f0_method,
                pitch,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        for t in opt_ts:
            t = t // self.window * self.window
            if pitch_guidance:
                audio_opt.append(
                    self.voice_conversion(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.voice_conversion(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if pitch_guidance:
            audio_opt.append(
                self.voice_conversion(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.voice_conversion(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(
                audio, self.sample_rate, audio_opt, self.tgt_sr, volume_envelope
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max
        if pitch_guidance:
            del pitch, pitchf
        del sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
