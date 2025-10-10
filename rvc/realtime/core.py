import os
import sys
import time
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
import numpy as np

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.utils.torch import circular_write
from rvc.realtime.utils.vad import VADProcessor
from rvc.realtime.pipeline import create_pipeline

SAMPLE_RATE = 16000
AUDIO_SAMPLE_RATE = 48000


class Realtime:
    def __init__(
        self,
        model_path: str = None,
        index_path: str = None,
        f0_method: str = "rmvpe",
        embedder_model: str = None,
        embedder_model_custom: str = None,
        silent_threshold: int = 0,
        vad_enabled: bool = False,
        vad_sensitivity: int = 3,
        vad_frame_ms: int = 30,
        sid: int = 0,
        # device: str = "cuda",
    ):
        self.sample_rate = SAMPLE_RATE
        self.convert_buffer = None
        self.pitch_buffer = None
        self.pitchf_buffer = None
        self.return_length = 0
        self.skip_head = 0
        self.silence_front = 0
        # Convert dB to RMS
        self.input_sensitivity = 10 ** (silent_threshold / 20)
        self.window_size = self.sample_rate // 100
        self.dtype = torch.float32  # torch.float16 if config.is_half else torch.float32

        self.vad = (
            VADProcessor(
                sensitivity_mode=vad_sensitivity,
                sample_rate=self.sample_rate,
                frame_duration_ms=vad_frame_ms,
            )
            if vad_enabled
            else None
        )
        # Create conversion pipelines
        self.pipeline = create_pipeline(
            model_path,
            index_path,
            f0_method,
            embedder_model,
            embedder_model_custom,
            # device,
            sid,
        )
        self.device = self.pipeline.device
        # Resampling of inputs and outputs.
        self.resample_in = tat.Resample(
            orig_freq=AUDIO_SAMPLE_RATE, new_freq=self.sample_rate, dtype=torch.float32
        ).to(self.device)
        self.resample_out = tat.Resample(
            orig_freq=self.pipeline.tgt_sr,
            new_freq=AUDIO_SAMPLE_RATE,
            dtype=torch.float32,
        ).to(self.device)

    def realloc(
        self,
        block_frame: int,
        extra_frame: int,
        crossfade_frame: int,
        sola_search_frame: int,
    ):
        # Calculate frame sizes based on DEVICE sample rate (f.e., 48000Hz) and convert to 16000Hz
        block_frame_16k = int(block_frame / AUDIO_SAMPLE_RATE * self.sample_rate)
        crossfade_frame_16k = int(
            crossfade_frame / AUDIO_SAMPLE_RATE * self.sample_rate
        )
        sola_search_frame_16k = int(
            sola_search_frame / AUDIO_SAMPLE_RATE * self.sample_rate
        )
        extra_frame_16k = int(extra_frame / AUDIO_SAMPLE_RATE * self.sample_rate)

        convert_size_16k = (
            block_frame_16k
            + sola_search_frame_16k
            + extra_frame_16k
            + crossfade_frame_16k
        )
        if (
            modulo := convert_size_16k % self.window_size
        ) != 0:  # Compensate for truncation due to hop size in model output.
            convert_size_16k = convert_size_16k + (self.window_size - modulo)
        self.convert_feature_size_16k = convert_size_16k // self.window_size

        self.skip_head = extra_frame_16k // self.window_size
        self.return_length = self.convert_feature_size_16k - self.skip_head
        self.silence_front = (
            extra_frame_16k - (self.window_size * 5) if self.silence_front else 0
        )
        # Audio buffer to measure volume between chunks
        audio_buffer_size = block_frame_16k + crossfade_frame_16k
        self.audio_buffer = torch.zeros(
            audio_buffer_size, dtype=self.dtype, device=self.device
        )
        # Audio buffer for conversion without silence
        self.convert_buffer = torch.zeros(
            convert_size_16k, dtype=self.dtype, device=self.device
        )
        # Additional +1 is to compensate for pitch extraction algorithm
        # that can output additional feature.
        self.pitch_buffer = torch.zeros(
            self.convert_feature_size_16k + 1, dtype=torch.int64, device=self.device
        )
        self.pitchf_buffer = torch.zeros(
            self.convert_feature_size_16k + 1, dtype=self.dtype, device=self.device
        )

    def inference(
        self,
        audio_input: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized.")

        # Input audio is always float32
        audio_input_16k = self.resample_in(
            torch.as_tensor(audio_input, dtype=torch.float32, device=self.device)
        ).to(self.dtype)
        circular_write(audio_input_16k, self.audio_buffer)

        vol_t = torch.sqrt(torch.square(self.audio_buffer).mean())
        vol = max(vol_t.item(), 0)

        if self.vad is not None:
            is_speech = self.vad.is_speech(audio_input_16k.cpu().numpy().copy())
            if not is_speech:
                # Busy wait to keep power manager happy and clocks stable. Running pipeline on-demand seems to lag when the delay between
                # voice changer activation is too high.
                # https://forums.developer.nvidia.com/t/why-kernel-calculate-speed-got-slower-after-waiting-for-a-while/221059/9
                self.pipeline.voice_conversion(
                    self.convert_buffer,
                    self.pitch_buffer,
                    self.pitchf_buffer,
                    f0_up_key,
                    index_rate,
                    self.convert_feature_size_16k,
                    self.silence_front,
                    self.skip_head,
                    self.return_length,
                    protect,
                    volume_envelope,
                    f0_autotune,
                    f0_autotune_strength,
                    proposed_pitch,
                    proposed_pitch_threshold,
                )
                return None, vol

        if vol < self.input_sensitivity:
            # Busy wait to keep power manager happy and clocks stable. Running pipeline on-demand seems to lag when the delay between
            # voice changer activation is too high.
            # https://forums.developer.nvidia.com/t/why-kernel-calculate-speed-got-slower-after-waiting-for-a-while/221059/9
            self.pipeline.voice_conversion(
                self.convert_buffer,
                self.pitch_buffer,
                self.pitchf_buffer,
                f0_up_key,
                index_rate,
                self.convert_feature_size_16k,
                self.silence_front,
                self.skip_head,
                self.return_length,
                protect,
                volume_envelope,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
            )

            return None, vol

        circular_write(audio_input_16k, self.convert_buffer)

        audio_model = self.pipeline.voice_conversion(
            self.convert_buffer,
            self.pitch_buffer,
            self.pitchf_buffer,
            f0_up_key,
            index_rate,
            self.convert_feature_size_16k,
            self.silence_front,
            self.skip_head,
            self.return_length,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
        )

        audio_out: torch.Tensor = self.resample_out(audio_model * torch.sqrt(vol_t))
        return audio_out, vol

    def __del__(self):
        del self.pipeline


class VoiceChanger:
    def __init__(
        self,
        read_chunk_size: int,
        cross_fade_overlap_size: float,
        extra_convert_size: float,
        model_path: str = None,
        index_path: str = None,
        f0_method: str = "rmvpe",
        embedder_model: str = None,
        embedder_model_custom: str = None,
        silent_threshold: int = 0,
        vad_enabled: bool = False,
        vad_sensitivity: int = 3,
        vad_frame_ms: int = 30,
        sid: int = 0,
        # device: str = "cuda",
    ):
        self.block_frame = read_chunk_size * 128
        self.crossfade_frame = int(cross_fade_overlap_size * AUDIO_SAMPLE_RATE)
        self.extra_frame = int(extra_convert_size * AUDIO_SAMPLE_RATE)
        self.sola_search_frame = AUDIO_SAMPLE_RATE // 100
        self.sola_buffer = None
        self.vc_model = Realtime(
            model_path,
            index_path,
            f0_method,
            embedder_model,
            embedder_model_custom,
            silent_threshold,
            vad_enabled,
            vad_sensitivity,
            vad_frame_ms,
            sid,
            # device
        )
        self.device = self.vc_model.device
        self.vc_model.realloc(
            self.block_frame,
            self.extra_frame,
            self.crossfade_frame,
            self.sola_search_frame,
        )
        self.generate_strength()

    def generate_strength(self):
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.crossfade_frame,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )

        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        # The size will change from the previous result, so the record will be deleted.
        self.sola_buffer = torch.zeros(
            self.crossfade_frame, device=self.device, dtype=torch.float32
        )

    def process_audio(
        self,
        audio_input: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        block_size = audio_input.shape[0]

        audio, vol = self.vc_model.inference(
            audio_input,
            f0_up_key,
            index_rate,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
        )

        if audio is None:
            # In case there's an actual silence - send full block with zeros
            return np.zeros(block_size, dtype=np.float32), vol

        conv_input = audio[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.crossfade_frame, device=self.device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

        audio = audio[sola_offset:]
        audio[: self.crossfade_frame] *= self.fade_in_window
        audio[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer[:] = audio[block_size : block_size + self.crossfade_frame]
        return audio[:block_size].detach().cpu().numpy(), vol

    @torch.no_grad()
    def on_request(
        self,
        audio_input: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        if self.vc_model is None:
            raise RuntimeError("Voice Changer is not selected.")

        start = (
            time.perf_counter()
        )  # Using perf_counter to measure real-time voice conversion latency.
        result, vol = self.process_audio(
            audio_input,
            f0_up_key,
            index_rate,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
        )
        end = time.perf_counter()

        return result, vol, [0, (end - start) * 1000, 0]
