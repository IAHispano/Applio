import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from rvc.realtime.audio import Audio
from rvc.realtime.core import AUDIO_SAMPLE_RATE
from rvc.realtime.worker import VoiceChangerWorker


class AudioCallbacks:
    def __init__(
        self,
        pass_through: bool = False,
        read_chunk_size: int = 192,
        cross_fade_overlap_size: float = 0.1,
        extra_convert_size: float = 0.5,
        model_path: str = None,
        index_path: str = None,
        f0_method: str = "rmvpe",
        embedder_model: str = None,
        embedder_model_custom: str = None,
        silent_threshold: int = -90,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        input_audio_gain: float = 1.0,
        output_audio_gain: float = 1.0,
        monitor_audio_gain: float = 1.0,
        monitor: bool = False,
        vad_enabled: bool = False,
        vad_sensitivity: int = 3,
        vad_frame_ms: int = 30,
        sid: int = 0,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        post_process: bool = False,
        record_audio: bool = False,
        record_audio_path: str = None,
        export_format: str = "WAV",
        **kwargs,
        # device: str = "cuda",
    ):
        self.pass_through = pass_through
        self._last_output = None
        self._last_vol = 0

        vc_kwargs = dict(
            read_chunk_size=read_chunk_size,
            cross_fade_overlap_size=cross_fade_overlap_size,
            extra_convert_size=extra_convert_size,
            model_path=model_path,
            index_path=index_path,
            f0_method=f0_method,
            embedder_model=embedder_model,
            embedder_model_custom=embedder_model_custom,
            silent_threshold=silent_threshold,
            vad_enabled=vad_enabled,
            vad_sensitivity=vad_sensitivity,
            vad_frame_ms=vad_frame_ms,
            sid=sid,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            post_process=post_process,
            record_audio=record_audio,
            record_audio_path=record_audio_path,
            export_format=export_format,
            **kwargs,
        )
        self.vc = VoiceChangerWorker(vc_kwargs)
        self.vc.start()

        self.audio = Audio(
            self,
            f0_up_key,
            index_rate,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
            input_audio_gain,
            output_audio_gain,
            monitor_audio_gain,
            monitor,
        )

    def change_voice(
        self,
        received_data: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        if self.pass_through:
            vol = float(np.sqrt(np.square(received_data).mean(dtype=np.float32)))
            return received_data, vol, [0, 0, 0], None

        params = dict(
            f0_up_key=f0_up_key,
            index_rate=index_rate,
            protect=protect,
            volume_envelope=volume_envelope,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            proposed_pitch=proposed_pitch,
            proposed_pitch_threshold=proposed_pitch_threshold,
        )
        self.vc.submit(received_data, params)

        result = self.vc.retrieve()
        if result is not None:
            audio, vol, perf_ms, _warmup = result
            self._last_output = audio
            self._last_vol = vol
            return audio, vol, [0, perf_ms, 0], None

        # No result ready yet; replay previous output to avoid underrun.
        if self._last_output is not None and self._last_output.shape[0] == received_data.shape[0]:
            return self._last_output, self._last_vol, [0, 0, 0], None

        return np.zeros(received_data.shape[0], dtype=np.float32), 0, [0, 0, 0], None
