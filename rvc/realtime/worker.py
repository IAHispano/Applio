import os
import sys
import multiprocessing as mp
import numpy as np
import time
from queue import Empty, Full

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.core import AUDIO_SAMPLE_RATE


def _worker_loop(vc_kwargs, input_q, output_q, config_q, stop_evt):
    """Entry point for the voice conversion worker process."""
    from rvc.realtime.core import VoiceChanger

    vc = VoiceChanger(**vc_kwargs)

    while not stop_evt.is_set():
        # Apply pending config updates.
        while True:
            try:
                cfg = config_q.get_nowait()
            except Empty:
                break
            _apply_config(vc, cfg)

        # Process one audio block.
        try:
            audio_in, params = input_q.get(timeout=0.05)
        except Empty:
            continue

        start = time.perf_counter()
        result, vol = vc.process_audio(audio_in, **params)
        perf_ms = (time.perf_counter() - start) * 1000

        warmup = vc.vc_model.warmup_blocks if vc.vc_model else 0

        # Drop stale output to keep queue depth at most 1.
        while True:
            try:
                output_q.get_nowait()
            except Empty:
                break
        try:
            output_q.put_nowait((result, vol, perf_ms, warmup))
        except Full:
            pass


def _apply_config(vc, cfg):
    """Apply a configuration update to the VoiceChanger in the worker process."""
    if "record_start" in cfg:
        vc.record_audio = True
        vc.record_audio_path = cfg.get("record_audio_path")
        vc.export_format = cfg.get("export_format", "WAV")
        vc.setup_soundfile_record()
        return

    if "record_stop" in cfg:
        vc.record_audio = False
        vc.record_audio_path = None
        vc.soundfile = None
        return

    # Realloc if crossfade/extra frame changed.
    cf = cfg.get("crossfade_frame")
    ef = cfg.get("extra_frame")
    if cf is not None and ef is not None:
        if vc.crossfade_frame != cf or vc.extra_frame != ef:
            del vc.fade_in_window, vc.fade_out_window, vc.sola_buffer
            vc.crossfade_frame = cf
            vc.extra_frame = ef
            vc.vc_model.realloc(
                vc.block_frame, vc.extra_frame,
                vc.crossfade_frame, vc.sola_search_frame,
            )
            vc.generate_strength()

    # Silence threshold.
    if "silent_threshold" in cfg:
        vc.vc_model.input_sensitivity = 10 ** (cfg["silent_threshold"] / 20)

    # VAD.
    if "vad_enabled" in cfg:
        if not cfg["vad_enabled"]:
            vc.vc_model.vad = None
        elif vc.vc_model.vad is None:
            from rvc.realtime.utils.vad import VADProcessor
            vc.vc_model.vad = VADProcessor(
                sensitivity_mode=cfg.get("vad_sensitivity", 3),
                sample_rate=vc.vc_model.sample_rate,
                frame_duration_ms=cfg.get("vad_frame_ms", 30),
            )

    # Noise reduction.
    if "clean_audio" in cfg:
        if not cfg["clean_audio"]:
            vc.vc_model.reduced_noise = None
        else:
            strength = cfg.get("clean_strength", 0.5)
            if vc.vc_model.reduced_noise is None:
                from noisereduce.torchgate import TorchGate
                vc.vc_model.reduced_noise = TorchGate(
                    vc.vc_model.pipeline.tgt_sr,
                    prop_decrease=strength,
                ).to(vc.vc_model.device)
            vc.vc_model.reduced_noise.prop_decrease = strength

    # Post-processing pedalboard.
    if "post_process" in cfg:
        kwargs = cfg.get("pedalboard_kwargs", {})
        if not cfg["post_process"]:
            vc.vc_model.board = None
            vc.vc_model.kwargs = None
        elif vc.vc_model.kwargs != kwargs:
            vc.vc_model.board = vc.vc_model.setup_pedalboard(**kwargs)
            vc.vc_model.kwargs = dict(kwargs)

    # Model hot-swap.
    model_path = cfg.get("model_path")
    if model_path and vc.vc_model.model_path != model_path:
        import torch
        import torchaudio.transforms as tat
        vc.vc_model.model_path = model_path
        vc.vc_model.pipeline.vc.load_model(model_path)
        vc.vc_model.pipeline.vc.setup_network()
        vc.vc_model.pipeline.version = vc.vc_model.pipeline.vc.version
        vc.vc_model.resample_out = tat.Resample(
            orig_freq=vc.vc_model.pipeline.tgt_sr,
            new_freq=AUDIO_SAMPLE_RATE,
            dtype=torch.float32,
        ).to(vc.vc_model.device)

    # SID change.
    sid = cfg.get("sid")
    if sid is not None and vc.vc_model.pipeline.sid != sid:
        import torch
        vc.vc_model.pipeline.torch_sid = torch.tensor(
            [sid], device=vc.vc_model.pipeline.device, dtype=torch.int64
        )

    # Index change.
    index_path = cfg.get("index_path")
    if index_path is not None:
        if index_path and vc.vc_model.index_path != index_path:
            from rvc.realtime.pipeline import load_faiss_index
            index, big_npy = load_faiss_index(
                index_path.strip().strip('"').strip("\n").strip('"')
                .strip().replace("trained", "added")
            )
            vc.vc_model.pipeline.index = index
            vc.vc_model.pipeline.big_npy = big_npy
            vc.vc_model.index_path = index_path
        elif not index_path:
            vc.vc_model.pipeline.index = None
            vc.vc_model.pipeline.big_npy = None
            vc.vc_model.index_path = None

    # F0 method change.
    f0_method = cfg.get("f0_method")
    if f0_method and vc.vc_model.pipeline.f0_method != f0_method:
        f0_model = vc.vc_model.pipeline.setup_f0(f0_method)
        vc.vc_model.pipeline.f0_model = f0_model
        vc.vc_model.pipeline.f0_method = f0_method

    # Embedder change.
    emb = cfg.get("embedder_model")
    emb_custom = cfg.get("embedder_model_custom")
    if emb is not None and (
        vc.vc_model.embedder_model != emb
        or vc.vc_model.embedder_model_custom != emb_custom
    ):
        from rvc.lib.utils import load_embedding
        old = vc.vc_model.pipeline.hubert_model
        del old
        hubert_model = load_embedding(emb, emb_custom)
        hubert_model = hubert_model.to(vc.vc_model.device).float()
        hubert_model.eval()
        vc.vc_model.pipeline.hubert_model = hubert_model
        vc.vc_model.embedder_model = emb
        vc.vc_model.embedder_model_custom = emb_custom


class _RealtimeState:
    """Cached subset of Realtime state readable from the main process."""

    def __init__(self):
        self.warmup_blocks = 0


class VoiceChangerWorker:
    """Runs VoiceChanger in a separate process for non-blocking audio callbacks."""

    def __init__(self, vc_kwargs):
        ctx = mp.get_context("spawn")
        self._input_q = ctx.Queue(maxsize=2)
        self._output_q = ctx.Queue(maxsize=2)
        self._config_q = ctx.Queue()
        self._stop = ctx.Event()
        self._vc_kwargs = vc_kwargs
        self._process = None
        # Cached state for main-process reads (change_callbacks_config / UI).
        self.crossfade_frame = int(
            vc_kwargs.get("cross_fade_overlap_size", 0.1) * AUDIO_SAMPLE_RATE
        )
        self.extra_frame = int(
            vc_kwargs.get("extra_convert_size", 0.5) * AUDIO_SAMPLE_RATE
        )
        self.block_frame = vc_kwargs.get("read_chunk_size", 192) * 128
        self.sola_search_frame = AUDIO_SAMPLE_RATE // 100
        self.vc_model = _RealtimeState()

    def start(self):
        self._process = mp.get_context("spawn").Process(
            target=_worker_loop,
            args=(
                self._vc_kwargs,
                self._input_q,
                self._output_q,
                self._config_q,
                self._stop,
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self):
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
            self._process = None

    def submit(self, audio_in, params):
        """Submit an audio block and inference params (non-blocking, drops if full)."""
        try:
            self._input_q.put_nowait((audio_in, params))
        except Full:
            pass

    def retrieve(self):
        """Get the latest processed result (non-blocking).

        Returns (audio, vol, perf_ms, warmup_blocks) or None.
        """
        result = None
        while True:
            try:
                result = self._output_q.get_nowait()
            except Empty:
                break
        if result is not None:
            self.vc_model.warmup_blocks = result[3]
        return result

    def send_config(self, cfg):
        """Send a configuration update dict to the worker process."""
        try:
            self._config_q.put_nowait(cfg)
        except Full:
            pass
