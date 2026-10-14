import os
import sys
import multiprocessing as mp
import time
from queue import Empty, Full

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.core import VoiceChanger, AUDIO_SAMPLE_RATE


def _worker_loop(vc_kwargs, input_q, output_q, config_q, stop_evt, ready_evt):
    """Entry point for the voice conversion worker process."""

    try:
        vc = VoiceChanger(**vc_kwargs)
        vc.warmup(index_rate=vc_kwargs.get("index_rate", 0))
    except (RuntimeError, NotImplementedError, TypeError) as error:
        from rvc.lib.tools.device import is_mps_error
        from rvc.configs.config import Config

        if Config().device != "mps" or not is_mps_error(error):
            raise
        print(f"[!] MPS realtime warmup failed ({error}); retrying on CPU.", flush=True)
        vc = VoiceChanger(**dict(vc_kwargs, device="cpu"))
        vc.warmup(index_rate=vc_kwargs.get("index_rate", 0))
    ready_evt.set()

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


def _apply_config(vc: VoiceChanger, cfg):
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
                vc.block_frame,
                vc.extra_frame,
                vc.crossfade_frame,
                vc.sola_search_frame,
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

    # Clean-audio noise reduction was removed (Voice Focus covers denoising).
    if "clean_audio" in cfg or "clean_strength" in cfg:
        vc.vc_model.reduced_noise = None

    # Formant shifting (offline-style STFT, applied to the 16 kHz input block).
    if "formant_shifting" in cfg:
        vc.vc_model.formant_shifting = bool(cfg["formant_shifting"])
    if "formant_qfrency" in cfg:
        try:
            vc.vc_model.formant_qfrency = float(cfg["formant_qfrency"])
        except (TypeError, ValueError):
            pass
    if "formant_timbre" in cfg:
        try:
            vc.vc_model.formant_timbre = float(cfg["formant_timbre"])
        except (TypeError, ValueError):
            pass

    # Post-processing (pedalboard) was removed; ignore legacy keys.

    # Model hot-swap.
    model_path = cfg.get("model_path")
    if model_path and vc.vc_model.model_path != model_path:
        import torch
        import torchaudio.transforms as tat

        vc.vc_model.model_path = model_path
        vc.vc_model.pipeline.vc.load_model(model_path)
        vc.vc_model.pipeline.vc.setup_network()
        vc.vc_model.pipeline.version = vc.vc_model.pipeline.vc.version
        vc.vc_model.pipeline.use_f0 = vc.vc_model.pipeline.vc.use_f0
        vc.vc_model.pipeline.tgt_sr = vc.vc_model.pipeline.vc.tgt_sr
        vc.vc_model.resample_out = tat.Resample(
            orig_freq=vc.vc_model.pipeline.tgt_sr,
            new_freq=AUDIO_SAMPLE_RATE,
            dtype=torch.float32,
        ).to(vc.vc_model.device)
        vc.vc_model.reduced_noise = None

    # SID change.
    sid = cfg.get("sid")
    if sid is not None and vc.vc_model.pipeline.sid != sid:
        import torch

        vc.vc_model.pipeline.torch_sid = torch.tensor(
            [sid], device=vc.vc_model.pipeline.device, dtype=torch.int64
        )

    # Keep path selection independent of retrieval strength. The audio-block
    # parameters activate a lazy index when the strength becomes positive.
    index_path = cfg.get("index_path")
    if index_path is not None and (index_path or "") != (vc.vc_model.index_path or ""):
        from rvc.realtime.utils.torch import IndexWrapper

        vc.vc_model.pipeline._graph_cache = None
        vc.vc_model.pipeline.big_tsr = None
        vc.vc_model.pipeline.index = (
            IndexWrapper(
                index_path.strip().strip('"').strip().replace("trained", "added"),
                device=vc.device,
                dtype=vc.vc_model.dtype,
            )
            if index_path
            else None
        )
        vc.vc_model.index_path = index_path or None
    if "index_rate" in cfg:
        vc.vc_model.pipeline.prepare_index(cfg["index_rate"])

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
        self._ready = ctx.Event()
        self._vc_kwargs = vc_kwargs
        self._process = None
        # Cached state for main-process reads (change_callbacks_config / UI).
        self.crossfade_frame = int(
            vc_kwargs.get("cross_fade_overlap_size", 0.1) * AUDIO_SAMPLE_RATE
        )
        self.extra_frame = int(
            vc_kwargs.get("extra_convert_size", 0.5) * AUDIO_SAMPLE_RATE
        )
        self.block_frame = vc_kwargs.get("block_frame", 24576)
        self.sola_search_frame = AUDIO_SAMPLE_RATE // 100
        self.vc_model = _RealtimeState()

    @property
    def ready(self):
        """True if the worker process has finished initializing the VoiceChanger model."""
        return self._ready.is_set()

    def start(self):
        self._process = mp.get_context("spawn").Process(
            target=_worker_loop,
            args=(
                self._vc_kwargs,
                self._input_q,
                self._output_q,
                self._config_q,
                self._stop,
                self._ready,
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
