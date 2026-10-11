"""Apply realtime settings at an audio block boundary, without reconnecting."""

import math
import os

import torch
import torchaudio.transforms as tat

from rvc.lib.user_config import resolve_logs_path
from rvc.lib.utils import load_embedding
from .core import AUDIO_SAMPLE_RATE
from .pipeline import RealtimeVoiceConverter, strip_parametrizations
from .utils.chunks import automatic_block_frame
from .utils.torch import IndexWrapper
from .utils.vad import VADProcessor


def apply_config_update(instance, current, message):
    model = instance.vc_model
    pipeline = model.pipeline
    changes = message.get("changes", {})
    effects = message.get("kwargs", {})
    if "key" in message:
        target = effects if message.get("if_kwargs", False) else changes
        target = {**target, message["key"]: message.get("value")}
        if message.get("if_kwargs", False):
            effects = target
        else:
            changes = target
    if not isinstance(changes, dict) or not isinstance(effects, dict):
        raise ValueError("Settings must be objects")
    if any(key not in current and key != "vad_sensitivity" for key in changes):
        raise ValueError("Unknown realtime setting")
    candidate = {
        **current,
        **changes,
        "kwargs": {**current.get("kwargs", {}), **effects},
    }
    ranges = {
        "f0_up_key": (-24, 24),
        "index_rate": (0, 1),
        "protect": (0, 0.5),
        "volume_envelope": (0, 1),
        "autotune_strength": (0, 1),
        "clean_strength": (0, 1),
        "input_audio_gain": (0, 200),
        "silent_threshold": (-90, 0),
        "cross_fade_overlap_size": (0.01, 0.5),
        "extra_convert_size": (0, 10),
        "vad_sensitivity": (0, 3),
    }
    for key, (low, high) in ranges.items():
        value = candidate.get(key, 1 if key == "vad_sensitivity" else current.get(key))
        if (key in candidate or value is not None) and (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"Invalid {key}")
    block = candidate.get("block_frame", instance.block_frame)
    if candidate.get("automatic_block_size") and (
        "automatic_block_size" in changes or "block_frame" in changes
    ):
        block = automatic_block_frame(getattr(instance, "last_processing_ms", 0))
    if not isinstance(block, int) or block < 480 or block > 48000 or block % 480:
        raise ValueError("Invalid block_frame")
    candidate["block_frame"] = block
    model_path = resolve_logs_path(candidate.get("model_path", model.model_path))
    index_path = resolve_logs_path(candidate.get("index_path", "")) or ""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("Model file not found")
    if index_path and not os.path.isfile(index_path):
        raise ValueError("Index file not found")
    candidate.update(model_path=model_path, index_path=index_path)

    # Prepare replacements first; a missing model/index/embedder must leave the
    # currently running voice and settings intact.
    replaced = model_path != model.model_path
    voice = RealtimeVoiceConverter(model_path) if replaced else pipeline.vc
    method = candidate.get("f0_method", pipeline.f0_method)
    if method not in ("crepe", "crepe-tiny", "rmvpe", "fcpe", "swift"):
        raise ValueError("Invalid f0_method")
    pitch_model = (
        pipeline.setup_f0(method) if method != pipeline.f0_method else pipeline.f0_model
    )
    embedder = candidate.get("embedder_model", model.embedder_model)
    custom = candidate.get("embedder_model_custom", model.embedder_model_custom)
    if embedder == "custom" and (not custom or not os.path.exists(custom)):
        raise ValueError("Custom embedder not found")
    hubert = pipeline.hubert_model
    if embedder != model.embedder_model or custom != model.embedder_model_custom:
        hubert = load_embedding(embedder, custom)
        strip_parametrizations(hubert)
        hubert = hubert.to(instance.device).float().eval()
    index, data = pipeline.index, pipeline.big_tsr
    if index_path != (model.index_path or ""):
        if index_path:
            index = IndexWrapper(index_path, device=instance.device, dtype=model.dtype)
            data, _ = index.read_index_tensor()
        else:
            index, data = None, None
    sid = candidate.get("sid", pipeline.sid)
    if not isinstance(sid, int) or not 0 <= sid < voice.net_g.emb_g.num_embeddings:
        raise ValueError("Invalid speaker ID")
    speaker = (
        torch.tensor([sid], device=pipeline.device, dtype=torch.int64)
        if sid != pipeline.sid
        else pipeline.torch_sid
    )
    sensitivity = int(candidate.get("vad_sensitivity", 1))
    vad = model.vad
    if not candidate.get("vad_enabled", True):
        vad = None
    elif vad is None or sensitivity != current.get("vad_sensitivity", 1):
        vad = VADProcessor(sensitivity_mode=sensitivity, sample_rate=model.sample_rate)
    noise = model.reduced_noise
    if not candidate.get("clean_audio", False):
        noise = None
    elif (
        noise is None
        or replaced
        or candidate.get("clean_strength") != current.get("clean_strength")
    ):
        from noisereduce.torchgate import TorchGate

        noise = TorchGate(
            voice.tgt_sr, prop_decrease=candidate.get("clean_strength", 0.5)
        ).to(instance.device)
    board = model.board
    kwargs = candidate["kwargs"]
    if not candidate.get("post_process", False):
        board = None
    elif board is None or kwargs != model.kwargs:
        board = model.setup_pedalboard(**kwargs)
    resample = (
        tat.Resample(voice.tgt_sr, AUDIO_SAMPLE_RATE, dtype=torch.float32).to(
            instance.device
        )
        if replaced
        else model.resample_out
    )

    if (
        voice is not pipeline.vc
        or pitch_model is not pipeline.f0_model
        or hubert is not pipeline.hubert_model
        or index is not pipeline.index
        or speaker is not pipeline.torch_sid
    ):
        pipeline._graph_cache = None
    pipeline.vc, pipeline.version, pipeline.use_f0, pipeline.tgt_sr = (
        voice,
        voice.version,
        voice.use_f0,
        voice.tgt_sr,
    )
    voice.decoder_fp16 = bool(
        candidate.get("decoder_fp16", False)
    ) and voice.vocoder in ("HiFi-GAN", "MRF HiFi-GAN")
    pipeline.f0_model, pipeline.f0_method = pitch_model, method
    pipeline.hubert_model, pipeline.index, pipeline.big_tsr = hubert, index, data
    pipeline.sid, pipeline.torch_sid = sid, speaker
    model.model_path, model.index_path = model_path, index_path
    model.embedder_model, model.embedder_model_custom = embedder, custom
    model.vad, model.reduced_noise, model.board = vad, noise, board
    model.kwargs = kwargs.copy() if board is not None else None
    model.resample_out = resample
    model.input_sensitivity = 10 ** (candidate.get("silent_threshold", -90) / 20)
    crossfade = round(candidate.get("cross_fade_overlap_size", 0.1) * AUDIO_SAMPLE_RATE)
    extra = round(candidate.get("extra_convert_size", 0.5) * AUDIO_SAMPLE_RATE)
    if (block, crossfade, extra) != (
        instance.block_frame,
        instance.crossfade_frame,
        instance.extra_frame,
    ):
        instance.block_frame, instance.crossfade_frame, instance.extra_frame = (
            block,
            crossfade,
            extra,
        )
        model.realloc(block, extra, crossfade, instance.sola_search_frame)
        instance.generate_strength()
    return candidate
