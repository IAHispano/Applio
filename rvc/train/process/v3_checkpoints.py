"""Two V3 package roles: exact training continuation and EMA inference export.

Headers bind architecture, feature/mel contracts and speaker vocabulary. Training
checkpoints retain live/EMA weights, optimizers, critics, scaler, settings and
per-rank RNG. Export selects EMA, merges adapters and removes training-only state.
An exported .pth supports conversion or new-stage initialization, not exact resume.
Atomic replacement prevents readers from observing partially written packages.
"""

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rvc.configs.architectures import BACKEND, FORMAT_VERSION
from rvc.configs.v3 import AcousticConfig, FeatureConfig, MelConfig, VocoderConfig
from rvc.lib.algorithm.v3.acoustic import AcousticModel
from rvc.lib.algorithm.v3.adapters import install_adapters, merge_adapters
from rvc.lib.algorithm.v3.vocoder import SpectralVocoder


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state():
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def load_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("backend") != BACKEND
        or payload.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("Not a supported Applio v3 model/checkpoint")
    MelConfig(**payload["mel"])
    if payload["kind"] == "acoustic":
        FeatureConfig(**payload["features"])
    elif payload["kind"] != "vocoder":
        raise ValueError("Unsupported v3 package kind")
    return payload


def construct(payload, device="cpu", use_ema=True):
    if payload["kind"] == "acoustic":
        model = AcousticModel(AcousticConfig(**payload["model_config"]))
        if payload.get("adapters"):
            install_adapters(model, **payload["adapters"])
    else:
        model = SpectralVocoder(VocoderConfig(**payload["model_config"]))
    weights = (
        payload.get("ema")
        if use_ema and payload.get("ema") is not None
        else payload["weights"]
    )
    model.load_state_dict(weights, strict=True)
    return model.to(device)


def header(model, kind, manifest, adapters=None):
    return {
        "backend": BACKEND,
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "torch_version": str(torch.__version__),
        "model_config": asdict(model.config),
        "mel": manifest["contract"]["mel"],
        "features": manifest["contract"]["features"],
        "speakers": manifest["speakers"],
        "dataset_id": manifest["dataset_id"],
        "adapters": adapters,
    }


def export_checkpoint(source, destination):
    payload = load_payload(source)
    model = construct(payload).eval()
    if payload.get("adapters"):
        merge_adapters(model)
    package = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "optimizer",
            "critic_optimizer",
            "critics",
            "scaler",
            "rng",
            "rank_rng",
            "ema",
            "phase",
            "loss_history",
            "weights",
        }
    }
    package["adapters"] = None
    package["weights"] = model.state_dict()
    package["inference_only"] = True
    if payload["kind"] == "acoustic":
        package["capabilities"] = {
            "predictor": bool(model.predictor_trained),
            "ordinary_flow": bool(model.flow_trained),
            "shortcuts": bool(model.shortcut_trained),
            "streaming": model.config.causal
            and payload["features"]["profile"] == "bounded",
        }
    atomic_save(destination, package)
    return destination
