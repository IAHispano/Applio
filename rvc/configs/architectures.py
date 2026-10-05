"""Checkpoint metadata selects classic v1/v2 or experimental V3 in CLI and GUI.

File extensions do not identify architecture. V3 headers also distinguish an
acoustic voice package from a universal vocoder so model discovery cannot offer
a vocoder as a target voice. Inspect with restricted Torch loading; cache entries
include file modification time and size to avoid stale metadata after replacement.
"""

from functools import lru_cache
import os
from pathlib import Path
from pickle import UnpicklingError

BACKEND = "applio-v3"
FORMAT_VERSION = 1
ARCHITECTURES = [("Classic RVC (v1/v2)", "classic"), ("Applio v3", "v3")]


@lru_cache(maxsize=256)
def _inspect(path, modified, size):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain model metadata")
    if payload.get("backend") == BACKEND:
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported v3 checkpoint format")
        return {
            k: payload[k]
            for k in (
                "backend",
                "format_version",
                "kind",
                "model_config",
                "mel",
                "features",
                "speakers",
                "step",
                "phase",
                "capabilities",
                "inference_only",
                "vocoder_backend",
            )
            if k in payload
        }
    if "weight" in payload and "config" in payload:
        return {
            "backend": "classic",
            "speakers_id": payload.get("speakers_id", 1),
            "version": payload.get("version", "v1"),
            "vocoder": payload.get("vocoder", "HiFi-GAN"),
        }
    raise ValueError("Unsupported voice model checkpoint")


def inspect_model(path):
    file = Path(path).resolve()
    if file.suffix.lower() == ".onnx":
        return {"backend": "classic", "version": "onnx"}
    info = file.stat()
    return dict(_inspect(str(file), info.st_mtime_ns, info.st_size))


def model_architecture(path):
    return "v3" if inspect_model(path)["backend"] == BACKEND else "classic"


def resolve_architecture(architecture, model=None):
    if architecture not in {"auto", "classic", "v3"}:
        raise ValueError("Architecture must be auto, classic or v3")
    actual = model_architecture(model) if model else None
    if architecture == "auto":
        return actual or "classic"
    if actual and actual != architecture:
        raise ValueError(
            f"Selected {architecture} architecture but the checkpoint is {actual}"
        )
    return architecture


def discover_models(root="logs", architecture="v3", kind="acoustic"):
    models = []
    files = []
    for directory, folders, names in os.walk(root):
        folders[:] = [name for name in folders if name != "_archive"]
        files.extend(Path(directory) / name for name in names)
    for file in sorted(files):
        if file.suffix.lower() not in {".pth", ".pt", ".onnx"} or file.name.startswith(
            ("G_", "D_")
        ):
            continue
        try:
            metadata = inspect_model(file)
            actual = "v3" if metadata["backend"] == BACKEND else "classic"
            if actual == architecture and (
                actual == "classic" or metadata.get("kind") == kind
            ):
                models.append((str(file.with_suffix("")), str(file)))
        except (ValueError, RuntimeError, OSError, EOFError, UnpicklingError):
            continue
    return models


def require_classic_payload(payload):
    if isinstance(payload, dict) and payload.get("backend") == BACKEND:
        raise ValueError(
            "This is an Applio v3 package. Select Applio v3 in Training/Inference, with its matching universal vocoder and content encoder."
        )
