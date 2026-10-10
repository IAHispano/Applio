"""Applio UVR orchestrator: model download, arch dispatch, single-file separation."""

import hashlib
import json
import logging
import os

from uvr import OUTPUT_MODELS_DIRNAME
from uvr import models as catalog
from uvr.demucs import DemucsSeparator
from uvr.mdx import MdxSeparator
from uvr.mdxc import MdxcSeparator
from uvr.roformer import RoformerSeparator
from uvr.vr import VrSeparator
from uvr.base import download_file, select_onnx_providers, select_torch_device

ARCH_CLASSES = {
    "vr": VrSeparator,
    "mdx": MdxSeparator,
    "demucs": DemucsSeparator,
    "roformer": RoformerSeparator,
    "mdxc": MdxcSeparator,
}


def _md5(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _yaml_in_dir(model_dir):
    yamls = sorted(f for f in os.listdir(model_dir) if f.endswith((".yaml", ".yml")))
    return os.path.join(model_dir, yamls[0]) if yamls else None


def _model_data_for(arch, weight_path):
    """Resolve arch params: yaml-driven archs read their config, VR/MDX look up the vendored hash tables."""
    here = os.path.dirname(os.path.abspath(__file__))
    if arch in ("demucs", "roformer", "mdxc"):
        import yaml

        yaml_path = _yaml_in_dir(os.path.dirname(weight_path))
        if yaml_path:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader) or {}
            training = data.get("training", {}) or {}
            instruments = [str(s) for s in training.get("instruments", [])]
            target = training.get("target_instrument") or (
                instruments[0] if instruments else "Vocals"
            )
            data.setdefault("primary_stem", target)
            return data
        return {"primary_stem": "Vocals"}
    digest = _md5(weight_path)
    table_file = os.path.join(
        here, "vr_model_data.json" if arch == "vr" else "mdx_model_data.json"
    )
    with open(table_file, encoding="utf-8") as f:
        table = json.load(f)
    if digest not in table:
        raise ValueError(
            f"Unsupported {arch.upper()} weights: MD5 {digest} not in the vendored model data table."
        )
    return table[digest]


def _default_arch_params(arch):
    if arch == "vr":
        return {
            "batch_size": 1,
            "window_size": 512,
            "aggression": 5,
            "enable_tta": False,
            "enable_post_process": False,
            "post_process_threshold": 0.2,
            "high_end_process": False,
        }
    if arch == "mdx":
        return {
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.25,
            "batch_size": 1,
            "enable_denoise": False,
        }
    if arch == "mdxc":
        return {
            "segment_size": 256,
            "overlap": 8,
            "batch_size": 1,
        }
    if arch == "demucs":
        return {
            "segment_size": "Default",
            "shifts": 2,
            "overlap": 0.25,
            "segments_enabled": True,
        }
    return {}


def separate_stems(
    input_path,
    model_key,
    models_dir=None,
    output_dir=None,
    output_format="wav",
    use_gpu=True,
    single_stem="all",
    arch_overrides=None,
    logger=None,
):
    """Separate one audio file. Returns {stem_name: absolute_path}."""
    logger = logger or logging.getLogger("uvr")
    model = catalog.resolve(model_key)
    if model is None:
        raise ValueError(
            f"Unknown UVR model '{model_key}'. Available: {', '.join(sorted(catalog.BY_KEY))}"
        )

    repo_root = os.getcwd()
    models_dir = models_dir or os.path.join(repo_root, OUTPUT_MODELS_DIRNAME, model.key)
    os.makedirs(models_dir, exist_ok=True)
    for filename, url in model.files:
        download_file(url, os.path.join(models_dir, filename), logger=logger)

    output_dir = os.path.abspath(output_dir) if output_dir else None
    torch_device = select_torch_device(use_gpu)
    logger.info(f"Separating with {model.label} on {torch_device}...")

    weight_path = os.path.join(models_dir, model.files[0][0])
    roformer_config = None
    if model.arch == "demucs":
        # Demucs loads from its yaml config (weights resolved beside it).
        yaml_path = _yaml_in_dir(models_dir)
        if not yaml_path:
            raise ValueError(
                f"Demucs model '{model.key}' is missing its yaml config in {models_dir}."
            )
        weight_path = yaml_path
    elif model.arch in ("roformer", "mdxc"):
        yaml_path = _yaml_in_dir(models_dir)
        if not yaml_path:
            raise ValueError(
                f"Model '{model.key}' is missing its yaml config in {models_dir}."
            )
        roformer_config = yaml_path
    model_data = _model_data_for(model.arch, weight_path)
    if model.arch == "demucs":
        model_data = {
            **model_data,
            "primary_stem": model_data.get("primary_stem", "Vocals"),
        }

    common_config = {
        "logger": logger,
        "log_level": logging.INFO,
        "torch_device": torch_device,
        "torch_device_cpu": select_torch_device(False),
        "torch_device_mps": None,
        "onnx_execution_provider": select_onnx_providers(
            use_gpu and torch_device.type == "cuda"
        ),
        "model_name": (
            model.key
            if model.arch in ("demucs", "roformer")
            else model.files[0][0].rsplit(".", 1)[0]
        ),
        "model_path": weight_path,
        "model_data": model_data,
        "roformer_class": model.cls,
        "roformer_config": roformer_config,
        "output_dir": output_dir,
        "output_format": output_format,
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.0,
        "output_single_stem": (
            None if str(single_stem).lower() == "all" else single_stem
        ),
        "sample_rate": 44100,
    }
    arch_params = _default_arch_params(model.arch)
    if arch_overrides:
        arch_params = {
            **arch_params,
            **{k: v for k, v in arch_overrides.items() if v is not None},
        }
    arch = ARCH_CLASSES[model.arch](common_config, arch_params)
    try:
        paths = arch.separate(input_path)
    finally:
        try:
            arch.clear_gpu_cache()
        except Exception:
            pass

    if model.arch == "demucs":
        labels = list(model.stems)
    elif model.arch == "roformer" and getattr(arch, "instruments", None):
        labels = list(arch.instruments)
    else:
        labels = [arch.primary_stem_name, arch.secondary_stem_name]
    stems = {}
    for path in paths:
        # Archs return paths relative to output_dir; resolve to absolute.
        full = (
            path if os.path.isabs(path) else os.path.join(output_dir or repo_root, path)
        )
        base = os.path.basename(full)
        lowered_base = base.lower()
        match = next((s for s in labels if f"_({s.lower()})_" in lowered_base), None)
        stems[match or os.path.splitext(base)[0]] = os.path.abspath(full)
    logger.info(f"Separation complete: {len(stems)} stems.")
    return stems
