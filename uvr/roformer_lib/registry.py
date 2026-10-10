"""Roformer implementation registry: variant key -> (module, class).

Mirrors the dispatch in mvsepless' inference loader (plain, windowed,
conformer, SW/FNO/HyperACE/conditional/siamese variants). Custom user
architectures live in ``uvr/roformer/custom/*.py`` — see README.md there.
"""

import importlib
import os

# variant key -> (module path inside uvr.roformer, class name)
BUILTIN_VARIANTS = {
    "mel_band": ("mel_band_roformer", "MelBandRoformer"),
    "bs": ("bs_roformer", "BSRoformer"),
    "windowed": ("windowed_model", "MelBandRoformerWSA"),
    "mel_conformer": ("mel_band_conformer", "MelBandConformer"),
    "bs_conformer": ("bs_conformer", "BSConformer"),
    "sw": ("bs_roformer_sw", "BSRoformer_SW"),
    "fno": ("bs_roformer_fno", "BSRoformer_FNO"),
    "hyperace": ("bs_roformer_hyperace", "BSRoformerHyperACE"),
    "hyperace2": ("bs_roformer_hyperace2", "BSRoformerHyperACE_2"),
    "conditional": ("bs_roformer_conditional", "BSRoformer_Conditional"),
    "siamese": ("bs_siamese_roformer", "BSSiameseRoformer"),
    "unwa_inst_large_2": ("bs_roformer_unwa_inst_large_2", "BSRoformer_2"),
}

CUSTOM_DIRNAME = "custom"


def _load_custom_variants():
    """Scan uvr/roformer/custom/*.py for user architectures.

    Contract: the module must define ``MODEL_CLASS`` (an nn.Module whose
    forward takes ``(batch, channels, samples)`` waveforms and returns stems)
    and may define ``VARIANT_KEY`` (defaults to the file stem).
    """
    found = {}
    custom_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), CUSTOM_DIRNAME
    )
    if not os.path.isdir(custom_dir):
        return found
    for fname in sorted(os.listdir(custom_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        stem = fname[:-3]
        try:
            module = importlib.import_module(
                f"uvr.roformer_lib.{CUSTOM_DIRNAME}.{stem}"
            )
        except Exception:
            continue
        cls = getattr(module, "MODEL_CLASS", None)
        if cls is None:
            continue
        key = getattr(module, "VARIANT_KEY", stem)
        found[key] = (f"{CUSTOM_DIRNAME}.{stem}", cls.__name__)
    return found


def resolve(variant_key):
    """Return the model class for a variant key (builtin or custom)."""
    table = dict(BUILTIN_VARIANTS)
    try:
        table.update(_load_custom_variants())
    except Exception:
        pass
    if variant_key not in table:
        raise ValueError(
            f"Unknown Roformer variant '{variant_key}'. "
            f"Builtins: {', '.join(sorted(BUILTIN_VARIANTS))}. "
            f"Add custom architectures under uvr/roformer_lib/custom/ (see README.md)."
        )
    module_path, class_name = table[variant_key]
    module = importlib.import_module(f"uvr.roformer_lib.{module_path}")
    return getattr(module, class_name)


def available_variants():
    table = dict(BUILTIN_VARIANTS)
    try:
        table.update(_load_custom_variants())
    except Exception:
        pass
    return sorted(table)
