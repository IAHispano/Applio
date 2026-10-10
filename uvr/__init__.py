"""Applio UVR — self-contained local stem separation (vocals, instrumentals, cleanup).

Vendored and distilled from the UVR / audio-separator ecosystem. No external
package imports from those libraries; only standard scientific Python
dependencies (torch, numpy, librosa, soundfile, scipy, tqdm, requests).

Heavy imports (torch and friends) stay inside ``uvr.separator`` so that
``uvr.models`` (catalog) and ``uvr.separate --list-models`` load instantly.
"""

__version__ = "1.0.0"

OUTPUT_MODELS_DIRNAME = "rvc/models/uvr"


def __getattr__(name):
    if name == "separate_stems":
        from uvr.separator import separate_stems

        return separate_stems
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OUTPUT_MODELS_DIRNAME", "separate_stems"]
