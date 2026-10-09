"""Optional compact derived caches; original recordings are never rewritten.

PCM24 FLAC quantizes resampled FLOAT audio. A per-segment power-of-two scale
prevents saturation, including resampling overshoot, and is undone before
feature extraction or waveform supervision. Content alone may be stored in FP16;
the loader restores FP32 and retains mel/pitch/voicing/energy precision.
Storage semantics belong in cache identities, not acoustic/vocoder contracts.
"""

import math

import numpy as np

COMPACT = {"version": 1, "waveform": "scaled-flac-pcm24", "content_dtype": "float16"}


def storage_contract(compact=False):
    return dict(COMPACT) if compact else None


def validate_storage(storage):
    if storage is not None and storage != COMPACT:
        raise ValueError("Unsupported derived-cache storage contract")


def waveform_scale(audio):
    peak = float(np.max(np.abs(audio)))
    if not math.isfinite(peak):
        raise ValueError("Cannot store nonfinite waveform")
    # Positive PCM24 tops out just below one. Reserve one code of headroom.
    scale = max(1.0, 2.0 ** math.ceil(math.log2(max(peak / (1 - 2**-23), 1.0))))
    if not math.isfinite(scale) or scale > 65536:
        raise ValueError("Waveform magnitude is outside compact-cache bounds")
    return scale


def read_segment(path, entry, sample_rate):
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    scale = entry.get("waveform_scale", 1.0)
    if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale < 1 or scale > 65536 or math.frexp(scale)[0] != 0.5:
        raise ValueError("Invalid waveform storage scale")
    if rate != sample_rate or audio.ndim != 1 or len(audio) != entry["samples"]:
        raise ValueError("Cached waveform dimensions or sample rate are invalid")
    audio = audio * np.float32(scale)
    if not np.isfinite(audio).all():
        raise ValueError("Cached waveform is nonfinite")
    return audio


def stored_features(features, storage):
    validate_storage(storage)
    if storage is None:
        return features
    result = dict(features)
    result["content"] = np.asarray(result["content"], dtype=np.float16)
    if not np.isfinite(result["content"]).all():
        raise ValueError("Content values cannot be represented by compact FP16 storage")
    return result
