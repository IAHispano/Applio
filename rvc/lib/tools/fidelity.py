"""Aligned saved-waveform fidelity proxies; no models, devices or synthesis.

These measurements quantify source preservation, not target identity or human
naturalness. A target's intended formant changes can increase envelope error.
No gain fitting, time warping, denoising or peak normalization is applied.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import get_window, resample_poly, stft


def _mono(audio):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Fidelity analysis requires nonempty finite mono audio")
    return audio


def compare_waveforms(source, prediction, sample_rate=44100, window=4096,
                      hop=512, smoothing_hz=(150, 300, 600)):
    """Compare physically aligned signals; spectral envelopes are proxies.

    Gaussian smoothing is applied to linear magnitude along frequency, then
    converted to dB. Multiple widths expose dependence on harmonic spacing and
    smoothing resolution. Spectral shape subtracts each frame's mean envelope;
    absolute envelope and weighted RMS errors retain level differences.
    """
    source, prediction = _mono(source), _mono(prediction)
    if len(source) != len(prediction):
        raise ValueError("Signals must have identical aligned sample counts")
    if (not isinstance(sample_rate, int) or sample_rate < 1000 or
            not isinstance(window, int) or window < 32 or window % 2 or
            not isinstance(hop, int) or not 1 <= hop <= window):
        raise ValueError("Invalid sample rate, even analysis window or hop")
    smoothing_hz = tuple(float(v) for v in smoothing_hz)
    if not smoothing_hz or any(not np.isfinite(v) or v <= 0 for v in smoothing_hz):
        raise ValueError("Envelope smoothing widths must be finite and positive")
    # Fixed padding/window, including short files; centers start at sample zero.
    if len(source) < window:
        source = np.pad(source, (0, window - len(source)))
        prediction = np.pad(prediction, (0, window - len(prediction)))
    hann = get_window("hann", window)
    spectra = [stft(x, fs=sample_rate, window=hann, nperseg=window,
                    noverlap=window - hop, boundary="zeros", padded=True)[2]
               for x in (source, prediction)]
    magnitudes = [np.abs(s) for s in spectra]
    del spectra
    parseval = hann.sum() ** 2 / (window * np.square(hann).sum())
    weights = np.full(window // 2 + 1, 2.0)
    weights[[0, -1]] = 1
    rms = [np.sqrt(np.sum(m * m * weights[:, None], axis=0) * parseval)
           for m in magnitudes]
    # Absolute -60 dBFS plus 40 dB below source peak; no output-derived mask.
    threshold = max(1e-3, float(rms[0].max()) * .01)
    active = rms[0] >= threshold
    inactive = ~active
    output_active = rms[1] >= threshold
    source_level, prediction_level = [20 * np.log10(np.maximum(v, 1e-5)) for v in rms]
    envelopes = {}
    for width in smoothing_hz:
        envelope = [20 * np.log10(np.maximum(
            gaussian_filter1d(m, width * window / sample_rate, axis=0,
                              mode="reflect"), 1e-5)) for m in magnitudes]
        difference = envelope[1] - envelope[0]
        absolute = np.mean(np.abs(difference), axis=0)
        shape = np.mean(np.abs(difference - difference.mean(axis=0)), axis=0)
        envelopes[str(width)] = dict(
            absolute_mae_db=float(absolute[active].mean()) if active.any() else None,
            shape_mae_db=float(shape[active].mean()) if active.any() else None,
            shape_p95_db=float(np.percentile(shape[active], 95)) if active.any() else None,
        )
    return dict(
        sample_rate=sample_rate, window=window, hop=hop,
        magnitude_floor_dbfs=-100, activity_threshold_amplitude=threshold,
        frames=len(active), active_source_frames=int(active.sum()),
        envelope_proxies=envelopes,
        active_level_mae_db=float(np.abs(prediction_level - source_level)[active].mean())
        if active.any() else None,
        inactive_output_level_dbfs=float(20 * np.log10(max(
            np.sqrt(np.mean(rms[1][inactive] ** 2)), 1e-5))) if inactive.any() else None,
        source_active_output_inactive_fraction=float((~output_active[active]).mean())
        if active.any() else None,
        source_inactive_output_active_fraction=float(output_active[inactive].mean())
        if inactive.any() else None,
        finite=True, gain_normalized=False, time_warped=False,
    )


def compare_files(source, prediction, **settings):
    """Read originals, resample for analysis only and retain file identities."""
    import soundfile as sf

    paths = [Path(source), Path(prediction)]
    sample_rate = settings.pop("sample_rate", 44100)
    if not isinstance(sample_rate, int) or sample_rate < 1000:
        raise ValueError("Invalid analysis sample rate")
    records, audio = [], []
    for path in paths:
        x, rate = sf.read(path, dtype="float64")
        x = _mono(x)
        records.append(dict(path=str(path),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            sample_rate=rate, samples=len(x), duration_seconds=len(x) / rate))
        divisor = math.gcd(rate, sample_rate)
        audio.append(resample_poly(x, sample_rate // divisor, rate // divisor)
                     if rate != sample_rate else x)
    difference = len(audio[1]) - len(audio[0])
    if abs(difference) > 1:
        raise ValueError("Duration mismatch exceeds one analysis sample; no alignment applied")
    size = min(map(len, audio))
    result = compare_waveforms(audio[0][:size], audio[1][:size], sample_rate, **settings)
    result.update(source=records[0], prediction=records[1],
                  analysis_sample_count=size, analysis_length_difference_samples=difference,
                  duration_difference_seconds=records[1]["duration_seconds"] - records[0]["duration_seconds"])
    return result
