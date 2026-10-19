import math

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

# Gating of the loudness (ITU-R BS.1770-4): 400 ms blocks every 100 ms, an
# absolute gate at -70 LUFS and a relative one 10 LU under the gated loudness
BLOCK_SECONDS = 0.4
STEP_SECONDS = 0.1
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0
# Largest boost to a target loudness, so a near-silent output is not blown up
MAX_BOOST_DB = 24.0
# The limiter holds its reduction this long around a peak and ramps it over
# the second, so it is down before the peak and never follows single cycles
LIMIT_HOLD_SECONDS = 0.05
LIMIT_RAMP_SECONDS = 0.02


def k_weighting(sample_rate: int):
    """
    K-weighting filter of the loudness measure, its high shelf and high pass,
    as second-order sections.

    Args:
        sample_rate (int): Sampling rate of the audio.
    """
    gain, q, frequency = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    k = math.tan(math.pi * frequency / sample_rate)
    high = 10.0 ** (gain / 20.0)
    low = high**0.499666774155
    a0 = 1.0 + k / q + k * k
    shelf = [
        (high + low * k / q + k * k) / a0,
        2.0 * (k * k - high) / a0,
        (high - low * k / q + k * k) / a0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ]
    q, frequency = 0.5003270373253953, 38.13547087613982
    k = math.tan(math.pi * frequency / sample_rate)
    a0 = 1.0 + k / q + k * k
    high_pass = [
        1.0,
        -2.0,
        1.0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ]
    return np.array([shelf, high_pass])


def integrated_loudness(audio: np.ndarray, sample_rate: int):
    """
    Integrated loudness in LUFS, -inf for audio shorter than a block or silent.

    Args:
        audio (np.ndarray): Audio of shape (samples,) or (channels, samples).
        sample_rate (int): Sampling rate of the audio.
    """
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    block = int(round(BLOCK_SECONDS * sample_rate))
    step = int(round(STEP_SECONDS * sample_rate))
    if audio.shape[-1] < block:
        return -math.inf
    power = signal.sosfilt(k_weighting(sample_rate), audio, axis=-1) ** 2
    total = np.concatenate(
        (np.zeros((power.shape[0], 1)), np.cumsum(power, axis=-1)), axis=-1
    )
    starts = np.arange(0, power.shape[-1] - block + 1, step)
    blocks = ((total[:, starts + block] - total[:, starts]) / block).sum(0)
    loudness = -0.691 + 10.0 * np.log10(blocks + 1e-20)
    kept = loudness > ABSOLUTE_GATE
    if not kept.any():
        return -math.inf
    relative = -0.691 + 10.0 * math.log10(blocks[kept].mean()) + RELATIVE_GATE
    kept &= loudness > relative
    return -0.691 + 10.0 * math.log10(blocks[kept].mean())


def loudness_gain(audio: np.ndarray, sample_rate: int, target: float):
    """
    Gain that brings the audio to a loudness, 1 for silent audio.

    Args:
        audio (np.ndarray): Audio of shape (samples,) or (channels, samples).
        sample_rate (int): Sampling rate of the audio.
        target (float): Loudness to reach, in LUFS.
    """
    measured = integrated_loudness(audio, sample_rate)
    if not math.isfinite(measured):
        return 1.0
    return 10.0 ** (min(target - measured, MAX_BOOST_DB) / 20.0)


def limit_peaks(audio: np.ndarray, sample_rate: int, ceiling_db: float = -1.0):
    """
    Look-ahead peak limiter that keeps every sample under a ceiling.

    Args:
        audio (np.ndarray): Audio of shape (samples,) or (channels, samples).
        sample_rate (int): Sampling rate of the audio.
        ceiling_db (float, optional): Ceiling in dBFS. Defaults to -1.0.
    """
    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = np.abs(audio) if audio.ndim == 1 else np.abs(audio).max(0)
    needed = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if needed.min() >= 1.0:
        return audio
    hold = max(1, int(LIMIT_HOLD_SECONDS * sample_rate)) | 1
    ramp = max(1, int(LIMIT_RAMP_SECONDS * sample_rate))
    gain = minimum_filter1d(needed, size=hold, mode="nearest")
    gain = uniform_filter1d(gain, size=ramp, mode="nearest")
    return (audio * gain).astype(audio.dtype)
