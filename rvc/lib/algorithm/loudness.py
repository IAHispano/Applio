import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

# The limiter holds its reduction this long around a peak and ramps it over
# the second, so it is down before the peak and never follows single cycles
LIMIT_HOLD_SECONDS = 0.05
LIMIT_RAMP_SECONDS = 0.02


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
