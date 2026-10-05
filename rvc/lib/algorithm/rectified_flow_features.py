import math

import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F

# Frame rate of the extracted pitch, of the content once doubled and of the curves.
FEATURE_RATE = 100
# Width of the sine window the loudness and breathiness curves are smoothed with.
SMOOTH_SECONDS = 0.06
# The tension's: long enough to follow the phrase and not each vowel.
TENSION_SMOOTH_SECONDS = 0.18

ENERGY_WINDOW_SECONDS = 0.032
ENERGY_FLOOR_DB = -70.0

# Three periods of an 80 Hz note, so its harmonics are resolved.
HARMONIC_WINDOW_SECONDS = 0.04
# The measured band, what a 16 kHz input has.
HARMONIC_MIN_HZ = 50.0
HARMONIC_MAX_HZ = 8000.0
# Spread of the band around each harmonic counted as periodic.
HARMONIC_SIGMA_HZ = 25.0
APERIODICITY_FLOOR_DB = -30.0


class LogMel(nn.Module):
    """
    Log mel spectrogram at one frame per hop, the same mel as OpenVPI's
    vocoders take.

    Args:
        sample_rate (int): Sampling rate of the audio.
        n_fft (int): FFT size.
        win_length (int): Window size.
        hop_length (int): Hop size.
        n_mels (int): Number of mel bins.
        fmin (float): Lowest frequency of the mel.
        fmax (float): Highest frequency of the mel.
    """

    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        basis = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        self.register_buffer("basis", torch.from_numpy(basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    @classmethod
    def from_config(cls, data):
        return cls(
            data["sample_rate"],
            data["n_fft"],
            data["win_length"],
            data["hop_length"],
            data["n_mels"],
            data["mel_fmin"],
            data["mel_fmax"],
        )

    def forward(self, audio, key_shift=0.0, hop_length=None):
        """
        Args:
            audio (torch.Tensor): Audio, shape (batch, samples).
            key_shift (float, optional): Semitones every frequency, pitch and formants, is scaled by. Defaults to 0.0.
            hop_length (int, optional): Hop size, which stretches time when it is not the configured one.
        """
        hop_length = int(hop_length or self.hop_length)
        factor = 2.0 ** (key_shift / 12.0)
        n_fft = int(round(self.n_fft * factor))
        win_length = int(round(self.win_length * factor))
        window = self.window
        if win_length != self.win_length:
            window = torch.hann_window(win_length, device=audio.device)
        pad = win_length - hop_length
        # Reflection needs more samples than it pads.
        short = (pad + 1) // 2 + 1 - audio.shape[-1]
        if short > 0:
            audio = F.pad(audio, (0, short))
        audio = F.pad(
            audio.float().unsqueeze(1), (pad // 2, (pad + 1) // 2), mode="reflect"
        ).squeeze(1)
        spec = torch.stft(
            audio,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        if n_fft != self.n_fft:
            bins = self.n_fft // 2 + 1
            spec = F.pad(spec, (0, 0, 0, max(0, bins - spec.shape[1])))[:, :bins]
            spec = spec * (self.win_length / win_length)
        return torch.log(torch.clamp(self.basis @ spec, min=1e-5))


def normalize_mel(mel, data):
    return (mel - data["mel_mean"]) / data["mel_std"]


def denormalize_mel(mel, data):
    return mel * data["mel_std"] + data["mel_mean"]


def upsample_content(features, mode="linear"):
    """
    Double the frames of the content features, from 50 Hz to FEATURE_RATE.

    Args:
        features (torch.Tensor): Content features, shape (frames, channels).
        mode (str, optional): "linear" or "nearest". Defaults to "linear".
    """
    x = features.unsqueeze(0).transpose(1, 2)
    if mode == "linear":
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
    else:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
    return x.transpose(1, 2)[0]


def smooth_curve(curve, seconds=SMOOTH_SECONDS):
    """
    Smooth a curve at FEATURE_RATE with a sine window.

    Args:
        curve (torch.Tensor): Curve, shape (batch, frames).
        seconds (float, optional): Width of the window. Defaults to SMOOTH_SECONDS.
    """
    width = int(round(seconds * FEATURE_RATE))
    kernel = torch.sin(
        torch.linspace(0, 1, width + 2, device=curve.device)[1:-1] * math.pi
    )
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    padded = F.pad(
        curve.unsqueeze(1), ((width - 1) // 2, width // 2), mode="replicate"
    )
    return F.conv1d(padded, kernel.to(curve.dtype)).squeeze(1)


def mel_frames(feature_frames, sample_rate, hop):
    """
    Number of mel frames covered by `feature_frames` frames at FEATURE_RATE.
    """
    return int(feature_frames * sample_rate / (hop * FEATURE_RATE))


def _positions(length, frames, sample_rate, hop, device):
    position = torch.arange(frames, device=device, dtype=torch.float64)
    position = (position * hop * FEATURE_RATE / sample_rate).clamp(max=length - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=length - 1)
    return left, right, (position - left).float()


def to_mel_rate(features, frames, sample_rate, hop):
    """
    Interpolate features at FEATURE_RATE to the mel frames, linearly.

    Args:
        features (torch.Tensor): Features, shape (..., time, channels).
        frames (int): Number of mel frames.
        sample_rate (int): Sampling rate of the audio.
        hop (int): Hop size of the mel.
    """
    left, right, weight = _positions(
        features.shape[-2], frames, sample_rate, hop, features.device
    )
    weight = weight.unsqueeze(-1)
    return features[..., left, :] * (1 - weight) + features[..., right, :] * weight


def curve_to_mel_rate(curve, frames, sample_rate, hop):
    """
    Interpolate a curve at FEATURE_RATE to the mel frames, linearly.

    Args:
        curve (torch.Tensor): Curve, shape (batch, time).
        frames (int): Number of mel frames.
        sample_rate (int): Sampling rate of the audio.
        hop (int): Hop size of the mel.
    """
    return to_mel_rate(curve.unsqueeze(-1), frames, sample_rate, hop)[..., 0]


def f0_to_mel_rate(f0, frames, sample_rate, hop):
    """
    Interpolate the pitch at FEATURE_RATE to the mel frames. Only between two
    voiced frames, elsewhere the nearer frame is taken.

    Args:
        f0 (torch.Tensor): Pitch in Hz, shape (..., time).
        frames (int): Number of mel frames.
        sample_rate (int): Sampling rate of the audio.
        hop (int): Hop size of the mel.
    """
    left, right, weight = _positions(f0.shape[-1], frames, sample_rate, hop, f0.device)
    a, b = f0[..., left], f0[..., right]
    nearest = torch.where(weight < 0.5, a, b)
    return torch.where((a > 0) & (b > 0), a * (1 - weight) + b * weight, nearest)


def frame_energy(audio, sample_rate, frames):
    """
    Log RMS per 10 ms frame, mapped from [ENERGY_FLOOR_DB, 0] dBFS to [-1, 1].

    Args:
        audio (torch.Tensor): Audio, shape (batch, samples).
        sample_rate (int): Sampling rate of the audio.
        frames (int): Number of frames.
    """
    hop = int(sample_rate) // FEATURE_RATE
    window = int(round(ENERGY_WINDOW_SECONDS * sample_rate)) | 1
    power = F.avg_pool1d(
        audio.float().unsqueeze(1).pow(2),
        kernel_size=window,
        stride=hop,
        padding=window // 2,
        count_include_pad=False,
    )
    if power.shape[-1] < frames:
        power = F.pad(power, (0, frames - power.shape[-1]), mode="replicate")
    db = 10.0 * torch.log10(power[..., :frames].clamp_min(1e-10))
    db = db.clamp(ENERGY_FLOOR_DB, 0.0)
    return (db / (-ENERGY_FLOOR_DB / 2.0) + 1.0).squeeze(1)


def _harmonics(audio, sample_rate, f0, frames):
    # Power per bin in the band, each bin's weight as part of a harmonic of
    # f0, and which bins sit on the fundamental.
    hop = int(sample_rate) // FEATURE_RATE
    window = int(round(HARMONIC_WINDOW_SECONDS * sample_rate))
    short = max(0, frames * hop - audio.shape[-1])
    audio = F.pad(audio.float(), (window // 2, window // 2 + short))
    power = (
        torch.stft(
            audio,
            window,
            hop_length=hop,
            window=torch.hann_window(window, device=audio.device),
            center=False,
            return_complex=True,
        )
        .abs()
        .square()[..., :frames]
    )
    freqs = torch.fft.rfftfreq(window, 1.0 / sample_rate).to(audio.device)
    band = (freqs >= HARMONIC_MIN_HZ) & (freqs <= HARMONIC_MAX_HZ)
    power, freqs = power[:, band], freqs[band]

    f0 = F.pad(f0.float(), (0, max(0, frames - f0.shape[-1])))[:, :frames]
    ratio = freqs[None, :, None] / f0.clamp_min(1.0)[:, None, :]
    nearest = ratio.round()
    distance = (ratio - nearest).abs() * f0[:, None, :]
    periodic = torch.exp(-0.5 * (distance / HARMONIC_SIGMA_HZ).square())
    periodic = periodic * ((nearest >= 1) & (f0[:, None, :] > 0))
    return power, periodic, nearest == 1


def aperiodicity(audio, sample_rate, f0, frames):
    """
    Aperiodic share of the energy per 10 ms frame, mapped from
    [APERIODICITY_FLOOR_DB, 0] dB to [-1, 1]. Unvoiced and silent frames read
    as fully aperiodic.

    Args:
        audio (torch.Tensor): Audio, shape (batch, samples).
        sample_rate (int): Sampling rate of the audio.
        f0 (torch.Tensor): Pitch of the audio in Hz at FEATURE_RATE, shape (batch, time).
        frames (int): Number of frames.
    """
    power, periodic, _ = _harmonics(audio, sample_rate, f0, frames)
    between = 1.0 - periodic
    noise = (power * between).sum(1) / between.sum(1).clamp_min(1e-3) * power.shape[1]
    total = power.sum(1)
    share = torch.where(
        total > 1e-8, noise / total.clamp_min(1e-10), torch.ones_like(total)
    )
    db = (10.0 * torch.log10(share.clamp(1e-10, 1.0))).clamp(APERIODICITY_FLOOR_DB, 0.0)
    return db / (-APERIODICITY_FLOOR_DB / 2.0) + 1.0


def tension(audio, sample_rate, f0, frames):
    """
    How much of the harmonic amplitude lies above the fundamental per 10 ms
    frame (DiffSinger's tension), taken from the median of the voiced frames.
    Unvoiced and silent frames read 0.

    Args:
        audio (torch.Tensor): Audio, shape (batch, samples).
        sample_rate (int): Sampling rate of the audio.
        f0 (torch.Tensor): Pitch of the audio in Hz at FEATURE_RATE, shape (batch, time).
        frames (int): Number of frames.
    """
    power, periodic, fundamental = _harmonics(audio, sample_rate, f0, frames)
    harmonic = (power * periodic).sum(1)
    above = (power * periodic * ~fundamental).sum(1)
    share = (above / harmonic.clamp_min(1e-10)).sqrt().clamp(1e-4, 1.0 - 1e-4)
    voiced = harmonic > 1e-8
    value = torch.logit(share) * 0.1
    median = (
        value.masked_fill(~voiced, float("nan")).nanmedian(dim=1, keepdim=True).values
    )
    return torch.where(voiced, value - median.nan_to_num(0.0), torch.zeros_like(value))
