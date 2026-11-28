"""Closed-form anti-aliased bandlimited harmonic and energy-gated noise excitation for HSVC."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_sum(
    x: torch.Tensor, phase: torch.Tensor, first: torch.Tensor, last: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate the closed-form trigonometric series sum_{k=first}^{last} cos(k * x + phase).

    Returns zero wherever last < first, and uses the analytical limit when sin(x / 2) -> 0
    to avoid numerical instability at phase wraps.

    Args:
        x (torch.Tensor): Angular step per harmonic index.
        phase (torch.Tensor): Base phase offset.
        first (torch.Tensor): Inclusive starting harmonic index.
        last (torch.Tensor): Inclusive ending harmonic index.

    Returns:
        torch.Tensor: Exact sum of cosines with the same shape as the broadcasted inputs.
    """
    count = (last - first + 1.0).clamp_min(0.0)
    x_wrap = torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi
    half = torch.sin(0.5 * x_wrap)
    small = half.abs() < 1e-6
    safe_half = torch.where(small, torch.ones_like(half), half)
    kernel = torch.where(small, count, torch.sin(0.5 * count * x_wrap) / safe_half)
    return kernel * torch.cos(0.5 * (first + last) * x_wrap + phase)


def bandlimited_pulse(
    turns: torch.Tensor,
    f0: torch.Tensor,
    phases: torch.Tensor,
    top_hz: float,
    taper_hz: float = 800.0,
) -> torch.Tensor:
    """
    Synthesize an alias-free, raised-cosine-tapered harmonic pulse train.

    Harmonics are summed analytically up to ``top_hz`` with a smooth raised-cosine
    rolloff over the top ``taper_hz`` band so pitch glides never click or alias across
    Nyquist. Providing multiple entries in ``phases`` scatters harmonic phases across
    residue classes (e.g., golden-ratio offsets) to reduce peak crest factor.

    Args:
        turns (torch.Tensor): Fractional instantaneous phase in [0, 1), shape [B, Samples].
        f0 (torch.Tensor): Sample-rate fundamental frequency in Hz, shape [B, Samples].
        phases (torch.Tensor): Per-class phase offsets in radians, shape [C].
        top_hz (float): Upper harmonic cutoff frequency in Hz (below Nyquist).
        taper_hz (float): Width of the upper raised-cosine rolloff band in Hz.

    Returns:
        torch.Tensor: Normalized bandlimited pulse waveform, shape [B, Samples].
    """
    classes = phases.shape[0]
    start_hz = top_hz - taper_hz
    idx = torch.arange(classes, device=f0.device, dtype=torch.float32).view(1, classes, 1)
    ph = phases.view(1, classes, 1)
    turns_c = turns.unsqueeze(1)
    x = (2.0 * math.pi * torch.remainder(classes * turns_c, 1.0)).float()
    base_ph = (2.0 * math.pi * torch.remainder(idx.double() * turns_c, 1.0)).float() + ph

    f0_u = f0.unsqueeze(1).clamp_min(35.0)
    first = (idx == 0).float()  # Omit DC component (k = 0)
    flat = torch.floor((start_hz / f0_u - idx) / classes)
    top = torch.ceil((top_hz / f0_u - idx) / classes) - 1.0
    slope = math.pi * classes * f0_u / taper_hz
    offset = math.pi * (idx * f0_u - start_hz) / taper_hz
    fade_first = torch.maximum(flat + 1.0, first)

    # Raised-cosine taper on the top harmonic band: 0.5 + 0.5 * cos(t)
    faded = (
        0.5 * _cosine_sum(x, base_ph, fade_first, top)
        + 0.25 * _cosine_sum(x + slope, base_ph + offset, fade_first, top)
        + 0.25 * _cosine_sum(x - slope, base_ph - offset, fade_first, top)
    )
    raw = (_cosine_sum(x, base_ph, first.expand_as(flat), flat) + faded).sum(dim=1)
    n_harm = torch.floor(top_hz / f0.clamp_min(35.0)).clamp_min(1.0)
    return raw * torch.rsqrt(0.5 * n_harm) * 0.25


class BandlimitedSourceGenerator(nn.Module):
    """
    Deterministic harmonic + energy-gated aperiodic source excitation module.

    Generates two bandlimited harmonic pulse channels (phase-aligned and 7-class
    golden-ratio phase-scattered) and three aperiodic noise channels (broadband white
    noise plus glottal-synchronized cosine/sine-modulated noise for natural breathiness).
    All channels are strictly multiplied by the continuous silence gate ``g_sil``.

    Args:
        sample_rate (int): Audio sampling rate in Hz (default: 48000).
        hop_in (int): Input frontend hop size in samples per 100 Hz frame (default: 480).
        nyquist_ratio (float): Fraction of Nyquist used as the harmonic ceiling (default: 0.99).
    """

    def __init__(self, sample_rate: int = 48000, hop_in: int = 480, nyquist_ratio: float = 0.99):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_in = hop_in
        self.top_hz = nyquist_ratio * (sample_rate / 2.0)
        self.register_buffer("aligned_phases", torch.zeros(1), persistent=False)
        # Golden-ratio phase offsets reduce waveform crest factor for smoother vocal timbre
        scatter = torch.tensor(
            [0.0, 2.39996, 4.79993, 0.91671, 3.31667, 5.71664, 1.83342], dtype=torch.float32
        )
        self.register_buffer("scatter_phases", scatter, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        f0_100hz: torch.Tensor,
        vuv_conf: torch.Tensor,
        g_sil: torch.Tensor,
        init_turns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synthesize 48 kHz harmonic and noise excitation waveforms from 100 Hz control curves.

        Args:
            f0_100hz (torch.Tensor): Fundamental frequency in Hz (0 for unvoiced), shape [B, T].
            vuv_conf (torch.Tensor): Voiced/unvoiced confidence in [0, 1], shape [B, T].
            g_sil (torch.Tensor): Smooth silence gate in [0, 1], shape [B, T].
            init_turns (Optional[torch.Tensor]): Initial phase in [0, 1) for streaming continuity, shape [B, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - harm_wav: Harmonic excitation [B, 2, T * hop_in].
                - noise_wav: Energy-gated aperiodic excitation [B, 3, T * hop_in].
                - end_turns: Final fractional phase [B, 1] for chaining streaming chunks.
        """
        B, T = f0_100hz.shape
        samples = T * self.hop_in
        device = f0_100hz.device

        # Clean F0 contour and upsample control curves linearly to audio sample rate
        voiced = (f0_100hz > 1.0).float()
        f0_clean = torch.where(
            f0_100hz > 1.0, f0_100hz.clamp(35.0, 2400.0), torch.full_like(f0_100hz, 150.0)
        )

        f0_sr = F.interpolate(
            f0_clean.unsqueeze(1), size=samples, mode="linear", align_corners=False
        ).squeeze(1)
        v_gate_100 = voiced * (0.35 + 0.65 * vuv_conf.clamp(0.0, 1.0)) * g_sil
        v_gate_sr = F.interpolate(
            v_gate_100.unsqueeze(1), size=samples, mode="linear", align_corners=False
        ).squeeze(1)
        sil_sr = F.interpolate(
            g_sil.unsqueeze(1), size=samples, mode="linear", align_corners=False
        ).squeeze(1)

        # Integrate instantaneous frequency in float64 to prevent phase drift on long clips
        inc = f0_sr.double() / float(self.sample_rate)
        turns = torch.cumsum(inc, dim=-1)
        if init_turns is not None:
            turns = turns + init_turns.double()
        end_turns = torch.remainder(turns[:, -1:], 1.0)
        turns = torch.remainder(turns, 1.0)

        p_aligned = bandlimited_pulse(turns, f0_sr, self.aligned_phases, self.top_hz) * v_gate_sr
        p_scatter = bandlimited_pulse(turns, f0_sr, self.scatter_phases, self.top_hz) * v_gate_sr
        harm_wav = torch.stack([p_aligned, p_scatter], dim=1)

        # Strictly energy-gated noise guarantees zero noise injection during digital silence
        white = torch.randn(B, samples, device=device, dtype=torch.float32) * 0.25 * sil_sr
        theta = (2.0 * math.pi * turns).float()
        n_cos = white * torch.cos(theta) * v_gate_sr
        n_sin = white * torch.sin(theta) * v_gate_sr
        noise_wav = torch.stack([white, n_cos, n_sin], dim=1)
        return harm_wav, noise_wav, end_turns
