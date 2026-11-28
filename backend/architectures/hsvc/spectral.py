"""Spectral basis, bounded-lookahead ConvNeXt-FiLM, FreqCoord 2D blocks, and Neural Timbre Bank for HSVC."""

import math
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_erb_interpolation_matrix(
    n_bands: int = 256, n_fft: int = 2048, win_length: int = 960, sr: int = 48000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a Mel/ERB-warped vocal-tract filter interpolation matrix and its minimum-phase basis.

    Maps compact perceptual band predictions (``n_bands``) smoothly onto linear FFT bins
    (``n_fft // 2 + 1``) while deriving the exact minimum-phase Hilbert cepstral response
    for every band so harmonic filtering preserves crisp glottal impulse alignment.

    Args:
        n_bands (int): Number of perceptual frequency bands (default: 256).
        n_fft (int): Synthesis STFT FFT size (default: 2048).
        win_length (int): Synthesis STFT Hann window length in samples (default: 960).
        sr (int): Sampling rate in Hz (default: 48000).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - W_erb: Partition-of-unity interpolation basis, shape [n_bands, n_fft // 2 + 1].
            - W_minphase: Minimum-phase imaginary cepstral basis, shape [n_bands, n_fft // 2 + 1].
    """
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float64)
    centers = librosa.mel_frequencies(n_mels=n_bands, fmin=0.0, fmax=sr / 2.0).astype(np.float64)
    centers[0] = 0.0
    centers[-1] = sr / 2.0

    # Triangular partition-of-unity basis across Mel-spaced band centers
    W = np.zeros((n_bands, n_bins), dtype=np.float64)
    for b in range(n_bands):
        c = centers[b]
        left = centers[b - 1] if b > 0 else -1.0
        right = centers[b + 1] if b < n_bands - 1 else sr / 2.0 + 1.0
        if b > 0:
            m_l = (freqs >= left) & (freqs <= c)
            W[b, m_l] = (freqs[m_l] - left) / max(c - left, 1e-8)
        if b < n_bands - 1:
            m_r = (freqs >= c) & (freqs <= right)
            W[b, m_r] = (right - freqs[m_r]) / max(right - c, 1e-8)
    W = W / np.maximum(W.sum(axis=0, keepdims=True), 1e-8)

    # Liftered real-cepstrum Hilbert transform for minimum-phase spectral response
    W_t = torch.from_numpy(W.astype(np.float32))
    cep = torch.fft.irfft(W_t, n=n_fft, dim=-1)
    fold = torch.zeros(n_fft, dtype=torch.float32)
    fold[0] = 1.0
    fold[1 : n_fft // 2] = 2.0
    fold[n_fft // 2] = 1.0
    quef = torch.arange(n_fft, dtype=torch.float32)
    lifter = torch.where(
        quef < win_length // 2,
        0.5 * (1.0 + torch.cos(math.pi * quef / (win_length // 2))),
        torch.zeros_like(quef),
    )
    lifter[0] = 1.0
    min_spec = torch.fft.rfft(cep * fold * lifter, n=n_fft, dim=-1)
    return W_t, min_spec.imag.contiguous()


class StreamingConvNeXtFiLMBlock(nn.Module):
    """
    1D ConvNeXt-GLU residual block with Speaker AdaLN/FiLM and bounded right lookahead.

    All excess padding is placed on the left (causal past) so real-time streaming latency
    remains strictly bounded by ``lookahead`` frames.

    Args:
        dim (int): Feature channel dimension.
        spk_dim (int): Speaker conditioning embedding dimension.
        kernel_size (int): Depthwise temporal convolution kernel size.
        dilation (int): Depthwise convolution dilation factor.
        lookahead (int): Maximum future frames visible to this block (0 = strictly causal).
        expansion (int): Pointwise GLU hidden channel expansion factor.
    """

    def __init__(
        self,
        dim: int,
        spk_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        lookahead: int = 0,
        expansion: int = 2,
    ):
        super().__init__()
        total_pad = (kernel_size - 1) * dilation
        self.pad_right = min(lookahead, total_pad)
        self.pad_left = total_pad - self.pad_right
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.film = nn.Linear(spk_dim, dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.pw1 = nn.Linear(dim, dim * expansion * 2)
        self.pw2 = nn.Linear(dim * expansion, dim)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), 0.1))

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """
        Apply bounded-lookahead ConvNeXt-FiLM transformation.

        Args:
            x (torch.Tensor): Input sequence, shape [B, C, T].
            spk (torch.Tensor): Global speaker embedding, shape [B, spk_dim].

        Returns:
            torch.Tensor: Residual output sequence, shape [B, C, T].
        """
        x_pad = F.pad(x, (self.pad_left, self.pad_right))
        h = self.dwconv(x_pad).transpose(1, 2)
        h = self.norm(h)
        scale, shift = self.film(spk).unsqueeze(1).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        a, b = self.pw1(h).chunk(2, dim=-1)
        h = self.pw2(F.silu(a) * b).transpose(1, 2)
        return x + self.gamma * h


class FreqCoordSpectralBlock(nn.Module):
    """
    2D Frequency-Time ConvNeXt-GLU block with Speaker FiLM conditioning.

    Uses symmetric padding along the frequency axis (full formant context) and bounded
    right-side lookahead along the time axis for streaming compatibility.

    Args:
        channels (int): Number of 2D feature map channels.
        spk_dim (int): Speaker conditioning embedding dimension.
        k_freq (int): Convolution kernel size along frequency axis.
        k_time (int): Convolution kernel size along time axis.
        lookahead (int): Future frame lookahead along time axis (0 = strictly causal).
    """

    def __init__(
        self, channels: int, spk_dim: int, k_freq: int = 7, k_time: int = 3, lookahead: int = 0
    ):
        super().__init__()
        self.pad_f = k_freq // 2
        self.pad_t_right = min(lookahead, k_time - 1)
        self.pad_t_left = (k_time - 1) - self.pad_t_right
        self.dw = nn.Conv2d(channels, channels, (k_freq, k_time), groups=channels)
        self.norm = nn.GroupNorm(8, channels)
        self.film = nn.Linear(spk_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """
        Refine a 2D time-frequency spectral feature map conditioned on speaker identity.

        Args:
            x (torch.Tensor): Spectral feature map, shape [B, C, F, T].
            spk (torch.Tensor): Global speaker embedding, shape [B, spk_dim].

        Returns:
            torch.Tensor: Refined spectral feature map, shape [B, C, F, T].
        """
        h = F.pad(x, (self.pad_t_left, self.pad_t_right, self.pad_f, self.pad_f))
        h = self.norm(self.dw(h))
        scale, shift = self.film(spk).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        a, b = self.pw1(h).chunk(2, dim=1)
        h = self.pw2(F.silu(a) * b)
        return x + self.gamma * h


class NeuralTimbreBank(nn.Module):
    """
    Differentiable per-speaker memory bank of phoneme- and register-aware timbre prototypes.

    Instead of relying on a single static speaker vector, each speaker stores ``num_protos``
    learnable timbre values queried frame-by-frame via multi-head attention from the fused
    content + pitch/dynamics representation. This allows different vowels, consonants, and
    vocal registers (chest/head/falsetto) to retrieve distinct speaker timbre nuances.

    Args:
        num_speakers (int): Capacity of the speaker embedding table.
        query_dim (int): Dimension of the frame-level query feature.
        spk_dim (int): Global speaker and prototype embedding dimension.
        num_protos (int): Number of timbre prototypes stored per speaker.
        n_heads (int): Number of cross-attention heads over prototypes.
    """

    def __init__(
        self,
        num_speakers: int,
        query_dim: int,
        spk_dim: int = 256,
        num_protos: int = 16,
        n_heads: int = 4,
    ):
        super().__init__()
        self.num_protos = num_protos
        self.spk_dim = spk_dim
        self.n_heads = n_heads
        self.head_dim = spk_dim // n_heads

        self.global_emb = nn.Embedding(num_speakers, spk_dim)
        nn.init.normal_(self.global_emb.weight, std=0.05)

        self.proto_keys = nn.Parameter(torch.randn(num_protos, spk_dim) * 0.05)
        self.proto_values = nn.Embedding(num_speakers, num_protos * spk_dim)
        nn.init.normal_(self.proto_values.weight, std=0.03)

        self.q_proj = nn.Linear(query_dim, spk_dim)
        self.out_proj = nn.Linear(spk_dim, query_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query_feat: torch.Tensor,
        sid: torch.Tensor,
        ref_timbre: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve global speaker embedding and frame-aligned local timbre features.

        Args:
            query_feat (torch.Tensor): Content + prosody query sequence, shape [B, T, query_dim].
            sid (torch.Tensor): Target speaker IDs, shape [B].
            ref_timbre (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional one-shot override
                ``(global_emb [B, spk_dim], proto_values [B, num_protos, spk_dim])`` from
                ``HSVCReferenceEncoder``. When None, looks up ``sid`` in the learned tables.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - g_spk: Global speaker embedding [B, spk_dim].
                - local_timbre: Frame-level prototype timbre residual [B, T, query_dim].
        """
        B, T, _ = query_feat.shape
        if ref_timbre is not None:
            g_spk, p_vals = ref_timbre
            g_spk = g_spk.to(dtype=query_feat.dtype, device=query_feat.device)
            p_vals = p_vals.to(dtype=query_feat.dtype, device=query_feat.device)
            if g_spk.shape[0] == 1 and B > 1:
                g_spk = g_spk.expand(B, -1)
            if p_vals.shape[0] == 1 and B > 1:
                p_vals = p_vals.expand(B, -1, -1)
            v = p_vals.reshape(B, self.num_protos, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            g_spk = self.global_emb(sid)
            v = self.proto_values(sid).view(B, self.num_protos, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(query_feat).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.proto_keys.view(1, self.num_protos, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.spk_dim)
        return g_spk, self.out_proj(out)
