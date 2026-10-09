"""Native full-band harmonic-prior complex-spectrum vocoder.

Inspired by Wavehax (Yoneyama et al., arXiv:2411.06807) and the small
This is an experimental native graph using Applio's physical mel/FFT contracts, not an official weight loader.
Unlike the existing multistream backend, it predicts full-band complex spectra
directly. Global spectrogram normalization and symmetric temporal kernels make
it file-only: callers must not advertise causal or realtime synthesis.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.acoustic.spectral import SameSTFT
from rvc.lib.algorithm.acoustic.vocoder import harmonic_prior


class SpectrogramNorm(nn.Module):
    """FP32 whole-spectrogram statistics; intentionally noncausal."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        x = x.float()
        variance, mean = torch.var_mean(x, dim=(1, 2, 3), keepdim=True, correction=0)
        return (x - mean) * torch.rsqrt(variance + 1e-6) * self.weight + self.bias


class FrequencyTimeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.channels
        self.depthwise = nn.Conv2d(
            c,
            c,
            (config.frequency_kernel, config.kernel_size),
            groups=c,
            padding=(config.frequency_kernel // 2, config.kernel_size // 2),
        )
        self.norm = SpectrogramNorm(c)
        self.expand = nn.Conv2d(c, c * config.expansion, 1)
        self.project = nn.Conv2d(c * config.expansion, c, 1)
        self.scale = nn.Parameter(torch.full((1, c, 1, 1), 1 / config.depth))

    def forward(self, x):
        return x + self.scale * self.project(
            F.gelu(self.expand(self.norm(self.depthwise(x))))
        )


class WavehaxVocoder(nn.Module):
    """Physical log-mel [B,M,T], F0/voicing [B,T] -> PCM [B,T*hop].

    Zero padding supports short validation clips. Fixed /6 mel scaling is part
    of this graph, not a claim of compatibility with an upstream normalization.
    Caller-supplied phase/noise make seeded comparisons repeatable. F0 and
    voicing drive excitation explicitly rather than relying on mel harmonics.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.backend != "wavehax" or config.causal or config.streams != 1:
            raise ValueError("Invalid full-band Wavehax configuration")
        self.stft = SameSTFT(config.n_fft, config.hop_length)
        bins = config.n_fft // 2 + 1
        self.prior_projection = nn.Conv1d(bins, bins, 7, padding=3)
        self.mel_projection = nn.Conv1d(config.mel_dim, bins, 7, padding=3)
        self.input = nn.Conv2d(5, config.channels, 1)
        self.input_norm = SpectrogramNorm(config.channels)
        self.blocks = nn.ModuleList(
            [FrequencyTimeBlock(config) for _ in range(config.depth)]
        )
        self.output_norm = SpectrogramNorm(config.channels)
        self.output = nn.Conv2d(config.channels, 2, 1)

    def forward(self, mel, f0, voiced, noise=None, phase=None):
        if (
            mel.ndim != 3
            or mel.shape[1] != self.config.mel_dim
            or f0.shape != mel[:, 0].shape
        ):
            raise ValueError("Mel and F0 shapes disagree")
        with torch.no_grad(), torch.autocast(mel.device.type, enabled=False):
            prior, _ = harmonic_prior(
                f0,
                voiced,
                self.config.hop_length,
                self.config.sample_rate,
                phase=phase,
                noise=noise,
            )
            spec = self.stft(prior)
        real, imag = spec.real, spec.imag
        x = torch.stack(
            [
                real,
                imag,
                self.prior_projection(real),
                self.prior_projection(imag),
                self.mel_projection(mel / 6),
            ],
            dim=1,
        )
        x = self.input_norm(self.input(x))
        for block in self.blocks:
            if self.config.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        predicted = self.output(self.output_norm(x)).float()
        return self.stft.inverse(torch.complex(predicted[:, 0], predicted[:, 1]))
