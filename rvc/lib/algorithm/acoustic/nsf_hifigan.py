"""Frozen pitch-controllable OpenVPI NSF-HiFiGAN at the shared mel boundary.
"""

from types import SimpleNamespace
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel, dilations):
        super().__init__()
        self.acts1 = nn.ModuleList(nn.LeakyReLU(.1) for _ in dilations)
        self.acts2 = nn.ModuleList(nn.LeakyReLU(.1) for _ in dilations)
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, dilation=d,
                      padding=(kernel * d - d) // 2) for d in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)
            for _ in dilations
        )

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            x = c2(a2(c1(a1(x)))) + x
        return x


class MiniNSFGenerator(nn.Module):
    """Weight-compatible inference graph, with the original source phase math."""

    def __init__(self, config):
        super().__init__()
        rates = config["upsample_rates"]
        self.source_sr = config["sample_rate"] / math.prod(rates[2:])
        self.upp = math.prod(rates[:2])
        self.num_kernels = len(config["resblock_kernel_sizes"])
        channels = config["upsample_initial_channel"]
        self.conv_pre = nn.Conv1d(config["num_mels"], channels, 7, padding=3)
        self.ups, self.resblocks = nn.ModuleList(), nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(rates, config["upsample_kernel_sizes"])):
            channels //= 2
            self.ups.append(nn.ConvTranspose1d(
                2 * channels, channels, kernel, rate, padding=(kernel - rate) // 2
            ))
            for res_kernel, dilations in zip(
                config["resblock_kernel_sizes"], config["resblock_dilation_sizes"]
            ):
                self.resblocks.append(ResidualBlock(channels, res_kernel, dilations))
            if i == 1:
                self.source_conv = nn.Conv1d(1, channels, 1)
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3)
        self.act_post = nn.LeakyReLU()

    def _fast_sine(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device)
        start = f0.unsqueeze(-1) / self.source_sr
        delta = F.pad(start[:, 1:] - start[:, :-1], (0, 0, 0, 1))
        phase = start * n + .5 * delta * n * (n - 1) / self.upp
        increment = torch.fmod(phase[..., -1:].float() + .5, 1.) - .5
        accumulated = increment.cumsum(dim=1).fmod(1.).to(f0)
        phase = phase + F.pad(accumulated[:, :-1], (0, 0, 1, 0))
        return torch.sin(2 * math.pi * phase.reshape(f0.shape[0], 1, -1))

    def forward(self, mel, f0):
        source = self._fast_sine(f0.float())
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, .1))
            if i == 1:
                x = x + self.source_conv(source)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        return torch.tanh(self.conv_post(self.act_post(x)))


class NSFHiFiGANVocoder(nn.Module):
    """Physical 40–16,000 Hz log-mel/F0 -> exactly frames*512 PCM samples.

    The supported mini-NSF graph has noise_sigma=0 and a deterministic source;
    caller noise is accepted for the shared file API but is unused. Streaming
    phase state is unsupported. Imported weights stay frozen in acoustic runs.
    """

    @staticmethod
    def validate_config(config):
        expected = dict(
            sample_rate=44100, num_mels=128, upsample_initial_channel=512,
            upsample_rates=[8, 8, 2, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4, 4],
            resblock="1", resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5]] * 3,
            mini_nsf=True, harmonic_num=8, noise_sigma=0.,
        )
        if config != expected:
            raise ValueError("Unsupported NSF-HiFiGAN graph; use the verified mini-NSF export")

    def __init__(self, config):
        super().__init__()
        self.validate_config(config)
        self.config = SimpleNamespace(sample_rate=44100, hop_length=512,
                                      mel_dim=128, causal=False)
        self.generator = MiniNSFGenerator(config)

    def forward(self, mel, f0, voiced=None, phase=None, noise=None, return_phase=False):
        if phase is not None or return_phase:
            raise ValueError("NSF-HiFiGAN supports file synthesis, not streaming phase state")
        if mel.ndim != 3 or mel.shape[1] != 128 or f0.shape != mel[:, 0].shape:
            raise ValueError("NSF-HiFiGAN requires aligned physical mel and pitch frames")
        if not torch.isfinite(f0).all() or torch.any(f0 < 0):
            raise ValueError("F0 must be finite and nonnegative")
        if voiced is not None:
            if voiced.shape != f0.shape:
                raise ValueError("Voicing and F0 shapes disagree")
            f0 = torch.where(voiced >= .5, f0, torch.zeros_like(f0))
        return self.generator(mel, f0)[:, 0]
