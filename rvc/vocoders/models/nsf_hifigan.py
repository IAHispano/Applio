"""NSF-HiFiGAN, as OpenVPI SingingVocoders trains it: HiFi-GAN with the NSF
sine excitation added at every stage, or, as ``mini_nsf``, a single sine at
the second stage's rate (PC-NSF-HiFiGAN's generator).

Ported from https://github.com/openvpi/SingingVocoders (MIT). Module names are
its own and those of ``rvc.lib.algorithm.generators.openvpi``, which loads the
exports.
"""

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LRELU_SLOPE = 0.1
#: ``architecture`` of an NSF-HiFiGAN rectified vocoder export.
ARCHITECTURE = "openvpi_nsf_hifigan"


def _padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
             for d in dilation]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, padding=_padding(kernel_size))
             for _ in dilation]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE)) + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
             for d in dilation]
        )

    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


class SourceModuleHnNSF(nn.Module):
    """Sine plus harmonics merged to one excitation."""

    def __init__(self, sample_rate, harmonic_num, sine_amp=0.1, noise_std=0.003):
        super().__init__()
        self.sample_rate = sample_rate
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.l_linear = nn.Linear(self.dim, 1)

    def forward(self, f0, upp):
        f0 = f0.unsqueeze(-1)
        rad = f0 / self.sample_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1) * torch.arange(1, self.dim + 1, device=f0.device)
        start = torch.rand(1, 1, self.dim, device=f0.device)
        start[..., 0] = 0
        sines = torch.sin(2 * np.pi * (rad + start)) * self.sine_amp
        uv = F.interpolate((f0 > 0).float().transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
        noise = (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sines)
        return torch.tanh(self.l_linear(sines * uv + noise))


class NSFHiFiGAN(nn.Module):
    """Log mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop]."""

    def __init__(self, sample_rate, num_mels, upsample_initial_channel, upsample_rates,
                 upsample_kernel_sizes, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, mini_nsf, harmonic_num=8, noise_sigma=0.0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.mini_nsf = mini_nsf
        self.noise_sigma = noise_sigma
        if mini_nsf:
            # The sine is made at the rate after the second stage and joins there.
            self.source_sr = sample_rate / int(np.prod(upsample_rates[2:]))
            self.upp = int(np.prod(upsample_rates[:2]))
        else:
            self.upp = int(np.prod(upsample_rates))
            self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)
            self.noise_convs = nn.ModuleList()

        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        block = ResBlock1 if resblock == "1" else ResBlock2
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(nn.ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(block(ch, kernel, dilation))
            if not mini_nsf:
                if i + 1 < len(upsample_rates):
                    stride = int(np.prod(upsample_rates[i + 1:]))
                    self.noise_convs.append(
                        nn.Conv1d(1, ch, kernel_size=stride * 2, stride=stride, padding=stride // 2)
                    )
                else:
                    self.noise_convs.append(nn.Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = nn.Conv1d(1, ch, 1)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        # Set by ``apply_precision_policy`` under AMP: the mel conv, the
        # source, the upsamplers and the output layer in FP32, and the stream
        # with them; the residual convs stay under autocast and add onto it.
        self.fp32_residuals = False

    def _fp32_region(self, x):
        """A context with autocast off, when ``fp32_residuals`` is set."""
        if not self.fp32_residuals:
            return nullcontext()
        return torch.autocast(x.device.type, enabled=False)

    def _fp32(self, x):
        return x.float() if self.fp32_residuals else x

    def _fast_sine(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        return torch.sin(2 * np.pi * rad.reshape(f0.shape[0], 1, -1))

    def forward(self, mel, f0):
        f0 = f0.float()
        with self._fp32_region(mel):
            source = self._fast_sine(f0) if self.mini_nsf else self.m_source(f0, self.upp).transpose(1, 2)
            x = self.conv_pre(self._fp32(mel))
        if self.noise_sigma > 0:
            x = x + self.noise_sigma * torch.randn_like(x)
        for i, up in enumerate(self.ups):
            with self._fp32_region(x):
                x = up(F.leaky_relu(self._fp32(x), LRELU_SLOPE))
                if not self.mini_nsf:
                    x = x + self.noise_convs[i](source)
                elif i == 1:
                    x = x + self.source_conv(source)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        # HiFi-GAN's last leaky ReLU is at PyTorch's default slope.
        with self._fp32_region(x):
            return torch.tanh(self.conv_post(F.leaky_relu(self._fp32(x))))
