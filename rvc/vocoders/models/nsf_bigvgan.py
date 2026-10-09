"""NSF-BigVGAN: BigVGAN's anti-aliased SnakeBeta blocks over HiFi-GAN's
transposed upsamplers, with the NSF sine excitation added at every stage.

Generator, source, AMP block and anti-aliasing filters ported from
https://github.com/PlayVoice/NSF-BigVGAN (MIT); with Triton installed, the
filtered SnakeBeta runs as one op on CUDA (``snake_alias_triton.py``).
"""

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from rvc.vocoders.models.snake_alias_triton import FusedResidualSnakeAlias, FusedSnakeAlias
except ImportError:
    FusedSnakeAlias = FusedResidualSnakeAlias = None

#: ``architecture`` of an NSF-BigVGAN rectified vocoder export.
ARCHITECTURE = "nsf_bigvgan"
#: The fixed merge of the sine and its ten overtones into one excitation.
MERGE_WEIGHT = (0.2942, -0.2243, 0.0033, -0.0056, -0.0020, -0.0046, 0.0221, -0.0083, -0.0241, -0.0036, -0.0581)
MERGE_BIAS = 0.0008
#: Taps of the 2x filters around each SnakeBeta, and their paddings: the
#: upsampler's on each side of its input, the lowpass's on the left.
ALIAS_KERNEL = 12
PHASE_PAD = (ALIAS_KERNEL // 4, ALIAS_KERNEL // 4)
DOWN_PAD = ALIAS_KERNEL // 2 - 1


def _padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def kaiser_sinc_filter(cutoff, half_width, kernel_size):
    """alias-free-torch's Kaiser-windowed sinc lowpass, [kernel_size] (even);
    ``cutoff`` and ``half_width`` as fractions of the sample rate."""
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half_size, half_size) + 0.5
    kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return kernel / kernel.sum()


class _SnakeBetaFunction(torch.autograd.Function):
    """SnakeBeta that saves only its input for backward, instead of the three
    oversampled tensors autograd keeps for the plain expression."""

    @staticmethod
    def forward(ctx, x, log_alpha, log_beta):
        ctx.save_for_backward(x, log_alpha, log_beta)
        alpha = log_alpha.exp()[:, None]
        inv_beta = torch.exp(-log_beta)[:, None]
        return torch.addcmul(x, torch.sin(x * alpha).square(), inv_beta)

    @staticmethod
    def backward(ctx, grad):
        x, log_alpha, log_beta = ctx.saved_tensors
        alpha = log_alpha.exp()[:, None]
        inv_beta = torch.exp(-log_beta)[:, None]
        sine = torch.sin(x * alpha)
        slope = grad * (2.0 * sine * torch.cos(x * alpha))
        grad_x = torch.addcmul(grad, slope, alpha * inv_beta).to(x.dtype)
        # The parameters are logs: each gradient carries the parameter itself.
        grad_log_alpha = (slope * x).sum((0, 2)) * (alpha * inv_beta)[:, 0]
        grad_log_beta = -(grad * sine.square()).sum((0, 2)) * inv_beta[:, 0]
        return grad_x, grad_log_alpha.to(log_alpha.dtype), grad_log_beta.to(log_beta.dtype)


class SnakeBeta(nn.Module):
    """BigVGAN v2's ``x + sin^2(alpha * x) / beta``, per-channel, log-scale."""

    def __init__(self, channels: int):
        super().__init__()
        # Logs of alpha and beta: zeros start both at 1.
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SnakeBetaFunction.apply(x, self.alpha, self.beta)


class SnakeAlias(nn.Module):
    """SnakeBeta at twice the rate, between the original's 2x Kaiser filters;
    one Triton op on CUDA."""

    def __init__(self, channels):
        super().__init__()
        self.act = SnakeBeta(channels)
        kernel = kaiser_sinc_filter(0.25, 0.3, ALIAS_KERNEL)
        self.register_buffer("filter", kernel, persistent=False)
        # The upsampler as the fused op takes it: per output phase, the weight
        # of each input sample from PHASE_PAD[0] before to PHASE_PAD[1] after.
        reach = PHASE_PAD[0]
        phase_weight = torch.zeros(2, 2 * reach + 1)
        for phase in range(2):
            for offset in range(-reach, reach + 1):
                tap = phase + (ALIAS_KERNEL - 2) // 2 - 2 * offset
                if 0 <= tap < ALIAS_KERNEL:
                    phase_weight[phase, offset + reach] = 2 * kernel[tap]
        self.register_buffer("phase_weight", phase_weight, persistent=False)

    def forward(self, x):
        if FusedSnakeAlias is not None and x.is_cuda:
            return FusedSnakeAlias.apply(x, *self._fused_args(x))
        channels = x.shape[1]
        kernel = self.filter.to(x.dtype)[None, None].expand(channels, -1, -1)
        pad = ALIAS_KERNEL // 2 - 1
        crop = 2 * pad + (ALIAS_KERNEL - 2) // 2
        x = F.pad(x, (pad, pad), mode="replicate")
        x = 2 * F.conv_transpose1d(x, kernel, stride=2, groups=channels)[..., crop:-crop]
        x = F.pad(self.act(x), (DOWN_PAD, ALIAS_KERNEL // 2), mode="replicate")
        return F.conv1d(x, kernel, stride=2, groups=channels)

    def forward_residual(self, x, r):
        """``(self(x + r), x + r)`` with the add inside the kernel; CUDA only."""
        return FusedResidualSnakeAlias.apply(x, r, *self._fused_args(x))

    def _fused_args(self, x):
        # Written in the dtype the next conv runs in rather than cast to it.
        if torch.is_autocast_enabled(x.device.type):
            dtype = torch.get_autocast_dtype(x.device.type)
        else:
            dtype = x.dtype
        return self.act.alpha, self.act.beta, self.phase_weight, self.filter, PHASE_PAD, DOWN_PAD, dtype


class SineGen(nn.Module):
    """The sine at f0 and its overtones, with noise in place of the unvoiced
    steps: f0 [B, T, 1] at the sample rate -> [B, T, harmonic_num + 1]."""

    def __init__(self, sample_rate, harmonic_num, sine_amp=0.1, noise_std=0.003):
        super().__init__()
        self.sample_rate = sample_rate
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std

    @torch.no_grad()
    def forward(self, f0):
        rad = (f0 * torch.arange(1, self.dim + 1, device=f0.device) / self.sample_rate) % 1
        # Random initial phase, but for the fundamental.
        start = torch.rand(f0.shape[0], self.dim, device=f0.device)
        start[:, 0] = 0
        rad[:, 0, :] = rad[:, 0, :] + start
        # -1 at every step where the running phase wraps, against cumsum overflow.
        wrapped = torch.cumsum(rad, 1) % 1
        shift = torch.zeros_like(rad)
        shift[:, 1:, :] = ((wrapped[:, 1:, :] - wrapped[:, :-1, :]) < 0) * -1.0
        sines = torch.sin(torch.cumsum(rad + shift, dim=1) * 2 * np.pi) * self.sine_amp
        uv = (f0 > 0).to(sines.dtype)
        noise = (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sines)
        return sines * uv + noise


class SourceModuleHnNSF(nn.Module):
    """``SineGen``'s harmonics merged to one excitation by fixed weights."""

    def __init__(self, sample_rate, sine_amp=0.1, noise_std=0.003):
        super().__init__()
        self.l_sin_gen = SineGen(sample_rate, len(MERGE_WEIGHT) - 1, sine_amp, noise_std)
        self.register_buffer("merge_w", torch.tensor([MERGE_WEIGHT]))
        self.register_buffer("merge_b", torch.tensor([MERGE_BIAS]))

    def forward(self, f0):
        return torch.tanh(F.linear(self.l_sin_gen(f0), self.merge_w) + self.merge_b)


class AMPBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
             for d in dilation]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, padding=_padding(kernel_size))
             for _ in dilation]
        )
        self.activations = nn.ModuleList(SnakeAlias(channels) for _ in range(2 * len(dilation)))

    def forward(self, x):
        units = list(zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]))
        if FusedResidualSnakeAlias is None or not x.is_cuda or x.dtype != torch.float32:
            for c1, c2, a1, a2 in units:
                x = c2(a2(c1(a1(x)))) + x
            return x
        # Each residual add is done inside the next activation's kernel.
        branch = None
        for c1, c2, a1, a2 in units:
            if branch is None:
                hidden = a1(x)
            else:
                hidden, x = a1.forward_residual(x, branch)
            branch = c2(a2(c1(hidden)))
        return branch + x


class NSFBigVGAN(nn.Module):
    """Log mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop]."""

    def __init__(self, sample_rate, num_mels, upsample_initial_channel, upsample_rates,
                 upsample_kernel_sizes, resblock_kernel_sizes, resblock_dilation_sizes):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.upp = int(np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate)
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(nn.ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2))
            if i + 1 < len(upsample_rates):
                stride = int(np.prod(upsample_rates[i + 1:]))
                self.noise_convs.append(
                    nn.Conv1d(1, ch, kernel_size=stride * 2, stride=stride, padding=stride // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, ch, kernel_size=1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, kernel, dilation))
        self.activation_post = SnakeAlias(ch)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # Set by ``apply_precision_policy`` under AMP: the excitation, the mel
        # conv, the upsamplers, the residual stream and the output layer stay
        # in FP32.
        self.fp32_residuals = False

    def _fp32_region(self, x):
        """A context with autocast off, when ``fp32_residuals`` is set."""
        if not self.fp32_residuals:
            return nullcontext()
        return torch.autocast(x.device.type, enabled=False)

    def _fp32(self, x):
        return x.float() if self.fp32_residuals else x

    def forward(self, mel, f0):
        f0 = F.interpolate(f0.float()[:, None], scale_factor=self.upp).transpose(1, 2)
        with self._fp32_region(mel):
            source = self.m_source(f0).transpose(1, 2)
            x = F.mish(self.conv_pre(self._fp32(mel)))
        for i, up in enumerate(self.ups):
            # The upsampler and the excitation in FP32 keep the stage's
            # residual stream there; the convs in the blocks stay under autocast.
            with self._fp32_region(x):
                x = up(self._fp32(x)) + self.noise_convs[i](source)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        # conv_post's few weights take the whole waveform's gradient, which
        # overflows FP16.
        with self._fp32_region(x):
            return torch.tanh(self.conv_post(self.activation_post(self._fp32(x))))
