import math

import numpy as np
import torch
from torch.nn import functional as F

ARCHITECTURE = "pupu_vocoder"
# Taps, cutoff and half width of the filters around each activation, the last
# two as fractions of the sampling rate. The released weights only render
# through the ones they were trained with.
ALIAS_KERNEL = 12
ALIAS_CUTOFF = 0.25
ALIAS_HALF_WIDTH = 0.3
# Zero crossings of the resampling filters on each side.
SINC_ZEROS = 8


def kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int):
    """
    Kaiser-windowed sinc lowpass filter.

    Args:
        cutoff (float): Cutoff as a fraction of the sampling rate.
        half_width (float): Half of the transition band, as a fraction of the sampling rate.
        kernel_size (int): Number of taps, an even number.
    """
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


def sinc_lowpass(cutoff: float):
    """
    Hann-windowed sinc lowpass filter.

    Args:
        cutoff (float): Cutoff as a fraction of the sampling rate.
    """
    half_size = int(SINC_ZEROS / cutoff / 2)
    time = torch.arange(-half_size, half_size + 1, dtype=torch.float64)
    window = torch.hann_window(2 * half_size + 1, periodic=False, dtype=torch.float64)
    kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (kernel / kernel.sum()).float()


def lowpass(x: torch.Tensor, kernel: torch.Tensor, stride: int = 1):
    """
    Filter every channel of a signal, behind replicate padding.

    Args:
        x (torch.Tensor): Signal of shape (batch, channels, time).
        kernel (torch.Tensor): Filter taps.
        stride (int, optional): Step between the output samples. Defaults to 1.
    """
    batch, channels, _ = x.shape
    half_size = kernel.shape[-1] // 2
    x = F.pad(
        x.reshape(batch * channels, 1, -1), (half_size, half_size), mode="replicate"
    )
    return F.conv1d(x, kernel[None, None], stride=stride).reshape(batch, channels, -1)


class ADAASnakeBeta(torch.nn.Module):
    """
    SnakeBeta activation by first-order anti-derivative anti-aliasing: the mean
    of the activation between each sample and the one before it.

    Args:
        channels (int): Number of channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Logarithms of alpha and beta
        self.alpha = torch.nn.Parameter(torch.zeros(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        previous = F.pad(x[..., :-1], (1, 0))
        x_sum, x_delta = x + previous, x - previous
        return x_sum / 2.0 + (1.0 / (2.0 * (beta + 1e-9))) * (
            1.0 - torch.cos(alpha * x_sum) * torch.sinc(alpha * x_delta / math.pi)
        )


class Activation1d(torch.nn.Module):
    """
    ADAASnakeBeta applied at twice the rate, between two lowpass filters, so
    the overtones it creates do not fold back into the band.

    Args:
        channels (int): Number of channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.act = ADAASnakeBeta(channels)
        kernel = kaiser_sinc_filter(ALIAS_CUTOFF, ALIAS_HALF_WIDTH, ALIAS_KERNEL)
        self.register_buffer("filter", kernel, persistent=False)

    def forward(self, x: torch.Tensor):
        channels = x.shape[1]
        # The activation reads differences between samples, so it runs in FP32
        with torch.autocast(x.device.type, enabled=False):
            kernel = self.filter[None, None].expand(channels, -1, -1)
            pad = ALIAS_KERNEL // 2 - 1
            crop = 2 * pad + (ALIAS_KERNEL - 2) // 2
            y = F.pad(x.float(), (pad, pad), mode="replicate")
            y = 2 * F.conv_transpose1d(y, kernel, stride=2, groups=channels)
            y = self.act(y[..., crop:-crop])
            y = F.pad(y, (ALIAS_KERNEL // 2 - 1, ALIAS_KERNEL // 2), mode="replicate")
            y = F.conv1d(y, kernel, stride=2, groups=channels)
        return y.to(x.dtype)


class ResBlock(torch.nn.Module):
    """
    Residual block of dilated convolutions with anti-aliased activations.

    Args:
        channels (int): Number of channels.
        kernel_size (int, optional): Kernel size of the convolutions. Defaults to 3.
        dilation (tuple, optional): Dilation of each first convolution. Defaults to (1, 3, 5).
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                )
                for d in dilation
            ]
        )
        self.convs2 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels, channels, kernel_size, padding=(kernel_size - 1) // 2
                )
                for _ in dilation
            ]
        )
        self.activations = torch.nn.ModuleList(
            [Activation1d(channels) for _ in range(2 * len(dilation))]
        )

    def forward(self, x: torch.Tensor):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            x = conv2(act2(conv1(act1(x)))) + x
        return x


class ResampleUpsampler(torch.nn.Module):
    """
    Upsampler without a transposed convolution: the signal is zero-stuffed and
    lowpassed, and the band the lowpass leaves empty is filled from the mel
    features, convolved and highpassed.

    Args:
        initial_channels (int): Number of channels of the mel features.
        rate (int): Upsampling rate.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(
        self, initial_channels: int, rate: int, in_channels: int, out_channels: int
    ):
        super().__init__()
        self.rate = rate
        self.convolution_after = torch.nn.Conv1d(in_channels, out_channels, 1)
        self.convolution_noise = torch.nn.Conv1d(
            initial_channels, in_channels, 7, padding=3
        )
        self.register_buffer("kernel", sinc_lowpass(0.5 / rate), persistent=False)

    def forward(self, x: torch.Tensor, x0: torch.Tensor, total_rate: int):
        # The convolution over the zero-stuffed mel features, as the transposed
        # convolution it is equal to
        y0 = F.conv_transpose1d(
            x0,
            self.convolution_noise.weight.transpose(0, 1).flip(-1),
            self.convolution_noise.bias,
            stride=total_rate,
            padding=3,
            output_padding=total_rate - 1,
        )
        with torch.autocast(x.device.type, enabled=False):
            y0 = y0.float()
            y0 = y0 - lowpass(y0, self.kernel)
            y = x.new_zeros(x.shape[0], x.shape[1], x.shape[2] * self.rate).float()
            y[:, :, :: self.rate] = x.float()
            y = lowpass(y, self.kernel)
        return self.convolution_after((y + y0).to(x.dtype))


def sample_f0(f0: torch.Tensor, upp: int):
    """
    Upsamples the fundamental frequency to the sample rate, with each frame's
    value at the frame's centre: linear between voiced frames, held up to an
    unvoiced one.

    Args:
        f0 (torch.Tensor): Fundamental frequency per frame, shape (batch, frames).
        upp (int): Samples per frame.

    Returns:
        The frequency per sample and the mask of the voiced samples.
    """
    after = torch.cat((f0[:, 1:], f0[:, -1:]), dim=1)
    left = torch.where(f0 > 0, f0, after)[..., None]
    right = torch.where(after > 0, after, f0)[..., None]
    along = torch.arange(upp, device=f0.device, dtype=f0.dtype) / upp
    voiced = torch.where(
        along < 0.5, (f0 > 0)[..., None], (after > 0)[..., None]
    ).flatten(1)
    return (left + (right - left) * along).flatten(1) * voiced, voiced


class SineSource(torch.nn.Module):
    """
    Harmonic excitation: a sine at the fundamental frequency and its overtones,
    noise where unvoiced, merged into one signal.

    Args:
        sample_rate (int): Sampling rate of the audio.
        harmonic_num (int, optional): Number of overtones above the fundamental. Defaults to 0.
        amp (float, optional): Amplitude of the sines. Defaults to 0.1.
        noise_std (float, optional): Standard deviation of the noise on voiced frames. Defaults to 0.003.
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        amp: float = 0.1,
        noise_std: float = 0.003,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.dim = harmonic_num + 1
        self.amp = amp
        self.noise_std = noise_std
        self.l_linear = torch.nn.Linear(self.dim, 1)

    @torch.no_grad()
    def _sines(self, f0: torch.Tensor, upp: int):
        rate, voiced = sample_f0(f0, upp)
        harmonics = rate.unsqueeze(-1) * torch.arange(
            1, self.dim + 1, device=f0.device
        )
        # Random initial phase, but for the fundamental
        start = torch.rand(f0.shape[0], 1, self.dim, device=f0.device)
        start[..., 0] = 0
        phase = torch.cumsum(harmonics.double() / self.sample_rate, dim=1) % 1
        sines = torch.sin((phase.float() + start) * 2 * np.pi) * self.amp
        uv = voiced.unsqueeze(-1).float()
        noise = uv * self.noise_std + (1 - uv) * self.amp / 3
        noise = noise * torch.randn_like(sines)
        # An overtone above Nyquist would fold back below it
        return sines * uv * (harmonics < self.sample_rate / 2) + noise

    def forward(self, f0: torch.Tensor, upp: int):
        return torch.tanh(self.l_linear(self._sines(f0, upp))).transpose(1, 2)


class PupuVocoder(torch.nn.Module):
    """
    Pupu-Vocoder (Aliasing-Free Neural Audio Synthesis, arXiv 2512.20211):
    BigVGAN with the transposed upsamplers replaced by resampling and
    SnakeBeta by its anti-aliased form. Renders a log mel spectrogram and the
    fundamental frequency into audio.

    The original takes the mel alone; here the sine at the fundamental
    frequency also joins every stage. Its convolutions are zeros in the
    converted official weights, which then render as the original.

    Args:
        sample_rate (int): Sampling rate of the audio.
        num_mels (int): Number of mel bins.
        upsample_initial_channel (int): Number of channels before the first upsampler.
        upsample_rates (list): Upsampling rate of each stage.
        resblock_kernel_sizes (list): Kernel size of each residual block of a stage.
        resblock_dilation_sizes (list): Dilations of each residual block of a stage.
        harmonic_num (int, optional): Number of overtones of the excitation. Defaults to 0.
    """

    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        upsample_initial_channel: int,
        upsample_rates: list,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        harmonic_num: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.upp = int(np.prod(upsample_rates))
        self.conv_pre = torch.nn.Conv1d(
            num_mels, upsample_initial_channel, 7, 1, padding=3
        )
        self.m_source = SineSource(sample_rate, harmonic_num)
        self.ups = torch.nn.ModuleList()
        self.source_convs = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()
        # The rate of each stage over the frame rate, and of the output over
        # the stage
        self.total_rates = [
            int(np.prod(upsample_rates[: i + 1])) for i in range(len(upsample_rates))
        ]
        self.source_rates = [self.upp // rate for rate in self.total_rates]
        channels = upsample_initial_channel
        for i, rate in enumerate(upsample_rates):
            self.ups.append(
                ResampleUpsampler(
                    upsample_initial_channel, rate, channels, channels // 2
                )
            )
            channels //= 2
            self.source_convs.append(torch.nn.Conv1d(1, channels, 7, padding=3))
            if self.source_rates[i] > 1:
                self.register_buffer(
                    f"source_kernel_{i}",
                    sinc_lowpass(0.4 / self.source_rates[i]),
                    persistent=False,
                )
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(channels, kernel, dilation))
        self.activation_post = Activation1d(channels)
        self.conv_post = torch.nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor):
        with torch.autocast(mel.device.type, enabled=False):
            source = self.m_source(f0.float(), self.upp)
            sources = []
            for i, rate in enumerate(self.source_rates):
                if rate > 1:
                    kernel = getattr(self, f"source_kernel_{i}")
                    sources.append(lowpass(source, kernel, stride=rate))
                else:
                    sources.append(source)
        x = x0 = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = up(x, x0, self.total_rates[i])
            x = x + self.source_convs[i](sources[i].to(x.dtype))
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0)
