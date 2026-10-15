import numpy as np
import torch

from rvc.lib.algorithm.generators.pupu_vocoder import (
    Activation1d,
    ResBlock,
    SineSource,
    lowpass,
    sinc_lowpass,
)

ARCHITECTURE = "bigvgan_v2"


class SnakeBeta(torch.nn.Module):
    """
    SnakeBeta activation of BigVGAN v2: x + sin^2(alpha * x) / beta, with
    alpha and beta per channel.

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
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class BigVGANv2(torch.nn.Module):
    """
    BigVGAN v2 (https://github.com/NVIDIA/BigVGAN, MIT) as its released
    checkpoints have it: transposed upsamplers and SnakeBeta between 2x
    lowpass filters. Renders a log mel spectrogram and the fundamental
    frequency into audio.

    The original takes the mel alone; here the sine at the fundamental
    frequency also joins every stage. Its convolutions are zeros in the
    converted official weights, which then render as the original.

    Args:
        sample_rate (int): Sampling rate of the audio.
        num_mels (int): Number of mel bins.
        upsample_initial_channel (int): Number of channels before the first upsampler.
        upsample_rates (list): Upsampling rate of each stage.
        upsample_kernel_sizes (list): Kernel size of each upsampler.
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
        upsample_kernel_sizes: list,
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
        # The rate of the output over each stage
        self.source_rates = [
            int(np.prod(upsample_rates[i + 1 :])) for i in range(len(upsample_rates))
        ]
        channels = upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                torch.nn.ConvTranspose1d(
                    channels, channels // 2, kernel, rate, padding=(kernel - rate) // 2
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
            for size, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(channels, size, dilation, SnakeBeta))
        self.activation_post = Activation1d(channels, SnakeBeta)
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
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = up(x) + self.source_convs[i](sources[i].to(x.dtype))
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0)
