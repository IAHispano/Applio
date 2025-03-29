import math
from typing import Optional

import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.generators.hifigan import SineGenerator
from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock


class SourceModuleHnNSF(torch.nn.Module):
    """
    Source Module for generating harmonic and noise components for audio synthesis.

    This module generates a harmonic source signal using sine waves and adds
    optional noise. It's often used in neural vocoders as a source of excitation.

    Args:
        sample_rate (int): Sampling rate of the audio in Hz.
        harmonic_num (int, optional): Number of harmonic overtones to generate above the fundamental frequency (F0). Defaults to 0.
        sine_amp (float, optional): Amplitude of the sine wave components. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of the additive white Gaussian noise. Defaults to 0.003.
        voiced_threshod (float, optional): Threshold for the fundamental frequency (F0) to determine if a frame is voiced. If F0 is below this threshold, it's considered unvoiced. Defaults to 0.
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGenerator(
            sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upsample_factor: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class HiFiGANNSFGenerator(torch.nn.Module):
    """
    Generator module based on the Neural Source Filter (NSF) architecture.

    This generator synthesizes audio by first generating a source excitation signal
    (harmonic and noise) and then filtering it through a series of upsampling and
    residual blocks. Global conditioning can be applied to influence the generation.

    Args:
        initial_channel (int): Number of input channels to the initial convolutional layer.
        resblock_kernel_sizes (list): List of kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): List of lists of dilation rates for the residual blocks, corresponding to each kernel size.
        upsample_rates (list): List of upsampling factors for each upsampling layer.
        upsample_initial_channel (int): Number of output channels from the initial convolutional layer, which is also the input to the first upsampling layer.
        upsample_kernel_sizes (list): List of kernel sizes for the transposed convolutional layers used for upsampling.
        gin_channels (int): Number of input channels for the global conditioning. If 0, no global conditioning is used.
        sr (int): Sampling rate of the audio.
        checkpointing (bool, optional): Whether to use gradient checkpointing to save memory during training. Defaults to False.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
    ):
        super(HiFiGANNSFGenerator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # handling odd upsampling rates
            if u % 2 == 0:
                # old method
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        channels[i],
                        k,
                        u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )
            """ handling odd upsampling rates
            #  s   k   p
            # 40  80  20
            # 32  64  16
            #  4   8   2
            #  2   3   1
            # 63 125  31
            #  9  17   4
            #  3   5   1
            #  1   1   0
            """
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2

            self.noise_convs.append(
                torch.nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        self.resblocks = torch.nn.ModuleList(
            [
                ResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        # new tensor
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            # Apply upsampling layer
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = x + noise_convs(har_source)
                xs = sum(
                    [
                        checkpoint(resblock, x, use_reentrant=False)
                        for j, resblock in enumerate(self.resblocks)
                        if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                    ]
                )
            else:
                x = ups(x)
                x = x + noise_convs(har_source)
                xs = sum(
                    [
                        resblock(x)
                        for j, resblock in enumerate(self.resblocks)
                        if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                    ]
                )
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        return self
