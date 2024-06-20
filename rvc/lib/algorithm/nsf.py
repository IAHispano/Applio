import math
import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional

from rvc.lib.algorithm.generators import SineGen
from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock1, ResBlock2
from rvc.lib.algorithm.commons import init_weights


class SourceModuleHnNSF(torch.nn.Module):
    """Source Module for harmonic-plus-noise excitation.

    Args:
        sampling_rate (int): Sampling rate in Hz.
        harmonic_num (int, optional): Number of harmonics above F0. Defaults to 0.
        sine_amp (float, optional): Amplitude of sine source signal. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of additive Gaussian noise. Defaults to 0.003.
        voiced_threshod (float, optional): Threshold to set voiced/unvoiced given F0. Defaults to 0.
        is_half (bool, optional): Whether to use half precision. Defaults to True.

    Inputs:
        x (torch.Tensor): F0 tensor with shape (batch_size, length, 1).
        upp (int): Upsampling factor.

    Outputs:
        sine_merge (torch.Tensor): Merged source harmonics with shape (batch_size, length, 1).
        None (None): Placeholder for noise source.
        None (None): Placeholder for voiced/unvoiced signal.
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
        is_half=True,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()
        # self.ddtype:int = -1

    def forward(self, x: torch.Tensor, upp: int = 1):
        """Generates the source excitation.

        Args:
            x (torch.Tensor): F0 tensor with shape (batch_size, length, 1).
            upp (int): Upsampling factor.

        Returns:
            sine_merge (torch.Tensor): Merged source harmonics with shape (batch_size, length, 1).
            None (None): Placeholder for noise source.
            None (None): Placeholder for voiced/unvoiced signal.
        """
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None  # noise, uv


class GeneratorNSF(torch.nn.Module):
    """Generator for synthesizing audio using the NSF (Neural Source Filter) approach.

    Args:
        initial_channel (int): Number of channels in the initial convolutional layer.
        resblock (str): Type of residual block to use (1 or 2).
        resblock_kernel_sizes (list): Kernel sizes of the residual blocks.
        resblock_dilation_sizes (list): Dilation rates of the residual blocks.
        upsample_rates (list): Upsampling rates.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes of the upsampling layers.
        gin_channels (int): Number of channels for the global conditioning input.
        sr (int): Sampling rate.
        is_half (bool, optional): Whether to use half precision. Defaults to False.

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, initial_channel, length).
        f0 (torch.Tensor): F0 tensor with shape (batch_size, length, 1).
        g (torch.Tensor, optional): Global conditioning input with shape (batch_size, gin_channels, length). Defaults to None.

    Outputs:
        x (torch.Tensor): Output tensor with shape (batch_size, 1, length).
    """

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sr,
        is_half=False,
    ):
        super(GeneratorNSF, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.noise_convs = torch.nn.ModuleList()
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    torch.nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(torch.nn.Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)

        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x, f0, g: Optional[torch.Tensor] = None):
        """Generates audio using the NSF approach.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, initial_channel, length).
            f0 (torch.Tensor): F0 tensor with shape (batch_size, length, 1).
            g (torch.Tensor, optional): Global conditioning input with shape (batch_size, gin_channels, length). Defaults to None.

        Returns:
            x (torch.Tensor): Output tensor with shape (batch_size, 1, length).
        """
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        # torch.jit.script() does not support direct indexing of torch modules
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            if i < self.num_upsamples:
                x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
                x = ups(x)
                x_source = noise_convs(har_source)
                x = x + x_source
                xs: Optional[torch.Tensor] = None
                l = [i * self.num_kernels + j for j in range(self.num_kernels)]
                for j, resblock in enumerate(self.resblocks):
                    if j in l:
                        if xs is None:
                            xs = resblock(x)
                        else:
                            xs += resblock(x)
                # This assertion cannot be ignored, if ignored, it will cause torch.jit.script() compilation errors
                assert isinstance(xs, torch.Tensor)
                x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        """Removes weight normalization from the upsampling and residual blocks."""
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
                    torch.nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            for hook in self.resblocks._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self
