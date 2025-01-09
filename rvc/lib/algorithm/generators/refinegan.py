import numpy as np
import torch
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import get_padding


class ResBlock(torch.nn.Module):
    """
    Residual block with multiple dilated convolutions.

    This block applies a sequence of dilated convolutional layers with Leaky ReLU activation.
    It's designed to capture information at different scales due to the varying dilation rates.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for the convolutional layers. Defaults to 7.
        dilation (tuple[int], optional): Tuple of dilation rates for the convolutional layers. Defaults to (1, 3, 5).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super(ResBlock, self).__init__()

        self.leaky_relu_slope = leaky_relu_slope
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs1 = torch.nn.ModuleList(
            [
                weight_norm(
                    torch.nn.Conv1d(
                        in_channels=in_channels if idx == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for idx, d in enumerate(dilation)
            ]
        )
        self.convs1.apply(self.init_weights)

        self.convs2 = torch.nn.ModuleList(
            [
                weight_norm(
                    torch.nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for idx, d in enumerate(dilation)
            ]
        )
        self.convs2.apply(self.init_weights)

    def forward(self, x: torch.Tensor):
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            # new tensor
            xt = torch.nn.functional.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            # in-place call
            xt = torch.nn.functional.leaky_relu_(xt, self.leaky_relu_slope)
            xt = c2(xt)

            if idx != 0 or self.in_channels == self.out_channels:
                x = xt + x
            else:
                x = xt

        return x

    def remove_parametrizations(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_parametrizations(c1)
            remove_parametrizations(c2)

    def init_weights(self, m):
        if type(m) == torch.nn.Conv1d:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0.0)


class AdaIN(torch.nn.Module):
    """
    Adaptive Instance Normalization layer.

    This layer applies a scaling factor to the input based on a learnable weight.

    Args:
        channels (int): Number of input channels.
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation applied after scaling. Defaults to 0.2.
    """

    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(channels))
        # safe to use in-place as it is used on a new x+gaussian tensor
        self.activation = torch.nn.LeakyReLU(leaky_relu_slope, inplace=True)

    def forward(self, x: torch.Tensor):
        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(torch.nn.Module):
    """
    Parallel residual block that applies multiple residual blocks with different kernel sizes in parallel.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_sizes (tuple[int], optional): Tuple of kernel sizes for the parallel residual blocks. Defaults to (3, 7, 11).
        dilation (tuple[int], optional): Tuple of dilation rates for the convolutional layers within the residual blocks. Defaults to (1, 3, 5).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    AdaIN(channels=out_channels),
                    ResBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(channels=out_channels),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)

        results = [block(x) for block in self.blocks]

        return torch.mean(torch.stack(results), dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block[1].remove_parametrizations()


class SineGenerator(torch.nn.Module):
    """
    Definition of sine generator

    Generates sine waveforms with optional harmonics and additive noise.
    Can be used to create harmonic noise source for neural vocoders.

    Args:
        samp_rate (int): Sampling rate in Hz.
        harmonic_num (int): Number of harmonic overtones (default 0).
        sine_amp (float): Amplitude of sine-waveform (default 0.1).
        noise_std (float): Standard deviation of Gaussian noise (default 0.003).
        voiced_threshold (float): F0 threshold for voiced/unvoiced classification (default 0).
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise * (1 - uv)
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """
    Source Module for generating harmonic and noise signals.

    This module uses a SineGenerator to produce harmonic signals based on the fundamental frequency (F0).

    Args:
        sampling_rate (int): Sampling rate of the audio.
        harmonic_num (int, optional): Number of harmonics to generate. Defaults to 0.
        sine_amp (float, optional): Amplitude of the sine wave. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of the additive noise. Defaults to 0.003.
        voiced_threshold (int, optional): F0 threshold for voiced/unvoiced classification. Defaults to 0.
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGenerator(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor):
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        return sine_merge, None, None


class RefineGANGenerator(torch.nn.Module):
    """
    RefineGAN generator for audio synthesis.

    This generator uses a combination of downsampling, residual blocks, and parallel residual blocks
    to refine an input mel-spectrogram and fundamental frequency (F0) into an audio waveform.
    It can also incorporate global conditioning.

    Args:
        sample_rate (int, optional): Sampling rate of the audio. Defaults to 44100.
        downsample_rates (tuple[int], optional): Downsampling rates for the downsampling blocks. Defaults to (2, 2, 8, 8).
        upsample_rates (tuple[int], optional): Upsampling rates for the upsampling blocks. Defaults to (8, 8, 2, 2).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
        num_mels (int, optional): Number of mel-frequency bins in the input mel-spectrogram. Defaults to 128.
        start_channels (int, optional): Number of channels in the initial convolutional layer. Defaults to 16.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 256.
        checkpointing (bool, optional): Whether to use checkpointing for memory efficiency. Defaults to False.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        downsample_rates: tuple[int] = (2, 2, 8, 8),
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,
        gin_channels: int = 256,
        checkpointing=False,
    ):
        super().__init__()

        self.downsample_rates = downsample_rates
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing

        self.f0_upsample = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num=8)

        # expands
        self.source_conv = weight_norm(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=start_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        channels = start_channels
        self.downsample_blocks = torch.nn.ModuleList([])
        for rate in downsample_rates:
            new_channels = channels * 2

            self.downsample_blocks.append(
                torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=1 / rate, mode="linear"),
                    ResBlock(
                        in_channels=channels,
                        out_channels=new_channels,
                        kernel_size=7,
                        dilation=(1, 3, 5),
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                )
            )

            channels = new_channels

        self.mel_conv = weight_norm(
            torch.nn.Conv1d(
                in_channels=num_mels,
                out_channels=channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(256, channels, 1)

        channels *= 2

        self.upsample_blocks = torch.nn.ModuleList([])
        self.upsample_conv_blocks = torch.nn.ModuleList([])

        for rate in upsample_rates:
            new_channels = channels // 2

            self.upsample_blocks.append(
                torch.nn.Upsample(scale_factor=rate, mode="linear")
            )

            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(
            torch.nn.Conv1d(
                in_channels=channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        f0 = self.f0_upsample(f0[:, None, :]).transpose(-1, -2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)

        # expanding pitch source to 16 channels
        # new tensor
        x = self.source_conv(har_source)
        # making a downscaled version to match upscaler stages
        downs = []
        for i, block in enumerate(self.downsample_blocks):
            # in-place call
            x = torch.nn.functional.leaky_relu_(x, self.leaky_relu_slope)
            downs.append(x)
            if self.training and self.checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # expanding spectrogram from 192 to 256 channels
        mel = self.mel_conv(mel)

        if g is not None:
            # adding expanded speaker embedding
            mel += self.cond(g)
        x = torch.cat([mel, x], dim=1)

        for ups, res, down in zip(
            self.upsample_blocks,
            self.upsample_conv_blocks,
            reversed(downs),
        ):
            # in-place call
            x = torch.nn.functional.leaky_relu_(x, self.leaky_relu_slope)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = torch.cat([x, down], dim=1)
                x = checkpoint(res, x, use_reentrant=False)
            else:
                x = ups(x)
                x = torch.cat([x, down], dim=1)
                x = res(x)
        # in-place call
        x = torch.nn.functional.leaky_relu_(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        # in-place call
        x = torch.tanh_(x)

        return x

    def remove_parametrizations(self):
        remove_parametrizations(self.source_conv)
        remove_parametrizations(self.mel_conv)
        remove_parametrizations(self.conv_post)

        for block in self.downsample_blocks:
            block[1].remove_parametrizations()

        for block in self.upsample_conv_blocks:
            block.remove_parametrizations()
