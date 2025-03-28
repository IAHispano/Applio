import torch
import numpy as np
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional

from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock
from rvc.lib.algorithm.commons import init_weights


class HiFiGANGenerator(torch.nn.Module):
    """
    HiFi-GAN Generator module for audio synthesis.

    This module implements the generator part of the HiFi-GAN architecture,
    which uses transposed convolutions for upsampling and residual blocks for
    refining the audio output. It can also incorporate global conditioning.

    Args:
        initial_channel (int): Number of input channels to the initial convolutional layer.
        resblock_kernel_sizes (list): List of kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): List of lists of dilation rates for the residual blocks, corresponding to each kernel size.
        upsample_rates (list): List of upsampling factors for each upsampling layer.
        upsample_initial_channel (int): Number of output channels from the initial convolutional layer, which is also the input to the first upsampling layer.
        upsample_kernel_sizes (list): List of kernel sizes for the transposed convolutional layers used for upsampling.
        gin_channels (int, optional): Number of input channels for the global conditioning. If 0, no global conditioning is used. Defaults to 0.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int = 0,
    ):
        super(HiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
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
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        # new tensor
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        # in-place call
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        # in-place call
        x = torch.tanh(x)

        return x

    def __prepare_scriptable__(self):
        for l in self.ups_and_resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SineGenerator(torch.nn.Module):
    """
    Sine wave generator with optional harmonic overtones and noise.

    This module generates sine waves for a fundamental frequency and its harmonics.
    It can also add Gaussian noise and apply a voiced/unvoiced mask.

    Args:
        sampling_rate (int): The sampling rate of the audio in Hz.
        num_harmonics (int, optional): The number of harmonic overtones to generate. Defaults to 0.
        sine_amplitude (float, optional): The amplitude of the sine wave components. Defaults to 0.1.
        noise_stddev (float, optional): The standard deviation of the additive Gaussian noise. Defaults to 0.003.
        voiced_threshold (float, optional): The threshold for the fundamental frequency (F0) to determine if a frame is voiced. Defaults to 0.0.
    """

    def __init__(
        self,
        sampling_rate: int,
        num_harmonics: int = 0,
        sine_amplitude: float = 0.1,
        noise_stddev: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super(SineGenerator, self).__init__()
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.sine_amplitude = sine_amplitude
        self.noise_stddev = noise_stddev
        self.voiced_threshold = voiced_threshold
        self.waveform_dim = self.num_harmonics + 1  # fundamental + harmonics

    def _compute_voiced_unvoiced(self, f0: torch.Tensor):
        """
        Generates a binary mask indicating voiced/unvoiced frames based on the fundamental frequency.

        Args:
            f0 (torch.Tensor): Fundamental frequency tensor of shape (batch_size, length).
        """
        uv_mask = (f0 > self.voiced_threshold).float()
        return uv_mask

    def _generate_sine_wave(self, f0: torch.Tensor, upsampling_factor: int):
        """
        Generates sine waves for the fundamental frequency and its harmonics.

        Args:
            f0 (torch.Tensor): Fundamental frequency tensor of shape (batch_size, length, 1).
            upsampling_factor (int): The factor by which to upsample the sine wave.
        """
        batch_size, length, _ = f0.shape

        # Create an upsampling grid
        upsampling_grid = torch.arange(
            1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device
        )

        # Calculate phase increments
        phase_increments = (f0 / self.sampling_rate) * upsampling_grid
        phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        phase_increments += torch.nn.functional.pad(
            cumulative_phase, (0, 0, 1, 0), mode="constant"
        )

        # Reshape to match the sine wave shape
        phase_increments = phase_increments.reshape(batch_size, -1, 1)

        # Scale for harmonics
        harmonic_scale = torch.arange(
            1, self.waveform_dim + 1, dtype=f0.dtype, device=f0.device
        ).reshape(1, 1, -1)
        phase_increments *= harmonic_scale

        # Add random phase offset (except for the fundamental)
        random_phase = torch.rand(1, 1, self.waveform_dim, device=f0.device)
        random_phase[..., 0] = 0  # Fundamental frequency has no random offset
        phase_increments += random_phase

        # Generate sine waves
        sine_waves = torch.sin(2 * np.pi * phase_increments)
        return sine_waves

    def forward(self, f0: torch.Tensor, upsampling_factor: int):
        with torch.no_grad():
            # Expand `f0` to include waveform dimensions
            f0 = f0.unsqueeze(-1)

            # Generate sine waves
            sine_waves = (
                self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude
            )

            # Compute voiced/unvoiced mask
            voiced_mask = self._compute_voiced_unvoiced(f0)

            # Upsample voiced/unvoiced mask
            voiced_mask = torch.nn.functional.interpolate(
                voiced_mask.transpose(2, 1),
                scale_factor=float(upsampling_factor),
                mode="nearest",
            ).transpose(2, 1)

            # Compute noise amplitude
            noise_amplitude = voiced_mask * self.noise_stddev + (1 - voiced_mask) * (
                self.sine_amplitude / 3
            )

            # Add Gaussian noise
            noise = noise_amplitude * torch.randn_like(sine_waves)

            # Combine sine waves and noise
            sine_waveforms = sine_waves * voiced_mask + noise

        return sine_waveforms, voiced_mask, noise
