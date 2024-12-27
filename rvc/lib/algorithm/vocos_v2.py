import torch
import numpy as np
from torch.nn.utils.parametrizations import weight_norm


class GlobalResponseNormalization(torch.nn.Module):
    """
    Implements Global Response Normalization (GRN).

    This module normalizes the input tensor along the channel dimension,
    scales it with learnable parameters, and adds a bias.

    Args:
        channel (int): Number of input channels.
    """

    def __init__(self, channel):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, channel), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, channel), requires_grad=True)

    def forward(self, x):
        gx = x.norm(dim=1, keepdim=True)
        nx = gx / gx.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.gamma * (nx * x) + self.bias + x


class ConvNeXtLayer(torch.nn.Module):
    """
    Implements a single ConvNeXt V2 layer. Reference: https://github.com/facebookresearch/ConvNeXt-V2

    The layer includes depthwise convolution, layer normalization, pointwise linear
    transformations, and Global Response Normalization (GRN).

    Args:
        channel (int): Number of input and output channels.
        h_channel (int): Number of hidden channels in the pointwise layers.
    """

    def __init__(self, channel, h_channel):
        super().__init__()
        self.dw_conv = torch.nn.Conv1d(
            channel, channel, kernel_size=7, padding=3, groups=channel
        )
        self.norm = torch.nn.LayerNorm(channel)
        self.pw1 = torch.nn.Linear(channel, h_channel)
        self.pw2 = torch.nn.Linear(h_channel, channel)
        self.grn = GlobalResponseNormalization(h_channel)

    def __init__(self, channel, h_channel):
        super().__init__()
        self.dw_conv = torch.nn.Conv1d(
            channel, channel, kernel_size=7, padding=3, groups=channel
        )
        self.norm = torch.nn.LayerNorm(channel)
        self.pw1 = torch.nn.Linear(channel, h_channel)
        self.pw2 = torch.nn.Linear(h_channel, channel)
        self.grn = GlobalResponseNormalization(h_channel)

    def forward(self, x):
        residual = x
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw1(x)
        x = torch.nn.functional.gelu(x)
        x = self.grn(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)
        return residual + x


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
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: torch.Tensor):
        """
        Generates voiced/unvoiced (UV) signal based on the fundamental frequency (F0).

        Args:
            f0 (torch.Tensor): Fundamental frequency tensor of shape (batch_size, length, 1).
        """
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values: torch.Tensor):
        """
        Generates sine waveforms based on the fundamental frequency (F0) and its harmonics.

        Args:
            f0_values (torch.Tensor): Tensor of fundamental frequency and its harmonics,
                                      shape (batch_size, length, dim), where dim indicates
                                      the fundamental tone and overtones.
        """
        # convert to F0 in rad. The integer part n can be ignored
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

    def forward(self, f0: torch.Tensor):
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

            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """
    Generates harmonic and noise source features.

    This module uses the SineGenerator to create harmonic signals based on the
    fundamental frequency (F0) and merges them into a single excitation signal.

    Args:
        sample_rate (int): Sampling rate in Hz.
        harmonic_num (int, optional): Number of harmonics above F0. Defaults to 0.
        sine_amp (float, optional): Amplitude of sine source signal. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of additive Gaussian noise. Defaults to 0.003.
        voiced_threshod (float, optional): Threshold to set voiced/unvoiced given F0. Defaults to 0.
    """

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
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


class VocosGenerator(torch.nn.Module):
    """
    Neural vocoder generator based on ConvNeXt-like blocks.

    Args:
        in_channel (int): Number of input channels.
        channel (int): Number of initial channels for convolutional layers.
        h_channel (int): Number of hidden channels in ConvNeXt layers.
        out_channel (int): Number of output channels.
        num_layers (int): Number of ConvNeXt layers.
        sample_rate (int): Sampling rate in Hz.
        gin_channels (int): Number of global conditioning channels (0 if not used).
    """

    def __init__(
        self,
        in_channel,  # inter_channels
        channel,  # upsample_initial_channel
        h_channel,  # upsample_initial_channel
        out_channel,  # (4 * sr // 100 + 2)
        num_layers,  # 8
        sample_rate,  # sr
        gin_channels,  # gin_channels
    ):
        super().__init__()

        self.pad = torch.nn.ReflectionPad1d([1, 0])
        self.in_conv = torch.nn.Conv1d(in_channel, channel, kernel_size=7, padding=3)
        self.norm = torch.nn.LayerNorm(channel)
        self.layers = torch.nn.ModuleList(
            [ConvNeXtLayer(channel, h_channel) for _ in range(num_layers)]
        )
        self.norm_last = torch.nn.LayerNorm(channel)
        self.out_conv = torch.nn.Conv1d(channel, out_channel, kernel_size=1)
        self.hop_size = sample_rate // 100
        self.n_fft = 4 * self.hop_size
        self.window = torch.hann_window(self.n_fft)

        self.cond = None
        if gin_channels > 0:
            self.cond = torch.nn.Conv1d(gin_channels, channel, kernel_size=1)

        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num=0)

        self.conv_pre_y = weight_norm(torch.nn.Conv1d(1, channel // 2, 7, 1, padding=3))
        self.fuse_y_mel = weight_norm(
            torch.nn.Conv1d(channel + channel // 2, channel, 1)
        )

    def forward(self, x: torch.Tensor, f0: torch.Tensor = None, g=None):
        # z = (8, 192, 36), p = (8, 36), g = (8, 256, 1)

        x = self.in_conv(x)  # (8, 192, 36) -> (8, 256, 36)

        if g is not None and self.cond:
            c = self.cond(g)
            x += c

        if f0 is not None:
            f0 = f0[:, None, :].transpose(-1, -2)  # (8, 36, 1)
            har_source, _, _ = self.m_source(f0)  # f0 to waveform
            har_source = har_source.transpose(-1, -2)  # (8, 1, 36)
            har_source = self.conv_pre_y(har_source)  # (8, 1, 36) -> (8, 128, 36)
            x = torch.cat(
                [x, har_source], dim=1
            )  # [(8, 256, 36), (8, 128, 36)] -> (8, 384, 36)
            x = self.fuse_y_mel(x)  # (8, 384, 36) -> (8, 256, 36)

        x = self.pad(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp(max=1e2)
        real = mag * phase.cos()
        imag = mag * phase.sin()
        s = torch.complex(real, imag)
        o = torch.istft(
            s,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.n_fft,
            window=self.window.to(s.device),
            center=True,
        ).unsqueeze(1)
        return o
