import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.generators.modules.stft import STFT

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.alpha1, self.alpha2):
            xt = x + (1 / a1) * (torch.sin(a1 * x) ** 2)  # Snake1D
            xt = c1(xt)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
			
class SineGenerator(nn.Module):
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

        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Tanh(),
        )

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
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

            sine_waves = sine_waves * uv + noise
        # correct DC offset
        sine_waves = sine_waves - sine_waves.mean(dim=1, keepdim=True)
        # merge with grad
        return self.merge(sine_waves)

class HiFTNetGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channel = 192,
        upsample_rates = [12, 10, 2, 2],
        upsample_initial_channel = 512,
        gin_channels = 256,
        sample_rate = 48000,
    ):
        super().__init__()
        self.n_fft = 1920
        self.n_bins = self.n_fft // 2 + 1   # 961
        self.hop_size = upsample_rates[2] * upsample_rates[3] # 2 final upsamples
        self.upp = sample_rate // 100
        
        self.m_source = SineGenerator(sample_rate, harmonic_num=8)
    
        self.conv_pre = weight_norm(nn.Conv1d(in_channel, upsample_initial_channel, 7, 1, padding=3))
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        # ups1
        up = upsample_rates[0]
        self.ups1 = nn.ConvTranspose1d(
            upsample_initial_channel, 
            upsample_initial_channel // 2, 
            kernel_size = up * 2, stride = up,  padding = (up + 1)//2, output_padding = up % 2)
        self.res1 = nn.ModuleList([
            ResBlock1(upsample_initial_channel // 2, 3, [1, 3, 5]),
            ResBlock1(upsample_initial_channel // 2, 7, [1, 3, 5]),
            ResBlock1(upsample_initial_channel // 2, 11, [1, 3, 5]),
            ])
        # ups2
        up = upsample_rates[1]
        self.ups2 = nn.ConvTranspose1d(
            upsample_initial_channel//2,
            upsample_initial_channel//4,
            kernel_size = up * 2, stride = up,  padding = (up + 1)//2, output_padding = up % 2)
        self.res2 = nn.ModuleList([
            ResBlock1(upsample_initial_channel//4, 3, [1, 3, 5]),
            ResBlock1(upsample_initial_channel//4, 7, [1, 3, 5]),
            ResBlock1(upsample_initial_channel//4, 11, [1, 3, 5]),
            ])
    
        stride = upsample_rates[1]
        self.noise_down1 = nn.Conv1d(
            self.n_bins * 2,
            upsample_initial_channel//2, 
            kernel_size=stride * 2 - stride % 2,
            stride = stride,
            padding = (stride + 1) // 2
            ) # 432
        self.noise_res1 = ResBlock1(upsample_initial_channel//2, 7, [1, 3, 5])
        
        self.noise_down2 = nn.Conv1d(
            self.n_bins * 2,
            upsample_initial_channel//4,
            1) # 4320

        self.noise_res2 = ResBlock1(upsample_initial_channel//4, 11, [1, 3, 5])
        
        self.conv_post = weight_norm(
            nn.Conv1d(upsample_initial_channel//4, self.n_bins * 2, 7, 1, padding=3)
        )
    
        self.stft = STFT(n_fft = self.n_fft, n_bins = self.n_bins, hop_length = self.hop_size)
        
        self.ups1.apply(init_weights)
        self.ups2.apply(init_weights)
        self.conv_post.apply(init_weights)
        
    def forward(self, mel, f0, g):
        # f0 to waveform to mag/phase
        f0 = F.interpolate(f0.unsqueeze(1), size=mel.shape[-1] * self.upp, mode="linear")
        har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)
        stft = self.stft(har_source)
        har = torch.cat([stft.real, stft.imag], dim=1)
        # input + conditioning
        x = self.conv_pre(mel)
        x = x + self.cond(g)
        # ups1
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.ups1(x)
        x = x + self.noise_res1(self.noise_down1(har))
        x = (self.res1[0](x) + self.res1[1](x) + self.res1[2](x)) / 3
        # ups2
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.ups2(x)
        x = x + self.noise_res2(self.noise_down2(har))
        x = (self.res2[0](x) + self.res2[1](x) + self.res2[2](x)) / 3
        # post
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        # istft last 2 upscales
        mag = torch.exp(x[:,:self.n_bins, :])
        phase = torch.sin(x[:, self.n_bins:, :])
        kernel = mag * torch.exp(1j * phase)
        out = self.stft.inverse(kernel)
        return out

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)