import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        alpha = self.alpha.exp()
        beta = self.beta.exp()
        x = x + (1.0 / (beta + 1e-9)) * (x * alpha).sin().pow(2)
        return x

class Block(nn.Module):
    def __init__(self, channels, kernelSize=3, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels, channels, kernelSize, dilation=dilation, padding="same"
        )
        self.act1 = Snake(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernelSize, padding="same")
        self.act2 = Snake(channels)

    def applyWeightNorm(self):
        self.conv1 = weight_norm(self.conv1)
        self.conv2 = weight_norm(self.conv2)

    def removeWeightNorm(self):
        self.conv1 = remove_weight_norm(self.conv1)
        self.conv2 = remove_weight_norm(self.conv2)

    def forward(self, x):
        res = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        x += res
        return x

class ResLayer(nn.Module):
    def __init__(self, channels, kernelSize=(3, 5, 7), dilation=(1, 3, 5)):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.kernelSize = kernelSize
        self.dilation = dilation
        for i in range(len(kernelSize)):
            for j in range(len(dilation)):
                self.blocks.append(Block(channels, kernelSize[i], dilation[j]))

    def applyWeightNorm(self):
        for i in range(len(self.blocks)):
            self.blocks[i].applyWeightNorm()

    def removeWeightNorm(self):
        for i in range(len(self.blocks)):
            self.blocks[i].removeWeightNorm()

    def forwardOneKernel(self, x, kernelID):
        out = self.blocks[kernelID * len(self.dilation)](x)
        for i in range(1, len(self.dilation)):
            out = self.blocks[kernelID * len(self.dilation) + i](out)
        return out

    def forward(self, x):
        sum = self.forwardOneKernel(x, 0)
        for i in range(1, len(self.kernelSize)):
            sum += self.forwardOneKernel(x, i)
        sum /= len(self.kernelSize)
        return sum

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

class Velocity(nn.Module):

    def __init__(
        self,
        channels=[512, 256, 128, 64, 32],
        upSampleRates=[12, 10, 2, 2],
        kernelSizesUp=[[3, 7, 11], [3, 7, 11], [3, 7, 11], [3, 7, 11]],
        dilationsUp=[[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
        kernelSizesDown=[[3], [3], [3], [3]],
        dilationsDown=[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        mel_bands=192,
        sample_rate=48000,
        gin_channels=256,
    ):
        super().__init__()
        
        self.upp = np.prod(upSampleRates)
        self.m_source = SineGenerator(sample_rate)
        
        self.upSampleRates = upSampleRates

        self.convDownIn = nn.Conv1d(1, channels[-1], 7, padding="same")
        
        self.convUpIn = nn.Conv1d(mel_bands, channels[0], 7, 1, padding="same")
        self.cond = nn.Conv1d(gin_channels, channels[0], 1)
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(len(upSampleRates)):
            self.ups.append(
                nn.ConvTranspose1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=2 * upSampleRates[i],
                    stride=upSampleRates[i],
                    padding=(upSampleRates[i]+1) // 2,
                    output_padding=upSampleRates[i] % 2
                ),
            )  # stride=2kernel=4padding

            kernel = 1 if upSampleRates[i] == 1 else upSampleRates[i] * 2 - upSampleRates[i] % 2
            padding = 0 if upSampleRates[i] == 1 else (kernel - upSampleRates[i]) // 2

            self.downs.append(
                nn.Conv1d(
                    channels[i + 1],
                    channels[i],
                    kernel_size=kernel,
                    stride=upSampleRates[i],
                    padding=padding,
                )
            )

        self.resLayerUps = nn.ModuleList()
        self.resLayerDowns = nn.ModuleList()

        for i in range(len(upSampleRates)):
            self.resLayerUps.append(ResLayer(channels[i + 1], kernelSizesUp[i], dilationsUp[i]))
            self.resLayerDowns.append(ResLayer(channels[i + 1], kernelSizesDown[i], dilationsDown[i]))

        self.convUpOut = nn.Conv1d(channels[-1], 1, 7, 1, padding="same")
        self.actUpOut = Snake(channels=channels[-1])

    def applyWeightNorm(self):
        self.convDownIn = weight_norm(self.convDownIn)
        self.convUpIn = weight_norm(self.convUpIn)
        self.convUpOut = weight_norm(self.convUpOut)

        for i in range(len(self.resLayerUps)):
            self.resLayerUps[i].applyWeightNorm()
        for i in range(len(self.resLayerDowns)):
            self.resLayerDowns[i].applyWeightNorm()
        for i in range(len(self.ups)):
            self.ups[i] = weight_norm(self.ups[i])
        for i in range(len(self.downs)):
            self.downs[i] = weight_norm(self.downs[i])

    def removeWeightNorm(self):
        self.convDownIn = remove_weight_norm(self.convDownIn)
        self.convUpIn = remove_weight_norm(self.convUpIn)
        self.convUpOut = remove_weight_norm(self.convUpOut)

        for i in range(len(self.resLayerUps)):
            self.resLayerUps[i].removeWeightNorm()
        for i in range(len(self.resLayerDowns)):
            self.resLayerDowns[i].removeWeightNorm()
        for i in range(len(self.ups)):
            self.ups[i] = remove_weight_norm(self.ups[i])
        for i in range(len(self.downs)):
            self.downs[i] = remove_weight_norm(self.downs[i])

    def forward(self, mel, f0, g):
        # (8, 36) -> (8, 1, 17280)
        f0 = F.interpolate(
            f0.unsqueeze(1), size=mel.shape[-1] * self.upp, mode="linear"
        )
        # to sine
        har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)

        # (8, 1, 17280) -> (8, 32, 17280)
        x = self.convDownIn(har_source)

        # downs
        # (8,   32, 17280)  -> ups 3
        # (8,   64,  8640)  -> ups 2
        # (8,  128,  4320)  -> ups 1
        # (8,  256,   432)  -> ups 0
        # (8,  512,    36)  -- merged with mel

        noise = [x]
        for i in range(len(self.downs) - 1, -1, -1):
            x = self.resLayerDowns[i](x)
            x = self.downs[i](x)
            noise.append(x)
        
        # (8, 512, 36)
        mel = self.convUpIn(mel)
        # speaker embedding
        mel += self.cond(g)
        # f0 embedding
        mel += noise[-1]

        # ups
        
        for i in range(len(self.ups)):
            mel = F.leaky_relu_(mel, 0.2)
            mel = self.ups[i](mel)
            mel += noise[-i - 2]
            mel = self.resLayerUps[i](mel)

        out = self.actUpOut(mel)
        out = self.convUpOut(out)
        out = torch.tanh(out)

        return out
