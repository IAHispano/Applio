import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ConvNeXtLayer(nn.Module):
    def __init__(self, channel, h_channel, scale):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            channel, channel, kernel_size=7, padding=3, groups=channel
        )
        self.norm = nn.LayerNorm(channel)
        self.pw_conv1 = nn.Linear(channel, h_channel)
        self.pw_conv2 = nn.Linear(h_channel, channel)
        self.scale = nn.Parameter(
            torch.full(size=(channel,), fill_value=scale), requires_grad=True
        )

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = x.transpose(1, 2)
        x = res + x
        return x

class Vocos(nn.Module):
    def __init__(self, in_channel, channel, h_channel, out_channel, num_layers, sample_rate, gin_channels):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.in_conv = nn.Conv1d(in_channel, channel, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channel)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channel, h_channel, scale) for _ in range(num_layers)]
        )
        self.norm_last = nn.LayerNorm(channel)
        self.out_conv = nn.Conv1d(channel, out_channel, 1)
        self.hop_size = sample_rate // 100
        
        self.window=torch.hann_window(4 * self.hop_size)
        
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, channel, 1)

    def forward(self, x, g: Optional):
        x = self.pad(x)
        x = self.in_conv(x)
       
        if g is not None:
            c = self.cond(g)
            x = x + c
            
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        phase = phase.cpu().float()
        mag = mag.cpu().float().exp().clamp_max(max=1e2)
        s = mag * (phase.cos() + 1j * phase.sin())
        o = torch.istft(
            s.to(x.device),
            n_fft=4 * self.hop_size,
            hop_length=self.hop_size,
            win_length=4 * self.hop_size,
            window=self.window.to(s.device),
            center = True,
            ).unsqueeze(1)
        return o
