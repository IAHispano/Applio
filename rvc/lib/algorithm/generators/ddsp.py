import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def oscillate_impluse(f0: torch.Tensor, sample_rate: float):
    '''
    f0: [N, 1, L]
    hop_size: int
    sample_rate: float

    Output: [N, 1, L * hop_size]
    '''
    
    I = torch.cumsum(f0, dim=2)
    sawtooth = (I / sample_rate) % 1.0
    impluse = sawtooth - sawtooth.roll(-1, dims=(2)) + (f0 / sample_rate)
    return impluse

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    # x: [BatchSize, cnannels, *]
    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x.mT, (self.channels,), self.gamma, self.beta, self.eps)
        return x.mT


# Global Resnponse Normalization for 1d Sequence (shape=[BatchSize, Channels, Length])
class GRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    # x: [batchsize, channels, length]
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# ConvNeXt v2
class ConvNeXtLayer(nn.Module):
    def __init__(self, channels=512, kernel_size=7, mlp_mul=2):
        super().__init__()
        padding = kernel_size // 2
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, padding, groups=channels)
        self.norm = LayerNorm(channels)
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.grn = GRN(channels * mlp_mul)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)

    # x: [batchsize, channels, length]
    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.gelu(x)
        x = self.grn(x)
        x = self.c3(x)
        x = x + res
        return x

class DDSPGenerator(nn.Module):
    def __init__(
        self,
        n_mels=192,
        internal_channels=512,
        num_layers=8,
        n_fft=4096, #1920,
        sample_rate=48000,
        gin_channels=256,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.checkpointing = checkpointing

        self.sample_rate = sample_rate
        self.hop_size = sample_rate // 100
        self.n_fft = n_fft
        self.window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)
        
        self.input_layer = nn.Conv1d(n_mels, internal_channels, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXtLayer(internal_channels) for _ in range(num_layers)])
        self.conv_post = nn.Conv1d(internal_channels, 1, 1)
        self.to_mag = nn.Conv1d(internal_channels, self.n_fft // 2 + 1, 1)
        self.to_phase = nn.Conv1d(internal_channels, self.n_fft // 2 + 1, 1)
        
        self.cond = nn.Conv1d(gin_channels, internal_channels, 1)
        
    def forward(self, x, f0, g):
        # x: (B, 192, 36), f0: (B, 36), g: (B, 256, 1)
        
        # (B, 192, 36) -> (B, 512, 36)
        x = self.input_layer(x)
        x += self.cond(g)
        x = self.mid_layers(x)

        # (B, 961, 36)
        mag = self.to_mag(x)
        # (B, 961, 36)
        phase = self.to_phase(x)
        # (B, 961, 36) complex64
        kernel = torch.exp_(mag) * torch.exp_(phase * 1j)
        del mag, phase
        #print(x.shape)
        # (B, 512, 36) -> (B, 1, 36)
        per = F.sigmoid(self.conv_post(x))
        # (B, 1, 36) -> (B, 1, 17280)
        per = F.interpolate(per, scale_factor=self.hop_size, mode='linear')
        # (B, 36) -> (B, 1, 17280)
        f0 = F.interpolate(f0.unsqueeze(1), scale_factor=self.hop_size, mode='linear')
        impulse = oscillate_impluse(f0, self.sample_rate)
        # voiced/unvoiced helper
        wave = (per * impulse + (1 - per) * torch.randn_like(impulse)).squeeze(1)
      
        # stft
        # in: (B, C, T)
        # out: (B, n_fft / 2, 1 + T / hop_size)
        wave_stft = torch.stft(
            wave, 
            self.n_fft,
            self.hop_size,
            window=self.window.to(wave.device),
            return_complex=True,
            # center = True -- default
            # onesided = True -- default
            )[:, :, 1:]
       
        out_stft = F.pad(wave_stft * kernel, [0, 1])
        out = torch.istft(
            out_stft, 
            self.n_fft,
            self.hop_size,
            window=self.window.to(wave.device)
        ).unsqueeze(-2)
        return out