import torch
from torch import nn
from torch.nn import functional as F

from rvc.lib.algorithm.rectified_flow.layers import (
    ConvNeXtBlock,
    atan_glu,
    timestep_embedding,
)

# Multiplies the guidance scale, less 1, before its sinusoids.
GUIDANCE_SCALE = 30.0


class LYNXNet2Block(nn.Module):
    """
    Depthwise conv, then two ATanGLU projections, pre-norm and residual.

    Args:
        channels (int): Number of channels.
        expansion (float): Expansion of the inner projections.
        kernel_size (int): Kernel size of the depthwise conv.
        adaln (bool, optional): Modulate the block by the time and speaker embedding (adaLN-Zero). Defaults to False.
    """

    def __init__(self, channels, expansion, kernel_size, adaln=False):
        super().__init__()
        inner = int(channels * expansion)
        self.norm = nn.LayerNorm(channels, elementwise_affine=not adaln)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.up = nn.Linear(channels, inner * 2)
        self.mid = nn.Linear(inner, inner * 2)
        self.down = nn.Linear(inner, channels)
        self.modulation = None
        if adaln:
            self.modulation = nn.Linear(channels, channels * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(self, x, mask, embedding=None, fused=True):
        y = self.norm(x)
        gate = None
        if self.modulation is not None:
            shift, scale, gate = self.modulation(F.silu(embedding)).chunk(3, dim=-1)
            # Not `1 + scale`: BF16 rounds small modulations to nothing.
            y = y + y * scale + shift
        y = self.depthwise((y * mask).transpose(1, 2)).transpose(1, 2)
        y = self.down(atan_glu(self.mid(atan_glu(self.up(y), fused)), fused))
        if gate is not None:
            y = y + gate * y
        return (x + y) * mask


class LYNXNet2Backbone(nn.Module):
    """
    DiffSinger's LYNXNet2: condition and time added at the input, then
    depthwise-separable gated blocks.

    Args:
        n_mels (int): Number of mel bins.
        cond_channels (int): Number of conditioning channels.
        channels (int, optional): Number of channels. Defaults to 1024.
        layers (int, optional): Number of blocks. Defaults to 6.
        expansion (float, optional): Expansion of the blocks. Defaults to 1.
        kernel_size (int, optional): Kernel size of the blocks. Defaults to 31.
        adaln (bool, optional): Modulate every block by time and speaker. Defaults to False.
        span (bool, optional): Add MeanFlow's second time input, the length of the step whose mean velocity is predicted, and the guidance scale that velocity is under. Defaults to False.
        time_scale (float, optional): Multiplier of the flow time before its sinusoids; Mean Flow needs it near 1. Defaults to 1000.0.
    """

    def __init__(
        self,
        n_mels,
        cond_channels,
        channels=1024,
        layers=6,
        expansion=1,
        kernel_size=31,
        adaln=False,
        span=False,
        time_scale=1000.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.time_scale = float(time_scale)
        self.input = nn.Linear(n_mels, channels)
        self.input_cond = nn.Conv1d(cond_channels, channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.layers = nn.ModuleList(
            [
                LYNXNet2Block(channels, expansion, kernel_size, adaln)
                for _ in range(layers)
            ]
        )
        self.voice = nn.Linear(cond_channels, channels) if adaln else None
        self.span_mlp = self.guide_mlp = None
        if span:
            # No biases, so a zero span and a guidance scale of 1 add exactly nothing.
            self.span_mlp, self.guide_mlp = (
                nn.Sequential(
                    nn.Linear(channels, channels * 4, bias=False),
                    nn.GELU(),
                    nn.Linear(channels * 4, channels, bias=False),
                )
                for _ in range(2)
            )
            nn.init.zeros_(self.span_mlp[-1].weight)
            nn.init.zeros_(self.guide_mlp[-1].weight)
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, n_mels)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.input_cond.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _vanishing(self, value, scale):
        # Sinusoids of `value` that are all 0 at 0.
        features = timestep_embedding(value.reshape(-1), self.channels, scale)
        half = self.channels // 2
        return torch.cat((features[:, :half], 1.0 - features[:, half:]), dim=-1)

    def forward(self, x, t, cond, mask, voice=None, span=None, guidance=None):
        """
        Args:
            x (torch.Tensor): Noisy mel, shape (batch, n_mels, frames).
            t (torch.Tensor): Flow time, shape (batch,) or (batch, frames).
            cond (torch.Tensor): Conditioning, shape (batch, cond_channels, frames).
            mask (torch.Tensor): Frame mask, shape (batch, 1, frames).
            voice (torch.Tensor, optional): Speaker embedding, shape (batch, cond_channels).
            span (torch.Tensor, optional): Length of the step whose mean velocity is wanted, shape (batch,), the velocity at `t` when None.
            guidance (torch.Tensor, optional): Speaker guidance scale the output is under, shape (batch,), unguided when None.
        """
        time = self.time_mlp(
            timestep_embedding(t.reshape(-1), self.channels, self.time_scale)
        )
        time = time.view(t.shape[0], -1, self.channels)
        if span is not None:
            features = self._vanishing(span, self.time_scale)
            time = time + self.span_mlp(features).view(
                span.shape[0], -1, self.channels
            )
        if guidance is not None:
            features = self._vanishing(guidance - 1.0, GUIDANCE_SCALE)
            time = time + self.guide_mlp(features).view(
                guidance.shape[0], -1, self.channels
            )
        frame_mask = mask.transpose(1, 2)
        # Full precision in: at late t the leftover noise is under BF16's step.
        with torch.autocast(x.device.type, enabled=False):
            h = self.input(x.transpose(1, 2).to(self.input.weight.dtype))
        h = h + self.input_cond(cond).transpose(1, 2) + time
        h = h * frame_mask
        embedding = None
        if self.voice is not None:
            embedding = time + self.voice(voice)[:, None, :]
        for layer in self.layers:
            h = layer(h, frame_mask, embedding, fused=span is None)
        h = self.norm(h)
        return (self.output(h) * frame_mask).transpose(1, 2)


class AuxDecoder(nn.Module):
    """
    DiffSinger's ConvNeXt aux decoder: a deterministic mel from the
    conditioning, where shallow sampling starts.

    Args:
        cond_channels (int): Number of conditioning channels.
        n_mels (int): Number of mel bins.
        channels (int, optional): Number of channels. Defaults to 512.
        layers (int, optional): Number of blocks. Defaults to 6.
        dropout (float, optional): Dropout of the blocks. Defaults to 0.1.
        speaker (bool, optional): Modulate every block by the speaker embedding. Defaults to False.
    """

    def __init__(
        self, cond_channels, n_mels, channels=512, layers=6, dropout=0.1, speaker=False
    ):
        super().__init__()
        self.input = nn.Conv1d(cond_channels, channels, 7, padding=3)
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    channels,
                    layer_scale=1e-6,
                    dropout=dropout,
                    speaker_channels=cond_channels if speaker else 0,
                )
                for _ in range(layers)
            ]
        )
        self.output = nn.Conv1d(channels, n_mels, 7, padding=3)
        self.output.use_adamw = True

    def forward(self, cond, mask, voice=None):
        x = self.input(cond) * mask
        for block in self.blocks:
            x = block(x, mask, voice)
        return self.output(x) * mask
