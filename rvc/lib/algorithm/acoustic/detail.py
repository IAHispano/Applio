"""Experimental pitch-aligned detail correction, independent of source PCM.

The frozen backbone supplies target tone; a frequency-local head predicts a
physical log-mel correction from that prediction, harmonic geometry and target
conditioning. It receives no reference mel or source spectral detail at inference.
Its final layer starts at zero to preserve the old model on initialization.
"""

import torch
from torch import nn
from torch.nn import functional as F


def fine_structure(mel, sigma=3.0):
    """Remove a broad mel-index envelope, with half-sample reflected boundaries.

    This Gaussian decomposition matches the listening diagnostic; it is not an
    exact physical separation of formants, harmonics or aperiodicity. Frequency
    filtering is frame-local and requires no future audio or streaming history.
    """
    with torch.autocast(device_type=mel.device.type, enabled=False):
        mel = mel.float()
        radius = int(4 * sigma + 0.5)
        axis = torch.arange(-radius, radius + 1, device=mel.device).float()
        weights = torch.exp(-0.5 * (axis / sigma).square())
        weights = weights / weights.sum()
        bands = mel.shape[1]
        indices = torch.arange(-radius, bands + radius, device=mel.device) % (2 * bands)
        indices = torch.where(indices < bands, indices, 2 * bands - 1 - indices)
        padded = mel.index_select(1, indices).transpose(1, 2)
        padded = padded.reshape(-1, 1, bands + 2 * radius)
        smooth = F.conv1d(padded, weights[None, None])
        smooth = smooth.reshape(mel.shape[0], mel.shape[-1], bands).transpose(1, 2)
        return mel - smooth


class HarmonicDetailHead(nn.Module):
    """Shared frequency kernels align local correction with supplied F0 geometry.

    Frequency coordinates and backbone mel convey band/tone context. FiLM from
    target-conditioned backbone features conveys voice and articulation. Kernels
    operate over frequency only, so packet boundaries add no temporal context.
    This is an unaccepted research architecture, not a naturalness guarantee.
    """

    def __init__(self, condition_width, mel_dim, width=32):
        super().__init__()
        self.input = nn.Conv2d(3, width, (5, 1), padding=(2, 0))
        self.condition = nn.Conv1d(condition_width, 2 * width, 1)
        self.layers = nn.ModuleList(
            nn.Conv2d(width, width, (5, 1), padding=(2, 0)) for _ in range(2)
        )
        self.output = nn.Conv2d(width, 1, (5, 1), padding=(2, 0))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("frequency", torch.linspace(-1, 1, mel_dim)[None, :, None])

    def forward(self, physical_mel, harmonic_geometry, condition):
        if physical_mel.shape != harmonic_geometry.shape:
            raise ValueError("Detail head requires aligned mel and harmonic geometry")
        coordinate = self.frequency.expand_as(physical_mel)
        inputs = torch.stack((physical_mel / 6, harmonic_geometry, coordinate), dim=1)
        x = F.silu(self.input(inputs))
        scale, bias = self.condition(condition).chunk(2, dim=1)
        x = x * (1 + 0.1 * scale.tanh()[:, :, None]) + bias[:, :, None]
        for layer in self.layers:
            x = x + 0.1 * F.silu(layer(x))
        return fine_structure(self.output(F.silu(x))[:, 0])
