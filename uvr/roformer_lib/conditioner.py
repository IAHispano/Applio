import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class BandEmbedder(nn.Module):
    """
    Embeds frequency bands into vector representations.
    """

    def __init__(
        self,
        num_bands,
        model_channels,
        band_channels,
    ):
        super().__init__()

        self.num_bands = num_bands
        self.null_bands_emb = nn.Parameter(torch.randn(1, model_channels))

        if num_bands is not None:
            self.band_emb = nn.Embedding(num_bands, model_channels)

        self.band_to_cond = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, band_channels),
            nn.SiLU(),
            nn.Linear(band_channels, band_channels),
        )

    def forward(self, bands, cond_drop_prob=0):

        bands_emb = self.band_emb(bands)

        if cond_drop_prob > 0:

            band_keep_mask = prob_mask_like(
                (bands.shape[0],), 1 - cond_drop_prob, device=bands.device
            )

            bands_emb = torch.where(
                rearrange(band_keep_mask, "b -> b 1"), bands_emb, self.null_bands_emb
            )

        bands_emb = self.band_to_cond(bands_emb)

        return bands_emb
