import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1

class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution,
        multiplier=1
    ):
        super().__init__()
       
        assert (len(resolution) == 3), "MRD layer requires list with len=3, got {}".format(resolution)

        self.resolution = resolution
        print(resolution)
        base_channels = int(32 * multiplier)

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(            1, base_channels, (3, 9),                padding=(1, 4))),
            weight_norm(nn.Conv2d(base_channels, base_channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(base_channels, base_channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(base_channels, base_channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(base_channels, base_channels, (3, 3),                padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(base_channels, 1, (3, 3), padding=(1, 1)))

        self.activation = nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x).unsqueeze(1)

        for layer in self.convs:
            x = self.activation(layer(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = (n_fft - hop_length) // 2
        x = F.pad(x, (pad, pad), mode="reflect",).squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.ones(win_length, device=x.device),
            center=False,
            return_complex=True,
        )
        # Compute magnitude from the complex STFT output
        x = torch.norm(torch.view_as_real(x), p=2, dim=-1)  # [B, F, TT]
        return x

class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, resolutions=None):
        super().__init__()
        if resolutions is None:
            resolutions = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
        assert all(len(r) == 3 for r in resolutions), (
            f"Each resolution must be a list of three integers, but got {resolutions}."
        )
        
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(r) for r in resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
