import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1

class DiscriminatorR(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        window,
        multiplier=1
    ):
        super().__init__()
       
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.pad = (n_fft - hop_length) // 2

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
        
        x = F.pad(x, (self.pad, self.pad), mode="reflect",).squeeze(1)
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
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
            [DiscriminatorR(n_fft, hop_length, win_length, torch.ones(win_length))
            for (n_fft, hop_length, win_length) in resolutions]
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
