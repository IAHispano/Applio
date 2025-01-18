import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
    ):
        super().__init__()

        self.period = period
        
        in_channels  = [ 1,   32,  128,  512, 1024]
        out_channels = [32,  128,  512, 1024, 1024]
        scales       = [ 3,    3,    3,    3,    1]
        
        self.convs = nn.ModuleList([
            weight_norm(
                nn.Conv2d(in_ch, out_ch, (5, 1), (s, 1), padding=(2, 0),)
            ) for in_ch, out_ch, s in zip(in_channels, out_channels, scales)
        ])

        self.output_conv = weight_norm(nn.Conv2d(out_channels[-1], 1, (3, 1), padding=(1, 0),))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        fmap = []
        for f in self.convs:
            x = self.lrelu(f(x))
            fmap.append(x)
        x = self.output_conv(x)
        out = torch.flatten(x, 1, -1)
        return out, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.periods = [2, 3, 5, 7, 11]
      
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in self.periods]
        )

    def forward(self, x):
        outs, fmaps = [], []
        for f in self.discriminators:
            out, fmap = f(x)
            outs.append(out)
            fmaps.extend(fmap)
        return outs, fmaps

class SpectralDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        win_length: int,
    ):

        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, "hann_window")(win_length))

        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 32, 1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        kernel_sizes = [[7,5],[5,3],[5,3],[3,3],[3,3],[3,3]]
        paddings     = [[3,2],[2,1],[2,1],[1,1],[1,1],[1,1]]
        strides      = [[2,2],[2,1],[2,2],[2,1],[2,2],[1,1]]

        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(32, 32, kernel_size=k, padding=p, stride=s))
                for k, p, s in zip(kernel_sizes, paddings, strides)
            ]
        )
            
        self.output_conv = weight_norm(nn.Conv2d(32, 1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = torch.stft(
            x.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs().unsqueeze(1)

        x = self.input_conv(x)
        fmap = []
        for f in self.convs:
            x = self.lrelu(f(x))
            fmap.append(x)
        x = self.output_conv(x)

        return x, fmap

class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes   = [1024, 2048,  512],
        hop_sizes   = [ 256,  512,  128],
        win_lengths = [1024, 2048,  512],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.discriminators = nn.ModuleList(
            [SpectralDiscriminator(f, h, w) for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, x):
        outs, fmaps = [], []
        for f in self.discriminators:
            out, fmap = f(x)
            outs.append(out)
            fmaps.extend(fmap)

        return outs, fmaps

class MultiResolutionMultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()

    def forward(self, y, y_hat):
        mpd_outs_r, mpd_fmaps_r = self.mpd(y)
        mrd_outs_r, mrd_fmaps_r = self.mrd(y)
        outs_r = mpd_outs_r + mrd_outs_r
        fmaps_r = mpd_fmaps_r + mrd_fmaps_r

        mpd_outs_g, mpd_fmaps_g = self.mpd(y_hat)
        mrd_outs_g, mrd_fmaps_g = self.mrd(y_hat)
        outs_g = mpd_outs_g + mrd_outs_g
        fmaps_g = mpd_fmaps_g + mrd_fmaps_g

        return outs_r, outs_g, fmaps_r, fmaps_g