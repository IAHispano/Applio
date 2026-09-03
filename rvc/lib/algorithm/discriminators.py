import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE
from rvc.lib.algorithm.univhd import UnivHDDiscriminator


#: The rate ``[2, 3, 5, 7, 11]`` was chosen at.  A period-``p`` branch folds
#: onto a grid at ``sr / p`` Hz and its receptive field spans ``647 * p / sr``
#: seconds, so both meanings of a period hold only if ``p`` scales with the
#: rate.  Carrying the set unchanged empties the *slow* end -- at 32 kHz the
#: longest branch drops from 323 ms to 222, and pitch structure lives there.
REFERENCE_SAMPLE_RATE = 22050

#: ``v3``'s periods at the reference rate, before scaling.  This is upstream's
#: ``[2, 3, 5, 7, 11]`` without its longest: the spectrogram branches are the
#: only part of this discriminator with resolution above 10 kHz, and a longer
#: period folds at a lower rate, so it is the branch least able to say anything
#: up there -- and it costs 8.2 M parameters, a fifth of the whole thing.
V3_BASE_PERIODS = (2, 3, 5, 7)


def rate_scaled_periods(periods, sample_rate, reference_rate=REFERENCE_SAMPLE_RATE):
    """The period set that keeps the branches' time scales at another rate.

    Targets are rounded to the nearest unused prime in log space -- prime
    because two periods sharing a factor fold onto overlapping samples and
    become one branch at two branches' cost, log because the quantity preserved
    is a ratio.  ``reference_rate`` returns the input unchanged, which makes
    this a derivation rather than a new design.

    A period leaves no trace in a parameter shape, so a checkpoint trained with
    one set loads into another without a murmur.
    """

    def is_prime(value):
        return value > 1 and all(value % f for f in range(2, int(value**0.5) + 1))

    candidates = [value for value in range(2, 512) if is_prime(value)]
    used, scaled = set(), []
    for period in periods:
        target = int(period) * float(sample_rate) / float(reference_rate)
        best = min(
            (value for value in candidates if value not in used),
            key=lambda value: abs(math.log(value / target)),
        )
        used.add(best)
        scaled.append(best)
    return tuple(sorted(scaled))


#: The three multi-resolution spectrogram branches.  The 512-point branch's
#: 50-sample hop is what reads frame-rate modulation the other two average
#: away.
V3_RESOLUTIONS = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        checkpointing: bool = False,
        version: str = "v2",
        sample_rate: int = 32000,
    ):
        super().__init__()

        univhd = False
        if version == "v1":
            periods = [2, 3, 5, 7, 11, 17]
            resolutions = []
        elif version == "v2":
            periods = [2, 3, 5, 7, 11, 17, 23, 37]
            resolutions = []
        elif version == "v3":
            # Rate-scaled periods, plus the harmonic branch: 0.33 M
            # parameters against this discriminator's 39 M, and 8% of the step.
            periods = rate_scaled_periods(V3_BASE_PERIODS, sample_rate)
            resolutions = V3_RESOLUTIONS
            univhd = True
        else:
            raise ValueError(f"Unknown discriminator version {version!r}.")

        self.version = version
        self.periods = list(periods)
        self.checkpointing = checkpointing
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
            + [
                DiscriminatorR(r, use_spectral_norm=use_spectral_norm)
                for r in resolutions
            ]
            + (
                [
                    UnivHDDiscriminator(
                        sample_rate=sample_rate,
                        use_spectral_norm=use_spectral_norm,
                    )
                ]
                if univhd
                else []
            )
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super().__init__()

        self.resolution = resolution
        self.lrelu_slope = 0.1
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        1,
                        32,
                        (3, 9),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x).unsqueeze(1)

        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(
            x,
            (pad, pad),
            mode="reflect",
        ).squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.ones(win_length, device=x.device),
            center=False,
            return_complex=True,
        )

        mag = torch.norm(torch.view_as_real(x), p=2, dim=-1)  # [B, F, TT]

        return mag
