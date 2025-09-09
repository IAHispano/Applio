import torch
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    """

    def __init__(self, checkpointing: bool = False):
        super().__init__()
        periods = [2, 3, 5, 7, 11, 17, 23, 37]
        self.checkpointing = checkpointing
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS()] + [DiscriminatorP(p) for p in periods]
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

    def __init__(self):
        super().__init__()

        self.convs = torch.nn.ModuleList(
            [
                weight_norm(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                weight_norm(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                weight_norm(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                weight_norm(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                weight_norm(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                weight_norm(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = weight_norm(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
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
    """

    def __init__(self, period: int, kernel_size: int = 5):
        super().__init__()
        self.period = period

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                weight_norm(
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

        self.conv_post = weight_norm(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
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
