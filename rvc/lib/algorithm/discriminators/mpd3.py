import torch
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        periods (str): Periods of the discriminator. V1 = [2, 3, 5, 7, 11, 17], V2 = [2, 3, 5, 7, 11, 17, 23, 37].
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(self, version: str, use_spectral_norm: bool = False, use_multiscale=False):
        super(MultiPeriodDiscriminator, self).__init__()
        
        self.use_multiscale = use_multiscale
        
        periods = (
            [2, 3, 5, 7, 11, 17] if version == "v1" else [2, 3, 5, 7, 11, 17, 23, 37]
        )
        self.discriminators = torch.nn.ModuleList([
            DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods
        ])
        
        if self.use_multiscale:
            self.scale_discriminators = torch.nn.ModuleList([
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ])
            self.meanpools = nn.ModuleList([
                torch.nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)
            ])
        else:
            self.discriminators.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))

    def forward(self, y, y_hat):
        """
        Forward pass of the multi-period discriminator.

        Args:
            y (torch.Tensor): Real audio signal.
            y_hat (torch.Tensor): Fake audio signal.
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        for i, d in enumerate(self.scale_discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
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
        super(DiscriminatorS, self).__init__()
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
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input audio signal.
        """
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x), inplace=True)
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
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input audio signal.
        """
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x), inplace=True)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap
