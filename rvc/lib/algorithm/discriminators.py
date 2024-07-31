import torch
import typing
from typing import List, Tuple
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from nnAudio import features
from torchaudio.transforms import Resample
from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE
from .san_modules import SANConv2d, SANConv1d

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

    def __init__(self, hps, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = getattr(hps, "mpd")
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        )

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

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiPeriodDiscriminatorV2(torch.nn.Module):
    """
    Multi-period discriminator V2.

    This class implements a multi-period discriminator V2, which is used
    to discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(self, hps,use_spectral_norm=False):
        super(MultiPeriodDiscriminatorV2, self).__init__()
        periods = getattr(hps, "mpd")
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        )

    def forward(self, y, y_hat):
        """
        Forward pass of the multi-period discriminator V2.

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

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorCQT(torch.nn.Module):
    def __init__(
        self,
        filters,
        max_filters,
        filters_scale,
        dilations,
        in_channels,
        out_channels,
        hop_length,
        n_octaves,
        bins_per_octave,
        sample_rate,
        cqtd_normalize_volume=False,
        is_san=False,
    ):
        super().__init__()

        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = (3, 9)
        self.dilations = dilations
        self.stride = (1, 2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fs = sample_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = torch.nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                torch.nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = torch.nn.ModuleList()

        self.convs.append(
            torch.nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(
                torch.nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding(
                        (self.kernel_size[0], self.kernel_size[0])
                    ),
                )
            )
        )

        if is_san:
            self.conv_post = SANConv2d(
                out_chs,
                self.out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            )
        else:
            self.conv_post = weight_norm(
                torch.nn.Conv2d(
                    out_chs,
                    self.out_channels,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding(
                        (self.kernel_size[0], self.kernel_size[0])
                    ),
                )
            )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = cqtd_normalize_volume

    def get_2d_padding(
        self,
        kernel_size: typing.Tuple[int, int],
        dilation: typing.Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x, is_san=False):
        fmap = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, layer in enumerate(self.convs):
            latent_z = layer(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        if is_san:
            x = self.conv_post(latent_z, is_san=is_san)
        else:
            x = self.conv_post(latent_z)

        if is_san:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            latent_z = [x_fun, x_dir]
        else:
            fmap.append(x)
            latent_z = torch.flatten(x, 1, -1)

        return latent_z, fmap

class MultiPeriodDiscriminatorV3(torch.nn.Module):
    def __init__(self, hps, use_spectral_norm=False):
        super(MultiPeriodDiscriminatorV3, self).__init__()
        # periods = [2, 3, 5, 7, 11, 17]
        periods = getattr(hps, "mpd")#[2, 3, 5, 7, 11, 17, 23, 37]
        # Using default values
        filters = getattr(hps, "filters", 32)
        max_filters = getattr(hps, "max_filters", 1024)
        filters_scale = getattr(hps, "filters_scale", 1)
        dilations = getattr(hps, "dilations", [1, 2, 4])
        in_channels = getattr(hps, "in_channels", 1)
        out_channels = getattr(hps, "out_channels", 1)
        hop_lengths = getattr(hps, "hop_lengths", [512, 256, 256])
        n_octaves = getattr(hps, "n_octaves", [9, 9, 9])
        bins_per_octaves = getattr(hps, "bins_per_octaves", [24, 36, 48])
        is_san = getattr(hps, "is_san", False)
        sample_rate = getattr(hps, "sample_rate", False)
        
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        discs = discs + [
                DiscriminatorCQT(
                    filters=filters,
                    max_filters=max_filters,
                    filters_scale=filters_scale,
                    dilations=dilations,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hop_length=hop_lengths[i],
                    n_octaves=n_octaves[i],
                    sample_rate=sample_rate,
                    bins_per_octave=bins_per_octaves[i],
                    is_san=is_san,
                )
                for i in range(len(hop_lengths))
            ]
        self.discriminators = torch.nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []  #
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
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

    def __init__(self, use_spectral_norm=False, is_san=False):
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
        if is_san:
            self.conv_post = SANConv1d(1024, 1, 3, 1, padding=1)
        else:
            self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        

    def forward(self, x, is_san=False):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input audio signal.
        """
        fmap = []
        for conv in self.convs:
            x = torch.nn.functional.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        if is_san:
            x = self.conv_post(x, is_san=is_san)
        else:
            x = self.conv_post(x)
        if is_san:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = [x_fun, x_dir]
        else:
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
        kernel_size (int): Kernel size of the convolutional layers.
            Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, is_san=False):
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
        if is_san:
            self.conv_post = SANConv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
        else:
            self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))


    def forward(self, x, is_san=False):
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
            x = torch.nn.functional.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)

        if is_san:
            x = self.conv_post(x, is_san=is_san)
        else:
            x = self.conv_post(x)
        if is_san:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = [x_fun, x_dir]
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
        return x, fmap

class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution, use_spectral_norm=False, is_san=False):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = torch.nn.ModuleList([
            norm_f(torch.nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(torch.nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        if is_san:
            self.conv_post = SANConv2d(32, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))
        else:
            self.conv_post = norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))
        

    def forward(self, x, is_san=False):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        if is_san:
            x = self.conv_post(x, is_san=is_san)
        else:
            x = self.conv_post(x)
        if is_san:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = [x_fun, x_dir]
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = torch.nn.functional.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False,
                       return_complex=False)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag