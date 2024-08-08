import typing
from typing import List, Tuple

import torch
import torch.nn as nn
from nnAudio import features
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import Resample
from rvc.lib.algorithm.layers.san import SANConv2d
import logging
logging.getLogger("nnAudio").setLevel(logging.ERROR)

class DiscriminatorCQT(nn.Module):
    def __init__(
        self,
        filters,
        max_filters,
        filters_scale,
        dilations,
        in_channels,
        out_channels,
        hop_lengths,
        n_octaves,
        bins_per_octaves,
        sample_rate,
        cqtd_normalize_volume=False,
        is_san=False,
    ):
        super().__init__()
        self.is_san = is_san
        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = (3, 9)
        self.dilations = dilations
        self.stride = (1, 2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fs = sample_rate
        self.hop_length = hop_lengths
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octaves

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
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
                    nn.Conv2d(
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
                nn.Conv2d(
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
                nn.Conv2d(
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

    def forward(self, x):
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

        if self.is_san:
            x = self.conv_post(latent_z, is_san=self.is_san)
        else:
            x = self.conv_post(latent_z)

        if self.is_san:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            latent_z = [x_fun, x_dir]
        else:
            fmap.append(x)
            latent_z = torch.flatten(x, 1, -1)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(
        self,
        filters=128,
        max_filters=1024,
        filters_scale=1,
        dilations=[1, 2, 4],
        in_channels=1,
        out_channels=1,
        hop_lengths=[512, 256, 256],
        n_octaves=[9, 9, 9],
        bins_per_octaves=[24, 36, 48],
        sample_rate=24000,
        is_san=False,
    ):
        super().__init__()
        self.is_san = is_san
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    filters=filters,
                    max_filters=max_filters,
                    filters_scale=filters_scale,
                    dilations=dilations,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hop_lengths=hop_lengths[i],
                    n_octaves=n_octaves[i],
                    sample_rate=sample_rate,
                    bins_per_octaves=bins_per_octaves[i],
                    is_san=is_san,
                )
                for i in range(len(hop_lengths))
            ]
        )

    def forward(
        self, y, y_hat
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y, is_san=self.is_san)
            y_d_g, fmap_g = disc(y_hat, is_san=self.is_san)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
