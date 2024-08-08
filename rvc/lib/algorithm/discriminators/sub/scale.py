import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from rvc.lib.algorithm.layers.san import SANConv1d

from rvc.lib.algorithm.residuals import LRELU_SLOPE


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, is_san=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.is_san = is_san
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        if is_san:
            self.conv_post = SANConv1d(1024, 1, 3, 1, padding=1)
        else:
            self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        if self.is_san:
            x = self.conv_post(x, is_san=self.is_san)
        else:
            x = self.conv_post(x)
        if self.is_san:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = [x_fun, x_dir]
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, is_san=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.is_san = is_san
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm, is_san=is_san)]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
