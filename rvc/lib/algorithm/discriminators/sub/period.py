import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from infer.lib.infer_pack.san_modules import SANConv2d
LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, is_san=False, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        1024, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0)
                    )
                ),
            ]
        )
        if is_san:
            self.conv_post = SANConv2d(
                1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)
            )
        else:
            self.conv_post = norm_f(
                nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
            )

    def forward(self, x, is_san=False):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
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


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11], is_san=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(p, is_san=is_san) for p in periods]
        )

    def forward(self, y, y_hat, is_san=False):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y, is_san=is_san)
            y_d_g, fmap_g = d(y_hat, is_san=is_san)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
