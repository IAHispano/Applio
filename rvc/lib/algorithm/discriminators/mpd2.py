import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1

class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3
    ):
        super().__init__()
        self.period = period
        
        in_channels =  [ 1,  32, 128,  512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_ch, out_ch, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0),))
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        
        self.activation = nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x):
        fmap = []
        
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = self.activation(layer(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11, 17, 23, 37]):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(p) for p in periods]
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
