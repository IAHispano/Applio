from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm
from typing import List

from rvc.lib.algorithm.pqmf import PQMF

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class MDC(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        dilations,
        use_spectral_norm=False
    ):
        super().__init__()
        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                weight_norm(Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_k,
                    dilation=_d,
                    padding=get_padding(_k, _d)
                ))
            )
        self.post_conv = weight_norm(Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides,
            padding=get_padding(_k, _d)
        ))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        _out = None
        for _l in self.d_convs:
            _x = torch.unsqueeze(_l(x), -1)
            _x = F.leaky_relu(_x, 0.2)
            if _out is None:
                _out = _x
            else:
                _out = torch.cat([_out, _x], axis=-1)
        x = torch.sum(_out, dim=-1)
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)  # @@

        return x


class SBDBlock(torch.nn.Module):
    def __init__(
        self,
        segment_dim,
        strides,
        filters,
        kernel_size,
        dilations,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])

        for _s, _f, _k, _d in zip(
            strides,
            filters_in_out,
            kernel_size,
            dilations
        ):
            self.convs.append(MDC(
                in_channels=_f[0],
                out_channels=_f[1],
                strides=_s,
                kernel_size=_k,
                dilations=_d,
            ))
        self.post_conv = weight_norm(Conv1d(
            in_channels=_f[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=3 // 2
        ))  # @@


    def forward(self, x):
        fmap = []
        for _l in self.convs:
            x = _l(x)
            fmap.append(x)
        x = self.post_conv(x)  # @@

        return x, fmap

class SBD(torch.nn.Module):
    def __init__(self, segment_size: int):
        super(SBD, self).__init__()
        
        self.bands = {
            "band1":
                {"filters":   [64, 128, 256, 256, 256],
                 "kernels":   [[7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]],
                 "dilations": [[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
                 "strides":   [1, 1, 3, 3, 1],
                 "range":     [0, 6],
                 "transpose": False,
                },
            "band2":
                {"filters":   [64, 128, 256, 256, 256],
                 "kernels":   [[5, 5, 5],[5, 5, 5],[5, 5, 5],[5, 5, 5],[5, 5, 5]],
                 "dilations": [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
                 "strides":   [1, 1, 3, 3, 1],
                 "range":     [0, 11],
                 "transpose": False,
                },     
            "band3":
                {"filters":   [64, 128, 256, 256, 256],
                 "kernels":   [[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]],
                 "dilations": [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                 "strides":   [1, 1, 3, 3, 1],
                 "range":     [0, 16],
                 "transpose": False,
                },
            "band4":
                {"filters":   [32, 64, 128, 128, 128],
                 "kernels":   [[5, 5, 5],[5, 5, 5],[5, 5, 5],[5, 5, 5],[5, 5, 5]],
                 "dilations": [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]],
                 "strides":   [1, 1, 3, 3, 1],
                 "range":     [0, 64],
                 "transpose": True,
                },
        }
                
        self.pqmf = PQMF(16, 256, 0.03, 10.0)
        self.pqmf_f = PQMF(64, 256, 0.1, 9.0)

        self.discriminators = torch.nn.ModuleList()

        for band in self.bands.values():
            if band["transpose"]:
                segment_dim = segment_size // 64
            else:
                segment_dim = band["range"][1] - band["range"][0]
            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim,
                filters=band["filters"],
                kernel_size=band["kernels"],
                dilations=band["dilations"],
                strides=band["strides"],
            ))

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)

        for d, band in zip(
            self.discriminators,
            self.bands.values()
        ):
            _br = band["range"]
            if band["transpose"]:
                y_in_f = self.pqmf_f.analysis(y)
                y_hat_in_f = self.pqmf_f.analysis(y_hat)
                y_d_r, fmap_r = d(y_in_f.transpose(1,2))
                y_d_g, fmap_g = d(y_hat_in_f.transpose(1,2))
            else:
                y_d_r, fmap_r = d(y_in[:, _br[0]:_br[1], :])
                y_d_g, fmap_g = d(y_hat_in[:, _br[0]:_br[1], :])
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs