# MIT License

# Copyright (c) 2023 Sony Research Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(tensor, dim):
    denom = tensor.norm(p=2.0, dim=dim, keepdim=True).clamp_min(1e-12)
    return tensor / denom


class SANConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super(SANConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        scale = self.weight.norm(p=2.0, dim=[1, 2], keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.zeros(in_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input, is_san=False):
        if self.bias is not None:
            input = input + self.bias.view(1, self.in_channels, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1)
        if is_san:
            out_fun = F.conv1d(
                input,
                normalized_weight.detach(),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out_dir = F.conv1d(
                input.detach(),
                normalized_weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = F.conv1d(
                input,
                normalized_weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=[1, 2])


class SANConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super(SANConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        scale = self.weight.norm(p=2.0, dim=[1, 2, 3], keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.zeros(in_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input, is_san=False):
        if self.bias is not None:
            input = input + self.bias.view(1, self.in_channels, 1, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1, 1)
        if is_san:
            out_fun = F.conv2d(
                input,
                normalized_weight.detach(),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out_dir = F.conv2d(
                input.detach(),
                normalized_weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = F.conv2d(
                input,
                normalized_weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=[1, 2, 3])
