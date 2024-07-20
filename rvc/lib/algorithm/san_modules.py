import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(tensor, dim):
    denom = tensor.norm(p=2.0, dim=dim, keepdim=True).clamp_min(1e-12)
    return tensor / denom


class SANLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype=None
                 ):
        super(SANLinear, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype)
        scale = self.weight.norm(p=2.0, dim=1, keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(in_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, flg_train=False):
        if self.bias is not None:
            input = input + self.bias
        normalized_weight = self._get_normalized_weight()
        scale = self.scale
        if flg_train:
            out_fun = F.linear(input, normalized_weight.detach(), None)
            out_dir = F.linear(input.detach(), normalized_weight, None)
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = F.linear(input, normalized_weight, None)
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=1)


class SANConv1d(nn.Conv1d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None
                 ):
        super(SANConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation,
            groups=1, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        scale = self.weight.norm(p=2.0, dim=[1, 2], keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(in_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, flg_train=False):
        if self.bias is not None:
            input = input + self.bias.view(self.in_channels, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1)
        if flg_train:
            out_fun = F.conv1d(input, normalized_weight.detach(), None, self.stride,
                               self.padding, self.dilation, self.groups)
            out_dir = F.conv1d(input.detach(), normalized_weight, None, self.stride,
                               self.padding, self.dilation, self.groups)
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = F.conv1d(input, normalized_weight, None, self.stride,
                           self.padding, self.dilation, self.groups)
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=[1, 2])


class SANConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None
                 ):
        super(SANConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation,
            groups=1, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        scale = self.weight.norm(p=2.0, dim=[1, 2, 3], keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(in_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, flg_train=False):
        if self.bias is not None:
            input = input + self.bias.view(self.in_channels, 1, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1, 1)
        if flg_train:
            out_fun = F.conv2d(input, normalized_weight.detach(), None, self.stride,
                               self.padding, self.dilation, self.groups)
            out_dir = F.conv2d(input.detach(), normalized_weight, None, self.stride,
                               self.padding, self.dilation, self.groups)
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = F.conv2d(input, normalized_weight, None, self.stride,
                           self.padding, self.dilation, self.groups)
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=[1, 2, 3])


class SANEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim,
                 scale_grad_by_freq=False,
                 sparse=False, _weight=None,
                 device=None, dtype=None):
        super(SANEmbedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx=None,
            max_norm=None, norm_type=2., scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse, _weight=_weight,
            device=device, dtype=dtype)
        scale = self.weight.norm(p=2.0, dim=1, keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale)

    def forward(self, input, flg_train=False):
        out = F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        out = _normalize(out, dim=-1)
        scale = F.embedding(
            input, self.scale, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        if flg_train:
            out_fun = out.detach()
            out_dir = out
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = _normalize(self.weight, dim=1)
