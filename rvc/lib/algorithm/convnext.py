import torch
import torch.nn.functional as F
from torch import nn

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        random_tensor = torch.bernoulli(torch.full_like(x, keep_prob))

        return x * random_tensor / keep_prob

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x               # (B, C, T)
        x = self.dwconv(x)      # (B, C, T)
        x = x.transpose(1, 2)   # (B, C, T) -> (B, T, C)
        x = self.norm(x)        # (B, T, C)
        x = self.pwconv1(x)     # (B, T, 4 * C)
        x = self.act(x)         # (B, T, 4 * C)
        x = self.pwconv2(x)     # (B, T, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)   # (B, T, C) -> (B, C, T)

        x = input + self.drop_path(x)
        return x
