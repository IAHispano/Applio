"""Low-rank 1x1 adapters with exact forward/merge scaling."""

import torch
import torch.nn.functional as F
from torch import nn


class AdaptedConv(nn.Module):
    """Frozen pointwise convolution plus (alpha/rank) * up(down(x)).

    Zero-initialized up weights preserve the base output initially. Export folds
    the low-rank matrices into the base convolution; runtime needs no adapter
    branches, and the universal vocoder is not adapted by this module.
    """

    def __init__(self, base, rank, alpha):
        super().__init__()
        if base.kernel_size != (1,) or base.groups != 1:
            raise ValueError("Adapters only support ungrouped pointwise convolutions")
        self.base = base
        self.scale = alpha / rank
        self.down = nn.Parameter(
            base.weight.new_empty(rank, base.in_channels, 1).normal_(0, 0.01)
        )
        self.up = nn.Parameter(base.weight.new_zeros(base.out_channels, rank, 1))

    def forward(self, x):
        return self.base(x) + F.conv1d(F.conv1d(x, self.down), self.up) * self.scale

    def merged(self):
        weight = (
            self.base.weight.detach()
            + (self.up[..., 0] @ self.down[..., 0])[..., None] * self.scale
        )
        with torch.no_grad():
            self.base.weight.copy_(weight)
        return self.base


def install_adapters(model, rank=8, alpha=8.0):
    if rank < 1 or alpha <= 0:
        raise ValueError("Adapter rank and alpha must be positive")
    model.requires_grad_(False)
    for name, child in list(model.named_children()):
        if (
            isinstance(child, nn.Conv1d)
            and child.kernel_size == (1,)
            and child.groups == 1
        ):
            setattr(model, name, AdaptedConv(child, rank, alpha))
        elif not isinstance(child, AdaptedConv):
            install_children(child, rank, alpha)
    model.speaker.weight.requires_grad_(True)


def install_children(parent, rank, alpha):
    for name, child in list(parent.named_children()):
        if (
            isinstance(child, nn.Conv1d)
            and child.kernel_size == (1,)
            and child.groups == 1
        ):
            setattr(parent, name, AdaptedConv(child, rank, alpha))
        elif not isinstance(child, AdaptedConv):
            install_children(child, rank, alpha)


def merge_adapters(model):
    for name, child in list(model.named_children()):
        if isinstance(child, AdaptedConv):
            setattr(model, name, child.merged())
        else:
            merge_adapters(child)
