import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional

import sys
import os

sys.path.append(os.getcwd())

from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock1, ResBlock2
from rvc.lib.algorithm.commons import init_weights


class Generator(torch.nn.Module):
    """Generator for synthesizing audio. Optimized for performance and quality.

    Args:
        initial_channel (int): Number of channels in the initial convolutional layer.
        resblock (str): Type of residual block to use (1 or 2).
        resblock_kernel_sizes (list): Kernel sizes of the residual blocks.
        resblock_dilation_sizes (list): Dilation rates of the residual blocks.
        upsample_rates (list): Upsampling rates.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes of the upsampling layers.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 0.
    """

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = ResBlock1 if resblock == 1 else ResBlock2

        self.ups_and_resblocks = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups_and_resblocks.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.ups_and_resblocks.append(resblock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups_and_resblocks.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
            x = self.conv_pre(x)
            if g is not None:
                x = x + self.cond(g)

            resblock_idx = 0
            for _ in range(self.num_upsamples):
                x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
                x = self.ups_and_resblocks[resblock_idx](x)
                resblock_idx += 1
                xs = 0
                for _ in range(self.num_kernels):
                    xs += self.ups_and_resblocks[resblock_idx](x)
                    resblock_idx += 1
                x = xs / self.num_kernels

            x = torch.nn.functional.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)

            return x

    def __prepare_scriptable__(self):
        """Prepares the module for scripting."""
        for l in self.ups_and_resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self

    def remove_weight_norm(self):
        """Removes weight normalization from the upsampling and residual blocks."""
        for l in self.ups_and_resblocks:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
