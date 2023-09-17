# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Differentiable, Pytorch based resampling.
Implementation of Julius O. Smith algorithm for resampling.
See https://ccrma.stanford.edu/~jos/resample/ for details.
This implementation is specially optimized for when new_sr / old_sr is a fraction
with a small numerator and denominator when removing the gcd (e.g. new_sr = 700, old_sr = 500).

Very similar to [bmcfee/resampy](https://github.com/bmcfee/resampy) except this implementation
is optimized for the case mentioned before, while resampy is slower but more general.

"""

import math
from typing import Optional

import torch
from torch.nn import functional as F

from .core import sinc
from .utils import simple_repr


class ResampleFrac(torch.nn.Module):
    """
    Resampling from the sample rate `old_sr` to `new_sr`.
    """
    def __init__(self, old_sr: int, new_sr: int, zeros: int = 24, rolloff: float = 0.945):
        """
        Args:
            old_sr (int): sample rate of the input signal x.
            new_sr (int): sample rate of the output.
            zeros (int): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce anti-aliasing, but will reduce some of the
                highest frequencies.

        Shape:

            - Input: `[*, T]`
            - Output: `[*, T']` with `T' = int(new_sr * T / old_sr)


        .. caution::
            After dividing `old_sr` and `new_sr` by their GCD, both should be small
            for this implementation to be fast.

        >>> import torch
        >>> resample = ResampleFrac(4, 5)
        >>> x = torch.randn(1000)
        >>> print(len(resample(x)))
        1250
        """
        super().__init__()
        if not isinstance(old_sr, int) or not isinstance(new_sr, int):
            raise ValueError("old_sr and new_sr should be integers")
        gcd = math.gcd(old_sr, new_sr)
        self.old_sr = old_sr // gcd
        self.new_sr = new_sr // gcd
        self.zeros = zeros
        self.rolloff = rolloff

        self._init_kernels()

    def _init_kernels(self):
        if self.old_sr == self.new_sr:
            return

        kernels = []
        sr = min(self.new_sr, self.old_sr)
        # rolloff will perform antialiasing filtering by removing the highest frequencies.
        # At first I thought I only needed this when downsampling, but when upsampling
        # you will get edge artifacts without this, the edge is equivalent to zero padding,
        # which will add high freq artifacts.
        sr *= self.rolloff

        # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
        # using the sinc interpolation formula:
        #   x(t) = sum_i x[i] sinc(pi * old_sr * (i / old_sr - t))
        # We can then sample the function x(t) with a different sample rate:
        #    y[j] = x(j / new_sr)
        # or,
        #    y[j] = sum_i x[i] sinc(pi * old_sr * (i / old_sr - j / new_sr))

        # We see here that y[j] is the convolution of x[i] with a specific filter, for which
        # we take an FIR approximation, stopping when we see at least `zeros` zeros crossing.
        # But y[j+1] is going to have a different set of weights and so on, until y[j + new_sr].
        # Indeed:
        # y[j + new_sr] = sum_i x[i] sinc(pi * old_sr * ((i / old_sr - (j + new_sr) / new_sr))
        #               = sum_i x[i] sinc(pi * old_sr * ((i - old_sr) / old_sr - j / new_sr))
        #               = sum_i x[i + old_sr] sinc(pi * old_sr * (i / old_sr - j / new_sr))
        # so y[j+new_sr] uses the same filter as y[j], but on a shifted version of x by `old_sr`.
        # This will explain the F.conv1d after, with a stride of old_sr.
        self._width = math.ceil(self.zeros * self.old_sr / sr)
        # If old_sr is still big after GCD reduction, most filters will be very unbalanced, i.e.,
        # they will have a lot of almost zero values to the left or to the right...
        # There is probably a way to evaluate those filters more efficiently, but this is kept for
        # future work.
        idx = torch.arange(-self._width, self._width + self.old_sr).float()
        for i in range(self.new_sr):
            t = (-i/self.new_sr + idx/self.old_sr) * sr
            t = t.clamp_(-self.zeros, self.zeros)
            t *= math.pi
            window = torch.cos(t/self.zeros/2)**2
            kernel = sinc(t) * window
            # Renormalize kernel to ensure a constant signal is preserved.
            kernel.div_(kernel.sum())
            kernels.append(kernel)

        self.register_buffer("kernel", torch.stack(kernels).view(self.new_sr, 1, -1))

    def forward(self, x: torch.Tensor, output_length: Optional[int] = None, full: bool = False):
        """
        Resample x.
        Args:
            x (Tensor): signal to resample, time should be the last dimension
            output_length (None or int): This can be set to the desired output length
                (last dimension). Allowed values are between 0 and
                ceil(length * new_sr / old_sr). When None (default) is specified, the
                floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the `output_length` only
                for the last one, while passing `full=True` to all the other ones.
        """
        if self.old_sr == self.new_sr:
            return x
        shape = x.shape
        length = x.shape[-1]
        x = x.reshape(-1, length)
        x = F.pad(x[:, None], (self._width, self._width + self.old_sr), mode='replicate')
        ys = F.conv1d(x, self.kernel, stride=self.old_sr)  # type: ignore
        y = ys.transpose(1, 2).reshape(list(shape[:-1]) + [-1])

        float_output_length = self.new_sr * length / self.old_sr
        max_output_length = int(math.ceil(float_output_length))
        default_output_length = int(float_output_length)
        if output_length is None:
            output_length = max_output_length if full else default_output_length
        elif output_length < 0 or output_length > max_output_length:
            raise ValueError(f"output_length must be between 0 and {max_output_length}")
        else:
            if full:
                raise ValueError("You cannot pass both full=True and output_length")
        return y[..., :output_length]

    def __repr__(self):
        return simple_repr(self)


def resample_frac(x: torch.Tensor, old_sr: int, new_sr: int,
                  zeros: int = 24, rolloff: float = 0.945,
                  output_length: Optional[int] = None, full: bool = False):
    """
    Functional version of `ResampleFrac`, refer to its documentation for more information.

    ..warning::
        If you call repeatidly this functions with the same sample rates, then the
        resampling kernel will be recomputed everytime. For best performance, you should use
        and cache an instance of `ResampleFrac`.
    """
    return ResampleFrac(old_sr, new_sr, zeros, rolloff).to(x)(x, output_length, full)


# Easier implementations for downsampling and upsampling by a factor of 2
# Kept for testing and reference

def _kernel_upsample2_downsample2(zeros):
    # Kernel for upsampling and downsampling by a factor of 2. Interestingly,
    # it is the same kernel used for both.
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def _upsample2(x, zeros=24):
    """
    Upsample x by a factor of two. The output will be exactly twice as long as the input.
    Args:
        x (Tensor): signal to upsample, time should be the last dimension
        zeros (int): number of zero crossing to keep in the sinc filter.

    This function is kept only for reference, you should use the more generic `resample_frac`
    one. This function does not perform anti-aliasing filtering.
    """
    *other, time = x.shape
    kernel = _kernel_upsample2_downsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = torch.stack([x, out], dim=-1)
    return y.view(*other, -1)


def _downsample2(x, zeros=24):
    """
    Downsample x by a factor of two. The output length is half of the input, ceiled.
    Args:
        x (Tensor): signal to downsample, time should be the last dimension
        zeros (int): number of zero crossing to keep in the sinc filter.

    This function is kept only for reference, you should use the more generic `resample_frac`
    one. This function does not perform anti-aliasing filtering.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = _kernel_upsample2_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)
