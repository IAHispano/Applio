# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
FIR windowed sinc lowpass filters.
"""

import math
from typing import Sequence, Optional

import torch
from torch.nn import functional as F

from .core import sinc
from .fftconv import fft_conv1d
from .utils import simple_repr


class LowPassFilters(torch.nn.Module):
    """
    Bank of low pass filters. Note that a high pass or band pass filter can easily
    be implemented by substracting a same signal processed with low pass filters with different
    frequencies (see `julius.bands.SplitBands` for instance).
    This uses a windowed sinc filter, very similar to the one used in
    `julius.resample`. However, because we do not change the sample rate here,
    this filter can be much more efficiently implemented using the FFT convolution from
    `julius.fftconv`.

    Args:
        cutoffs (list[float]): list of cutoff frequencies, in [0, 0.5] expressed as `f/f_s` where
            f_s is the samplerate and `f` is the cutoff frequency.
            The upper limit is 0.5, because a signal sampled at `f_s` contains only
            frequencies under `f_s / 2`.
        stride (int): how much to decimate the output. Keep in mind that decimation
            of the output is only acceptable if the cutoff frequency is under `1/ (2 * stride)`
            of the original sampling rate.
        pad (bool): if True, appropriately pad the input with zero over the edge. If `stride=1`,
            the output will have the same length as the input.
        zeros (float): Number of zero crossings to keep.
            Controls the receptive field of the Finite Impulse Response filter.
            For lowpass filters with low cutoff frequency, e.g. 40Hz at 44.1kHz,
            it is a bad idea to set this to a high value.
            This is likely appropriate for most use. Lower values
            will result in a faster filter, but with a slower attenuation around the
            cutoff frequency.
        fft (bool or None): if True, uses `julius.fftconv` rather than PyTorch convolutions.
            If False, uses PyTorch convolutions. If None, either one will be chosen automatically
            depending on the effective filter size.


    ..warning::
        All the filters will use the same filter size, aligned on the lowest
        frequency provided. If you combine a lot of filters with very diverse frequencies, it might
        be more efficient to split them over multiple modules with similar frequencies.

    ..note::
        A lowpass with a cutoff frequency of 0 is defined as the null function
        by convention here. This allows for a highpass with a cutoff of 0 to
        be equal to identity, as defined in `julius.filters.HighPassFilters`.

    Shape:

        - Input: `[*, T]`
        - Output: `[F, *, T']`, with `T'=T` if `pad` is True and `stride` is 1, and
            `F` is the numer of cutoff frequencies.

    >>> lowpass = LowPassFilters([1/4])
    >>> x = torch.randn(4, 12, 21, 1024)
    >>> list(lowpass(x).shape)
    [1, 4, 12, 21, 1024]
    """

    def __init__(self, cutoffs: Sequence[float], stride: int = 1, pad: bool = True,
                 zeros: float = 8, fft: Optional[bool] = None):
        super().__init__()
        self.cutoffs = list(cutoffs)
        if min(self.cutoffs) < 0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if max(self.cutoffs) > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.stride = stride
        self.pad = pad
        self.zeros = zeros
        self.half_size = int(zeros / min([c for c in self.cutoffs if c > 0]) / 2)
        if fft is None:
            fft = self.half_size > 32
        self.fft = fft
        window = torch.hann_window(2 * self.half_size + 1, periodic=False)
        time = torch.arange(-self.half_size, self.half_size + 1)
        filters = []
        for cutoff in cutoffs:
            if cutoff == 0:
                filter_ = torch.zeros_like(time)
            else:
                filter_ = 2 * cutoff * window * sinc(2 * cutoff * math.pi * time)
                # Normalize filter to have sum = 1, otherwise we will have a small leakage
                # of the constant component in the input signal.
                filter_ /= filter_.sum()
            filters.append(filter_)
        self.register_buffer("filters", torch.stack(filters)[:, None])

    def forward(self, input):
        shape = list(input.shape)
        input = input.view(-1, 1, shape[-1])
        if self.pad:
            input = F.pad(input, (self.half_size, self.half_size), mode='replicate')
        if self.fft:
            out = fft_conv1d(input, self.filters, stride=self.stride)
        else:
            out = F.conv1d(input, self.filters, stride=self.stride)
        shape.insert(0, len(self.cutoffs))
        shape[-1] = out.shape[-1]
        return out.permute(1, 0, 2).reshape(shape)

    def __repr__(self):
        return simple_repr(self)


class LowPassFilter(torch.nn.Module):
    """
    Same as `LowPassFilters` but applies a single low pass filter.

    Shape:

        - Input: `[*, T]`
        - Output: `[*, T']`, with `T'=T` if `pad` is True and `stride` is 1.

    >>> lowpass = LowPassFilter(1/4, stride=2)
    >>> x = torch.randn(4, 124)
    >>> list(lowpass(x).shape)
    [4, 62]
    """

    def __init__(self, cutoff: float, stride: int = 1, pad: bool = True,
                 zeros: float = 8, fft: Optional[bool] = None):
        super().__init__()
        self._lowpasses = LowPassFilters([cutoff], stride, pad, zeros, fft)

    @property
    def cutoff(self):
        return self._lowpasses.cutoffs[0]

    @property
    def stride(self):
        return self._lowpasses.stride

    @property
    def pad(self):
        return self._lowpasses.pad

    @property
    def zeros(self):
        return self._lowpasses.zeros

    @property
    def fft(self):
        return self._lowpasses.fft

    def forward(self, input):
        return self._lowpasses(input)[0]

    def __repr__(self):
        return simple_repr(self)


def lowpass_filters(input: torch.Tensor,  cutoffs: Sequence[float],
                    stride: int = 1, pad: bool = True,
                    zeros: float = 8, fft: Optional[bool] = None):
    """
    Functional version of `LowPassFilters`, refer to this class for more information.
    """
    return LowPassFilters(cutoffs, stride, pad, zeros, fft).to(input)(input)


def lowpass_filter(input: torch.Tensor,  cutoff: float,
                   stride: int = 1, pad: bool = True,
                   zeros: float = 8, fft: Optional[bool] = None):
    """
    Same as `lowpass_filters` but with a single cutoff frequency.
    Output will not have a dimension inserted in the front.
    """
    return lowpass_filters(input, [cutoff], stride, pad, zeros, fft)[0]
