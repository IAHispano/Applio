# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2021
"""
FIR windowed sinc highpass and bandpass filters.
Those are convenience wrappers around the filters defined in `julius.lowpass`.
"""

from typing import Sequence, Optional

import torch

# Import all lowpass filters for consistency.
from .lowpass import lowpass_filter, lowpass_filters, LowPassFilter,  LowPassFilters  # noqa
from .utils import simple_repr


class HighPassFilters(torch.nn.Module):
    """
    Bank of high pass filters. See `julius.lowpass.LowPassFilters` for more
    details on the implementation.

    Args:
        cutoffs (list[float]): list of cutoff frequencies, in [0, 0.5] expressed as `f/f_s` where
            f_s is the samplerate and `f` is the cutoff frequency.
            The upper limit is 0.5, because a signal sampled at `f_s` contains only
            frequencies under `f_s / 2`.
        stride (int): how much to decimate the output. Probably not a good idea
            to do so with a high pass filters though...
        pad (bool): if True, appropriately pad the input with zero over the edge. If `stride=1`,
            the output will have the same length as the input.
        zeros (float): Number of zero crossings to keep.
            Controls the receptive field of the Finite Impulse Response filter.
            For filters with low cutoff frequency, e.g. 40Hz at 44.1kHz,
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

    Shape:

        - Input: `[*, T]`
        - Output: `[F, *, T']`, with `T'=T` if `pad` is True and `stride` is 1, and
            `F` is the numer of cutoff frequencies.

    >>> highpass = HighPassFilters([1/4])
    >>> x = torch.randn(4, 12, 21, 1024)
    >>> list(highpass(x).shape)
    [1, 4, 12, 21, 1024]
    """

    def __init__(self, cutoffs: Sequence[float], stride: int = 1, pad: bool = True,
                 zeros: float = 8, fft: Optional[bool] = None):
        super().__init__()
        self._lowpasses = LowPassFilters(cutoffs, stride, pad, zeros, fft)

    @property
    def cutoffs(self):
        return self._lowpasses.cutoffs

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
        lows = self._lowpasses(input)

        # We need to extract the right portion of the input in case
        # pad is False or stride > 1
        if self.pad:
            start, end = 0, input.shape[-1]
        else:
            start = self._lowpasses.half_size
            end = -start
        input = input[..., start:end:self.stride]
        highs = input - lows
        return highs

    def __repr__(self):
        return simple_repr(self)


class HighPassFilter(torch.nn.Module):
    """
    Same as `HighPassFilters` but applies a single high pass filter.

    Shape:

        - Input: `[*, T]`
        - Output: `[*, T']`, with `T'=T` if `pad` is True and `stride` is 1.

    >>> highpass = HighPassFilter(1/4, stride=1)
    >>> x = torch.randn(4, 124)
    >>> list(highpass(x).shape)
    [4, 124]
    """

    def __init__(self, cutoff: float, stride: int = 1, pad: bool = True,
                 zeros: float = 8, fft: Optional[bool] = None):
        super().__init__()
        self._highpasses = HighPassFilters([cutoff], stride, pad, zeros, fft)

    @property
    def cutoff(self):
        return self._highpasses.cutoffs[0]

    @property
    def stride(self):
        return self._highpasses.stride

    @property
    def pad(self):
        return self._highpasses.pad

    @property
    def zeros(self):
        return self._highpasses.zeros

    @property
    def fft(self):
        return self._highpasses.fft

    def forward(self, input):
        return self._highpasses(input)[0]

    def __repr__(self):
        return simple_repr(self)


def highpass_filters(input: torch.Tensor,  cutoffs: Sequence[float],
                     stride: int = 1, pad: bool = True,
                     zeros: float = 8, fft: Optional[bool] = None):
    """
    Functional version of `HighPassFilters`, refer to this class for more information.
    """
    return HighPassFilters(cutoffs, stride, pad, zeros, fft).to(input)(input)


def highpass_filter(input: torch.Tensor,  cutoff: float,
                    stride: int = 1, pad: bool = True,
                    zeros: float = 8, fft: Optional[bool] = None):
    """
    Functional version of `HighPassFilter`, refer to this class for more information.
    Output will not have a dimension inserted in the front.
    """
    return highpass_filters(input, [cutoff], stride, pad, zeros, fft)[0]


class BandPassFilter(torch.nn.Module):
    """
    Single band pass filter, implemented as a the difference of two lowpass filters.

    Args:
        cutoff_low (float): lower cutoff frequency, in [0, 0.5] expressed as `f/f_s` where
            f_s is the samplerate and `f` is the cutoff frequency.
            The upper limit is 0.5, because a signal sampled at `f_s` contains only
            frequencies under `f_s / 2`.
        cutoff_high (float): higher cutoff frequency, in [0, 0.5] expressed as `f/f_s`.
            This must be higher than cutoff_high. Note that due to the fact
            that filter are not perfect, the output will be non zero even if
            cutoff_high == cutoff_low.
        stride (int): how much to decimate the output.
        pad (bool): if True, appropriately pad the input with zero over the edge. If `stride=1`,
            the output will have the same length as the input.
        zeros (float): Number of zero crossings to keep.
            Controls the receptive field of the Finite Impulse Response filter.
            For filters with low cutoff frequency, e.g. 40Hz at 44.1kHz,
            it is a bad idea to set this to a high value.
            This is likely appropriate for most use. Lower values
            will result in a faster filter, but with a slower attenuation around the
            cutoff frequency.
        fft (bool or None): if True, uses `julius.fftconv` rather than PyTorch convolutions.
            If False, uses PyTorch convolutions. If None, either one will be chosen automatically
            depending on the effective filter size.


    Shape:

        - Input: `[*, T]`
        - Output: `[*, T']`, with `T'=T` if `pad` is True and `stride` is 1.

    ..Note:: There is no BandPassFilters (bank of bandpasses) because its
        signification would be the same as `julius.bands.SplitBands`.

    >>> bandpass = BandPassFilter(1/4, 1/3)
    >>> x = torch.randn(4, 12, 21, 1024)
    >>> list(bandpass(x).shape)
    [4, 12, 21, 1024]
    """

    def __init__(self, cutoff_low: float, cutoff_high: float, stride: int = 1, pad: bool = True,
                 zeros: float = 8, fft: Optional[bool] = None):
        super().__init__()
        if cutoff_low > cutoff_high:
            raise ValueError(f"Lower cutoff {cutoff_low} should be less than "
                             f"higher cutoff {cutoff_high}.")
        self._lowpasses = LowPassFilters([cutoff_low, cutoff_high], stride, pad, zeros, fft)

    @property
    def cutoff_low(self):
        return self._lowpasses.cutoffs[0]

    @property
    def cutoff_high(self):
        return self._lowpasses.cutoffs[1]

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
        lows = self._lowpasses(input)
        return lows[1] - lows[0]

    def __repr__(self):
        return simple_repr(self)


def bandpass_filter(input: torch.Tensor,  cutoff_low: float, cutoff_high: float,
                    stride: int = 1, pad: bool = True,
                    zeros: float = 8, fft: Optional[bool] = None):
    """
    Functional version of `BandPassfilter`, refer to this class for more information.
    Output will not have a dimension inserted in the front.
    """
    return BandPassFilter(cutoff_low, cutoff_high, stride, pad, zeros, fft).to(input)(input)
