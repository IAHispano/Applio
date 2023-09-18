# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020

"""
Implementation of a FFT based 1D convolution in PyTorch.
While FFT is used in CUDNN for small kernel sizes, it is not the case for long ones, e.g. 512.
This module implements efficient FFT based convolutions for such convolutions. A typical
application is for evaluationg FIR filters with a long receptive field, typically
evaluated with a stride of 1.
"""
from typing import Optional

import torch
try:
    import torch.fft as new_fft
except ImportError:
    new_fft = None  # type: ignore
from torch.nn import functional as F

from .core import pad_to, unfold
from .utils import simple_repr


# This is quite verbose, but sadly needed to make TorchScript happy.
def _new_rfft(x: torch.Tensor):
    z = new_fft.rfft(x, dim=-1)
    return torch.view_as_real(z)


def _old_rfft(x: torch.Tensor):
    return torch.rfft(x, 1)  # type: ignore


def _old_irfft(x: torch.Tensor, length: int):
    result = torch.irfft(x, 1, signal_sizes=(length,))  # type: ignore
    return result


def _new_irfft(x: torch.Tensor, length: int):
    x = torch.view_as_complex(x)
    return new_fft.irfft(x, length, dim=-1)


if new_fft is None:
    _rfft = _old_rfft
    _irfft = _old_irfft
else:
    _rfft = _new_rfft
    _irfft = _new_irfft


def _compl_mul_conjugate(a: torch.Tensor, b: torch.Tensor):
    """
    Given a and b two tensors of dimension 4
    with the last dimension being the real and imaginary part,
    returns a multiplied by the conjugate of b, the multiplication
    being with respect to the second dimension.

    """
    # PyTorch 1.7 supports complex number, but not for all operations.
    # Once the support is widespread, this can likely go away.

    op = "bcft,dct->bdft"
    return torch.stack([
        torch.einsum(op, a[..., 0], b[..., 0]) + torch.einsum(op, a[..., 1], b[..., 1]),
        torch.einsum(op, a[..., 1], b[..., 0]) - torch.einsum(op, a[..., 0], b[..., 1])
    ],
                       dim=-1)


def fft_conv1d(
        input: torch.Tensor, weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None, stride: int = 1, padding: int = 0,
        block_ratio: float = 5):
    """
    Same as `torch.nn.functional.conv1d` but using FFT for the convolution.
    Please check PyTorch documentation for more information.

    Args:
        input (Tensor): input signal of shape `[B, C, T]`.
        weight (Tensor): weight of the convolution `[D, C, K]` with `D` the number
            of output channels.
        bias (Tensor or None): if not None, bias term for the convolution.
        stride (int): stride of convolution.
        padding (int): padding to apply to the input.
        block_ratio (float): can be tuned for speed. The input is splitted in chunks
            with a size of `int(block_ratio * kernel_size)`.

    Shape:

        - Inputs: `input` is `[B, C, T]`, `weight` is `[D, C, K]` and bias is `[D]`.
        - Output: `(*, T)`


    ..note::
        This function is faster than `torch.nn.functional.conv1d` only in specific cases.
        Typically, the kernel size should be of the order of 256 to see any real gain,
        for a stride of 1.

    ..Warning::
        Dilation and groups are not supported at the moment. This function might use
        more memory than the default Conv1d implementation.
    """
    input = F.pad(input, (padding, padding))
    batch, channels, length = input.shape
    out_channels, _, kernel_size = weight.shape

    if length < kernel_size:
        raise RuntimeError(f"Input should be at least as large as the kernel size {kernel_size}, "
                           f"but it is only {length} samples long.")
    if block_ratio < 1:
        raise RuntimeError("Block ratio must be greater than 1.")

    # We are going to process the input blocks by blocks, as for some reason it is faster
    # and less memory intensive (I think the culprit is `torch.einsum`.
    block_size: int = min(int(kernel_size * block_ratio), length)
    fold_stride = block_size - kernel_size + 1
    weight = pad_to(weight, block_size)
    weight_z = _rfft(weight)

    # We pad the input and get the different frames, on which
    frames = unfold(input, block_size, fold_stride)

    frames_z = _rfft(frames)
    out_z = _compl_mul_conjugate(frames_z, weight_z)
    out = _irfft(out_z, block_size)
    # The last bit is invalid, because FFT will do a circular convolution.
    out = out[..., :-kernel_size + 1]
    out = out.reshape(batch, out_channels, -1)
    out = out[..., ::stride]
    target_length = (length - kernel_size) // stride + 1
    out = out[..., :target_length]
    if bias is not None:
        out += bias[:, None]
    return out


class FFTConv1d(torch.nn.Module):
    """
    Same as `torch.nn.Conv1d` but based on `fft_conv1d`.
    Please check PyTorch documentation for more information.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (int): kernel size of convolution.
        stride (int): stride of convolution.
        padding (int): padding to apply to the input.
        bias (bool): if True, use a bias term.

    ..note::
        This module is faster than `torch.nn.Conv1d` only in specific cases.
        Typically, `kernel_size` should be of the order of 256 to see any real gain,
        for a stride of 1.

    ..warning::
        Dilation and groups are not supported at the moment. This module might use
        more memory than the default Conv1d implementation.

    >>> fftconv = FFTConv1d(12, 24, 128, 4)
    >>> x = torch.randn(4, 12, 1024)
    >>> print(list(fftconv(x).shape))
    [4, 24, 225]
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        self.weight = conv.weight
        self.bias = conv.bias

    def forward(self, input: torch.Tensor):
        return fft_conv1d(
            input, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return simple_repr(self, overrides={"bias": self.bias is not None})
