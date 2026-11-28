import torch
from typing import Optional


def init_weights(m, mean=0.0, std=0.01):
    """
    Initialize the weights of a module.

    Args:
        m: The module to initialize.
        mean: The mean of the normal distribution.
        std: The standard deviation of the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding needed for a convolution.

    Args:
        kernel_size: The size of the kernel.
        dilation: The dilation of the convolution.
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    """
    Convert the pad shape to a list of integers.

    Args:
        pad_shape: The pad shape..
    """
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4, dim: int = 2
):
    """
    Slice segments from a tensor, handling tensors with different numbers of dimensions.

    Args:
        x (torch.Tensor): The tensor to slice.
        ids_str (torch.Tensor): The starting indices of the segments.
        segment_size (int, optional): The size of each segment. Defaults to 4.
        dim (int, optional): The dimension to slice across (2D or 3D tensors). Defaults to 2.
    """
    if dim == 2:
        ret = torch.zeros_like(x[:, :segment_size])
    elif dim == 3:
        ret = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size
        if dim == 2:
            ret[i] = x[i, idx_str:idx_end]
        else:
            ret[i] = x[i, :, idx_str:idx_end]

    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    Randomly slice segments from a tensor.

    Args:
        x: The tensor to slice.
        x_lengths: The lengths of the sequences.
        segment_size: The size of each segment.
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size, dim=3)
    return ret, ids_str


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    Fused add tanh sigmoid multiply operation.

    Args:
        input_a: The first input tensor.
        input_b: The second input tensor.
        n_channels: The number of channels.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    """
    Generate a sequence mask.

    Args:
        length: The lengths of the sequences.
        max_length: The maximum length of the sequences.
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def grad_norm(parameters, norm_type: float = 2.0):
    """
    Calculates norm of parameter gradients

    Args:
        parameters: The list of parameters to clip.
        norm_type: The type of norm to use for clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if not parameters:
        return 0.0

    return torch.linalg.vector_norm(
        torch.stack([p.grad.norm(norm_type) for p in parameters]), ord=norm_type
    ).item()
