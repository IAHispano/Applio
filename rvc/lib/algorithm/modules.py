import math

import torch
from torch import nn
from torch.nn import functional as F


from rvc.lib.algorithm.transforms import piecewise_rational_quadratic_transform
from rvc.lib.algorithm.commons import fused_add_tanh_sigmoid_multiply


class LayerNorm(nn.Module):
    """Layer normalization module.

    Args:
        channels (int): Number of channels.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.

    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).

        Returns:
            torch.Tensor: Layer-normalized tensor of shape (batch_size, channels, time_steps).

        """
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    """Convolutional layer with ReLU activation and layer normalization.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        n_layers (int): Number of convolutional layers.
        p_dropout (float): Dropout probability.

    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, time_steps).
            x_mask (torch.Tensor): Mask tensor of shape (batch_size, 1, time_steps).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, time_steps).

        """
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Module):
    """Dilated depth-separable convolution module.

    Args:
        channels (int): Number of channels.
        kernel_size (int): Size of the convolutional kernel.
        n_layers (int): Number of convolutional layers.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.

    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super(DDSConv, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        layers = []
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        groups=channels,
                        dilation=dilation,
                        padding=padding,
                    ),
                    LayerNorm(channels),
                    nn.GELU(),
                    nn.Conv1d(channels, channels, 1),
                    LayerNorm(channels),
                    nn.GELU(),
                    nn.Dropout(p_dropout),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x, x_mask, g=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
            x_mask (torch.Tensor): Mask tensor of shape (batch_size, 1, time_steps).
            g (torch.Tensor, optional): Conditioning tensor of shape (batch_size, gin_channels, time_steps).
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, time_steps).

        """
        if g is not None:
            x = x + g

        for layer in self.layers:
            y = layer(x * x_mask)
            x = x + y

        return x * x_mask


class WN(torch.nn.Module):
    """Weight-normalized convolution module.

    Args:
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation_rate (int): Dilation rate of the convolution.
        n_layers (int): Number of convolutional layers.
        gin_channels (int, optional): Number of conditioning channels. Defaults to 0.
        p_dropout (float, optional): Dropout probability. Defaults to 0.

    """

    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(
                cond_layer, name="weight"
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.parametrizations.weight_norm(
                in_layer, name="weight"
            )
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.parametrizations.weight_norm(
                res_skip_layer, name="weight"
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_channels, time_steps).
            x_mask (torch.Tensor): Mask tensor of shape (batch_size, 1, time_steps).
            g (torch.Tensor, optional): Conditioning tensor of shape (batch_size, gin_channels, time_steps).
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_channels, time_steps).

        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        """Remove weight normalization from the module."""
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class ConvFlow(nn.Module):
    """Convolutional flow layer for flow-based models.

    Args:
        in_channels (int): Number of input channels.
        filter_channels (int): Number of filter channels.
        kernel_size (int): Size of the convolutional kernel.
        n_layers (int): Number of convolutional layers.
        num_bins (int, optional): Number of bins for the piecewise rational quadratic transform.
            Defaults to 10.
        tail_bound (float, optional): Tail bound for the piecewise rational quadratic transform.
            Defaults to 5.0.

    """

    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        tail_bound=5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, time_steps).
            x_mask (torch.Tensor): Mask tensor of shape (batch_size, 1, time_steps).
            g (torch.Tensor, optional): Conditioning tensor of shape (batch_size, gin_channels, time_steps).
                Defaults to None.
            reverse (bool, optional): Whether to reverse the operation. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape (batch_size, in_channels, time_steps).
                - torch.Tensor: Log determinant tensor.

        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x
