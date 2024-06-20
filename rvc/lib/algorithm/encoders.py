import math
import torch
from typing import Optional

from rvc.lib.algorithm.commons import sequence_mask
from rvc.lib.algorithm.modules import WaveNet
from rvc.lib.algorithm.normalization import LayerNorm
from rvc.lib.algorithm.attentions import FFN, MultiHeadAttention


class Encoder(torch.nn.Module):
    """
    Encoder module for the Transformer model.

    Args:
        hidden_channels (int): Number of hidden channels in the encoder.
        filter_channels (int): Number of filter channels in the feed-forward network.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        kernel_size (int, optional): Kernel size of the convolution layers in the feed-forward network. Defaults to 1.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        window_size (int, optional): Window size for relative positional encoding. Defaults to 10.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_channels, time_steps).
        x_mask (torch.Tensor): Mask tensor of shape (batch_size, time_steps), indicating valid time steps.

    Returns:
        torch.Tensor: Encoded tensor of shape (batch_size, hidden_channels, time_steps).
    """

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=10,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TextEncoderV1(torch.nn.Module):
    """Text Encoder with 256 embedding dimension.

    Args:
        out_channels (int): Output channels of the encoder.
        hidden_channels (int): Hidden channels of the encoder.
        filter_channels (int): Filter channels of the encoder.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        kernel_size (int): Kernel size of the convolutional layers.
        p_dropout (float): Dropout probability.
        f0 (bool, optional): Whether to use F0 embedding. Defaults to True.

    Inputs:
        phone (torch.Tensor): Phoneme tensor with shape (batch_size, length, 256).
        pitch (torch.Tensor, optional): Pitch tensor with shape (batch_size, length). Defaults to None.
        lengths (torch.Tensor): Lengths of the input sequences.

    Outputs:
        m (torch.Tensor): Mean tensor with shape (batch_size, out_channels, length).
        logs (torch.Tensor): Log variance tensor with shape (batch_size, out_channels, length).
        x_mask (torch.Tensor): Mask tensor with shape (batch_size, 1, length).
    """

    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        f0=True,
    ):
        super(TextEncoderV1, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = torch.nn.Linear(256, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = torch.nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, phone: torch.Tensor, pitch: Optional[torch.Tensor], lengths: torch.Tensor
    ):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class TextEncoderV2(torch.nn.Module):
    """Text Encoder with 768 embedding dimension.

    Args:
        out_channels (int): Output channels of the encoder.
        hidden_channels (int): Hidden channels of the encoder.
        filter_channels (int): Filter channels of the encoder.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        kernel_size (int): Kernel size of the convolutional layers.
        p_dropout (float): Dropout probability.
        f0 (bool, optional): Whether to use F0 embedding. Defaults to True.

    Inputs:
        phone (torch.Tensor): Phoneme tensor with shape (batch_size, length, 768).
        pitch (torch.Tensor, optional): Pitch tensor with shape (batch_size, length). Defaults to None.
        lengths (torch.Tensor): Lengths of the input sequences.

    Outputs:
        m (torch.Tensor): Mean tensor with shape (batch_size, out_channels, length).
        logs (torch.Tensor): Log variance tensor with shape (batch_size, out_channels, length).
        x_mask (torch.Tensor): Mask tensor with shape (batch_size, 1, length).
    """

    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        f0=True,
    ):
        super(TextEncoderV2, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = torch.nn.Linear(768, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = torch.nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor, lengths: torch.Tensor):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class PosteriorEncoder(torch.nn.Module):
    """Posterior Encoder for inferring latent representation.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        hidden_channels (int): Number of hidden channels in the encoder.
        kernel_size (int): Kernel size of the convolutional layers.
        dilation_rate (int): Dilation rate of the convolutional layers.
        n_layers (int): Number of layers in the encoder.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 0.

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, in_channels, length).
        x_lengths (torch.Tensor): Lengths of the input sequences.
        g (torch.Tensor, optional): Global conditioning input with shape (batch_size, gin_channels, 1). Defaults to None.

    Outputs:
        z (torch.Tensor): Latent representation with shape (batch_size, out_channels, length).
        m (torch.Tensor): Mean tensor with shape (batch_size, out_channels, length).
        logs (torch.Tensor): Log variance tensor with shape (batch_size, out_channels, length).
        x_mask (torch.Tensor): Mask tensor with shape (batch_size, 1, length).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super(PosteriorEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        """Removes weight normalization from the encoder."""
        self.enc.remove_weight_norm()

    def __prepare_scriptable__(self):
        """Prepares the module for scripting."""
        for hook in self.enc._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.enc)
        return self
