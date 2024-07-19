import math
import torch
from typing import Optional

from rvc.lib.algorithm.commons import sequence_mask
from rvc.lib.algorithm.modules import WaveNet
from rvc.lib.algorithm.normalization import LayerNorm
from rvc.lib.algorithm.attentions import FFN, FFNV2, MultiHeadAttention


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
        vocoder_type="hifigan",
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
        self.vocoder_type = vocoder_type
        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if vocoder_type == "hifigan":
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
            elif vocoder_type == "bigvgan":
                self.attn_layers.append(
                    torch.nn.MultiheadAttention(
                        hidden_channels, n_heads, dropout=p_dropout, batch_first=True
                    )
                )
                self.norm_layers_1.append(torch.nn.LayerNorm(hidden_channels))
            
            if vocoder_type == "hifigan":
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
            elif vocoder_type == "bigvgan":
                self.ffn_layers.append(
                    FFNV2(
                        hidden_channels,
                        hidden_channels,
                        filter_channels,
                        kernel_size,
                        p_dropout=p_dropout,
                    )
                )
                self.norm_layers_2.append(torch.nn.LayerNorm(hidden_channels))

            

    def forward(self, x, x_mask):
        if self.vocoder_type == "hifigan":
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

        elif self.vocoder_type == "bigvgan":
            attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(2)
            attn_mask = attn_mask[0]
            attn_mask = attn_mask == 0
            x = x * x_mask.unsqueeze(-1)
            for attn_layer, norm_layer_1, ffn_layer, norm_layer_2 in zip(
                self.attn_layers,
                self.norm_layers_1,
                self.ffn_layers,
                self.norm_layers_2,
            ):
                y, _ = attn_layer(x, x, x, attn_mask=attn_mask)
                y = self.drop(y)
                x = norm_layer_1(x + y)
                y = ffn_layer(x, x_mask)
                y = self.drop(y)
                x = norm_layer_2(x + y)
            x = x * x_mask.unsqueeze(-1)

        return x


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        embedding_dim,
        vocoder_type="hifigan",
        f0=True,
    ):
        super(TextEncoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.vocoder_type = vocoder_type
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.encoder = Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            vocoder_type=vocoder_type,
        )
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)
        if f0:
            self.emb_pitch = torch.nn.Embedding(256, hidden_channels)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor, lengths: torch.Tensor):
        if self.vocoder_type == "hifigan":
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

        elif self.vocoder_type == "bigvgan":
            if pitch is None:
                x = self.emb_phone(phone)
            else:
                x = self.emb_phone(phone) + self.emb_pitch(pitch)

            x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
            x = self.lrelu(x)
            x_mask = sequence_mask(lengths, x.size(1)).to(x.dtype)
            x = self.encoder(x, x_mask)
            x_mask = x_mask.unsqueeze(-1)
            stats = self.proj(x.transpose(1, 2)).transpose(1, 2) * x_mask
            stats = stats.transpose(1, 2)
            x_mask = x_mask.transpose(1, 2)
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
