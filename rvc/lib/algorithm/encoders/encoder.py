import os
import sys
import torch

sys.path.append(os.getcwd())

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
            elif vocoder_type in ["bigvgan", "bigvsan"]:
                self.attn_layers.append(
                    torch.nn.MultiheadAttention(
                        hidden_channels, n_heads, dropout=p_dropout, batch_first=True
                    )
                )
                self.norm_layers_1.append(torch.nn.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    vocoder_type=vocoder_type
                    )
                )
            if vocoder_type == "hifigan":
                self.norm_layers_2.append(LayerNorm(hidden_channels))
            elif vocoder_type in ["bigvgan", "bigvsan"]:
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

        elif self.vocoder_type in ["bigvgan", "bigvsan"]:
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
