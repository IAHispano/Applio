import math
import torch


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention module with optional relative positional encoding and proximal bias.

    Args:
        channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_heads (int): Number of attention heads.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        window_size (int, optional): Window size for relative positional encoding. Defaults to None.
        heads_share (bool, optional): Whether to share relative positional embeddings across heads. Defaults to True.
        block_length (int, optional): Block length for local attention. Defaults to None.
        proximal_bias (bool, optional): Whether to use proximal bias in self-attention. Defaults to False.
        proximal_init (bool, optional): Whether to initialize the key projection weights the same as query projection weights. Defaults to False.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int = None,
        heads_share: bool = True,
        block_length: int = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert (
            channels % n_heads == 0
        ), "Channels must be divisible by the number of heads."

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias

        # Define projections
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)

        self.drop = torch.nn.Dropout(p_dropout)

        # Relative positional encodings
        if window_size:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = torch.nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels)
                * rel_stddev
            )

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        torch.nn.init.xavier_uniform_(self.conv_o.weight)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        # Compute query, key, value projections
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)

        # Compute attention
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        # Final output projection
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        # Reshape and compute scaled dot-product attention
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.window_size:
            assert t_s == t_t, "Relative attention only supports self-attention."
            scores += self._compute_relative_scores(query, t_s)

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias only supports self-attention."
            scores += self._attention_bias_proximal(t_s).to(scores.device, scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length:
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)

        # Apply softmax and dropout
        p_attn = self.drop(torch.nn.functional.softmax(scores, dim=-1))

        # Compute attention output
        output = torch.matmul(p_attn, value)

        if self.window_size:
            output += self._apply_relative_values(p_attn, t_s)

        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn

    def _compute_relative_scores(self, query, length):
        rel_emb = self._get_relative_embeddings(self.emb_rel_k, length)
        rel_logits = self._matmul_with_relative_keys(
            query / math.sqrt(self.k_channels), rel_emb
        )
        return self._relative_position_to_absolute_position(rel_logits)

    def _apply_relative_values(self, p_attn, length):
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        rel_emb = self._get_relative_embeddings(self.emb_rel_v, length)
        return self._matmul_with_relative_values(rel_weights, rel_emb)

    # Helper methods
    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, embeddings, length):
        pad_length = (length - (self.window_size + 1)).clamp(min=0)
        start = ((self.window_size + 1) - length).clamp(min=0)
        end = start + 2 * length - 1

        # Padding via concatenation to workaround ONNX padding limitations
        x = torch.cat(
            [
                torch.zeros(
                    embeddings.size(0),
                    pad_length,
                    embeddings.size(2),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                ),
                embeddings,
            ],
            dim=1,
        )

        embeddings = torch.cat(
            [
                x,
                torch.zeros(
                    x.size(0), pad_length, x.size(2), device=x.device, dtype=x.dtype
                ),
            ],
            dim=1,
        )

        return embeddings[:, start:end]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        # Reshaping trick via manual padding to perform coordinate shifting
        x = self.padding(x, [0, 1])
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = self.padding(x_flat, [0, length - 1])

        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = self.padding(x, [0, length - 1])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = self.padding(x_flat, [length, 0])

        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = r.unsqueeze(0) - r.unsqueeze(1)
        return -torch.log1p(torch.abs(diff)).unsqueeze(0).unsqueeze(0)

    def padding(self, x, pad):
        """Alternative zero padding function using concatenation to ensure ONNX support."""

        left, right = pad[-2], pad[-1]
        shape = list(x.shape[:-1])

        # Concatenate zeros to the edges of the last dimension
        x = torch.cat(
            [torch.zeros(*shape, left, device=x.device, dtype=x.dtype), x], dim=-1
        )
        x = torch.cat(
            [x, torch.zeros(*shape, right, device=x.device, dtype=x.dtype)], dim=-1
        )

        return x


class FFN(torch.nn.Module):
    """
    Feed-forward network module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        filter_channels (int): Number of filter channels in the convolution layers.
        kernel_size (int): Kernel size of the convolution layers.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        activation (str, optional): Activation function to use. Defaults to None.
        causal (bool, optional): Whether to use causal padding in the convolution layers. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str = None,
        causal: bool = False,
    ):
        super().__init__()
        self.padding_fn = self._causal_padding if causal else self._same_padding

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)

        self.activation = activation
        self.kernel_size = kernel_size

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding_fn(x * x_mask))
        x = self._apply_activation(x)
        x = self.drop(x)
        x = self.conv_2(self.padding_fn(x * x_mask))
        return x * x_mask

    def _apply_activation(self, x):
        if self.activation == "gelu":
            return x * torch.sigmoid(1.702 * x)
        return torch.relu(x)

    def _causal_padding(self, x):
        return torch.nn.functional.pad(x, [self.kernel_size, 0, 0, 0, 0, 0])

    def _same_padding(self, x):
        return torch.nn.functional.pad(
            x,
            [(self.kernel_size - 1) // 2, self.kernel_size // 2, 0, 0, 0, 0],
        )


class LayerNorm(torch.nn.Module):
    """
    Layer normalization module.

    Args:
        channels (int): Number of channels.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # Transpose to (batch_size, time_steps, channels) for layer_norm
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(
            x, (self.channels,), self.gamma, self.beta, self.eps
        )
        # Transpose back to (batch_size, channels, time_steps)
        return x.transpose(1, -1)


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
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = torch.nn.Dropout(p_dropout)

        self.attn_layers = torch.nn.ModuleList(
            [
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_1 = torch.nn.ModuleList(
            [LayerNorm(hidden_channels) for _ in range(n_layers)]
        )
        self.ffn_layers = torch.nn.ModuleList(
            [
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_2 = torch.nn.ModuleList(
            [LayerNorm(hidden_channels) for _ in range(n_layers)]
        )

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

        return x * x_mask
