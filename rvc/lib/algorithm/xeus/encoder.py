import torch
import torch.nn as nn
import math

from rvc.lib.algorithm.xeus.utils import make_pad_mask

class MultiSequential(nn.Sequential):
    def __init__(self, *args, layer_drop_rate=0.0):
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training or (_probs[idx] >= self.layer_drop_rate):
                args = m(*args)
        return args


def repeat(N, fn, layer_drop_rate=0.0):
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LayerNorm(nn.LayerNorm):
    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )

class ConvolutionalPositionalEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        max_len: int = 5000,
        kernel_size: int = 128,
        groups: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups,)

        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        for hook in self.conv._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                _LG.warning("Removing weight_norm from %s", self.__class__.__name__)
                nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
    ):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate

        # LayerNorm for q and k
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        q = self.q_norm(q)
        k = self.k_norm(k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        # This wastes a lot of GPU memory
        # TO DO: add a flag to check if this should be saved
        self.attn = attn

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class ConvolutionalSpatialGatingUnit(nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(self, size: int, kernel_size: int, dropout_rate: float,):
        super().__init__()
        n_channels = size // 2  # split input channels
        self.norm = LayerNorm(n_channels)
        self.conv = nn.Conv1d(n_channels, n_channels, kernel_size, 1, (kernel_size - 1) // 2, groups=n_channels,)
        self.act = nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_r, x_g = x.chunk(2, dim=-1)
        x_g = torch.utils.checkpoint.checkpoint(self.norm, x_g, use_reentrant=False)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        x_g = torch.utils.checkpoint.checkpoint(self.act, x_g, use_reentrant=False)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        del x_g, x_r
        return out

class ConvolutionalGatingMLP(nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.channel_proj1 = nn.Sequential(nn.Linear(size, linear_units), nn.GELU())
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )
        self.channel_proj2 = nn.Linear(linear_units // 2, size)

    def forward(self, x, mask):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None
        del x

        xs_pad = torch.utils.checkpoint.checkpoint(self.channel_proj1, xs_pad, use_reentrant=False)
        xs_pad = self.csgu(xs_pad)  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)  # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        idim,
        hidden_units,
        dropout_rate,
    ):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation=Swish()

    def forward(self, x):
        x = torch.utils.checkpoint.checkpoint(self.w_1, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.activation, x, use_reentrant=False)

        return self.w_2(self.dropout(x))

class EBranchformerEncoderLayer(nn.Module):

    def __init__(
        self,
        size: int,
        attn: nn.Module,
        cgmlp: nn.Module,
        feed_forward: nn.Module,
        feed_forward_macaron: nn.Module,
        dropout_rate: float,
        merge_conv_kernel: int = 31,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        self.norm_ff = LayerNorm(size)
        self.ff_scale = 0.5
        self.norm_ff_macaron = LayerNorm(size)

        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.norm_mlp = LayerNorm(size)  # for the MLP module
        self.norm_final = LayerNorm(size)  # for the final output of the block

        self.dropout = nn.Dropout(dropout_rate)

        self.depthwise_conv_fusion = nn.Conv1d(
            size + size,
            size + size,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=size + size,
            bias=True,
        )
        self.merge_proj = nn.Linear(size + size, size)


    def forward(self, x_input, mask):

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        residual = x
        x = torch.utils.checkpoint.checkpoint(self.norm_ff_macaron, x, use_reentrant=False)
        x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
        del residual

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = torch.utils.checkpoint.checkpoint(self.norm_mha, x1, use_reentrant=False)

        if pos_emb is not None:
            x_att = self.attn(x1, x1, x1, pos_emb, mask)
        else:
            x_att = self.attn(x1, x1, x1, mask)

        x1 = self.dropout(x_att)
        del x_att

        # Branch 2: convolutional gating mlp
        x2 = torch.utils.checkpoint.checkpoint(self.norm_mlp, x2, use_reentrant=False)

        if pos_emb is not None:
            x2 = (x2, pos_emb)
        x2 = self.cgmlp(x2, mask)
        if isinstance(x2, tuple):
            x2 = x2[0]

        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        del x1, x2

        x_tmp = x_concat.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + self.dropout(self.merge_proj(x_concat + x_tmp))
        del x_tmp, x_concat

        # feed forward module
        residual = x
        x = torch.utils.checkpoint.checkpoint(self.norm_ff, x, use_reentrant=False)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        del residual

        x = torch.utils.checkpoint.checkpoint(self.norm_final, x, use_reentrant=False)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask

class EBranchformerEncoder(nn.Module):
    def __init__(
        self,
        output_size: int = 1024,
        positional_dropout_rate: float = 0.1,
        max_pos_emb_len: int = 5000,
        attention_heads: int = 8,
        attention_dropout_rate: float = 0.1,
        cgmlp_linear_units: int = 4096,
        cgmlp_conv_kernel: int = 31,
        num_blocks: int = 19,
        dropout_rate: float = 0.1,
        layer_drop_rate: float = 0.0,
        linear_units: int = 4096,
        merge_conv_kernel: int = 31,
    ):
        super().__init__()

        self.embed = nn.Sequential(
                ConvolutionalPositionalEmbedding(output_size, positional_dropout_rate, max_pos_emb_len),
                nn.Dropout(dropout_rate),
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EBranchformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size, attention_dropout_rate,),
                ConvolutionalGatingMLP(output_size, cgmlp_linear_units, cgmlp_conv_kernel, dropout_rate,),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate,),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate,),
                dropout_rate,
                merge_conv_kernel,
            ),
            layer_drop_rate,
        )

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad = self.embed(xs_pad)

        for layer_idx, encoder_layer in enumerate(self.encoders):
            xs_pad, masks = encoder_layer(xs_pad, masks)
        return xs_pad
