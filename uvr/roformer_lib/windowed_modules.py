import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, pack, unpack
from typing import Tuple
from functools import wraps
from packaging import version
from collections import namedtuple
import os

# PyTorch version check
TORCH_VERSION = tuple(map(int, torch.__version__.split(".")[:2]))
IS_TORCH_LT_2_5 = TORCH_VERSION < (2, 5)
IS_TORCH_LT_2_0 = TORCH_VERSION < (2, 0)

# Conditional import for flex attention
if not IS_TORCH_LT_2_5:
    from .flex_attention_utils import (
        FlexAttention,
        generate_sliding_window_with_sinks,
    )
else:
    FlexAttention = None
    generate_sliding_window_with_sinks = None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Manual SDPA for PyTorch < 2.0
def manual_scaled_dot_product_attention(
    q, k, v, attn_mask=None, is_causal=False, scale=None, dropout_p=0.0
):
    """SDPA implementation for PyTorch < 2.0"""
    if scale is None:
        scale = q.shape[-1] ** -0.5

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        seq_len = attn_weights.shape[-1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn_weights.device), diagonal=1
        ).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attn_mask

    attn_weights = torch.softmax(attn_weights, dim=-1)

    if dropout_p > 0.0 and q.requires_grad:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)

    return torch.matmul(attn_weights, v)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
        flash=True,
        wsa_window_len=None,
        n_wsa_sinks=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(
            flash=flash and not IS_TORCH_LT_2_0,  # Disable flash for old PyTorch
            dropout=dropout,
            wsa_window_len=wsa_window_len,
            n_wsa_sinks=n_wsa_sinks,
        )
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x, return_attn=False):
        x = self.norm(x)

        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        if return_attn:
            out, attn = self.attend(q, k, v, return_attn=True)
        else:
            out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")
        result = self.to_out(out)

        if return_attn:
            return result, attn
        return result


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth: int = 1,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        use_flash=True,
        wsa_window_len=None,
        n_wsa_sinks=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                rotary_embed=rotary_embed,
                flash=use_flash,
                wsa_window_len=wsa_window_len,
                n_wsa_sinks=n_wsa_sinks,
            )

            self.layers.append(
                ModuleList(
                    [attn, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(Module):
    def __init__(self, dim: int, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = dim_hidden if dim_hidden is not None else dim_in

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1)
            )
            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        flash=False,
        scale=None,
        wsa_window_len=None,
        n_wsa_sinks=None,
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.wsa_window_len = wsa_window_len
        self.n_wsa_sinks = n_wsa_sinks
        self.use_flash = flash

        # Flex attention only for PyTorch >= 2.5
        if wsa_window_len is not None and n_wsa_sinks is not None and n_wsa_sinks > 0:
            if IS_TORCH_LT_2_5:
                print_once(
                    f"Warning: WSA (windowed sliding attention) requires PyTorch >= 2.5.0, got {torch.__version__}. "
                    "Disabling WSA and falling back to standard attention."
                )
                self.flex_attn = None
                self.wsa_window_len = None
                self.n_wsa_sinks = None
            else:
                assert not (
                    version.parse(torch.__version__) < version.parse("2.5.0")
                ), "in order to use flex attention, you must be using pytorch 2.5 or above"
                mask_mod = generate_sliding_window_with_sinks(
                    wsa_window_len, n_wsa_sinks
                )

                self.flex_attn = FlexAttention(
                    mask_mod=mask_mod,
                    dropout=dropout,
                    scale=scale,
                    compile=True,
                )
        else:
            self.flex_attn = None

            # Flash attention warning for old PyTorch
            if self.use_flash and IS_TORCH_LT_2_0:
                print_once(
                    f"Warning: Flash attention requires PyTorch >= 2.0.0, got {torch.__version__}. "
                    "Falling back to standard attention."
                )
                self.use_flash = False

            self.cpu_config = FlashAttentionConfig(True, True, True)
            self.cuda_config = None

            if not torch.cuda.is_available() or not self.use_flash:
                return

            device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
            device_version = version.parse(
                f"{device_properties.major}.{device_properties.minor}"
            )

            if device_version >= version.parse("8.0"):
                if os.name == "nt":
                    print_once(
                        "Windows OS detected, using math or mem efficient attention if input tensor is on cuda"
                    )
                    self.cuda_config = FlashAttentionConfig(False, True, True)
                else:
                    print_once(
                        "GPU Compute Capability equal or above 8.0, using flash attention if input tensor is on cuda"
                    )
                    self.cuda_config = FlashAttentionConfig(True, False, False)
            else:
                print_once(
                    "GPU Compute Capability below 8.0, using math or mem efficient attention if input tensor is on cuda"
                )
                self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        if self.scale is not None:
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        config = self.cuda_config if is_cuda else self.cpu_config

        # For PyTorch < 2.0, use manual attention
        if IS_TORCH_LT_2_0:
            return manual_scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        return out

    def forward(self, q, k, v, return_attn=False):
        # Flex attention path
        if self.flex_attn is not None:
            return self.flex_attn(q, k, v)

        # Flash attention path (PyTorch >= 2.0)
        if self.use_flash and not IS_TORCH_LT_2_0:
            return self.flash_attn(q, k, v)

        # Manual attention path (PyTorch < 2.0 or fallback)
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5

        sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)

        if return_attn:
            return out, attn
        return out
