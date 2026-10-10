from functools import lru_cache
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# PyTorch version check
TORCH_VERSION = tuple(map(int, torch.__version__.split(".")[:2]))
IS_TORCH_LT_2_5 = TORCH_VERSION < (2, 5)
IS_TORCH_LT_2_0 = TORCH_VERSION < (2, 0)

# Flex attention available only in PyTorch >= 2.5
if not IS_TORCH_LT_2_5:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
        _mask_mod_signature,
    )
else:
    # Dummy implementations for older PyTorch
    flex_attention = None
    create_block_mask = None
    _mask_mod_signature = None


def manual_sliding_window_attention(
    q,
    k,
    v,
    window_size: int,
    num_sink_tokens: int = 0,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
):
    """
    Manual sliding window attention implementation for PyTorch 1.13
    This emulates the behavior of flex attention with sliding window + sink tokens
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    if scale is None:
        scale = head_dim**-0.5

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Create sliding window mask
    half_window = window_size // 2

    # Initialize mask (False = attend, True = mask out)
    mask = torch.zeros(seq_len, seq_len, device=q.device, dtype=torch.bool)

    # Apply sliding window: mask out positions outside the window
    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        mask[i, :start] = True
        mask[i, end:] = True

    # Add sink tokens (always attend to first num_sink_tokens)
    if num_sink_tokens > 0:
        # Don't mask out sink tokens in queries
        mask[:, :num_sink_tokens] = False
        # Don't mask out when query is sink token
        mask[:num_sink_tokens, :] = False

    # Apply mask
    scores = scores.masked_fill(mask, float("-inf"))

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # Dropout
    if dropout_p > 0 and q.requires_grad:
        attn = F.dropout(attn, p=dropout_p, training=True)

    # Apply attention to values
    out = torch.matmul(attn, v)

    return out


def manual_causal_sliding_window_attention(
    q,
    k,
    v,
    window_size: int,
    num_sink_tokens: int = 0,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
):
    """
    Manual causal sliding window attention (for autoregressive models)
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    if scale is None:
        scale = head_dim**-0.5

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Create causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device), diagonal=1
    ).bool()

    # Create sliding window mask
    half_window = window_size // 2

    # Initialize mask with causal
    mask = causal_mask.clone()

    # Override with sliding window (allow attending to tokens within window)
    for i in range(seq_len):
        start = max(0, i - half_window)
        # For causal, we can only attend to past tokens
        mask[i, start : i + 1] = False

    # Add sink tokens
    if num_sink_tokens > 0:
        # Always attend to sink tokens (if they are in the past for causal)
        mask[:, : min(num_sink_tokens, seq_len)] = False
        mask[:num_sink_tokens, :] = False

    # Apply mask
    scores = scores.masked_fill(mask, float("-inf"))

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # Dropout
    if dropout_p > 0 and q.requires_grad:
        attn = F.dropout(attn, p=dropout_p, training=True)

    # Apply attention to values
    out = torch.matmul(attn, v)

    return out


@lru_cache(maxsize=128)
def create_block_mask_cached(
    mask_mod,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    device: torch.device,
):
    """Create block mask - for PyTorch < 2.5 returns None"""
    if not IS_TORCH_LT_2_5:
        block_mask = create_block_mask(
            mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
        )
        return block_mask

    # For older PyTorch, return None and use manual implementation
    return None


def get_compiled_flex_attention(compile: bool = True, mode: str = "default"):
    """Return None for PyTorch < 2.5 as flex attention is not available"""
    if IS_TORCH_LT_2_5:
        return None

    if compile:
        return torch.compile(flex_attention, dynamic=False, mode=mode)
    return flex_attention


def generate_sliding_window_with_sinks(window_size: int, num_sink_tokens: int):
    """Generate a mask function for sliding window with sink tokens"""

    def sliding_window_with_global_sinks(b, h, q_idx, kv_idx):
        # For PyTorch >= 2.5, this is used by flex_attention
        # For older versions, we'll use manual implementation
        if not IS_TORCH_LT_2_5:
            half_window = window_size // 2
            is_query_sink = q_idx < num_sink_tokens
            is_kv_sink = kv_idx < num_sink_tokens
            is_in_window = torch.abs(q_idx - kv_idx) <= half_window
            return is_query_sink | is_kv_sink | is_in_window
        return True

    sliding_window_with_global_sinks.__name__ = (
        f"sliding_window_w{window_size}_sinks{num_sink_tokens}"
    )
    return sliding_window_with_global_sinks


class FlexAttention(nn.Module):
    """
    FlexAttention implementation with automatic fallback to manual attention
    for PyTorch versions < 2.5
    """

    def __init__(
        self,
        mask_mod,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        compile: bool = False,
        compile_mode: str = "max-autotune",
        causal: bool = False,
    ):
        super().__init__()

        self.mask_mod = mask_mod
        self.scale = scale
        self.dropout = dropout
        self.causal = causal
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Parse window size and sink tokens from mask_mod name
        self.window_size = 128  # Default
        self.num_sink_tokens = 0

        if mask_mod is not None and hasattr(mask_mod, "__name__"):
            name = mask_mod.__name__
            match_w = re.search(r"w(\d+)", name)
            match_s = re.search(r"sinks(\d+)", name)
            if match_w:
                self.window_size = int(match_w.group(1))
            if match_s:
                self.num_sink_tokens = int(match_s.group(1))

        # For PyTorch >= 2.5, use native flex attention
        if not IS_TORCH_LT_2_5:
            self.flex_attn_fn = get_compiled_flex_attention(compile, mode=compile_mode)
        else:
            self.flex_attn_fn = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, num_heads, seq_len, head_dim = q.shape

        # Use native flex attention for PyTorch >= 2.5
        if not IS_TORCH_LT_2_5 and self.flex_attn_fn is not None:
            block_mask = create_block_mask_cached(
                self.mask_mod,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=q.device.type,
            )

            if self.scale is not None:
                default_scale = head_dim**-0.5
                q = q * (self.scale / default_scale)

            out = self.flex_attn_fn(q, k, v, block_mask=block_mask, scale=self.scale)

            if self.training and self.attn_dropout is not None:
                out = self.attn_dropout(out)

            return out

        # Fallback to manual implementation for PyTorch < 2.5
        if self.causal:
            out = manual_causal_sliding_window_attention(
                q,
                k,
                v,
                window_size=self.window_size,
                num_sink_tokens=self.num_sink_tokens,
                scale=self.scale,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            out = manual_sliding_window_attention(
                q,
                k,
                v,
                window_size=self.window_size,
                num_sink_tokens=self.num_sink_tokens,
                scale=self.scale,
                dropout_p=self.dropout if self.training else 0.0,
            )

        if self.training and self.attn_dropout is not None:
            out = self.attn_dropout(out)

        return out
