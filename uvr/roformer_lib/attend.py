from functools import wraps
from packaging import version
from collections import namedtuple

import os
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


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
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        self.use_torch_2_sdpa = False
        self._config_checked = False

        # Проверяем версию PyTorch при первом вызове
        if flash and not self._config_checked:
            if version.parse(torch.__version__) >= version.parse("2.0.0"):
                print_once("PyTorch >= 2.0 detected, will use SDPA if available.")
                self.use_torch_2_sdpa = True

                # Настройки для PyTorch >= 2.0
                self.cpu_config = FlashAttentionConfig(True, True, True)
                self.cuda_config = None

                if torch.cuda.is_available():
                    device_properties = torch.cuda.get_device_properties(
                        torch.device("cuda")
                    )
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
            else:
                print_once(
                    "PyTorch < 2.0 detected, flash attention will use einsum fallback."
                )
                self.use_torch_2_sdpa = False

            self._config_checked = True

    def flash_attn_torch2(self, q, k, v):
        """SDPA для PyTorch >= 2.0"""
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        is_cuda = q.is_cuda
        config = self.cuda_config if is_cuda else self.cpu_config

        old_sdp_kernel = False
        if hasattr(torch, "backends"):
            if hasattr(torch.backends, "cuda"):
                if hasattr(torch.backends.cuda, "sdp_kernel"):
                    old_sdp_kernel = True
        new_sdp_kernel = False
        if hasattr(torch, "nn"):
            if hasattr(torch.nn, "attention"):
                if hasattr(torch.nn.attention, "sdpa_kernel") and hasattr(
                    torch.nn.attention, "SDPBackend"
                ):
                    new_sdp_kernel = True

        if old_sdp_kernel and not new_sdp_kernel:
            with torch.backends.cuda.sdp_kernel(**config._asdict()):
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0
                )
        else:
            backends = []
            if config.enable_flash:
                backends.append(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
            if config.enable_mem_efficient:
                backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
            if config.enable_math:
                backends.append(torch.nn.attention.SDPBackend.MATH)
            with torch.nn.attention.sdpa_kernel(backends):
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0
                )
        return out

    def forward(self, q, k, v):
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash and self.use_torch_2_sdpa:
            try:
                return self.flash_attn_torch2(q, k, v)
            except Exception as e:
                print(f"Flash attention failed: {e}. Falling back to einsum.")
                self.use_torch_2_sdpa = False

        # Fallback для PyTorch < 2.0 или если flash отключен
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
