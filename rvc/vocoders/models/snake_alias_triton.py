"""Triton kernels for NSF-BigVGAN's ``SnakeAlias``: ``snake_triton``'s fused
op with the original's filters, a ``DTAPS``-tap stride-2 lowpass behind
replicate padding in place of the 65-tap one behind reflect padding.

The upsampler side is ``snake_triton``'s and so are the names; the activated
upsampled signal ``u`` is kept unpadded here, as two phases of T, and the
lowpass reads it clamped.  ``DPAD`` is the lowpass's left padding.
"""

import torch
import triton
import triton.language as tl

from rvc.vocoders.models.snake_triton import BLOCK, _bwd_up, _grid, _preact, _snake_bwd


@triton.jit
def _fwd_up(x_ptr, r_ptr, w_ptr, la_ptr, lb_ptr, u_ptr, xn_ptr, T, C, PAD_L,
            TAPS: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """Upsample and activate: ``x`` (plus ``r``) -> the scratch ``u``.
    With ``HAS_RES`` also writes ``x + r`` to ``xn``."""
    row = tl.program_id(0)
    ch = row % C
    alpha = tl.exp(tl.load(la_ptr + ch).to(tl.float32))
    ib = tl.exp(-tl.load(lb_ptr + ch).to(tl.float32))
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = k < T
    x_row = x_ptr + row.to(tl.int64) * T
    r_row = r_ptr + row.to(tl.int64) * T
    v0, v1 = _preact(x_row, r_row, w_ptr, k, T, PAD_L, mask, TAPS, HAS_RES)
    if HAS_RES:
        xn = (tl.load(x_row + k, mask=mask, other=0.0).to(tl.float32)
              + tl.load(r_row + k, mask=mask, other=0.0).to(tl.float32))
        tl.store(xn_ptr + row.to(tl.int64) * T + k, xn.to(xn_ptr.dtype.element_ty), mask=mask)
    # SnakeBeta: v + sin^2(alpha * v) / beta.
    s0 = tl.sin(v0 * alpha)
    s1 = tl.sin(v1 * alpha)
    u_row = u_ptr + row.to(tl.int64) * 2 * T
    tl.store(u_row + k, v0 + ib * s0 * s0, mask=mask)
    tl.store(u_row + T + k, v1 + ib * s1 * s1, mask=mask)


@triton.jit
def _fwd_down(u_ptr, d_ptr, y_ptr, T, DPAD: tl.constexpr, DTAPS: tl.constexpr,
              BLOCK: tl.constexpr):
    """``y[n] = sum_j d[j] * u[2n + j - DPAD]``, replicate-padded."""
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < T
    u_row = u_ptr + row.to(tl.int64) * 2 * T
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(DTAPS):
        # Clamped position in the upsampled signal, then its phase and index.
        m = tl.minimum(tl.maximum(2 * n + j - DPAD, 0), 2 * T - 1)
        acc += tl.load(d_ptr + j) * tl.load(u_row + (m % 2) * T + m // 2, mask=mask, other=0.0)
    tl.store(y_ptr + row.to(tl.int64) * T + n, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_snake(gy_ptr, x_ptr, w_ptr, d_ptr, la_ptr, lb_ptr, gv_ptr, dla_ptr, dlb_ptr,
               T, C, PAD_L, TAPS: tl.constexpr, DPAD: tl.constexpr, DTAPS: tl.constexpr,
               EDGE: tl.constexpr, BLOCK: tl.constexpr):
    """Backward through the lowpass and SnakeBeta: ``gy`` -> ``gv``, the
    gradient at the upsampled signal, (rows, 2, T), and ``dla``/``dlb``."""
    row = tl.program_id(0)
    ch = row % C
    alpha = tl.exp(tl.load(la_ptr + ch).to(tl.float32))
    ib = tl.exp(-tl.load(lb_ptr + ch).to(tl.float32))
    start = tl.program_id(1) * BLOCK
    k = (start + tl.arange(0, BLOCK))[None, :]
    mask = k < T
    x_row = x_ptr + row.to(tl.int64) * T
    # Recomputed rather than saved by the forward.
    v0, v1 = _preact(x_row, x_row, w_ptr, k, T, PAD_L, mask, TAPS, False)
    gy_row = gy_ptr + row.to(tl.int64) * T
    g0 = tl.zeros((1, BLOCK), dtype=tl.float32)
    g1 = tl.zeros((1, BLOCK), dtype=tl.float32)
    # The lowpass's adjoint: tap j of output n read upsampled sample
    # 2n + j - DPAD, so each tap reaches one phase. Offsets are shifted by
    # DTAPS to divide a non-negative number.
    for j in tl.static_range(DTAPS):
        if (DPAD + j) % 2 == 0:
            n = k + (DPAD - j + 2 * DTAPS) // 2 - DTAPS
            g0 += tl.load(d_ptr + j) * tl.load(
                gy_row + n, mask=mask & (n >= 0) & (n < T), other=0.0).to(tl.float32)
        else:
            n = k + (DPAD + 1 - j + 2 * DTAPS) // 2 - DTAPS
            g1 += tl.load(d_ptr + j) * tl.load(
                gy_row + n, mask=mask & (n >= 0) & (n < T), other=0.0).to(tl.float32)
    # Replicate padding: every read left of the signal landed on sample 0,
    # every one right of it on sample 2T - 1.
    p = tl.arange(0, EDGE)
    if start == 0:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for j in tl.static_range(DTAPS):
            edge += tl.load(d_ptr + j) * tl.load(
                gy_row + p, mask=(2 * p + j < DPAD) & (p < T), other=0.0).to(tl.float32)
        g0 += tl.where(k == 0, tl.sum(edge, 0), 0.0)
    if start + BLOCK >= T:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for j in tl.static_range(DTAPS):
            edge += tl.load(d_ptr + j) * tl.load(
                gy_row + T - 1 - p, mask=(2 * p + DPAD + 1 < j) & (p < T), other=0.0).to(tl.float32)
        g1 += tl.where(k == T - 1, tl.sum(edge, 0), 0.0)
    _snake_bwd(v0, v1, g0, g1, alpha, ib, gv_ptr + row.to(tl.int64) * 2 * T, k, T, mask,
               dla_ptr + row + tl.arange(0, 1), dlb_ptr + row + tl.arange(0, 1),
               tl.full((1,), True, tl.int1), False)


def _forward(x, r, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype):
    """``(y, x + r)``, or ``(y, None)`` without a residual ``r``."""
    B, C, T = x.shape
    u = torch.empty(B * C, 2, T, device=x.device, dtype=torch.float32)
    xn = None if r is None else torch.empty_like(x, dtype=torch.promote_types(x.dtype, r.dtype))
    # Pointers the kernel does not use without a residual still need a tensor.
    _fwd_up[_grid(B * C, T)](x, x if r is None else r, up_w, log_alpha, log_beta, u,
                             x if xn is None else xn, T, C, pad[0], TAPS=up_w.shape[1],
                             HAS_RES=r is not None, BLOCK=BLOCK)
    y = torch.empty(B, C, T, device=x.device, dtype=dtype)
    _fwd_down[_grid(B * C, T)](u, down_w, y, T, DPAD=down_pad, DTAPS=down_w.shape[0], BLOCK=BLOCK)
    return y, xn


def _backward(gy, x, gxn, log_alpha, log_beta, up_w, down_w, pad, down_pad, x_dtype, r_dtype):
    """``x`` is what the activation read: the input, or ``x + r`` when fused.
    Returns ``(gx, gr, dla, dlb)``; ``gr`` is None without a residual."""
    B, C, T = x.shape
    taps, down_taps = up_w.shape[1], down_w.shape[0]
    gv = torch.empty(B * C, 2, T, device=x.device, dtype=torch.float32)
    dla = torch.zeros(B * C, device=x.device, dtype=torch.float32)
    dlb = torch.zeros(B * C, device=x.device, dtype=torch.float32)
    _bwd_snake[_grid(B * C, T)](gy.contiguous(), x, up_w, down_w, log_alpha, log_beta, gv, dla, dlb,
                                T, C, pad[0], TAPS=taps, DPAD=down_pad, DTAPS=down_taps,
                                EDGE=triton.next_power_of_2(down_taps), BLOCK=BLOCK)
    # Upsampler, and the residual's gradient when there is one.
    res = r_dtype is not None
    gx = torch.empty(B, C, T, device=x.device, dtype=x_dtype)
    gr = torch.empty(B, C, T, device=x.device, dtype=r_dtype) if res else None
    _bwd_up[_grid(B * C, T)](gv, up_w, gxn.contiguous() if res else gv, gx,
                             gr if res else gx, T, pad[0], pad[1], TAPS=taps, HAS_RES=res,
                             EDGE=triton.next_power_of_2(max(pad[0], pad[1], 1)), BLOCK=BLOCK)
    # Per-row sums -> per-channel gradients.
    return gx, gr, dla.view(B, C).sum(0), dlb.view(B, C).sum(0)


class FusedSnakeAlias(torch.autograd.Function):
    """``(x, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype) -> y`` in ``dtype``.

    ``up_w`` is the upsampler's polyphase weights, (2, taps), and ``pad`` its
    replicate padding; ``down_w`` the lowpass and ``down_pad`` its left padding.
    """

    @staticmethod
    def forward(ctx, x, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype):
        x = x.contiguous()
        y, _ = _forward(x, None, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype)
        ctx.save_for_backward(x, log_alpha, log_beta, up_w, down_w)
        ctx.pads = (pad, down_pad)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, log_alpha, log_beta, up_w, down_w = ctx.saved_tensors
        gx, _, dla, dlb = _backward(gy, x, None, log_alpha, log_beta, up_w, down_w, *ctx.pads,
                                    x.dtype, None)
        return gx, dla, dlb, None, None, None, None, None


class FusedResidualSnakeAlias(torch.autograd.Function):
    """``(x, r, ...) -> (y, x + r)``: ``FusedSnakeAlias`` of ``x + r``, with the
    sum computed in the kernel and returned for the residual stream."""

    @staticmethod
    def forward(ctx, x, r, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype):
        x, r = x.contiguous(), r.contiguous()
        y, xn = _forward(x, r, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype)
        ctx.save_for_backward(xn, log_alpha, log_beta, up_w, down_w)
        ctx.pads = (pad, down_pad)
        ctx.dtypes = (x.dtype, r.dtype)
        return y, xn

    @staticmethod
    def backward(ctx, gy, gxn):
        xn, log_alpha, log_beta, up_w, down_w = ctx.saved_tensors
        gx, gr, dla, dlb = _backward(gy, xn, gxn, log_alpha, log_beta, up_w, down_w, *ctx.pads,
                                     *ctx.dtypes)
        return gx, gr, dla, dlb, None, None, None, None, None
