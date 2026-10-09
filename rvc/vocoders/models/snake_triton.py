"""Triton kernels for an anti-aliased SnakeBeta, whose helpers ``snake_alias_triton`` builds on.

One op for ``AntiAliasedActivation(SnakeBeta)`` at factor 2, width 16: the
2x polyphase upsampler, SnakeBeta and the 65-tap stride-2 lowpass.  The
activated upsampled signal goes through one FP32 scratch buffer and is
recomputed from the input in backward.  The two phases of the upsampled signal
share every read of the input and of the output gradient.

Both paddings of the unfused modules are folded into the kernels: replicate on
the input (clamped reads, and its adjoint on the two edge samples) and reflect
on the upsampled signal (mirrored reads, and its adjoint on the edge lanes).
``FusedResidualSnakeBeta`` also takes the residual add in front of the
activation.

Every kernel sees the tensors as ``B * C`` rows of ``T`` samples; a program is
one ``BLOCK`` of one row, but for the edge pass.  Names, with ``g`` in front for a gradient:

    x, r, xn    input, residual and their sum
    w           upsampler weights, (2, TAPS): one row per output phase
    la, lb      log alpha and log beta, per channel; ``ib`` is 1 / beta
    v0, v1      upsampled signal at 2k and 2k + 1, before SnakeBeta
    u0, u1      the same after it
    d           the lowpass: even taps read phase 0 (33), odd ones phase 1 (32)
    y           output
    dla, dlb    gradients of log alpha and log beta, per row
"""

import torch
import triton
import triton.language as tl

BLOCK = 256
#: Rows per program of ``_bwd_snake_edges``, whose rows are 32 samples each.
EDGE_ROWS = 4


@triton.jit
def _preact(x_row, r_row, w_ptr, k, T, PAD_L, mask, TAPS: tl.constexpr,
            HAS_RES: tl.constexpr):
    """The upsampled input at 2k and 2k + 1, from one read per tap."""
    v0 = tl.zeros(k.shape, dtype=tl.float32)
    v1 = tl.zeros(k.shape, dtype=tl.float32)
    for t in tl.static_range(TAPS):
        idx = tl.minimum(tl.maximum(k + t - PAD_L, 0), T - 1)
        xv = tl.load(x_row + idx, mask=mask, other=0.0).to(tl.float32)
        if HAS_RES:
            xv += tl.load(r_row + idx, mask=mask, other=0.0).to(tl.float32)
        v0 += tl.load(w_ptr + t) * xv
        v1 += tl.load(w_ptr + TAPS + t) * xv
    return v0, v1


@triton.jit
def _fwd_up(x_ptr, r_ptr, w_ptr, la_ptr, lb_ptr, u_ptr, xn_ptr, T, C, PAD_L,
            TAPS: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """Upsample and activate: ``x`` (plus ``r``) -> the padded scratch ``u``.
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
    u0 = v0 + ib * s0 * s0
    u1 = v1 + ib * s1 * s1
    # ``u`` is the upsampled signal reflect-padded by 32, as two phases of
    # T + 32: each sample goes to its own slot and, near the edges, to the
    # mirrored one.
    KP = T + 32
    u_row = u_ptr + row.to(tl.int64) * 2 * KP
    tl.store(u_row + k + 16, u0, mask=mask)
    tl.store(u_row + KP + k + 16, u1, mask=mask)
    tl.store(u_row + 16 - k, u0, mask=mask & (k >= 1) & (k <= 16))
    tl.store(u_row + KP + 15 - k, u1, mask=mask & (k <= 15))
    tl.store(u_row + 2 * T + 15 - k, u0, mask=mask & (k >= T - 16))
    tl.store(u_row + KP + 2 * T + 14 - k, u1, mask=mask & (k >= T - 17) & (k <= T - 2))


@triton.jit
def _fwd_down(u_ptr, d_ptr, y_ptr, T, BLOCK: tl.constexpr):
    """``y[n] = sum_j d[j] * u[2n + j]`` over the padded signal ``_fwd_up`` wrote."""
    row = tl.program_id(0)
    KP = T + 32
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < T
    u_row = u_ptr + row.to(tl.int64) * 2 * KP
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Stride 2 over the interleaved signal: each phase is read at stride 1.
    for i in tl.static_range(33):
        acc += tl.load(d_ptr + 2 * i) * tl.load(u_row + n + i, mask=mask, other=0.0)
    for i in tl.static_range(32):
        acc += tl.load(d_ptr + 2 * i + 1) * tl.load(u_row + KP + n + i, mask=mask, other=0.0)
    tl.store(y_ptr + row.to(tl.int64) * T + n, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _snake_bwd(v0, v1, g0, g1, alpha, ib, gv_row, k, T, mask, dla_ptr, dlb_ptr, row_mask,
               ACCUMULATE: tl.constexpr):
    """SnakeBeta's backward for both phases, from the gradient at its output.
    Tensors are (rows, lanes); ``dla_ptr``/``dlb_ptr`` point at each row's sum."""
    a0 = v0 * alpha
    a1 = v1 * alpha
    s0 = tl.sin(a0)
    s1 = tl.sin(a1)
    # d/da sin^2(a) = 2 sin(a) cos(a)
    t0 = 2.0 * s0 * tl.cos(a0)
    t1 = 2.0 * s1 * tl.cos(a1)
    gv0 = g0 * (1.0 + ib * alpha * t0)
    gv1 = g1 * (1.0 + ib * alpha * t1)
    # The edge pass adds to what the main pass stored.
    if ACCUMULATE:
        tl.atomic_add(gv_row + k, gv0, mask=mask)
        tl.atomic_add(gv_row + T + k, gv1, mask=mask)
    else:
        tl.store(gv_row + k, gv0, mask=mask)
        tl.store(gv_row + T + k, gv1, mask=mask)
    dla = tl.where(mask, ib * (g0 * t0 * a0 + g1 * t1 * a1), 0.0)
    dlb = tl.where(mask, -ib * (g0 * s0 * s0 + g1 * s1 * s1), 0.0)
    tl.atomic_add(dla_ptr, tl.sum(dla, 1), mask=row_mask)
    tl.atomic_add(dlb_ptr, tl.sum(dlb, 1), mask=row_mask)


@triton.jit
def _bwd_snake(gy_ptr, x_ptr, w_ptr, d_ptr, la_ptr, lb_ptr, gv_ptr, dla_ptr, dlb_ptr,
               T, C, PAD_L, TAPS: tl.constexpr, BLOCK: tl.constexpr):
    """Backward through the lowpass and SnakeBeta: ``gy`` -> ``gv``, the
    gradient at the upsampled signal, (rows, 2, T), and ``dla``/``dlb``."""
    row = tl.program_id(0)
    ch = row % C
    alpha = tl.exp(tl.load(la_ptr + ch).to(tl.float32))
    ib = tl.exp(-tl.load(lb_ptr + ch).to(tl.float32))
    k = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK))[None, :]
    mask = k < T
    x_row = x_ptr + row.to(tl.int64) * T
    # Recomputed rather than saved by the forward.
    v0, v1 = _preact(x_row, x_row, w_ptr, k, T, PAD_L, mask, TAPS, False)
    gy_row = gy_ptr + row.to(tl.int64) * T
    # The lowpass's adjoint at padded position 2k + e + 32.
    g0 = tl.zeros((1, BLOCK), dtype=tl.float32)
    g1 = tl.zeros((1, BLOCK), dtype=tl.float32)
    for i in tl.static_range(33):
        n = k + 16 - i
        g = tl.load(gy_row + n, mask=mask & (n >= 0) & (n < T), other=0.0).to(tl.float32)
        g0 += tl.load(d_ptr + 2 * i) * g
        if i < 32:
            g1 += tl.load(d_ptr + 2 * i + 1) * g
    _snake_bwd(v0, v1, g0, g1, alpha, ib, gv_ptr + row.to(tl.int64) * 2 * T, k, T, mask,
               dla_ptr + row + tl.arange(0, 1), dlb_ptr + row + tl.arange(0, 1),
               tl.full((1,), True, tl.int1), False)


@triton.jit
def _bwd_snake_edges(gy_ptr, x_ptr, w_ptr, d_ptr, la_ptr, lb_ptr, gv_ptr, dla_ptr, dlb_ptr,
                     R, T, C, PAD_L, TAPS: tl.constexpr, ROWS: tl.constexpr):
    """The reflect padding's second read: upsampled samples 1..32 and
    2T-33..2T-2 also sit at a mirrored padded position.  Program (i, 0) takes
    the first 32 samples of ``ROWS`` rows, (i, 1) their last 32."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    right = tl.program_id(1)
    row_mask = rows < R
    ch = rows % C
    alpha = tl.exp(tl.load(la_ptr + ch, mask=row_mask, other=0.0).to(tl.float32))[:, None]
    ib = tl.exp(-tl.load(lb_ptr + ch, mask=row_mask, other=0.0).to(tl.float32))[:, None]
    k = tl.zeros((ROWS, 32), dtype=tl.int32) + (tl.arange(0, 32) + right * (T - 32))[None, :]
    mask = row_mask[:, None] & (k >= 0) & (k < T)
    x_row = x_ptr + rows.to(tl.int64)[:, None] * T
    v0, v1 = _preact(x_row, x_row, w_ptr, k, T, PAD_L, mask, TAPS, False)
    gy_row = gy_ptr + rows.to(tl.int64)[:, None] * T
    g0 = tl.zeros((ROWS, 32), dtype=tl.float32)
    g1 = tl.zeros((ROWS, 32), dtype=tl.float32)
    # The lowpass's adjoint at each sample's mirrored position.
    if right == 0:
        for i in tl.static_range(33):
            n = 16 - k - i
            g = tl.load(gy_row + n, mask=mask & (k >= 1) & (n >= 0), other=0.0).to(tl.float32)
            g0 += tl.load(d_ptr + 2 * i) * g
            if i < 32:
                n = 15 - k - i
                g = tl.load(gy_row + n, mask=mask & (n >= 0), other=0.0).to(tl.float32)
                g1 += tl.load(d_ptr + 2 * i + 1) * g
    else:
        for i in tl.static_range(33):
            n = 2 * T + 15 - k - i
            g = tl.load(gy_row + n, mask=mask & (n >= 0) & (n < T), other=0.0).to(tl.float32)
            g0 += tl.load(d_ptr + 2 * i) * g
            if i < 32:
                n = 2 * T + 14 - k - i
                g = tl.load(gy_row + n, mask=mask & (k <= T - 2) & (n >= 0) & (n < T),
                            other=0.0).to(tl.float32)
                g1 += tl.load(d_ptr + 2 * i + 1) * g
    _snake_bwd(v0, v1, g0, g1, alpha, ib, gv_ptr + rows.to(tl.int64)[:, None] * 2 * T, k, T,
               mask, dla_ptr + rows, dlb_ptr + rows, row_mask, True)


@triton.jit
def _bwd_up(gv_ptr, w_ptr, gxn_ptr, gx_ptr, gr_ptr, T, PAD_L, PAD_R, TAPS: tl.constexpr,
            HAS_RES: tl.constexpr, EDGE: tl.constexpr, BLOCK: tl.constexpr):
    """Backward through the upsampler: ``gv`` -> ``gx``.  With ``HAS_RES``,
    ``gxn`` (the gradient of the returned ``x + r``) is added and the result
    also written to ``gr``."""
    row = tl.program_id(0)
    start = tl.program_id(1) * BLOCK
    j = start + tl.arange(0, BLOCK)
    mask = j < T
    gv_row = gv_ptr + row.to(tl.int64) * 2 * T
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # The upsampler's adjoint: both phases, every tap that read x[j].
    for b in tl.static_range(2):
        for t in tl.static_range(TAPS):
            a = j + PAD_L - t
            acc += tl.load(w_ptr + b * TAPS + t) * tl.load(
                gv_row + b * T + a, mask=mask & (a >= 0) & (a < T), other=0.0)
    # Replicate padding: every padded position left of the signal read x[0],
    # every one right of it x[T - 1].
    p = tl.arange(0, EDGE)
    if start == 0:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for b in tl.static_range(2):
            for t in tl.static_range(TAPS):
                q = p - t
                edge += tl.load(w_ptr + b * TAPS + t) * tl.load(
                    gv_row + b * T + q, mask=(p < PAD_L) & (q >= 0) & (q < T), other=0.0)
        acc += tl.where(j == 0, tl.sum(edge, 0), 0.0)
    if start + BLOCK >= T:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for b in tl.static_range(2):
            for t in tl.static_range(TAPS):
                q = PAD_L + T + p - t
                edge += tl.load(w_ptr + b * TAPS + t) * tl.load(
                    gv_row + b * T + q, mask=(p < PAD_R) & (q >= 0) & (q < T), other=0.0)
        acc += tl.where(j == T - 1, tl.sum(edge, 0), 0.0)
    out = row.to(tl.int64) * T + j
    if HAS_RES:
        acc += tl.load(gxn_ptr + out, mask=mask, other=0.0).to(tl.float32)
        tl.store(gr_ptr + out, acc.to(gr_ptr.dtype.element_ty), mask=mask)
    tl.store(gx_ptr + out, acc.to(gx_ptr.dtype.element_ty), mask=mask)


def _grid(rows, length):
    """Launch grid: one program per row and per ``BLOCK`` samples."""
    return (rows, triton.cdiv(length, BLOCK))


def _forward(x, r, log_alpha, log_beta, up_w, down_w, pad, dtype):
    """``(y, x + r)``, or ``(y, None)`` without a residual ``r``."""
    B, C, T = x.shape
    # The activated 2x signal: two phases, each reflect-padded by 16.
    u = torch.empty(B * C, 2, T + 32, device=x.device, dtype=torch.float32)
    xn = None if r is None else torch.empty_like(x, dtype=torch.promote_types(x.dtype, r.dtype))
    # Pointers the kernel does not use without a residual still need a tensor.
    _fwd_up[_grid(B * C, T)](x, x if r is None else r, up_w, log_alpha, log_beta, u,
                             x if xn is None else xn, T, C, pad[0], TAPS=up_w.shape[1],
                             HAS_RES=r is not None, BLOCK=BLOCK)
    y = torch.empty(B, C, T, device=x.device, dtype=dtype)
    _fwd_down[_grid(B * C, T)](u, down_w, y, T, BLOCK=BLOCK)
    return y, xn


def _backward(gy, x, gxn, log_alpha, log_beta, up_w, down_w, pad, x_dtype, r_dtype):
    """``x`` is what the activation read: the input, or ``x + r`` when fused.
    Returns ``(gx, gr, dla, dlb)``; ``gr`` is None without a residual."""
    B, C, T = x.shape
    taps = up_w.shape[1]
    gv = torch.empty(B * C, 2, T, device=x.device, dtype=torch.float32)
    dla = torch.zeros(B * C, device=x.device, dtype=torch.float32)
    dlb = torch.zeros(B * C, device=x.device, dtype=torch.float32)
    gy = gy.contiguous()
    # Lowpass and SnakeBeta: the main pass stores ``gv``, the edge pass adds
    # the reflect padding's share to it.
    args = (gy, x, up_w, down_w, log_alpha, log_beta, gv, dla, dlb)
    _bwd_snake[_grid(B * C, T)](*args, T, C, pad[0], TAPS=taps, BLOCK=BLOCK)
    _bwd_snake_edges[(triton.cdiv(B * C, EDGE_ROWS), 2)](*args, B * C, T, C, pad[0], TAPS=taps,
                                                         ROWS=EDGE_ROWS)
    # Upsampler, and the residual's gradient when there is one.
    res = r_dtype is not None
    gx = torch.empty(B, C, T, device=x.device, dtype=x_dtype)
    gr = torch.empty(B, C, T, device=x.device, dtype=r_dtype) if res else None
    _bwd_up[_grid(B * C, T)](gv, up_w, gxn.contiguous() if res else gv, gx,
                             gr if res else gx, T, pad[0], pad[1], TAPS=taps, HAS_RES=res,
                             EDGE=triton.next_power_of_2(max(pad[0], pad[1], 1)), BLOCK=BLOCK)
    # Per-row sums -> per-channel gradients.
    return gx, gr, dla.view(B, C).sum(0), dlb.view(B, C).sum(0)


class FusedSnakeBeta(torch.autograd.Function):
    """``(x, log_alpha, log_beta, up_w, down_w, pad, dtype) -> y`` in ``dtype``.

    ``up_w`` is the upsampler's polyphase weights, (2, taps), and ``pad`` its
    replicate ``phase_pad``; ``down_w`` the 65-tap lowpass.
    """

    @staticmethod
    def forward(ctx, x, log_alpha, log_beta, up_w, down_w, pad, dtype):
        x = x.contiguous()
        y, _ = _forward(x, None, log_alpha, log_beta, up_w, down_w, pad, dtype)
        ctx.save_for_backward(x, log_alpha, log_beta, up_w, down_w)
        ctx.pad = pad
        return y

    @staticmethod
    def backward(ctx, gy):
        x, log_alpha, log_beta, up_w, down_w = ctx.saved_tensors
        gx, _, dla, dlb = _backward(gy, x, None, log_alpha, log_beta, up_w, down_w, ctx.pad,
                                    x.dtype, None)
        return gx, dla, dlb, None, None, None, None


class FusedResidualSnakeBeta(torch.autograd.Function):
    """``(x, r, ...) -> (y, x + r)``: ``FusedSnakeBeta`` of ``x + r``, with the
    sum computed in the kernel and returned for the residual stream."""

    @staticmethod
    def forward(ctx, x, r, log_alpha, log_beta, up_w, down_w, pad, dtype):
        x, r = x.contiguous(), r.contiguous()
        y, xn = _forward(x, r, log_alpha, log_beta, up_w, down_w, pad, dtype)
        ctx.save_for_backward(xn, log_alpha, log_beta, up_w, down_w)
        ctx.pad = pad
        ctx.dtypes = (x.dtype, r.dtype)
        return y, xn

    @staticmethod
    def backward(ctx, gy, gxn):
        xn, log_alpha, log_beta, up_w, down_w = ctx.saved_tensors
        gx, gr, dla, dlb = _backward(gy, xn, gxn, log_alpha, log_beta, up_w, down_w, ctx.pad,
                                     *ctx.dtypes)
        return gx, gr, dla, dlb, None, None, None, None
