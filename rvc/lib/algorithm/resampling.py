"""Windowed-sinc resamplers used by the RefineGAN decoder.

Two jobs, both about band limits:

* :class:`AntiAliasedUpsample1d` interpolates between stages.  Zero-stuffing
  copies the input spectrum to every multiple of the input rate, and whatever
  the filter leaves of those copies is an image at ``|k*R_in +- f|``.
* :class:`AntiAliasedActivation` evaluates a pointwise nonlinearity at twice
  the rate and filters back down, so the harmonics it creates above Nyquist do
  not fold into the audible band.

``width``, ``rolloff`` and ``filter_beta`` are one filter design, not three
independent knobs, so none of them has a default.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def cache_scope():
    """Context for building a tensor that outlives the call that built it.

    Inference runs under ``torch.inference_mode()``, and a tensor first
    materialised there carries no version counter, so autograd refuses to save
    it for backward.  A kernel cached on the module survives into the next
    training step, so it must be built outside inference mode whichever mode
    happened to fill the cache.
    """

    return torch.inference_mode(False)


def _safe_pad(x: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return x
    mode = "reflect" if x.shape[-1] > padding else "replicate"
    return F.pad(x, (padding, padding), mode=mode)


def lowpass_kernel(
    factor: int,
    width: int,
    rolloff: float,
    filter_beta: float,
) -> Tensor:
    """A Kaiser-windowed sinc, normalised to unit gain, as ``[1, 1, taps]``."""

    half = max(1, int(width) * max(1, int(factor)))
    positions = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = 0.5 * float(rolloff) / max(1, int(factor))
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)
    kernel = kernel * torch.kaiser_window(
        kernel.numel(), periodic=False, beta=float(filter_beta), dtype=kernel.dtype
    )
    return (kernel / kernel.sum()).view(1, 1, -1)


def filter_schedule(
    value: "float | Sequence[float]",
    stages: int,
    name: str,
    minimum: float | None = None,
) -> tuple[float, ...]:
    """Normalise a scalar-or-per-stage filter setting into one value per stage."""

    if isinstance(value, (int, float)):
        schedule = (float(value),) * stages
    else:
        schedule = tuple(float(item) for item in value)
    if len(schedule) != stages:
        raise ValueError(
            f"{name} has {len(schedule)} entries for {stages} stages; "
            f"give one per stage or a single value."
        )
    if minimum is not None and any(item < minimum for item in schedule):
        raise ValueError(f"{name} values must be >= {minimum}.")
    return schedule


class FixedLowPass1d(nn.Module):
    """Depthwise low-pass, optionally decimating by ``stride``."""

    def __init__(
        self,
        factor: int,
        width: int,
        rolloff: float,
        filter_beta: float,
        stride: int = 1,
    ):
        super().__init__()
        self.stride = int(stride)
        self.register_buffer(
            "kernel",
            lowpass_kernel(factor, width, rolloff, filter_beta),
            persistent=False,
        )

    def _grouped_kernel(self, x: Tensor) -> Tensor:
        channels = int(x.shape[1])
        key = (channels, x.dtype, x.device)
        if getattr(self, "_kernel_key", None) != key:
            with cache_scope():
                self._kernel_cache = (
                    self.kernel.to(device=x.device, dtype=x.dtype)
                    .expand(channels, -1, -1)
                    .contiguous()
                )
            self._kernel_key = key
        return self._kernel_cache

    def forward(self, x: Tensor) -> Tensor:
        kernel = self._grouped_kernel(x)
        padding = (kernel.shape[-1] - 1) // 2
        return F.conv1d(
            _safe_pad(x, padding),
            kernel,
            stride=self.stride,
            groups=x.shape[1],
        )


class AntiAliasedUpsample1d(nn.Module):
    """Upsample by ``factor`` with a windowed-sinc interpolation filter.

    The kernels are non-persistent buffers, so this module adds nothing to a
    state dict and a checkpoint cannot tell one filter design from another.
    """

    def __init__(
        self,
        factor: int,
        filter_width: int,
        rolloff: float,
        filter_beta: float,
    ):
        super().__init__()
        self.factor = int(factor)
        # The ``factor`` gain that compensates zero-stuffing is folded into the
        # kernel, so no full-rate tensor is multiplied afterwards.
        kernel = lowpass_kernel(self.factor, filter_width, rolloff, filter_beta)
        self.register_buffer("kernel", kernel * self.factor, persistent=False)

        kernel_size = int(kernel.shape[-1])
        self.pad = kernel_size // self.factor - 1
        # The kernel is a symmetric sinc of odd length, so its group delay is
        # ``(kernel_size - 1) / 2`` at the output rate.  Cropping exactly that
        # is what puts input sample ``n`` on output ``n * factor``; anything
        # less delays the trunk against the skip connections it is summed with.
        self.pad_left = self.pad * self.factor + (kernel_size - 1) // 2

        # Polyphase bookkeeping.  All of it is constant, so none of it may be
        # derived from ``x.shape[-1]``: under ``torch.compile`` that makes the
        # pad widths symbolic and the branch on them data-dependent.  The
        # length cancels out of the right-hand pad,
        # ``max(0, max(shift) + L - (L + 2*pad + 1))``.
        taps = -(-kernel_size // self.factor)
        whole, offset = divmod(self.pad_left, self.factor)
        shifts = tuple(
            whole + (1 if phase + offset >= self.factor else 0)
            for phase in range(self.factor)
        )
        self.taps = taps
        self.shifts = shifts
        self.phase_offset = offset
        self.extra_left = max(0, taps - 1 - min(shifts))
        self.extra_right = max(0, max(shifts) - 2 * self.pad - 1)
        self.starts = tuple(self.extra_left + shift - taps + 1 for shift in shifts)

    def _polyphase(self, x: Tensor) -> Tensor:
        """The kernel split into ``factor`` phases, cached per device and dtype.

        With ``pad_left = a*F + b``, the transposed convolution's tap index
        ``qF + p - nF + pad_left`` splits into phase ``(p + b) mod F`` and tap
        ``q - n + a`` (plus one when ``p + b >= F``), which is where
        ``shifts`` comes from.
        """

        channels = int(x.shape[1])
        key = (channels, x.dtype, x.device)
        if getattr(self, "_poly_key", None) != key:
            kernel = self.kernel.to(device=x.device, dtype=x.dtype)[0, 0]
            weight = kernel.new_zeros(self.factor, 1, self.taps)
            for phase in range(self.factor):
                index = (phase + self.phase_offset) % self.factor
                part = kernel[index :: self.factor]
                # ``w[taps-1-j] = phase[j]``, so a phase shorter than ``taps``
                # is right-aligned; left-aligning it shifts that phase alone by
                # one sample.
                weight[phase, 0, self.taps - part.numel() :] = part.flip(-1)
            with cache_scope():
                self._poly_cache = weight.repeat(channels, 1, 1).contiguous()
            self._poly_key = key
        return self._poly_cache

    def _transposed(self, x: Tensor) -> Tensor:
        """The same filter as one grouped transposed convolution."""

        channels, length = x.shape[1], x.shape[-1]
        key = (channels, x.dtype, x.device)
        if getattr(self, "_dense_key", None) != key:
            with cache_scope():
                self._dense_cache = (
                    self.kernel.to(device=x.device, dtype=x.dtype)
                    .expand(channels, -1, -1)
                    .contiguous()
                )
            self._dense_key = key
        padded = F.pad(x, (self.pad, self.pad + 1), mode="replicate")
        out = F.conv_transpose1d(
            padded,
            self._dense_cache,
            stride=self.factor,
            padding=self.pad_left,
            groups=channels,
        )
        return out[..., : length * self.factor]

    def forward(self, x: Tensor) -> Tensor:
        if self.factor == 1:
            return x
        # A stride-F transposed convolution is F convolutions of K/F taps whose
        # outputs interleave: the same multiplies, none at the output rate, and
        # 5-15x faster because grouped ``conv_transpose1d`` is cuDNN's slow
        # path.  But interleaving needs a modular index however it is written,
        # and Inductor cannot lower the one this builds, so the compiled graph
        # gets the transposed form instead.  The two agree to ~4e-7.
        if torch.compiler.is_compiling():
            return self._transposed(x)

        batch, channels, length = x.shape[0], x.shape[1], x.shape[-1]
        weight = self._polyphase(x)
        # Replicate rather than zero: the filter should extend the edge, not
        # invent a discontinuity at it.  The extra sample on the right is what
        # makes the output reach ``length * factor``.
        padded = F.pad(
            x,
            (self.pad + self.extra_left, self.pad + 1 + self.extra_right),
            mode="replicate",
        )
        phases = F.conv1d(padded, weight, groups=channels)
        phases = phases.view(batch, channels, self.factor, -1)
        out = x.new_empty(batch, channels, length * self.factor)
        for phase, start in enumerate(self.starts):
            out[..., phase :: self.factor] = phases[
                :, :, phase, start : start + length
            ]
        return out


class AntiAliasedActivation(nn.Module):
    """A pointwise nonlinearity evaluated at 2x the rate, then filtered back.

    A nonlinearity at the stage rate creates harmonics above that stage's
    Nyquist, and those fold back inharmonically -- audible as partials whose
    vibrato runs backwards, since a component folded about ``F`` sits at
    ``2F - k*f0``.  Oversampling is the mechanism: a smoother curve is not a
    substitute, because a smooth activation is no better than ``leaky_relu`` on
    a realistic multi-partial input and is not homogeneous, so whatever edge it
    has vanishes as the trunk's amplitude grows.

    ``rolloff`` is the setting to be careful with.  The round trip cuts at
    ``rolloff`` of the stage's own Nyquist, so this band-limits as well as
    anti-aliases; too low and it removes more signal than alias.  The alias
    floor at 2x is about -55 dB whatever the filter, because ``leaky_relu``
    makes products of every order and only second-order ones fit under 2x.

    Both resamplers' kernels are non-persistent, so wrapping an activation adds
    no state-dict key.
    """

    def __init__(
        self,
        activation: nn.Module | None = None,
        *,
        leaky_relu_slope: float = 0.2,
        factor: int = 2,
        filter_width: int = 16,
        rolloff: float = 0.99,
        filter_beta: float = 6.0,
    ):
        super().__init__()
        self.factor = int(factor)
        self.activation = (
            nn.LeakyReLU(leaky_relu_slope) if activation is None else activation
        )
        self.design = (
            self.factor,
            int(filter_width),
            float(rolloff),
            float(filter_beta),
        )
        self.up = AntiAliasedUpsample1d(
            self.factor,
            filter_width=filter_width,
            rolloff=rolloff,
            filter_beta=filter_beta,
        )
        self.down = FixedLowPass1d(
            self.factor,
            width=filter_width,
            rolloff=rolloff,
            filter_beta=filter_beta,
            stride=self.factor,
        )

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-1]
        # In-place on the upsampler's own output: it is this module's largest
        # intermediate and nothing else holds a reference to it.
        upsampled = self.up(x)
        if type(self.activation) is nn.LeakyReLU:
            upsampled = F.leaky_relu(
                upsampled, self.activation.negative_slope, inplace=True
            )
        else:
            upsampled = self.activation(upsampled)
        x = self.down(upsampled)
        # The two resamplers round their padding independently, so the round
        # trip can land a sample either side.  This module sits inside a
        # residual add, where a lost sample is a shape error.
        if x.shape[-1] > length:
            return x[..., :length]
        if x.shape[-1] < length:
            return F.pad(x, (0, length - x.shape[-1]), mode="replicate")
        return x
