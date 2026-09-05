"""
Fixed windowed-sinc interpolation for the decoder's up path.

``rolloff`` is the fraction of the stage's Nyquist the filter keeps, so
``1 - rolloff`` is the whole transition band. Width, rolloff and beta are one
design and not three independent knobs: a narrow band asked of a short kernel
simply is not realised.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

def cache_scope():
    """
    Context for building a tensor that outlives the call that built it.

    A tensor first materialised under ``torch.inference_mode()`` carries no
    version counter, so autograd refuses to save it for backward. A kernel
    cached on the module during an eval pass would poison every training step
    after it.
    """

    return torch.inference_mode(False)


def _safe_pad(x: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return x
    mode = "reflect" if x.shape[-1] > padding else "replicate"
    return F.pad(x, (padding, padding), mode=mode)


# ``filter_beta`` has no default on purpose: it belongs to the same design as
# the width and the rolloff, so a caller that supplies two of the three and
# inherits the third has not chosen a design at all.


def lowpass_kernel(
    factor: int,
    width: int,
    rolloff: float,
    filter_beta: float,
) -> Tensor:
    half = max(1, int(width) * max(1, int(factor)))
    positions = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = 0.5 * float(rolloff) / max(1, int(factor))
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)
    kernel = kernel * torch.kaiser_window(
        kernel.numel(), periodic=False, beta=float(filter_beta), dtype=kernel.dtype
    )
    return (kernel / kernel.sum()).view(1, 1, -1)


class AntiAliasedUpsample1d(nn.Module):
    def __init__(
        self,
        factor: int,
        filter_width: int,
        rolloff: float,
        filter_beta: float,
    ):
        super().__init__()
        self.factor = int(factor)
        # The gain that compensates the zero-stuffing is folded into the kernel
        # rather than applied to the output, which lives at the upsampled rate
        # and would cost a second full-rate tensor on every call.
        kernel = lowpass_kernel(self.factor, filter_width, rolloff, filter_beta)
        self.register_buffer("kernel", kernel * self.factor, persistent=False)

        kernel_size = int(kernel.shape[-1])
        self.pad = kernel_size // self.factor - 1
        # The kernel is a symmetric windowed sinc of odd length, so its group
        # delay is ``(kernel_size - 1) / 2`` samples at the output rate, and
        # that is what the transposed convolution's padding has to crop for
        # input sample n to land on output ``n * factor``. It was
        # ``(kernel_size - factor) // 2``, short by ``factor // 2``, so every
        # instance delayed its output and the trunk arrived late against the
        # skips it is concatenated with. Weights trained against the old offset
        # encode the misalignment, so this is a pretrain-from-zero change.
        self.pad_left = self.pad * self.factor + (kernel_size - 1) // 2

        # Everything the polyphase forward needs about padding is a constant,
        # so it is computed here rather than from ``x.shape[-1]``: under
        # torch.compile the latter makes the pad widths symbolic and the branch
        # on them data-dependent. The length cancels out of the right-hand pad.
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
        self.starts = tuple(
            self.extra_left + shift - taps + 1 for shift in shifts
        )

    def _polyphase(self, x: Tensor):
        """
        The kernel split into ``factor`` phases, cached per device/dtype.

        A transposed convolution of stride F with a K-tap kernel is F
        convolutions of K/F taps whose outputs interleave -- the same
        multiplies, none of them at the output rate. Grouped conv1d is not the
        slow path in cuDNN that grouped conv_transpose1d is, so this is 5-14x
        faster for the same result.
        """

        channels = int(x.shape[1])
        key = (channels, x.dtype, x.device)
        if getattr(self, "_poly_key", None) != key:
            kernel = self.kernel.to(device=x.device, dtype=x.dtype)[0, 0]
            taps = self.taps
            weight = kernel.new_zeros(self.factor, 1, taps)
            for phase in range(self.factor):
                index = (phase + self.phase_offset) % self.factor
                part = kernel[index :: self.factor]
                # A phase shorter than ``taps`` is right-aligned; left-aligning
                # it shifts that one phase by a sample.
                weight[phase, 0, taps - part.numel() :] = part.flip(-1)
            with cache_scope():
                self._poly_cache = weight.repeat(channels, 1, 1).contiguous()
            self._poly_key = key
        return self._poly_cache

    def _transposed(self, x: Tensor) -> Tensor:
        """
        The original formulation, kept for torch.compile. Identical output to
        the polyphase form and about 4x slower in eager.
        """

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
        # Interleaving the phases needs a modular index however it is written,
        # and Inductor crashes on the one this builds. So: the fast form in
        # eager, the transposed form under compile. The branch is folded at
        # trace time, and the two agree to 4e-7.
        if torch.compiler.is_compiling():
            return self._transposed(x)

        batch, channels, length = x.shape[0], x.shape[1], x.shape[-1]
        weight = self._polyphase(x)

        # The same replicate pad the transposed form uses: the extra input
        # sample on the right is what makes the output reach ``length * factor``,
        # and replicating keeps the filter from inventing an edge.
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


def filter_schedule(
    value: "float | Sequence[float]",
    stages: int,
) -> tuple[float, ...]:
    """
    Normalise a scalar-or-per-stage filter setting into one value per stage.

    Width, rolloff and beta all take either form, so the broadcast lives in one
    place rather than three.
    """

    if isinstance(value, (int, float)):
        return (float(value),) * stages
    return tuple(float(item) for item in value)
