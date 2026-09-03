from typing import Sequence

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights, get_padding
from rvc.lib.algorithm.resampling import (
    AntiAliasedActivation,
    AntiAliasedUpsample1d,
    filter_schedule,
)


#: How much of a ``ParallelResBlock`` gets anti-aliased activations, ordered by
#: coverage.  The flag reaches the ``AdaIN`` activations wrapped around each
#: ``ResBlock`` as well as the block's own conv pairs:
#:
#:     "none"    nothing
#:     "adain"   the 6 AdaIN per stage, and nothing else
#:     "half"    those, plus the first activation of each of the 9 conv pairs
#:     "full"    those, plus both activations of each pair (18)
#:
#: A/B renders put the inharmonic lines entirely in the ``AdaIN`` activations
#: at 2 and 8 kHz: covering the loops and the conv pairs without them leaves
#: the lines, and covering those six alone removes them.  Cost is per
#: activation, so "adain" is also 6 sites a stage instead of 24.
ANTIALIAS_MODES = ("none", "adain", "half", "full")


#: Interpolation filter for the trunk's upsamplers, one entry per stage.
#:
#: Only the last stage gets a long filter.  What matters is the image a stage
#: makes times the path from that stage to the output, and the last stage's
#: image both is the loudest at the output and mirrors about 4 kHz.  Extending
#: the earlier stages moves numbers that are already 30 dB down and costs edge:
#: ``AntiAliasedUpsample1d`` replicate-pads, so a longer kernel on a short
#: input reads more invented continuation than signal -- at a 0.4 s training
#: segment stage 0 gets 40 samples.
#: RefineGAN's stage rates, by sample rate.  Not the config's
#: ``upsample_rates``: that key is HiFi-GAN's and ascending, while this decoder
#: wants descending -- a stage's anti-image filter keeps ``rolloff`` of the
#: rate it reads, so the last residual block synthesises everything above
#: ``rolloff * rate[-2] / 2`` from scratch, and ``[10, 6, 2, 2]`` at 24 kHz
#: puts that ceiling at 900 Hz against 3600 for ``[5, 4, 4, 3]``.
#:
#: The rates leave no trace in the state dict -- channel counts follow the
#: stage index, not the rate -- so a decoder built with the wrong ones loads a
#: checkpoint silently and renders a different signal.  Hence a table rather
#: than a config key.
#:
#: 32 kHz is the trained one.  24 kHz is chosen to reproduce its internal
#: ladder exactly -- both give loop rates 100/500/2000/8000, so the two
#: anti-aliased stages land at 2 and 8 kHz either way -- and differ only in the
#: final expansion.  40 and 48 kHz cannot match it and take the descending
#: factorisation with the highest last-stage ceiling instead.
REFINEGAN_UPSAMPLE_RATES = {
    24000: (5, 4, 4, 3),
    32000: (5, 4, 4, 4),
    40000: (5, 5, 4, 4),
    48000: (6, 5, 4, 4),
}


def upsample_rates_for(sample_rate: int, hop_length: int):
    """The stage rates for a sample rate, checked against the hop."""

    rates = REFINEGAN_UPSAMPLE_RATES.get(int(sample_rate))
    if rates is None:
        raise ValueError(
            f"RefineGAN has no stage layout for {sample_rate} Hz; known rates "
            f"are {sorted(REFINEGAN_UPSAMPLE_RATES)}."
        )
    product = 1
    for rate in rates:
        product *= rate
    if product != int(hop_length):
        raise ValueError(
            f"RefineGAN's stages {rates} multiply to {product}, but the hop "
            f"length at {sample_rate} Hz is {hop_length}."
        )
    return rates


DEFAULT_UPSAMPLE_WIDTH = (12, 12, 12, 48)
DEFAULT_UPSAMPLE_ROLLOFF = (0.90, 0.90, 0.90, 0.99)
DEFAULT_UPSAMPLE_BETA = (6.0, 6.0, 6.0, 9.0)


def loop_rates(sample_rate: int, upsample_rates: "Sequence[int]"):
    """The rate each of the two loops' activations runs at, in Hz.

    Every pointwise nonlinearity folds about the Nyquist of its own rate, so
    the rate decides whether its aliasing lands in the audible band.
    ``antialias_stages`` indexes the residual blocks and cannot reach the
    ``downs[]`` activations at all; ``antialias_rates`` is what does.
    """

    total = 1
    for rate in upsample_rates:
        total *= int(rate)
    frame_rate = int(sample_rate) // total

    down, rate = [], int(sample_rate)
    for factor in reversed([int(r) for r in upsample_rates]):
        down.append(rate)
        rate //= factor

    up, rate = [], frame_rate
    for factor in [int(r) for r in upsample_rates]:
        up.append(rate)
        rate *= factor

    return tuple(down), tuple(up)


class ResBlock(nn.Module):
    """Residual block of dilated convolutions at multiple dilation rates.

    ``antialias`` wraps the activations in :class:`AntiAliasedActivation`.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
        antialias: str = "none",
    ):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope
        if antialias not in ANTIALIAS_MODES:
            raise ValueError(
                f"antialias must be one of {ANTIALIAS_MODES}, not {antialias!r}."
            )
        self.antialias = antialias

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2.apply(init_weights)

        # ``"adain"`` covers the activations wrapping this block, not the ones
        # inside it, so here it counts as ``"none"``.
        count = {
            "none": 0,
            "adain": 0,
            "half": len(self.convs1),
            "full": 2 * len(self.convs1),
        }[antialias]
        # One instance each rather than one shared: they are stateless, but
        # each caches an expanded per-channel kernel.
        self.activations = nn.ModuleList(
            [
                AntiAliasedActivation(leaky_relu_slope=leaky_relu_slope)
                for _ in range(count)
            ]
        )

    def forward(self, x: torch.Tensor):
        index = 0
        # ``"adain"`` wraps the activations *around* this block, not the ones
        # inside it, so here it is ``"none"``.
        wraps_pairs = self.antialias in ("half", "full")
        for c1, c2 in zip(self.convs1, self.convs2):
            if not wraps_pairs:
                xt = F.leaky_relu(x, self.leaky_relu_slope)
            else:
                xt = self.activations[index](x)
                index += 1
            xt = c1(xt)
            if self.antialias == "full":
                xt = self.activations[index](xt)
                index += 1
            else:
                xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x

        return x


class AdaIN(nn.Module):
    """Noise-regularised activation, two per ``ResBlock``.

    ``antialias`` follows the ``ParallelResBlock`` mode rather than being its
    own flag: these six per stage are where the inharmonic lines come from, and
    a setting that covered the conv pairs while leaving these raw was the
    configuration that shipped with the artefact.
    """

    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
        antialias: bool = False,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        self.antialias = bool(antialias)
        # safe to use in-place as it is used on a new x+gaussian tensor
        self.activation = (
            AntiAliasedActivation(leaky_relu_slope=leaky_relu_slope)
            if self.antialias
            else nn.LeakyReLU(leaky_relu_slope)
        )

    def forward(self, x: torch.Tensor):
        # The noise is a training-time regulariser and pure cost at
        # inference: this runs six times per stage, and dropping it in eval
        # took the generator from 418 ms to 313 on CPU at batch 4.  Inference
        # stays stochastic anyway -- the source draws its own noise, and that
        # one *is* the unvoiced content.
        if not self.training:
            return self.activation(x)

        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    """Runs several ResBlocks (different kernel sizes) in parallel and averages them."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
        antialias: str = "none",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.antialias = antialias

        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.input_conv.apply(init_weights)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(
                        channels=out_channels,
                        leaky_relu_slope=leaky_relu_slope,
                        antialias=antialias != "none",
                    ),
                    ResBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                        antialias=antialias,
                    ),
                    AdaIN(
                        channels=out_channels,
                        leaky_relu_slope=leaky_relu_slope,
                        antialias=antialias != "none",
                    ),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)


class SineGenerator(nn.Module):
    """Sine + additive-noise harmonic excitation source.

    The only one left.  ``comb`` and ``bank`` were removed on 2026-09-03: the
    inharmonic lines they were being traded against turned out to be the
    ``AdaIN`` activations, and on a fixed trunk the bank had been *negative*
    against the sine once ``source_gain`` was on (multi-scale mel 1.7357 ->
    1.8057 for the bank, 1.9714 -> 1.7418 for the sine).
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Tanh(),
        )

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim), dim = fundamental + overtones."""
        # rad_values is F0 in rad mod 1 (the integer cycle count doesn't affect phase)
        rad_values = (f0_values / self.sampling_rate) % 1

        # random initial phase per harmonic, none for the fundamental
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    # Kept out of the compiled graph: ``_f02sine`` is a cumsum over the sample
    # axis, which Inductor lowers to a ``SplitScan`` whose codegen raises.
    # Everything up to ``merge`` runs under ``no_grad`` and is a pure function
    # of f0, so excluding it costs no fusion.
    @torch.compiler.disable
    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise

        # merge with grad
        return self.merge(sine_waves)


class RefineGAN2Generator(nn.Module):
    """
    RefineGAN2: the RefineGAN decoder with its signal-path defects fixed.

    Not interchangeable with :class:`RefineGANGenerator`.  Every tensor has the
    same name and shape, so ``load_state_dict`` accepts either into either; the
    differences live entirely in code, and ``source_gain`` is the only key that
    tells a checkpoint of one from the other.

    What changed:

    * Descending stage rates from :data:`REFINEGAN_UPSAMPLE_RATES`, so the last
      residual block is not left synthesising most of its band from scratch.
    * A windowed-sinc interpolation filter instead of linear upsampling, whose
      triangular kernel rejects the first image by only 1.7-9.6 dB.
    * The upsampler crops its own group delay, so the trunk is not delayed
      against the skip connections it is summed with.
    * Anti-aliased ``AdaIN`` activations at the two stages running at 2 and
      8 kHz, which is where the inharmonic lines came from.
    * An excitation gain projected from the conditioning.

    Args:
        source_gain (bool, optional): Scale the excitation by an intensity
            envelope projected from the conditioning, as RefineGAN's paper
            does with the mel. Defaults to False.
        antialias_rates (Sequence[int], optional): Which of the two loops'
            activation rates, in Hz, get anti-aliased activations -- see
            ``loop_rates``. This is the knob that matters: it selects by the
            rate a nonlinearity actually runs at, so it can reach the
            ``downs[]`` activation at 8 kHz where the fold at ``8000 - k*f0``
            is created. Protecting every rate the decoder has costs 12% of the
            step against raw activations; the shipped config does that.
            Defaults to none.
        antialias_stages (Sequence[int], optional): Which upsampling stages get
            anti-aliased activations in their residual blocks. Defaults to none,
            and the shipped config leaves it there: it addresses the residual
            blocks' *internal* activations, never the loops', it did nothing
            measurable against the mirroring, and at ``"full"`` on one stage it
            cost 47 ms and 795 MiB at batch 8 -- more than protecting all five
            loop rates with a filter three times as long.
        antialias (str, optional): ``"none"``, ``"half"`` or ``"full"`` -- how
            many of each conv pair's two activations are anti-aliased in those
            stages. Defaults to ``"half"``, and is forced to ``"none"`` when
            ``antialias_stages`` is empty.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 32000,
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,
        gin_channels: int = 256,
        checkpointing: bool = False,
        upsample_initial_channel=512,
        filter_width: "int | Sequence[int]" = DEFAULT_UPSAMPLE_WIDTH,
        rolloff: "float | Sequence[float]" = DEFAULT_UPSAMPLE_ROLLOFF,
        filter_beta: "float | Sequence[float]" = DEFAULT_UPSAMPLE_BETA,
        antialias_stages: "Sequence[int] | None" = None,
        antialias: str = "adain",
        source_gain: bool = True,
        antialias_rates: "Sequence[int] | None" = None,
    ):
        super().__init__()
        # No config keys for any of this in Applio, so the defaults *are* the
        # shipped configuration -- the one the A/B renders settled.  Absent
        # stages means the two that run at 2 and 8 kHz, which for the four
        # stages every supported rate uses is [1, 2].
        if antialias_stages is None:
            antialias_stages = [len(upsample_rates) - 3, len(upsample_rates) - 2]
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing

        # The down path doubles ``start_channels`` once per stage and the up
        # path concatenates ``downs[]`` into ``channels + channels // 4``, so
        # the two only meet at one value.  It was a config knob that produced a
        # shape error deep in ``forward`` for every other value; this says so
        # at construction instead.
        required = upsample_initial_channel // (4 * 2 ** (len(upsample_rates) - 1))
        if int(start_channels) != required:
            raise ValueError(
                f"start_channels must be {required} for "
                f"upsample_initial_channel={upsample_initial_channel} over "
                f"{len(upsample_rates)} stages, not {start_channels}: the down "
                f"path doubles it per stage and the up path expects the skip to "
                f"be a quarter of the trunk."
            )

        # Scalar or one-per-stage, normalised in one place -- see
        # ``DEFAULT_UPSAMPLE_WIDTH`` for why these are a schedule and not a
        # single number.  ``filter_schedule`` is what refuses a list of the
        # wrong length, which is the only way to get this silently wrong.
        count = len(upsample_rates)
        self.filter_width = filter_schedule(filter_width, count, "filter_width", 1)
        self.rolloff = filter_schedule(rolloff, count, "rolloff", 0.0)
        self.filter_beta = filter_schedule(filter_beta, count, "filter_beta", 0.0)
        if any(value > 1.0 for value in self.rolloff):
            raise ValueError(
                f"rolloff is a fraction of the stage's Nyquist and cannot "
                f"exceed 1.0, received {self.rolloff}."
            )

        stages = () if antialias_stages is None else tuple(int(s) for s in antialias_stages)
        if any(s < 0 or s >= len(upsample_rates) for s in stages):
            raise ValueError(
                f"antialias_stages must index the {len(upsample_rates)} "
                f"upsampling stages, received {stages}."
            )
        self.antialias_stages = tuple(sorted(set(stages)))
        self.antialias = antialias if self.antialias_stages else "none"

        # Anti-aliasing selected by rate rather than by block: this is the
        # only thing that reaches the ``downs[]`` activations.
        self.down_rates, self.up_rates = loop_rates(sample_rate, upsample_rates)
        protected = {int(r) for r in (antialias_rates or ())}
        unknown = protected - set(self.down_rates) - set(self.up_rates)
        if unknown:
            raise ValueError(
                f"antialias_rates {sorted(unknown)} match no activation rate; "
                f"this decoder runs its down loop at {self.down_rates} and its "
                f"up loop at {self.up_rates} Hz."
            )
        self.antialias_rates = tuple(sorted(protected))
        # ``Identity`` where a rate is not protected, so the forward stays a
        # plain index and each site keeps its own kernel cache.
        self.down_activations = nn.ModuleList(
            [
                AntiAliasedActivation(leaky_relu_slope=leaky_relu_slope)
                if rate in protected
                else nn.Identity()
                for rate in self.down_rates
            ]
        )
        self.up_activations = nn.ModuleList(
            [
                AntiAliasedActivation(leaky_relu_slope=leaky_relu_slope)
                if rate in protected
                else nn.Identity()
                for rate in self.up_rates
            ]
        )

        # ``int``, not the ``np.int64`` ``np.prod`` returns: Dynamo wraps a
        # numpy scalar as a CPU tensor, and one CPU node makes Inductor emit a
        # C++ kernel, which needs a compiler that Windows may not have.
        self.upp = int(np.prod(upsample_rates))

        # Sine is the only source; the state dict is what a cross-load would
        # silently accept, so nothing else may be built here.
        self.source_type = "sine"
        self.m_source = SineGenerator(sample_rate)

        # ``start_channels``, not a literal 16: the down path below is built
        # from it, so a hardcoded width here is a mismatch waiting to happen.
        self.pre_conv = weight_norm(
            nn.Conv1d(1, start_channels, 7, 1, padding=3)
        )

        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList([])
        self.df0 = []
        for i, u in enumerate(upsample_rates):

            new_size = int(size / upsample_rates[-i - 1])
            # T dimension factors for torchaudio.functional.resample
            self.df0.append([size, new_size])
            size = new_size

            new_channels = channels * 2
            self.downsample_blocks.append(
                weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3))
            )
            channels = new_channels

        channels = upsample_initial_channel

        self.mel_conv = weight_norm(
            nn.Conv1d(
                num_mels,
                channels // 2,
                7,
                1,
                padding=3,
            )
        )

        self.mel_conv.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(256, channels // 2, 1)

        # The paper scales its template by intensity read off the mel; this
        # decoder is handed ``z``, from which the log intensity is recoverable.
        # Projecting the source's envelope from the conditioning also puts
        # ``z`` back on the critical path for harmonic content: with a flat,
        # f0-driven source the trunk gets its harmonics without consulting
        # ``z`` at all.  193 parameters.
        self.has_source_gain = bool(source_gain)
        if self.has_source_gain:
            self.source_gain = nn.Conv1d(num_mels, 1, 1)
            # Identity at initialisation, so a run that switches this on
            # starts from exactly the excitation it had before.
            nn.init.zeros_(self.source_gain.weight)
            nn.init.constant_(self.source_gain.bias, 0.5413248546129181)

            # The gain multiplies the excitation, so a residual image in it
            # stamps a sideband onto every harmonic -- hence a filtered
            # upsample rather than ``F.interpolate``.
            self.source_gain_ups = nn.ModuleList(
                [
                    AntiAliasedUpsample1d(
                        rate,
                        filter_width=self.filter_width[stage],
                        rolloff=self.rolloff[stage],
                        filter_beta=self.filter_beta[stage],
                    )
                    for stage, rate in enumerate(upsample_rates)
                ]
            )

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for stage, rate in enumerate(upsample_rates):
            new_channels = channels // 2

                # A windowed sinc, not ``nn.Upsample(mode="linear")``: a
            # triangular kernel rejects the first image by 1.7-9.6 dB, which
            # stamps the frame grid into the waveform as a mirrored partial
            # either side of every harmonic.
            self.upsample_blocks.append(
                AntiAliasedUpsample1d(
                    rate,
                    filter_width=self.filter_width[stage],
                    rolloff=self.rolloff[stage],
                    filter_beta=self.filter_beta[stage],
                )
            )

            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                    antialias=(
                        self.antialias
                        if stage in self.antialias_stages
                        else "none"
                    ),
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        )
        self.conv_post.apply(init_weights)

    # Kept out of the compiled graph: ``torchaudio.functional.resample``
    # builds its kernel from Python ints, which Inductor lowers to a CPU
    # kernel needing a C++ compiler.  Not replaced with a ``FixedLowPass1d``,
    # because torchaudio's filter is 385/953 taps against 73/169 and its
    # stopband is 135-156 dB against 68-78.
    @torch.compiler.disable
    def _decimate(self, x: torch.Tensor, orig_freq: int, new_freq: int):
        return torchaudio.functional.resample(
            x.contiguous(),
            orig_freq=orig_freq,
            new_freq=new_freq,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )

    def _apply_source_gain(self, har_source: torch.Tensor, mel: torch.Tensor):
        """Scale the excitation by an intensity envelope read off ``mel``.

        ``mel`` is this decoder's conditioning -- ``z``, despite the name -- at
        the frame rate; ``har_source`` is (batch, 1, frames * upp).
        """

        if not self.has_source_gain:
            return har_source
        gain = F.softplus(self.source_gain(mel))
        for ups in self.source_gain_ups:
            gain = ups(gain)
        length = har_source.shape[-1]
        if gain.shape[-1] > length:
            gain = gain[..., :length]
        elif gain.shape[-1] < length:
            gain = F.pad(gain, (0, length - gain.shape[-1]), mode="replicate")
        return har_source * gain

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        f0_size = mel.shape[-1]
        f0 = F.interpolate(f0.unsqueeze(1), size=f0_size * self.upp, mode="linear")
        har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)
        har_source = self._apply_source_gain(har_source, mel)
        x = self.pre_conv(har_source)
        downs = []
        for index, (block, (old_size, new_size)) in enumerate(
            zip(self.downsample_blocks, self.df0)
        ):
            activation = self.down_activations[index]
            x = (
                F.leaky_relu(x, self.leaky_relu_slope)
                if isinstance(activation, nn.Identity)
                else activation(x)
            )
            downs.append(x)
            x = self._decimate(x, int(f0_size * old_size), int(f0_size * new_size))
            x = block(x)

        mel = self.mel_conv(mel)
        if g is not None:
            mel = mel + self.cond(g)

        x = torch.cat([mel, x], dim=1)

        for index, (ups, res, down) in enumerate(
            zip(
                self.upsample_blocks,
                self.upsample_conv_blocks,
                reversed(downs),
            )
        ):
            activation = self.up_activations[index]
            x = (
                F.leaky_relu(x, self.leaky_relu_slope)
                if isinstance(activation, nn.Identity)
                else activation(x)
            )

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = torch.cat([x, down], dim=1)
                x = checkpoint(res, x, use_reentrant=False)
            else:
                x = ups(x)
                x = torch.cat([x, down], dim=1)
                x = res(x)

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        """Fold every weight norm back into its weight, by walking the modules.

        Walking rather than listing layers by name: a hand-written list goes
        stale the moment one is added, and nothing catches it because
        ``Synthesizer`` walks the decoder itself and never calls this.
        """

        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
