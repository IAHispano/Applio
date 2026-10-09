import math
import traceback
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.vocoders.models.commons import get_padding
from rvc.vocoders.models.san import SANConv1d, SANConv2d, san_tail
from rvc.vocoders.models.residuals import LRELU_SLOPE
from rvc.vocoders.messages import (
    DISCRIMINATOR_COMPILE_ENABLE_FAILED,
    DISCRIMINATOR_COMPILE_RUNTIME_FAILED,
)
from rvc.vocoders.terminal import warning

#: Applio's branch layouts, under their names so a diff is a diff.  ``v2`` is
#: HiFi-GAN's; ``v3`` trades its three widest period branches for three
#: multi-resolution spectrogram branches.  ``v4`` is this fork's: ``v3``'s
#: periods with pre-emphasis and multi-band spectrogram branches.
#:
#: ``DiscriminatorR``'s frequency axis can be decimated -- ``(1, 2, 2)`` in the
#: last two layers measured no worse than Applio's ``(1, 1, 1)`` on both probe
#: defects and 24% cheaper (81.1 vs 107.0 ms, 342 vs 396 MiB, batch 8 / 0.4 s).
#: Reach it through ``d_frequency_strides``; it is not a version.
#:
#: What must not be touched is the 512-point branch's 50-sample hop.  The
#: branch catches frame-rate mirroring as a *temporal* modulation, so 100 Hz
#: needs a frame shorter than its 10 ms period; a 4096-point branch (10.9 ms)
#: drops held-out accuracy to chance, and so does ChouwaGAN's 128/256/512.
REFERENCE_SAMPLE_RATE = 22050


def rate_scaled_periods(periods, sample_rate, reference_rate=REFERENCE_SAMPLE_RATE):
    """The period set that keeps HiFi-GAN's *time scales* at another rate.

    A period-``p`` branch folds onto a grid at ``sr / p`` Hz and its receptive
    field spans ``647 * p / sr`` seconds, so both meanings of a period hold only
    if ``p`` scales with the rate.  ``[2, 3, 5, 7, 11]`` was chosen at 22.05 kHz
    and carried everywhere unchanged, which empties the *slow* end: at 32 kHz
    the longest branch drops from 323 ms to 222, and pitch structure lives
    there.

    Targets are rounded to the nearest unused prime in log space -- prime
    because two periods sharing a factor fold onto overlapping samples and
    become one branch at two branches' cost, log because the quantity preserved
    is a ratio.  ``reference_rate`` returns the input unchanged, which is what
    makes this a derivation rather than a new design.
    """

    def is_prime(value):
        return value > 1 and all(value % f for f in range(2, int(value**0.5) + 1))

    candidates = [value for value in range(2, 512) if is_prime(value)]
    used, scaled = set(), []
    for period in periods:
        target = int(period) * float(sample_rate) / float(reference_rate)
        best = min(
            (value for value in candidates if value not in used),
            key=lambda value: abs(math.log(value / target)),
        )
        used.add(best)
        scaled.append(best)
    return tuple(sorted(scaled))


#: The reviewed, frozen result of ``rate_scaled_periods`` for the versions and
#: rates that ship.  A *pin*, not a source: the constructor derives, and a test
#: asserts the two agree -- that is what keeps the rule and the numbers together.
#:
#: Membership marks a version as rate-scaled.  ``v1``/``v2`` are absent on
#: purpose: v2's eight periods are Applio's and every RVC v2 pretrained D is
#: trained against them, so they stay verbatim at every rate.
#:
#: Deriving instead of writing the list into each config costs one thing: an
#: unkeyed checkpoint from before the scaling no longer resumes into a
#: ``d_version``-only config.  It fails in ``assert_periods_match``, naming the
#: ``d_periods`` to set.  A period is invisible in every weight
#: (``DiscriminatorP``'s kernels are ``(k, 1)`` whatever ``p`` is), so that
#: guard is the only thing that can tell.
PERIODS_BY_RATE = {
    "v3": {32000: (3, 5, 7, 11, 17)},
}


#: The three multi-resolution spectrogram branches, shared by ``v3`` and
#: ``v4``.  The 512-point branch's 50-sample hop is the one that must not be
#: touched -- see the note above ``REFERENCE_SAMPLE_RATE``.
#:
#: The two spectral branches window the whole transform: UnivNet's shorter
#: windows set the resolution from ``win_length``, too coarse at 32 kHz to
#: separate a harmonic from the valley beside it.  The 512 keeps its 7.5 ms
#: window -- that is what makes it the temporal branch.
V3_RESOLUTIONS = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
V4_RESOLUTIONS = [[1024, 120, 1024], [2048, 240, 2048], [512, 50, 240]]

DISCRIMINATOR_VERSIONS = {
    "v1": ([2, 3, 5, 7, 11, 17], [], (1, 1, 1)),
    "v2": ([2, 3, 5, 7, 11, 17, 23, 37], [], (1, 1, 1)),
    "v3": ([2, 3, 5, 7, 11], V3_RESOLUTIONS, (1, 1, 1)),
    # Not rate-scaled: period 2 is kept on purpose, it is the only branch that
    # folds near Nyquist.  The spectral and pre-emphasis changes are in the
    # tables below.
    "v4": ([2, 3, 5, 7, 11], V4_RESOLUTIONS, (1, 2, 2)),
}

#: Versions whose spectrogram branches are ``MultiBandDiscriminatorR``.
MULTIBAND_MRD_VERSIONS = {"v4"}

#: Pre-emphasis coefficient applied to the period branches' input, per version.
#: Off on v4: at 0.97 periods 5, 7 and 11 stayed near chance for 12k steps.
MPD_PRE_EMPHASIS_BY_VERSION = {"v4": 0.0}

#: R1 strength per version; absent or 0 turns the penalty off.  Opt in with
#: ``d_r1_gamma`` and tune it from ``r1/penalty``.
R1_GAMMA_BY_VERSION = {}


#: How much of the objective UnivHD is allowed to be, per version.
#:
#: ``1.0`` -- the paper's additive setting, and what every version outside this
#: table gets -- was wrong for the branch set ``v4`` first ran (four
#: rate-scaled periods, single-band spectrogram branches), and the pretrain
#: that showed it is the argument for the number here; it has not been
#: re-measured on the current layout.  ``disc_sep``
#: is ``mean(real logit) - mean(fake logit)`` per branch, i.e. how decisively a
#: head is separating.  Measured on a 32 kHz pretrain over these nine branches,
#: SAN on, rolling mean over 50 steps, between steps 2k and 8.5k:
#:
#:     branch            separation
#:     univhd               5.1 - 7.7
#:     msd                  0.7 - 2.1
#:     period_3             0.8 - 1.0
#:     period_5             0.4 - 0.6
#:     period_7             0.3 - 0.7
#:     period_11            0.5 - 0.8
#:     resolution_512       0.3
#:     resolution_1024      0.2 - 0.3
#:     resolution_2048      0.2 - 0.3
#:
#: An order of magnitude clear of the other eight, and it stayed there for the
#: whole window rather than converging toward them.  Because the generator's
#: term is ``(1 - dg)^2``, a head separating by ~6 contributes a term ~10x an
#: average branch's, so one of nine heads was most of ``loss_adv`` -- and the
#: gradient it put into the decoder is most of what drove that run's decoder
#: grad norm to 8-15 x 10^3.
#:
#: ``0.15`` is chosen to leave UnivHD the loudest single head without leaving
#: it the only one: it is roughly the ratio between that separation and the
#: rest, so the branch stops being most of the term while still outweighing
#: any one of the other eight.  It is a starting point, not a fixed point --
#: ``d_univhd_weight`` overrides it on any version, and ``disc_sep_50/univhd``
#: against the other branches is the series that says whether it landed.
#:
#: ``v3`` takes the same weight: the shipped config runs it with UnivHD,
#: which separates several times more than its other branches there too.
#: ``v1`` and ``v2`` predate UnivHD here and keep the paper's 1.0.
UNIVHD_WEIGHT_BY_VERSION = {
    "v3": 0.15,
    "v4": 0.15,
}

#: What a branch with no entry in a weight table is worth: the plain sum every
#: HiFi-GAN-lineage discriminator has always used.
DEFAULT_BRANCH_WEIGHT = 1.0


def univhd_weight_for(version: str) -> float:
    """The default UnivHD weight for a version; 1.0 where none is pinned."""

    return float(UNIVHD_WEIGHT_BY_VERSION.get(str(version), DEFAULT_BRANCH_WEIGHT))


class MPD_MSD_Combined(torch.nn.Module):
    """Multi-period, multi-scale (and optionally multi-resolution / UnivHD) discriminators combined."""

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        version: str = "v2",
        periods=None,
        resolutions=None,
        frequency_strides=None,
        use_msd: bool = True,
        use_fast_mpd: bool = False,
        sample_rate: int = 32000,
        use_univhd: bool = False,
        use_san: bool = False,
        univhd_n_fft: int = 2048,
        univhd_hop_length: int = 256,
        univhd_harmonics: int = 10,
        univhd_bins_per_octave: int = 24,
        univhd_f_min: float = 80.0,
        univhd_channels: int = 32,
        univhd_half_harmonic: bool = True,
        univhd_max_hz: Optional[float] = None,
        univhd_weight: Optional[float] = None,
        msd_weight: float = DEFAULT_BRANCH_WEIGHT,
        mrd_fp32_input: bool = True,
        wave_fp32_input: bool = False,
        mrd_multiband: Optional[bool] = None,
        mrd_channels: int = 32,
        mpd_pre_emphasis: Optional[float] = None,
        r1_gamma: Optional[float] = None,
        r1_interval: int = 16,
        r1_batch_fraction: float = 0.5,
        r1_segment: int = 6400,
        mrd_mel_cond: Optional[dict] = None,
    ):
        """``version`` picks a preset; the overrides edit it branch by branch.

        ``mrd_mel_cond`` conditions the multi-band spectrogram branches on the
        generator's input mel (see ``MultiBandDiscriminatorR``); ``forward``
        and ``real_score`` then take it as ``cond``.

        ``mrd_multiband`` and ``mpd_pre_emphasis`` default to what the version
        says (``MULTIBAND_MRD_VERSIONS``, ``MPD_PRE_EMPHASIS_BY_VERSION``);
        ``mrd_channels`` is the width of the multi-band branches only.

        ``r1_gamma`` defaults to ``R1_GAMMA_BY_VERSION``.  The trainer penalises
        one branch at a time, each about every ``r1_interval`` steps, on the
        loudest ``r1_batch_fraction`` of the real batch cropped to
        ``r1_segment`` samples (0 keeps the whole segment); the discriminator
        only provides :meth:`real_score`.

        ``None`` means "whatever the version says"; an empty list means "none of
        this family" -- a distinction a falsy check would lose.

        Branch costs, fwd+bwd at batch 8 over 0.4 s: scale 16 ms / 168 MiB,
        each period ~9 ms / ~250 MiB, each spectrogram ~30 ms / ~430 MiB.
        Dropping a period is a bigger lever than anything inside a branch.

        ``use_fast_mpd`` swaps every period branch for
        :class:`FastDiscriminatorP` -- 15x fewer parameters at indistinguishable
        probe accuracy; see that class for the numbers and for what the probe
        cannot see.  It changes every period branch's shapes, so a strict load
        of a checkpoint trained the other way fails on its own.

        ``sample_rate`` selects branch *frequencies*, not just UnivHD's
        filterbank: for the versions in ``PERIODS_BY_RATE`` the periods are
        derived from it.  ``use_univhd`` appends the harmonic branch (arXiv
        2512.03486) for +9% time and memory and +0.33 M parameters -- additive,
        as the paper runs it.

        ``mrd_fp32_input`` runs each spectrogram branch's STFT and first conv
        outside FP16 autocast; see :meth:`DiscriminatorR.forward`.  It adds no
        parameters, so a checkpoint loads either way, and without autocast it
        changes nothing.
        """

        super().__init__()
        if version not in DISCRIMINATOR_VERSIONS:
            raise ValueError(
                f"Unknown discriminator version {version!r}: "
                f"expected one of {sorted(DISCRIMINATOR_VERSIONS)}."
            )
        preset_periods, preset_resolutions, preset_strides = DISCRIMINATOR_VERSIONS[
            version
        ]
        if periods is None:
            # ``version in PERIODS_BY_RATE`` is the rate-scaled marker; v1 and
            # v2 keep Applio's set at every rate.  Derived rather than looked
            # up so an unfrozen rate still gets a correct set instead of an
            # exception -- the frozen table is the reviewed pin, not the only
            # legal answer, and ``rate_scaled_periods`` is the rule it pins.
            periods = (
                list(rate_scaled_periods(preset_periods, sample_rate))
                if version in PERIODS_BY_RATE
                else preset_periods
            )
        else:
            periods = list(periods)
        resolutions = (
            preset_resolutions if resolutions is None else [list(r) for r in resolutions]
        )
        frequency_strides = (
            preset_strides if frequency_strides is None else tuple(frequency_strides)
        )
        if any(len(r) != 3 for r in resolutions):
            raise ValueError(
                "Each resolution is [n_fft, hop_length, win_length]; "
                f"received {resolutions}."
            )
        self.version = version
        self.periods = tuple(int(p) for p in periods)
        self.resolutions = tuple(tuple(int(v) for v in r) for r in resolutions)
        self.frequency_strides = tuple(int(s) for s in frequency_strides)
        self.use_msd = bool(use_msd)
        self.use_fast_mpd = bool(use_fast_mpd)
        self.mrd_fp32_input = bool(mrd_fp32_input)
        self.mrd_multiband = (
            version in MULTIBAND_MRD_VERSIONS
            if mrd_multiband is None
            else bool(mrd_multiband)
        )
        self.mrd_channels = int(mrd_channels)
        self.mpd_pre_emphasis = float(
            MPD_PRE_EMPHASIS_BY_VERSION.get(version, 0.0)
            if mpd_pre_emphasis is None
            else mpd_pre_emphasis
        )
        self.r1_gamma = float(
            R1_GAMMA_BY_VERSION.get(version, 0.0) if r1_gamma is None else r1_gamma
        )
        if self.r1_gamma < 0.0:
            raise ValueError(f"r1_gamma cannot be negative; received {self.r1_gamma}.")
        self.r1_interval = max(1, int(r1_interval))
        self.r1_batch_fraction = min(1.0, max(0.0, float(r1_batch_fraction)))
        self.r1_segment = max(0, int(r1_segment))
        self.use_univhd = bool(use_univhd)
        # ``None`` means "whatever the version pins", the same convention the
        # branch overrides above use; an explicit number wins on any version.
        self.univhd_weight = (
            univhd_weight_for(version)
            if univhd_weight is None
            else float(univhd_weight)
        )
        if self.univhd_weight < 0.0:
            raise ValueError(
                f"univhd_weight is a loss weight and cannot be negative; "
                f"received {self.univhd_weight}."
            )
        self.msd_weight = float(msd_weight)
        if self.msd_weight < 0.0:
            raise ValueError(
                f"msd_weight is a loss weight and cannot be negative; "
                f"received {self.msd_weight}."
            )
        # ``train.py`` reads this to decide whether the losses take their SAN
        # form; it is an attribute rather than a lookup on the branches so a
        # discriminator that is *asked* for SAN and could not build it cannot
        # report otherwise.
        self.use_san = bool(use_san)
        self.supports_san = self.use_san
        self.sample_rate = int(sample_rate)
        # Read by ``forward``: spectral norm's power iteration advances once
        # per weight access, so a batched pass would run it once where two
        # separate passes run it twice.  That is a change to the training
        # dynamics, not an optimisation, so the batched path is off there.
        self.use_spectral_norm = bool(use_spectral_norm)
        self.use_checkpointing = use_checkpointing
        branches = []
        if self.use_msd:
            branches.append(
                DiscriminatorS(use_spectral_norm=use_spectral_norm, use_san=self.use_san)
            )
        period_branch = FastDiscriminatorP if self.use_fast_mpd else DiscriminatorP
        branches += [
            period_branch(
                p,
                use_spectral_norm=use_spectral_norm,
                use_san=self.use_san,
                pre_emphasis=self.mpd_pre_emphasis,
            )
            for p in self.periods
        ]
        for branch in branches:
            branch.wave_fp32_input = bool(wave_fp32_input)
        if mrd_mel_cond and not self.mrd_multiband:
            raise ValueError("mrd_mel_cond needs the multi-band spectrogram branches (v4).")
        self.mrd_mel_cond = dict(mrd_mel_cond) if mrd_mel_cond else None
        #: Branch indices that take ``cond``.
        self.cond_branches = (
            frozenset(range(len(branches), len(branches) + len(self.resolutions)))
            if self.mrd_mel_cond
            else frozenset()
        )
        if self.mrd_multiband:
            branches += [
                MultiBandDiscriminatorR(
                    list(r),
                    channels=self.mrd_channels,
                    frequency_strides=self.frequency_strides,
                    use_spectral_norm=use_spectral_norm,
                    use_san=self.use_san,
                    mel_cond=self.mrd_mel_cond,
                )
                for r in self.resolutions
            ]
        else:
            branches += [
                DiscriminatorR(
                    list(r),
                    use_spectral_norm=use_spectral_norm,
                    frequency_strides=self.frequency_strides,
                    use_san=self.use_san,
                    fp32_input=self.mrd_fp32_input,
                )
                for r in self.resolutions
            ]
        if self.use_univhd:
            from rvc.vocoders.models.univhd import UnivHDDiscriminator

            branches.append(
                UnivHDDiscriminator(
                    sample_rate=self.sample_rate,
                    n_fft=int(univhd_n_fft),
                    hop_length=int(univhd_hop_length),
                    harmonics=int(univhd_harmonics),
                    bins_per_octave=int(univhd_bins_per_octave),
                    f_min=float(univhd_f_min),
                    channels=int(univhd_channels),
                    half_harmonic=bool(univhd_half_harmonic),
                    max_hz=univhd_max_hz,
                    use_spectral_norm=use_spectral_norm,
                    use_san=self.use_san,
                )
            )
        if not branches:
            raise ValueError(
                "A discriminator needs at least one branch; the scale branch, "
                "the periods and the resolutions were all turned off."
            )
        self.discriminators = torch.nn.ModuleList(branches)

    @property
    def branch_labels(self) -> tuple:
        """One name per entry of ``discriminators``, in the same order.

        Built from what ``__init__`` actually assembled rather than from a
        preset, so a config that turns a family off or replaces a period set
        still gets labels that line up with the branches -- which is the whole
        point of having them, since they exist to name a per-branch diagnostic.
        """

        labels = ["msd"] if self.use_msd else []
        labels += [f"period_{p}" for p in self.periods]
        labels += [f"resolution_{n_fft}" for n_fft, _, _ in self.resolutions]
        if self.use_univhd:
            labels.append("univhd")
        return tuple(labels)

    @property
    def branch_weights(self) -> tuple:
        """One loss weight per entry of ``discriminators``, in the same order.

        ``None`` is not an option here even when every weight is 1.0: the
        losses take this as a positional list and a tuple that is one entry
        short would silently weight the wrong branches.  Built from the same
        assembly ``branch_labels`` walks, so the two cannot drift apart.

        The generator's adversarial term, the feature-matching term and the
        discriminator's own loss all take these.  See
        ``UNIVHD_WEIGHT_BY_VERSION`` for UnivHD's and the measurements behind
        it; the scale branch takes ``msd_weight`` (``d_msd_weight``).
        """

        count = len(self.discriminators)
        weights = [DEFAULT_BRANCH_WEIGHT] * count
        if self.use_msd:
            # The scale branch is built first, as ``branch_labels`` records.
            weights[0] = self.msd_weight
        if self.use_univhd:
            # UnivHD appends itself last -- see ``__init__`` -- which is also
            # what ``branch_labels`` records.
            weights[-1] = self.univhd_weight
        return tuple(weights)

    @property
    def uses_branch_weights(self) -> bool:
        """Whether any branch is weighted away from 1.0.

        Lets a caller skip passing the weights entirely on the common case, so
        an unweighted run builds exactly the graph it built before this
        existed.
        """

        return any(w != DEFAULT_BRANCH_WEIGHT for w in self.branch_weights)

    def enable_compile(self, mode: str = "default") -> bool:
        """Compile the paired real/fake forward, replacing ``forward`` in place.

        ``train.py`` looks this method up with ``getattr(model, "enable_compile",
        None)`` and silently reports "not supported" when it is missing, which
        is what ``compile_discriminator`` did for every run after the ChouwaGAN
        discriminator -- the only class that ever defined it -- was removed.

        Worth compiling: at batch 1 over 0.4 s this forward dispatches 558 ATen
        ops, and the training step runs it twice (once for the discriminator
        update, once with ``no_grad_real`` for the generator's), so 1116 of the
        ~2283 the step still spends outside the compiled decoder are here.  Of
        those 558, 132 are ``_weight_norm_interface`` -- the parametrisation
        recomputing ``g * v / ||v||`` for every convolution on every forward,
        which is exactly what a fused graph stops paying separately.

        ``no_grad_real`` and ``combine_inputs`` are Python ``bool``s, so Dynamo
        guards on them and keeps one graph per combination rather than
        branching inside any of them.  Checkpointing
        is the case that is *not* compiled: it is a fallback for a card that
        cannot hold the activations, and pairing it with compilation trades a
        known-good path for an untested one.
        """

        if getattr(self, "_compile_enabled", False):
            return getattr(self, "_compile_mode", mode) == mode

        eager_forward = self.forward
        try:
            compiled_forward = torch.compile(eager_forward, dynamic=False, mode=mode)
        except Exception as error:
            warning(
                f"{DISCRIMINATOR_COMPILE_ENABLE_FAILED} {error}\n"
                f"{traceback.format_exc()}",
                tag="[INIT]",
            )
            return False
        compile_failed = False

        def training_forward(*args, **kwargs):
            nonlocal compile_failed
            if not self.training or compile_failed or self.use_checkpointing:
                return eager_forward(*args, **kwargs)
            try:
                return compiled_forward(*args, **kwargs)
            except Exception as error:
                compile_failed = True
                # The traceback, for the same reason the decoder's fallback
                # keeps one: this fires once and then stays eager, so without
                # it the run reports a failure nobody can act on.
                warning(
                    f"{DISCRIMINATOR_COMPILE_RUNTIME_FAILED} {error}\n"
                    f"{traceback.format_exc()}",
                    tag="[TRAIN]",
                )
                return eager_forward(*args, **kwargs)

        self.forward = training_forward
        self._compile_enabled = True
        self._compile_mode = mode
        return True

    @property
    def spectral_branch_indices(self) -> tuple:
        """Indices of the spectrogram and UnivHD branches, in branch order."""
        return tuple(
            index
            for index, label in enumerate(self.branch_labels)
            if label.startswith("resolution_") or label == "univhd"
        )

    def _cond_kwargs(self, index, cond):
        return {"cond": cond} if index in self.cond_branches else {}

    def set_mel_cond_scale(self, scale: float) -> None:
        """Scale of the conditioned branches' mel projection, 0 to 1."""
        for index in self.cond_branches:
            self.discriminators[index].mel_scale.fill_(float(scale))

    def real_score(self, x, index, cond=None):
        """Per-sample R1 score of branch ``index``: its weighted mean logit.

        Calls the branch directly, so it stays eager under ``enable_compile``
        -- the penalty needs a double backward the compiled graph cannot give.
        Under SAN it reads the function output, the one the generator is
        scored by.
        """
        logits, _ = self.discriminators[index](x, **self._cond_kwargs(index, cond))
        return self.branch_weights[index] * logits.float().mean(dim=1)

    def forward(
        self,
        y,
        y_hat,
        no_grad_real: bool = False,
        san_training: bool = False,
        combine_inputs: bool = False,
        extra=None,
        extra_branches=(),
        cond=None,
    ):
        """``no_grad_real`` runs the real branch under ``no_grad``.

        The generator update needs the real side only as a *target*: its logits
        are thrown away and its feature maps are the constant the feature
        matching loss measures against.  Left differentiable it still builds a
        full activation graph, and the feature loss then backwards through it
        into discriminator weights whose gradients are zeroed before they are
        ever stepped -- the generator update runs after the discriminator's,
        and ``optim_d.zero_grad`` brackets it on both sides.  So the whole real
        backward is work with no consumer.

        Off by default because the discriminator update *does* need it: that is
        the pass whose gradient trains ``net_d``.

        ``combine_inputs`` runs the real and the fake side as one batch of
        ``2B`` instead of two batches of ``B``.  Nothing in any branch mixes
        samples -- there is no batch normalisation here, and ``weight_norm``,
        the STFTs, the harmonic bank and ``san_tail`` are all per-sample -- so
        the outputs are the same numbers, reached in half the kernel launches
        and with one ``weight_norm`` recompute per convolution instead of two.
        Measured on an RTX 5060 at batch 8 over 0.4 s, discriminator update
        only, fwd+bwd: v2 124 -> 113 ms, v3 161 -> 150, v4 149 -> 136, v4 with
        SAN 151 -> 140, all at peak VRAM within 1% of the paired path
        (``torch.split`` returns views, and the real side's activations were
        already being held alive across the fake pass).  Checkpointing is the
        one case that trades memory for the time: 194 -> 178 ms at +360 MiB,
        because one checkpoint boundary now holds a ``2B`` input instead of two
        boundaries holding ``B`` each.

        Upstream pairs this with ``parametrize.cached()``.  That was measured
        here too and is worth nothing once the passes are batched (-0.0%,
        because batching already halves the ``weight_norm`` recomputes) while
        holding every materialised weight alive for the whole forward
        (+250 MiB on v2), so it is not used.

        It is mutually exclusive with ``no_grad_real`` -- half a batch cannot
        be under ``no_grad`` -- which is why only the discriminator update
        asks for it, and it is off under spectral norm for the reason given at
        ``self.use_spectral_norm``.

        ``extra`` is a third batch run through the branches in
        ``extra_branches`` only; their logits come back as a fifth element, one
        per listed branch.  With ``combine_inputs`` it joins those branches'
        batch, so it adds no kernel launches.

        ``cond`` is the generator's input mel, shared by ``y`` and ``y_hat``,
        for the branches in ``cond_branches``.
        """
        if self.cond_branches and cond is None:
            raise ValueError("This discriminator is mel-conditioned; pass cond.")
        if extra is not None and self.cond_branches.intersection(extra_branches):
            raise ValueError("The extra batch has no mel for the conditioned branches.")
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        extra_outputs = []
        checkpointing = self.training and self.use_checkpointing
        # Only the discriminator update asks for the direction output.  In the
        # generator's pass the direction is not something the generator may
        # move, so requesting it would build a graph with no consumer -- the
        # same waste ``no_grad_real`` exists to avoid.
        san = bool(san_training) and self.use_san
        combined = combine_inputs and not no_grad_real and not self.use_spectral_norm

        if combined:
            paired = torch.cat((y, y_hat), dim=0)
            paired_cond = None if cond is None else torch.cat((cond, cond), dim=0)
            for index, d in enumerate(self.discriminators):
                with_extra = extra is not None and index in extra_branches
                batch = torch.cat((paired, extra), dim=0) if with_extra else paired
                sizes = (y.shape[0], y_hat.shape[0]) + (
                    (extra.shape[0],) if with_extra else ()
                )
                kwargs = self._cond_kwargs(index, paired_cond)
                if checkpointing:
                    y_d, fmap = checkpoint(
                        d, batch, san_training=san, use_reentrant=False, **kwargs
                    )
                else:
                    y_d, fmap = d(batch, san_training=san, **kwargs)
                # Under SAN a branch returns ``[function, direction]`` rather
                # than one tensor, and both halves have to be split.
                if isinstance(y_d, (list, tuple)):
                    split = [torch.split(part, sizes, dim=0) for part in y_d]
                    y_d_r = [part[0] for part in split]
                    y_d_g = [part[1] for part in split]
                    if with_extra:
                        extra_outputs.append([part[2] for part in split])
                else:
                    split = torch.split(y_d, sizes, dim=0)
                    y_d_r, y_d_g = split[0], split[1]
                    if with_extra:
                        extra_outputs.append(split[2])
                fmap_r, fmap_g = [], []
                for feature in fmap:
                    if isinstance(feature, tuple):
                        parts = [torch.split(band, sizes, dim=0) for band in feature]
                        fmap_r.append(tuple(part[0] for part in parts))
                        fmap_g.append(tuple(part[1] for part in parts))
                        continue
                    parts = torch.split(feature, sizes, dim=0)
                    fmap_r.append(parts[0])
                    fmap_g.append(parts[1])

                y_d_rs.append(y_d_r)
                y_d_gs.append(y_d_g)
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

            if extra is not None:
                return y_d_rs, y_d_gs, fmap_rs, fmap_gs, extra_outputs
            return y_d_rs, y_d_gs, fmap_rs, fmap_gs

        for index, d in enumerate(self.discriminators):
            kwargs = self._cond_kwargs(index, cond)
            # The other two arms add no context manager at all, and not
            # ``enable_grad``: an outer ``no_grad`` (the validation path) must
            # stay in force.
            if no_grad_real:
                with torch.no_grad():
                    y_d_r, fmap_r = d(y, san_training=san, **kwargs)
            elif checkpointing:
                y_d_r, fmap_r = checkpoint(
                    d, y, san_training=san, use_reentrant=False, **kwargs
                )
            else:
                y_d_r, fmap_r = d(y, san_training=san, **kwargs)

            if checkpointing:
                y_d_g, fmap_g = checkpoint(
                    d, y_hat, san_training=san, use_reentrant=False, **kwargs
                )
            else:
                y_d_g, fmap_g = d(y_hat, san_training=san, **kwargs)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        if extra is not None:
            extra_outputs = [
                self.discriminators[i](extra, san_training=san)[0]
                for i in extra_branches
            ]
            return y_d_rs, y_d_gs, fmap_rs, fmap_gs, extra_outputs
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """Multi-scale discriminator branch, operating directly on the waveform."""

    #: Set by ``MPD_MSD_Combined``; see ``waveform_conv``.
    wave_fp32_input = False

    def __init__(self, use_spectral_norm: bool = False, use_san: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.use_san = bool(use_san)
        # No ``norm_f`` on a SAN head: it normalises its own weight, and the two
        # reparametrisations would fight over the same tensor.
        self.conv_post = (
            SANConv1d(1024, 1, 3, 1, padding=1)
            if self.use_san
            else norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        )
        # In place on each conv's own output: autograd keeps one tensor per
        # layer instead of two.
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x, san_training: bool = False):
        fmap = []
        for index, conv in enumerate(self.convs):
            x = self.lrelu(conv(x) if index else waveform_conv(conv, x, self.wave_fp32_input))
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)


def waveform_conv(conv, x, fp32: bool):
    """A branch's first conv, on the waveform; out of autocast when ``fp32``,
    where BF16 would keep 8 bits of each sample."""
    if not fp32:
        return conv(x)
    with torch.autocast(x.device.type, enabled=False):
        return conv(x.float())


def pre_emphasize(x, coefficient):
    """``x[t] - coefficient * x[t-1]``; the waveform is otherwise dominated by
    its low band, which leaves a period branch nearly blind above a few kHz."""
    if not coefficient:
        return x
    return torch.cat((x[..., :1], x[..., 1:] - coefficient * x[..., :-1]), dim=-1)


class DiscriminatorP(torch.nn.Module):
    """Multi-period discriminator branch: reshapes the waveform onto a period-`p` grid."""

    #: Set by ``MPD_MSD_Combined``; see ``waveform_conv``.
    wave_fp32_input = False

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        use_san: bool = False,
        pre_emphasis: float = 0.0,
    ):
        super().__init__()
        self.period = period
        self.pre_emphasis = float(pre_emphasis)
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(1024, 1, (3, 1), 1, padding=(1, 0))
            if self.use_san
            else norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        )
        # In place on each conv's own output: autograd keeps one tensor per
        # layer instead of two.
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x, san_training: bool = False):
        fmap = []
        x = pre_emphasize(x, self.pre_emphasis)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for index, conv in enumerate(self.convs):
            x = self.lrelu(conv(x) if index else waveform_conv(conv, x, self.wave_fp32_input))
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)


class FastDiscriminatorP(torch.nn.Module):
    """A period branch at a fraction of ``DiscriminatorP``'s width.

    Same reshape and the same six feature maps; the channel schedule is
    ``32, 64, 128, 256`` capped rather than ``32, 128, 512, 1024, 1024``, and
    there are four strided layers instead of five plus a stride-1 layer at the
    end.  Ported from KazeFlow's ChouwaGAN discriminator.

    Why it is worth having: the period family is 32.88 M of the 38.81 M
    parameters in a ``v4`` discriminator, so this is where the memory is.  At
    32 kHz over the shipped period set, 300 steps / 3 seeds, held-out accuracy
    at separating real audio from one defect:

        defect             stock (32.88 M)   this at 256 (2.18 M)
        >9 kHz shelf loss   66.2 +- 8.2       62.1 +- 13.1  (at 128)
        f0 jitter           74.2 +- 6.2       71.7 +-  5.2  (at 128)
        slow dynamics       72.1 +- 3.9       70.4 +-  3.3  (at 128)

    Every gap is smaller than at least one side's seed spread.  A capacity
    sweep on the same probe was flat from 128 to 512 channels (78.5 / 80.0 /
    80.0 / 80.6 on frame-rate AM against the stock schedule's 83.1), which is
    why 256 is the shipped width: it is the knee, not a compromise.

    What the probe cannot see, stated because it is the reason this is opt-in:
    it measures detection of a fixed defect by a freshly trained branch, not
    whether a 15x smaller adversary keeps providing gradient against a
    generator adapting to it for 100k steps.  The failure mode there is
    saturation, not blindness.  It also changes ``loss_fm``'s scale -- six maps
    at <=256 channels instead of <=1024 -- which the feature-matching governor,
    the adversarial ceiling and the per-branch R1 all read.
    """

    #: Set by ``MPD_MSD_Combined``; see ``waveform_conv``.
    wave_fp32_input = False

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        channels: int = 32,
        max_channels: int = 256,
        n_layers: int = 4,
        use_spectral_norm: bool = False,
        use_san: bool = False,
        pre_emphasis: float = 0.0,
    ):
        super().__init__()
        self.period = period
        self.pre_emphasis = float(pre_emphasis)
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList()
        in_channels = 1
        for layer in range(n_layers):
            out_channels = min(channels * (2 ** layer), max_channels)
            self.convs.append(
                norm_f(
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
            )
            in_channels = out_channels

        self.conv_final = norm_f(
            torch.nn.Conv2d(
                in_channels,
                in_channels,
                (kernel_size, 1),
                1,
                padding=(get_padding(kernel_size, 1), 0),
            )
        )
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(in_channels, 1, (3, 1), 1, padding=(1, 0))
            if self.use_san
            else norm_f(torch.nn.Conv2d(in_channels, 1, (3, 1), 1, padding=(1, 0)))
        )
        # ``LRELU_SLOPE`` and not KazeFlow's 0.1: this is a capacity swap, and
        # an activation that differs from the branch it replaces would make it
        # two changes wearing one name.  In place, as in ``DiscriminatorP``.
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x, san_training: bool = False):
        fmap = []
        x = pre_emphasize(x, self.pre_emphasis)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for index, conv in enumerate(self.convs):
            x = self.lrelu(conv(x) if index else waveform_conv(conv, x, self.wave_fp32_input))
            fmap.append(x)
        x = self.lrelu(self.conv_final(x))
        fmap.append(x)
        return san_tail(self, x, fmap, san_training)


class DiscriminatorR(torch.nn.Module):
    """Multi-resolution spectrogram discriminator (Applio's, verbatim).

    A period branch reshapes the waveform and looks at it in the time domain, so
    a defect that is narrow in frequency and stationary in time -- an image, a
    tonal artefact, a missing band -- is spread across its receptive field and
    barely visible.  This branch takes the STFT magnitude at three resolutions
    instead, which is where such a defect is a single bright or missing line.

    The two full-window branches use a Hann: a boxcar's -13 dB sidelobes fill
    the inter-harmonic valleys in the real and the generated input alike, which
    hides the one defect those branches are here to catch.  The fine-hop branch
    reads modulation in time and keeps the boxcar.
    """

    def __init__(
        self,
        resolution,
        use_spectral_norm: bool = False,
        frequency_strides=(1, 1, 1),
        use_san: bool = False,
        fp32_input: bool = True,
    ):
        super().__init__()

        self.resolution = resolution
        self.fp32_input = bool(fp32_input)
        self.lrelu_slope = 0.1
        self.frequency_strides = tuple(int(s) for s in frequency_strides)
        if len(self.frequency_strides) != 3:
            raise ValueError(
                "DiscriminatorR has three strided layers; "
                f"received {len(self.frequency_strides)} frequency strides."
            )
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList(
            [norm_f(torch.nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))]
            + [
                norm_f(
                    torch.nn.Conv2d(32, 32, (3, 9), stride=(s, 2), padding=(1, 4))
                )
                for s in self.frequency_strides
            ]
            + [norm_f(torch.nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))]
        )
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(32, 1, (3, 3), padding=(1, 1))
            if self.use_san
            else norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))
        )

        # ``win_length`` is fixed at construction, so the window is a constant
        # and was being rebuilt on every call -- three resolution branches,
        # both the real and the fake pass, and both the discriminator and the
        # generator update, i.e. twelve allocations of the same vector per
        # training step.  Non-persistent so no checkpoint gains a key.
        #
        # Hann only where the window spans the whole transform: on the short
        # temporal branch it doubles a mainlobe already wider than f0 and takes
        # its frequency contrast to 0.3 dB.  That branch's job is the hop.
        n_fft, _hop, win_length = self.resolution
        self.register_buffer(
            "window",
            torch.hann_window(int(win_length))
            if int(win_length) == int(n_fft)
            else torch.ones(int(win_length)),
            persistent=False,
        )

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        # Floored for the same reason as ``UnivHDDiscriminator.spectrogram``;
        # in FP32 because the floor underflows in FP16.
        x = torch.view_as_real(x).float()
        return torch.sqrt(x.square().sum(dim=-1) + 1e-8)

    def forward(self, x, san_training: bool = False):
        fmap = []
        if self.fp32_input:
            # The magnitude is linear and unnormalised -- a full-scale sine
            # reads 120 to 512 here, against the [-1, 1] every period branch
            # sees -- so under FP16 autocast this
            # stage is where the activations and the weight gradient of the
            # first conv run out of range first, and at a raised learning rate
            # the branch diverges.  Only the STFT and ``convs[0]`` leave
            # autocast: after one conv and a LeakyReLU the scale is the other
            # branches', and the rest goes back to FP16.  Measured on an RTX
            # 5060, v4 + UnivHD + SAN at batch 8 over 0.4 s, D update plus the
            # generator's pass through D: 140.1 -> 148.5 ms, +0.8 GiB peak;
            # the whole branch in FP32 was 200.6 ms, +1.2 GiB.
            with torch.autocast(x.device.type, enabled=False):
                x = self.spectrogram(x.float()).unsqueeze(1)
                x = F.leaky_relu(self.convs[0](x), self.lrelu_slope, inplace=True)
        else:
            x = self.spectrogram(x).unsqueeze(1)
            x = F.leaky_relu(self.convs[0](x), self.lrelu_slope, inplace=True)
        fmap.append(x)
        # In place on each conv's own output: autograd keeps one tensor per
        # layer instead of two.
        for layer in self.convs[1:]:
            x = F.leaky_relu(layer(x), self.lrelu_slope, inplace=True)
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)


def mel_to_linear_bins(n_bins, sample_rate, n_mels, fmin, fmax):
    """(n_bins, n_mels) matrix interpolating a log mel between band centres
    onto the linear STFT bins, held flat past the first and last centre."""
    import librosa
    import numpy as np

    centres = librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]
    freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    eye = np.eye(n_mels)
    matrix = np.stack([np.interp(freqs, centres, eye[j]) for j in range(n_mels)], axis=1)
    return torch.from_numpy(matrix).float()


def _keep_new_mel_embed(
    module, state_dict, prefix, local_metadata, strict, missing_keys,
    unexpected_keys, error_msgs,
):
    """Load hook: a branch saved without mel conditioning keeps its fresh
    projection, whose small starting scale barely moves the loaded scores."""
    for key, value in module.state_dict().items():
        if key.startswith("mel_embed.") or key == "mel_proj_logit":
            state_dict.setdefault(prefix + key, value)


class MultiBandDiscriminatorR(torch.nn.Module):
    """Spectrogram branch on a compressed complex STFT, one conv stack per band.

    Three input channels: log magnitude, and the real and imaginary parts of
    the STFT with its magnitude raised to ``compression`` (phase kept).  The
    linear magnitude ``DiscriminatorR`` reads leaves the noise floor and the
    upper bands numerically near zero, and it has no phase at all.  The
    frequency axis is split into ``BANDS`` (fractions of the bins), each with
    its own stack, as in DAC's MRD, so the low band cannot claim every filter.
    ``frequency_strides`` and the window rule are ``DiscriminatorR``'s.

    ``mel_cond`` (``sample_rate``, ``n_mels``, ``fmin``, ``fmax``) conditions
    the branch on the generator's input mel by projection: the mel, resampled
    onto this branch's bins and frames, goes through a small stack with the
    bands' geometry, and the result is a per-location direction, normalised
    and scaled like SAN's ``conv_post``, whose inner product with the last
    feature map is added to the logits.  Under SAN the function term reads it
    detached and the direction term reads the features detached, so the
    conditioning cannot inflate the logits past SAN's bound.  The scale starts
    near zero, so the branch starts unconditional.  ``forward`` then needs
    ``cond`` [batch, n_mels, frames].
    """

    #: Width of the mel embedding stack.
    MEL_EMBED_CHANNELS = 16
    #: Projection scale = ``SCALE_MAX * sigmoid(mel_proj_logit)``: bounded like
    #: SAN's, and a sigmoid rather than a clamp so it cannot stick at a bound.
    MEL_PROJ_SCALE_MAX = 4.0
    MEL_PROJ_LOGIT_INIT = -4.0

    BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
    # Power floor: keeps log and the compression gain finite in silence,
    # about 94 dB under a full-scale sine at these magnitudes.
    POWER_EPS = 1e-4

    def __init__(
        self,
        resolution,
        channels: int = 32,
        use_spectral_norm: bool = False,
        use_san: bool = False,
        compression: float = 0.3,
        frequency_strides=(1, 1, 1),
        mel_cond: Optional[dict] = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.compression = float(compression)
        self.lrelu_slope = 0.1
        self.frequency_strides = tuple(int(s) for s in frequency_strides)
        if len(self.frequency_strides) != 3:
            raise ValueError(
                "MultiBandDiscriminatorR has three strided layers; "
                f"received {len(self.frequency_strides)} frequency strides."
            )
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        n_fft, _hop, win_length = self.resolution
        n_bins = int(n_fft) // 2 + 1
        self.band_edges = tuple(
            (int(round(lo * n_bins)), int(round(hi * n_bins))) for lo, hi in self.BANDS
        )
        self.mel_cond = dict(mel_cond) if mel_cond else None

        def band_stack(in_channels, width, last):
            # The layers the bands share, so a stack built with the same
            # strides lands on the same grid as the band's last feature map.
            return [norm_f(torch.nn.Conv2d(in_channels, width, (3, 9), padding=(1, 4)))] + [
                norm_f(torch.nn.Conv2d(width, width, (3, 9), stride=(s, 2), padding=(1, 4)))
                for s in self.frequency_strides
            ] + [last]

        self.bands = torch.nn.ModuleList(
            torch.nn.ModuleList(
                band_stack(
                    3, channels,
                    norm_f(torch.nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
                )
            )
            for _ in self.BANDS
        )
        if self.mel_cond:
            self.register_buffer(
                "mel_to_bins",
                mel_to_linear_bins(n_bins, **self.mel_cond),
                persistent=False,
            )
            width = self.MEL_EMBED_CHANNELS
            stacks = []
            for _ in self.BANDS:
                last = torch.nn.Conv2d(width, channels, (3, 3), padding=(1, 1))
                stacks.append(torch.nn.ModuleList(band_stack(1, width, last)))
            self.mel_embed = torch.nn.ModuleList(stacks)
            self.mel_proj_logit = torch.nn.Parameter(
                torch.tensor(self.MEL_PROJ_LOGIT_INIT)
            )
            # Scales the projection; ``set_mel_cond_scale`` can ramp it in.
            self.register_buffer("mel_scale", torch.ones(()), persistent=False)
            self.register_load_state_dict_pre_hook(_keep_new_mel_embed)
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(channels, 1, (3, 3), padding=(1, 1))
            if self.use_san
            else norm_f(torch.nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))
        )
        self.register_buffer(
            "window",
            torch.hann_window(int(win_length))
            if int(win_length) == int(n_fft)
            else torch.ones(int(win_length)),
            persistent=False,
        )

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        # Written through the power rather than ``abs`` so the gradient stays
        # finite at a zero bin.
        power = x.real.square() + x.imag.square() + self.POWER_EPS
        gain = power ** ((self.compression - 1.0) / 2.0)
        return torch.stack(
            (0.5 * torch.log(power), x.real * gain, x.imag * gain), dim=1
        )

    def forward(self, x, san_training: bool = False, cond=None):
        # Only the STFT leaves autocast; after compression the input is in the
        # range the other branches see.
        with torch.autocast(x.device.type, enabled=False):
            x = self.spectrogram(x.float())
            if self.mel_cond:
                if cond is None:
                    raise ValueError("This branch is mel-conditioned; pass cond.")
                # Both framings centre frame t near (t + 0.5) * hop, so a
                # linear resize lines them up to within a fraction of a frame.
                cond = F.interpolate(
                    cond.float(), size=x.shape[-1], mode="linear", align_corners=False
                )
                mel = (self.mel_to_bins @ cond).unsqueeze(1)
        layers = [[] for _ in self.bands[0]]
        for (lo, hi), stack in zip(self.band_edges, self.bands):
            h = x[:, :, lo:hi]
            for index, layer in enumerate(stack):
                # In place on the conv's own output, as in ``DiscriminatorR``.
                h = F.leaky_relu(layer(h), self.lrelu_slope, inplace=True)
                layers[index].append(h)
        # One entry per layer, a tuple of its band maps: ``feature_loss`` takes
        # their joint mean, so the branch weighs what ``DiscriminatorR`` does
        # without copying the activations into one tensor.
        fmap = [tuple(maps) for maps in layers]
        features = torch.cat(layers[-1], dim=2)
        projection = None
        if self.mel_cond:
            directions = []
            for (lo, hi), stack in zip(self.band_edges, self.mel_embed):
                e = mel[:, :, lo:hi]
                for layer in stack[:-1]:
                    e = F.leaky_relu(layer(e), self.lrelu_slope, inplace=True)
                directions.append(stack[-1](e))
            with torch.autocast(features.device.type, enabled=False):
                h = features.float()
                e = torch.cat(directions, dim=2).float()
                # Unit norm over the channels at every location.
                e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-12)
                scale = (
                    self.MEL_PROJ_SCALE_MAX
                    * torch.sigmoid(self.mel_proj_logit.float())
                    * self.mel_scale
                )
                if self.use_san and san_training:
                    # SAN's split: the function term moves the scale and the
                    # trunk, the direction term moves only the embedding.
                    projection = (
                        scale * (e.detach() * h).sum(1, keepdim=True),
                        scale.detach() * (e * h.detach()).sum(1, keepdim=True),
                    )
                else:
                    projection = scale * (e * h).sum(1, keepdim=True)
        return san_tail(self, features, fmap, san_training, projection)
