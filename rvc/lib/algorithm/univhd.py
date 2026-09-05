"""UnivHD -- the Universal Harmonic Discriminator, arXiv 2512.03486.

A single branch, built to be *added* to an existing set rather than replace
one: the paper's headline configuration is HiFi-GAN with MS-STFT and UnivHD,
and its Table II reports the pair beating either alone.

What it is: an STFT magnitude re-indexed by *harmonic order* before any
convolution sees it.  The bank has one learnable triangular filter per
(harmonic order ``h``, center ``fc``) pair,

    nabla_h(f) = [1 - 2 |f - h*fc| / fbw]+

so the conv stack is handed ``[H, K, T]`` rather than ``[F, T]``: row ``h`` of
column ``k`` is the energy the frame put at the ``h``-th harmonic of ``fc_k``.
A harmonic series with a missing or smeared partial is then a defect along one
axis of the input, rather than a pattern the conv stack must assemble first.

Two design points, each the paper's answer to a specific alternative:

* **Fixed STFT window, not CQT.**  A real CQT resolves each bin at its own hop;
  the paper's objection is temporal asynchronisation, and its answer is one
  window with the log spacing moved into the filterbank.
* **Centers scale linearly in ``h``.**  ``h * fc`` with integer ``h`` is what
  puts *odd* harmonics on the grid; a constant-Q bank spaces geometrically and
  lands on octaves.  ``h = 0.5`` catches sub-harmonic and period-doubling
  behaviour -- creak, growl, octave errors in f0.
"""

import math

import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.san import SANConv2d, san_tail


#: Glasberg & Moore's equivalent rectangular bandwidth, as the paper writes
#: it: ``fbw ~= (0.1079 * fc + 24.7) / gamma``.  This is what makes the bank's
#: resolution dynamic -- 33 Hz at 80 Hz, 565 Hz at 5 kHz -- without a per-bin
#: hop, so a low harmonic is resolved sharply and a high one read as a band.
ERB_SLOPE = 0.1079
ERB_OFFSET = 24.7

#: Channel width of every conv in the stack, chosen to reproduce the paper's
#: 0.31M parameters.
CHANNELS = 32

#: Dilation rates of one MDC block, applied on the *frequency* axis.  The
#: paper gives 5x5 kernels at rates [1, 2, 4] and does not say which axis
#: carries the dilation; the structure this branch reads is inter-harmonic,
#: and dilating time as well would give a rate-4 kernel a 116 ms footprint.
DILATIONS = (1, 2, 4)

LRELU_SLOPE = 0.1


def harmonic_orders(harmonics: int, half_harmonic: bool = True):
    """``[0.5, 1, 2, ..., H]`` -- the paper's ``H`` orders plus the half."""

    orders = [0.5] if half_harmonic else []
    orders.extend(float(h) for h in range(1, int(harmonics) + 1))
    return orders


def center_frequencies(f_min: float, f_max: float, bins_per_octave: int):
    """Log-spaced centers of the *first* harmonic, ``fc_1 .. fc_K``.

    Only the first harmonic's centers are chosen; every other order reuses them
    scaled by ``h``.  ``f_max`` is where the caller has already applied the
    paper's Nyquist criterion ``fs / (2H)``, which is what keeps ``H * fc_K``
    below Nyquist -- the highest-order filter is the binding one, not the
    highest center.
    """

    if f_max <= f_min:
        raise ValueError(
            f"UnivHD needs f_max > f_min; received f_min={f_min}, f_max={f_max}. "
            "f_max is sample_rate / (2 * harmonics), so too many harmonics at a "
            "low sample rate collapses the bank."
        )
    octaves = math.log2(f_max / f_min)
    count = int(math.floor(octaves * int(bins_per_octave))) + 1
    return [f_min * 2.0 ** (k / float(bins_per_octave)) for k in range(count)]


class HarmonicFilterBank(torch.nn.Module):
    """The learnable triangular bank, ``|X|[B,F,T] -> [B,H,K,T]``.

    ``gamma`` is rebuilt into the filters on every forward rather than cached,
    because it is a parameter and the bank has to stay differentiable through
    it.  That costs one ``[H, K, F]`` tensor per call -- 1.4M floats at the
    defaults, against the spectrogram itself, so it does not show up.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        harmonics: int = 10,
        bins_per_octave: int = 24,
        f_min: float = 80.0,
        half_harmonic: bool = True,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        orders = harmonic_orders(harmonics, half_harmonic)
        # ``fs / (2H)``: the paper's Nyquist criterion.  It is stated on the
        # *centers*, so the top order H sits exactly at Nyquist and the half
        # harmonic sits an octave below the bottom center -- both deliberate.
        f_max = self.sample_rate / (2.0 * float(int(harmonics)))
        centers = center_frequencies(float(f_min), f_max, int(bins_per_octave))
        self.harmonics = int(harmonics)
        self.bins_per_octave = int(bins_per_octave)
        self.f_min = float(f_min)
        self.f_max = float(f_max)

        self.register_buffer(
            "orders", torch.tensor(orders, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "centers", torch.tensor(centers, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "bin_hz",
            torch.linspace(0.0, self.sample_rate / 2.0, self.n_fft // 2 + 1),
            persistent=False,
        )
        # ``gamma >= 1`` via ``1 + softplus``, initialised so gamma ~= 1: the
        # constraint is the paper's, and it is one-sided on purpose.  gamma
        # divides the bandwidth, so the bank may sharpen away from the
        # psychoacoustic law but never blur past it, which is what stops it
        # from degenerating into a handful of wide bands.
        self.gamma_raw = torch.nn.Parameter(torch.tensor(-6.0))

    @property
    def n_orders(self):
        return int(self.orders.numel())

    @property
    def n_filters(self):
        return int(self.centers.numel())

    def gamma(self):
        return 1.0 + F.softplus(self.gamma_raw)

    def filters(self):
        """``[H, K, F]``, triangular, rebuilt from the current ``gamma``."""

        # [H, K]: the actual center of every (order, filter) pair.
        centers = self.orders[:, None] * self.centers[None, :]
        bandwidth = (ERB_SLOPE * centers + ERB_OFFSET) / self.gamma()
        distance = (self.bin_hz[None, None, :] - centers[..., None]).abs()
        return torch.clamp(1.0 - 2.0 * distance / bandwidth[..., None], min=0.0)

    def forward(self, magnitude):
        # [B, F, T] -> [B, H, K, T].  One matmul over F, which is why the bank
        # is affordable at 11 x 115 filters: it is a single [HK, F] x [F, T].
        weights = self.filters()
        flat = weights.reshape(-1, weights.shape[-1])
        out = torch.matmul(flat, magnitude)
        return out.reshape(
            magnitude.shape[0], weights.shape[0], weights.shape[1], -1
        )


class _HybridConvBlock(torch.nn.Module):
    """The paper's HCB: a depthwise-separable path and a plain path, concatenated.

    The depthwise conv has one kernel per harmonic order and cannot mix them,
    so it reads *intra*-harmonic structure; the plain conv sees every order at
    once and reads *inter*-harmonic structure.  The ablation says removing the
    depthwise path costs MCD and F0RMSE while removing the plain path costs
    PESQ, so they are not redundant.
    """

    def __init__(self, in_channels: int, channels: int, norm_f):
        super().__init__()
        self.depthwise = norm_f(
            torch.nn.Conv2d(
                in_channels, in_channels, (7, 7), padding=(3, 3), groups=in_channels
            )
        )
        self.pointwise = norm_f(torch.nn.Conv2d(in_channels, channels, (1, 1)))
        self.standard = norm_f(
            torch.nn.Conv2d(in_channels, channels, (7, 7), padding=(3, 3))
        )
        # The concatenation is 2 * channels wide and every MDC below is
        # ``channels`` wide; the paper's parameter count only works if the two
        # meet, so they meet here rather than by widening the first MDC.
        self.project = norm_f(torch.nn.Conv2d(2 * channels, channels, (1, 1)))

    def forward(self, x):
        separable = self.pointwise(F.leaky_relu(self.depthwise(x), LRELU_SLOPE))
        standard = self.standard(x)
        joined = torch.cat([separable, standard], dim=1)
        return self.project(F.leaky_relu(joined, LRELU_SLOPE))


class _MultiScaleDilatedBlock(torch.nn.Module):
    """Three dilated 5x5 convs summed, then a stride-(2,1) conv.

    The stride decimates *frequency* only.  That is the paper's, and it is also
    the one thing this branch must not get wrong in the other direction: the
    time axis is already coarse at hop ``hop_length``, and striding it would
    put this branch in the same trap that costs the 4096-point resolution
    branch its detection -- see the module docstring.
    """

    def __init__(self, in_channels: int, channels: int, norm_f):
        super().__init__()
        self.dilated = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_channels,
                        channels,
                        (5, 5),
                        dilation=(d, 1),
                        padding=(2 * d, 2),
                    )
                )
                for d in DILATIONS
            ]
        )
        # The three are summed, so they are one sparse 17x5 convolution.
        # Merging them is 13% more multiplies for one launch instead of three,
        # but ``weight_norm`` forces the merged weight to be rebuilt each call
        # and that costs more than the launches save: fwd+bwd 10.55 -> 11.46 ms.
        self.down = norm_f(
            torch.nn.Conv2d(channels, channels, (5, 5), stride=(2, 1), padding=(2, 2))
        )

    def forward(self, x):
        scales = sum(conv(x) for conv in self.dilated)
        return self.down(F.leaky_relu(scales, LRELU_SLOPE))


def _strided_size(size: int) -> int:
    return (int(size) + 2 * 2 - 5) // 2 + 1


class UnivHDDiscriminator(torch.nn.Module):
    """The full branch: STFT -> harmonic bank -> HCB -> 3x MDC -> score.

    Returns ``(score, fmap)`` like every other branch here, so
    ``MPD_MSD_Combined.forward`` and the feature-matching loss need to know
    nothing about it.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 256,
        harmonics: int = 10,
        bins_per_octave: int = 24,
        f_min: float = 80.0,
        channels: int = CHANNELS,
        half_harmonic: bool = True,
        use_spectral_norm: bool = False,
        use_san: bool = False,
    ):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.bank = HarmonicFilterBank(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            harmonics=harmonics,
            bins_per_octave=bins_per_octave,
            f_min=f_min,
            half_harmonic=half_harmonic,
        )
        # A real analysis window, unlike ``DiscriminatorR``'s deliberate boxcar:
        # the bank's triangles are only as selective as the bins they read, and
        # a rectangular window's leakage would fill in exactly the gaps between
        # harmonics that this branch is built to look at.
        self.register_buffer(
            "window", torch.hann_window(self.n_fft), persistent=False
        )

        channels = int(channels)
        self.hcb = _HybridConvBlock(self.bank.n_orders, channels, norm_f)
        self.mdc = torch.nn.ModuleList(
            [
                _MultiScaleDilatedBlock(channels, channels, norm_f)
                for _ in range(len(DILATIONS))
            ]
        )
        # Each MDC halves the frequency axis, so a bank with fewer filters than
        # the product of the strides collapses before the last block and the
        # dilated kernels above it are reading padding.  ``_strided_size``
        # floors at 1 rather than raising, so nothing downstream notices -- the
        # branch would build, run, and be a very expensive constant.
        if self.bank.n_filters < 2 ** len(self.mdc):
            raise ValueError(
                f"UnivHD's filterbank collapses: {self.bank.n_filters} filters "
                f"under {len(self.mdc)} stride-2 blocks. Raise "
                "bins_per_octave or lower f_min."
            )
        rows = self.bank.n_filters
        for _ in self.mdc:
            rows = _strided_size(rows)
        # "Final convolution layer kernel size matches MDC output frequency
        # dimension": the score is a 1-D series over time, not a map, so the
        # frequency axis is folded in one kernel rather than pooled.
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(channels, 1, (rows, 1))
            if self.use_san
            else norm_f(torch.nn.Conv2d(channels, 1, (rows, 1)))
        )
        self.output_rows = rows

    def spectrogram(self, x):
        window = self.window.to(device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        return torch.abs(spec)

    def forward(self, x, san_training: bool = False):
        fmap = []
        # ``torch.stft`` has no half precision path on every backend, and the
        # bank is a matmul over F where fp32 costs nothing measurable.
        x = self.bank(self.spectrogram(x.float()))
        x = self.hcb(x)
        fmap.append(x)
        for block in self.mdc:
            x = F.leaky_relu(block(x), LRELU_SLOPE)
            # The paper takes the feature-matching loss from each MDC output,
            # which is what these three entries are.
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)
