import math

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "chouwagan"
SOURCE_F0_FLOOR = 50.0
LOG_GAIN_RANGE = (-12.0, 6.0)
PHASES = ("mixed", "minimum", "zero")
INTERPOLATIONS = ("linear", "cubic")
# Scale of the band contrast input, from nepers to about [0, 1.5]
CONTRAST_SCALE = 4.0
# Below `phase_head_hz` the phase head's turn fades in over this much
PHASE_HEAD_RAMP_HZ = 1000.0
# Below `source_phase_hz` the harmonics' phases fade in over this much
SOURCE_PHASE_RAMP_HZ = 1000.0
SOURCE_PHASE_SEED = 0


def continuous_f0(f0: torch.Tensor):
    """
    Fill the unvoiced zeros of a pitch track linearly between voiced frames,
    holding the first and last voiced values past the ends.

    Args:
        f0 (torch.Tensor): Pitch of shape (batch, frames), 0 where unvoiced.
    """
    frames = f0.shape[-1]
    voiced = f0 > 0
    index = torch.arange(frames, device=f0.device).expand_as(f0)
    last = torch.where(voiced, index, -1).cummax(-1).values
    following = torch.where(voiced, index, frames).flip(-1).cummin(-1).values.flip(-1)
    f_last = f0.gather(-1, last.clamp(min=0))
    f_next = f0.gather(-1, following.clamp(max=frames - 1))
    weight = (index - last) / (following - last).clamp(min=1)
    return torch.where(
        (last >= 0) & (following < frames),
        f_last + (f_next - f_last) * weight,
        torch.where(
            last >= 0,
            f_last,
            torch.where(following < frames, f_next, torch.zeros_like(f0)),
        ),
    )


def band_centers(sample_rate: int, min_hz: float, ratio: float):
    """
    Band centres from 0 to Nyquist, `min_hz` apart at the bottom and a `ratio`
    of their frequency apart above, which keeps every gain's impulse response short.

    Args:
        sample_rate (int): Sampling rate.
        min_hz (float): Smallest spacing between centres.
        ratio (float): Spacing as a share of the centre's frequency.
    """
    nyquist = sample_rate / 2
    centers = [0.0]
    while centers[-1] < nyquist:
        centers.append(centers[-1] + max(min_hz, ratio * centers[-1]))
    centers[-1] = nyquist
    return torch.tensor(centers, dtype=torch.float64)


def interpolation_weights(centers: torch.Tensor, freqs: torch.Tensor):
    """
    For each of `freqs`, the indices of the `centers` either side and the weight
    of the upper one, held flat past both ends.

    Args:
        centers (torch.Tensor): Frequencies the values are given at, ascending.
        freqs (torch.Tensor): Frequencies to interpolate to.
    """
    freqs = freqs.clamp(centers[0], centers[-1])
    upper = torch.searchsorted(centers, freqs, right=True).clamp(1, len(centers) - 1)
    lower = upper - 1
    return lower, upper, (freqs - centers[lower]) / (centers[upper] - centers[lower])


def interpolation_matrix(centers: torch.Tensor, freqs: torch.Tensor):
    """
    Linear interpolation of values at `centers` to `freqs`, held flat past both
    ends, of shape (len(centers), len(freqs)).

    Args:
        centers (torch.Tensor): Frequencies the values are given at, ascending.
        freqs (torch.Tensor): Frequencies to interpolate to.
    """
    lower, upper, t = interpolation_weights(centers, freqs)
    matrix = torch.zeros(len(centers), len(freqs), dtype=torch.float64)
    columns = torch.arange(len(freqs))
    matrix[lower, columns] = 1 - t
    matrix[upper, columns] += t
    return matrix


def band_matrix(centers: torch.Tensor, n_fft: int, sample_rate: int):
    """
    Linear interpolation of band values to the bins of an FFT, of shape
    (bands, n_fft // 2 + 1).

    Args:
        centers (torch.Tensor): Band centres in Hz.
        n_fft (int): FFT size.
        sample_rate (int): Sampling rate.
    """
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64)
    return interpolation_matrix(centers, freqs)


def minimum_phase_basis(matrix: torch.Tensor, n_fft: int):
    """
    Each row of a log-magnitude basis as the complex log spectrum of its
    minimum-phase version. Folding the real cepstrum is linear, so a weighted
    sum of these rows is the minimum-phase log spectrum of the weighted sum of
    log magnitudes; its conjugate is the maximum-phase one.

    Args:
        matrix (torch.Tensor): Log-magnitude basis of shape (bands, bins).
        n_fft (int): FFT size.
    """
    cepstrum = torch.fft.irfft(matrix, n=n_fft, dim=-1)
    fold = torch.zeros(n_fft, dtype=cepstrum.dtype)
    fold[0] = 1.0
    fold[1 : n_fft // 2] = 2.0
    fold[n_fft // 2] = 1.0
    return torch.fft.rfft(cepstrum * fold, n=n_fft, dim=-1)


def to_steps(
    x: torch.Tensor, steps: int, step: int, frame: int, offset: int, mode: str = "linear"
):
    """
    Interpolation of values at the centres `n * frame + offset` to `steps`
    points `step` apart, linear or Catmull-Rom.

    Args:
        x (torch.Tensor): Values of shape (batch, channels, frames).
        steps (int): Number of points.
        step (int): Spacing of the points, in samples.
        frame (int): Spacing of the values, in samples.
        offset (int): Position of the first value, in samples.
        mode (str): "linear" or "cubic".
    """
    if mode not in INTERPOLATIONS:
        raise ValueError(f"mode must be one of {INTERPOLATIONS}, not {mode!r}.")
    count = x.shape[-1]
    position = ((torch.arange(steps, device=x.device) * step - offset) / frame).clamp(
        0, count - 1
    )
    lower = position.floor().long()
    upper = (lower + 1).clamp(max=count - 1)
    weight = (position - lower).to(x.dtype)
    if mode == "linear":
        return x[..., lower] * (1 - weight) + x[..., upper] * weight
    before = x[..., (lower - 1).clamp(min=0)]
    after = x[..., (lower + 2).clamp(max=count - 1)]
    a, b = x[..., lower], x[..., upper]
    return a + 0.5 * weight * (
        (b - before)
        + weight
        * ((2 * before - 5 * a + 4 * b - after) + weight * (3 * (a - b) + after - before))
    )


def cosine_sum(x: torch.Tensor, phase, first: torch.Tensor, last: torch.Tensor):
    """
    `sum(cos(k * x + phase) for k in first..last)` elementwise, in closed form;
    zero where `last < first`.

    Args:
        x (torch.Tensor): Angle.
        phase (float or torch.Tensor): Phase offset.
        first (torch.Tensor): First harmonic number.
        last (torch.Tensor): Last harmonic number.
    """
    count = (last - first + 1).clamp(min=0)
    x = torch.remainder(x + math.pi, 2 * math.pi) - math.pi
    half = torch.sin(x / 2)
    small = half.abs() < 1e-9
    kernel = torch.where(
        small,
        count,
        torch.sin(count * x / 2) / torch.where(small, torch.ones_like(half), half),
    )
    return kernel * torch.cos((first + last) * x / 2 + phase)



def known_contrast(contrast: torch.Tensor):
    """
    The vocoder's contrast input from band contrasts already divided by
    `CONTRAST_SCALE`: a channel of ones after them says they are known. Zeros,
    the absence of the input, read as unknown.

    Args:
        contrast (torch.Tensor): Band contrast of shape (batch, bands, frames).
    """
    return torch.cat((contrast, torch.ones_like(contrast[:, :1])), dim=1)


class BandContrast(nn.Module):
    """
    Per frame and band, the mean log contrast of the harmonics of the audio over
    the points halfway between them; 0 where a band has none.

    Args:
        sample_rate (int): Sampling rate.
        hop_length (int): Samples between frames.
        band_edges (list): Edges between the bands, in Hz.
        fft_size (int, optional): FFT size. Defaults to four frames.
        nyquist_margin (float, optional): Share of Nyquist the harmonics are read up to. Defaults to 0.95.
    """

    def __init__(
        self, sample_rate, hop_length, band_edges, fft_size=None, nyquist_margin=0.95
    ):
        super().__init__()
        fft_size = int(fft_size or 4 * hop_length)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fft_size = fft_size
        self.top = nyquist_margin * sample_rate / 2
        self.bin_hz = sample_rate / fft_size
        self.bands = len(band_edges) + 1
        self.register_buffer("window", torch.hann_window(fft_size), persistent=False)
        self.register_buffer(
            "band_edges",
            torch.tensor([float(hz) for hz in band_edges], dtype=torch.float32),
            persistent=False,
        )

    def log_spectrum(self, x, frames, hop):
        # Frames centred on the centres of the mel frames
        pad = self.fft_size // 2 - hop // 2
        x = F.pad(x, (pad, pad))
        spectrum = torch.stft(
            x,
            self.fft_size,
            hop,
            self.fft_size,
            self.window,
            center=False,
            return_complex=True,
        ).abs()
        return torch.log(spectrum[..., :frames] + 1e-5).transpose(1, 2)

    @torch.no_grad()
    def forward(self, audio, f0, hop=None, scale=1.0):
        """
        Args:
            audio (torch.Tensor): Audio of shape (batch, samples) or (batch, 1, samples).
            f0 (torch.Tensor): Pitch of the audio at the frame rate, shape (batch, frames).
            hop (int, optional): Samples between frames. Defaults to the mel's hop.
            scale (float, optional): Factor the pitch and formants of the audio are read as shifted up by. Defaults to 1.0.
        """
        hop = hop or self.hop_length
        with torch.autocast(audio.device.type, enabled=False):
            f0 = f0.float()
            frames = f0.shape[-1]
            voiced = f0 > 0
            if not voiced.any():
                return f0.new_zeros(f0.shape[0], self.bands, frames)
            audio = audio.reshape(audio.shape[0], -1).float()
            audio = F.pad(audio, (0, max(0, frames * hop - audio.shape[-1])))
            spectrum = self.log_spectrum(audio, frames, hop)
            size = spectrum.shape[-1]

            top = self.top / scale
            low = f0[voiced].min().clamp(min=SOURCE_F0_FLOOR / scale)
            harmonics = torch.arange(
                1, int(top / low) + 1, device=f0.device, dtype=f0.dtype
            )
            peaks = f0.unsqueeze(-1) * harmonics
            valid = voiced.unsqueeze(-1) & (peaks + 0.5 * f0.unsqueeze(-1) < top)

            # The peak is the largest bin within a quarter of f0 of the harmonic
            width = int(math.ceil(float(f0.max()) / 4 / self.bin_hz))
            offsets = torch.arange(-width, width + 1, device=f0.device)
            centre = torch.round(peaks / self.bin_hz).long()
            index = (centre.unsqueeze(-1) + offsets).clamp(0, size - 1)
            inside = offsets.abs() <= (f0 / 4 / self.bin_hz).unsqueeze(-1).unsqueeze(-1)
            peak = spectrum.gather(-1, index.flatten(2)).view(index.shape)
            peak = peak.masked_fill(~inside, -1e4).amax(-1)

            # The valley is read halfway to the next harmonic
            position = ((peaks + 0.5 * f0.unsqueeze(-1)) / self.bin_hz).clamp(
                0, size - 1
            )
            lower = position.floor().long().clamp(max=size - 2)
            weight = position - lower
            valley = (
                spectrum.gather(-1, lower) * (1 - weight)
                + spectrum.gather(-1, lower + 1) * weight
            )

            members = F.one_hot(
                torch.bucketize(peaks, self.band_edges / scale), self.bands
            ) * valid.unsqueeze(-1)
            members = members.to(f0.dtype)
            contrast = torch.einsum(
                "bth,bthn->bnt", (peak - valley).masked_fill(~valid, 0), members
            )
            return contrast / members.sum(2).transpose(1, 2).clamp(min=1)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block over frames.

    Args:
        dim (int): Channels.
        intermediate_dim (int): Channels of the pointwise expansion.
        kernel_size (int): Size of the depthwise convolution.
        layer_scale (float): Initial scale of the residual branch.
    """

    def __init__(self, dim, intermediate_dim, kernel_size, layer_scale):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(layer_scale)))

    def forward(self, x):
        y = self.dwconv(x).transpose(1, 2)
        y = self.pwconv2(F.gelu(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).transpose(1, 2)


class ChouwaGAN(nn.Module):
    """
    ChouwaGAN vocoder: a frame-rate ConvNeXt predicts per-band gains of a
    phase-aligned band-limited pulse train and of noise that follows the
    glottal cycle, both synthesised by overlap-add of short-time spectra.

    Args:
        sample_rate (int): Sampling rate.
        num_mels (int): Number of mel bins.
        hop_length (int): Samples per mel frame.
        contrast_band_edges (list, optional): Edges of the harmonic contrast bands it reads, None for no contrast input.
        max_phase_spread (float, optional): Largest random phase, in radians, the harmonics of a band take per synthesis step; 0 keeps them phase-aligned.
        phase_head_hz (float, optional): Frequency above which the network turns the phase of every harmonic bin, None for no phase head.
        phase_noise_mix (bool, optional): Whether the phase head also blends, per bin, the phase of a white noise's spectrogram into its turn.
        source_phase_hz (float, optional): Frequency from which each harmonic keeps a fixed random phase of its own, set per bin from the harmonic of f0 nearest to it; None keeps them phase-aligned.
        Other arguments are the model's settings, as the export stores them.
    """

    def __init__(
        self,
        sample_rate,
        num_mels,
        hop_length,
        dim=256,
        intermediate_dim=768,
        num_blocks=8,
        kernel_size=7,
        band_min_hz=100.0,
        band_ratio=0.03,
        glottal_band_min_hz=400.0,
        glottal_band_ratio=0.15,
        glottal_max_hz=None,
        harmonic_window=1024,
        harmonic_fft=2048,
        kaiser_beta=8.0,
        noise_window=512,
        noise_fft=1024,
        synthesis_hop=128,
        harmonic_substeps=2,
        harmonic_amp=0.01,
        taper_hz=2000.0,
        max_jitter=0.02,
        max_shimmer=0.1,
        max_f0_correction=20.0,
        phase="mixed",
        residual_bound=0.1,
        block_steps=4096,
        noise_curve_hz=None,
        aspiration_regions_hz=None,
        max_aspiration_sharpness=16.0,
        contrast_band_edges=None,
        max_phase_spread=0.0,
        phase_head_hz=None,
        phase_noise_mix=False,
        control_interpolation="linear",
        source_phase_hz=None,
    ):
        super().__init__()
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, not {phase!r}.")
        if control_interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"control_interpolation must be one of {INTERPOLATIONS}, "
                f"not {control_interpolation!r}."
            )
        self.control_interpolation = control_interpolation
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.synthesis_hop = int(synthesis_hop)
        self.noise_substeps = self.hop_length // self.synthesis_hop
        self.harmonic_substeps = int(harmonic_substeps)
        self.harmonic_fft = int(harmonic_fft)
        self.noise_fft = int(noise_fft)
        self.harmonic_amp = float(harmonic_amp)
        self.taper_hz = float(taper_hz)
        self.max_jitter = float(max_jitter)
        self.max_shimmer = float(max_shimmer)
        self.max_f0_correction = float(max_f0_correction) / 1200.0
        self.residual_bound = float(residual_bound)
        self.block_steps = int(block_steps)
        self.max_phase_spread = float(max_phase_spread)
        self.source_phase_hz = None
        if source_phase_hz is not None:
            self.source_phase_hz = float(source_phase_hz)
            generator = torch.Generator().manual_seed(SOURCE_PHASE_SEED)
            count = int(sample_rate / 2 / SOURCE_F0_FLOOR) + 2
            phases = math.pi * (2 * torch.rand(count, generator=generator) - 1)
            bins = torch.linspace(0, sample_rate / 2, self.harmonic_fft // 2 + 1)
            bins = bins.view(1, -1, 1)
            ramp = (bins - self.source_phase_hz) / SOURCE_PHASE_RAMP_HZ + 1
            self.register_buffer("source_phases", phases, persistent=False)
            self.register_buffer("source_bins", bins, persistent=False)
            self.register_buffer("source_ramp", ramp.clamp(0, 1), persistent=False)

        band_hz = band_centers(sample_rate, band_min_hz, band_ratio)
        tract = band_matrix(band_hz, self.harmonic_fft, sample_rate)
        self.bands = tract.shape[0]
        self.glottal_bands = 0
        if phase == "zero":
            tract_log = tract.to(torch.complex128)
        else:
            tract_log = minimum_phase_basis(tract, self.harmonic_fft)
        self.register_buffer(
            "tract_log", tract_log.to(torch.complex64), persistent=False
        )
        self.register_buffer("tract_bands", tract.float(), persistent=False)
        if self.max_phase_spread > 0:
            bins = torch.linspace(
                0, sample_rate / 2, self.harmonic_fft // 2 + 1, dtype=torch.float64
            )
            lower, upper, t = interpolation_weights(band_hz, bins)
            self.register_buffer("spread_lower", lower, persistent=False)
            self.register_buffer("spread_upper", upper, persistent=False)
            self.register_buffer(
                "spread_weight", t.float().view(1, -1, 1), persistent=False
            )
            # Overlap-add sums random phases in power: sum(w^4) / sum(w^2)^2 of
            # the frames over each sample, as a share of the coherent sum
            hop = self.synthesis_hop
            squares = torch.kaiser_window(
                int(harmonic_window),
                periodic=True,
                beta=kaiser_beta,
                dtype=torch.float64,
            )
            squares = F.pad(squares.square(), (0, -int(harmonic_window) % hop))
            squares = squares.view(-1, hop)
            rho = squares.square().sum(0) / squares.sum(0).square()
            self.spread_rho = float(rho.mean())
        self.phase_bins = 0
        if phase_head_hz is not None:
            bins = torch.linspace(0, sample_rate / 2, self.harmonic_fft // 2 + 1)
            ramp = (bins - float(phase_head_hz)) / PHASE_HEAD_RAMP_HZ + 1
            ramp = ramp.clamp(0, 1)
            self.phase_first_bin = int((ramp > 0).nonzero()[0])
            self.phase_bins = len(bins) - self.phase_first_bin
            self.register_buffer(
                "phase_ramp",
                ramp[self.phase_first_bin :].view(1, -1, 1),
                persistent=False,
            )
        self.phase_noise_mix = bool(phase_noise_mix) and self.phase_bins > 0
        if phase == "mixed":
            centers = band_centers(sample_rate, glottal_band_min_hz, glottal_band_ratio)
            glottal = band_matrix(centers, self.harmonic_fft, sample_rate)
            if glottal_max_hz is not None:
                glottal = glottal[centers <= float(glottal_max_hz)]
            self.glottal_bands = glottal.shape[0]
            glottal_log = minimum_phase_basis(glottal, self.harmonic_fft).conj()
            self.register_buffer(
                "glottal_log", glottal_log.to(torch.complex64), persistent=False
            )
            bins = torch.linspace(
                0, sample_rate / 2, self.harmonic_fft // 2 + 1, dtype=torch.float64
            )
            nearest = torch.searchsorted(bins, band_hz).clamp(max=len(bins) - 1)
            self.register_buffer(
                "glottal_at_bands", glottal[:, nearest].float(), persistent=False
            )
        self.curve_knots = 0
        if noise_curve_hz:
            knots = torch.tensor(
                sorted(float(hz) for hz in noise_curve_hz), dtype=torch.float64
            )
            self.curve_knots = len(knots)
            self.register_buffer(
                "curve_bands",
                interpolation_matrix(knots, band_hz).float(),
                persistent=False,
            )
        noise = band_matrix(band_hz, self.noise_fft, sample_rate)
        self.register_buffer("noise_bands", noise.float(), persistent=False)
        self.aspiration_regions = 0
        if aspiration_regions_hz:
            knots = torch.tensor(
                sorted(float(hz) for hz in aspiration_regions_hz), dtype=torch.float64
            )
            self.aspiration_regions = len(knots)
            self.max_aspiration_sharpness = float(max_aspiration_sharpness)
            bins = torch.linspace(
                0, sample_rate / 2, self.noise_fft // 2 + 1, dtype=torch.float64
            )
            # Each region draws its own noise, so amplitudes whose squares sum to one
            self.register_buffer(
                "aspiration_masks",
                interpolation_matrix(knots, bins).sqrt().float(),
                persistent=False,
            )
        self.contrast = (
            BandContrast(sample_rate, hop_length, contrast_band_edges)
            if contrast_band_edges
            else None
        )
        contrast_inputs = self.contrast.bands + 1 if self.contrast is not None else 0
        self.register_buffer(
            "harmonic_window",
            torch.kaiser_window(int(harmonic_window), periodic=True, beta=kaiser_beta),
            persistent=False,
        )
        self.register_buffer(
            "noise_window", torch.hann_window(int(noise_window)), persistent=False
        )
        # Where in the glottal cycle the aspiration peaks; the pulse is at 0
        self.aspiration_phase = nn.Parameter(
            torch.full((self.aspiration_regions,), math.pi)
            if self.aspiration_regions
            else torch.tensor(math.pi)
        )

        self.embed = nn.Conv1d(num_mels + 2 + contrast_inputs, dim, 7, padding=3)
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim, kernel_size, 1 / num_blocks)
            for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        # Tract, glottal and noise gains, the residual, the voiced noise curve,
        # then aspiration depth (and sharpness, per region), jitter, shimmer and
        # the f0 correction, then the phase spread per tract band
        self.splits = [
            self.bands * self.harmonic_substeps,
            self.glottal_bands,
            self.bands * self.noise_substeps,
            2 * self.bands if self.residual_bound > 0 else 0,
            self.curve_knots,
            2 * self.aspiration_regions + 3 if self.aspiration_regions else 4,
            self.bands if self.max_phase_spread > 0 else 0,
        ]
        self.head = nn.Linear(dim, sum(self.splits))
        # Each turned bin's (cos, sin), and with the noise mix its logit, per
        # harmonic sub-step
        self.phase_outputs = 3 if self.phase_noise_mix else 2
        self.phase_head = None
        if self.phase_bins:
            self.phase_head = nn.Linear(
                dim, self.phase_outputs * self.phase_bins * self.harmonic_substeps
            )

    def _controls(self, mel, cf0, voiced, contrast):
        inputs = [
            mel,
            torch.log(cf0.clamp(min=SOURCE_F0_FLOOR) / 200.0).unsqueeze(1),
            voiced.unsqueeze(1).to(mel.dtype),
        ]
        if self.contrast is not None:
            if contrast is None:
                contrast = mel.new_zeros(
                    mel.shape[0], self.contrast.bands + 1, mel.shape[-1]
                )
            inputs.append(contrast.to(mel.dtype))
        x = self.embed(torch.cat(inputs, dim=1))
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x.transpose(1, 2))
        turns = None
        if self.phase_head is not None:
            turns = self.phase_head(x).transpose(1, 2).float()
        return self.head(x).transpose(1, 2).float(), turns

    def _per_step(self, x, substeps, steps):
        batch, channels, frames = x.shape
        channels //= substeps
        x = x.reshape(batch, channels, substeps, frames).transpose(2, 3)
        x = x.reshape(batch, channels, frames * substeps)
        size = self.hop_length // substeps
        return to_steps(
            x, steps, self.synthesis_hop, size, size // 2, self.control_interpolation
        )

    def _pulse(self, f0, voiced, shimmer):
        # Every harmonic below Nyquist, those in the top `taper_hz` faded by a
        # raised cosine in the harmonic number: three shifted closed-form sums
        nyquist = self.sample_rate / 2
        start = nyquist - self.taper_hz
        f0 = f0.double().clamp(min=SOURCE_F0_FLOOR)
        theta = 2 * math.pi * (torch.cumsum(f0 / self.sample_rate, dim=-1) % 1.0)
        flat = torch.floor(start / f0)
        top = torch.ceil(nyquist / f0) - 1
        slope = math.pi * f0 / self.taper_hz
        offset = -math.pi * start / self.taper_hz
        faded = (
            0.5 * cosine_sum(theta, 0.0, flat + 1, top)
            + 0.25 * cosine_sum(theta + slope, offset, flat + 1, top)
            + 0.25 * cosine_sum(theta - slope, -offset, flat + 1, top)
        )
        pulse = cosine_sum(theta, 0.0, torch.ones_like(flat), flat) + faded
        pulse = pulse.float() * self.harmonic_amp
        return pulse * voiced * (1 + shimmer), theta.float()

    def _phase_spread(self, spread):
        # Random phases drawn per band within +-spread and interpolated across the
        # bins between band centres, so each harmonic turns as a whole; scaled by
        # the power they keep through overlap-add
        lower, upper, t = self.spread_lower, self.spread_upper, self.spread_weight
        drawn = spread * (2 * torch.rand_like(spread) - 1)
        phase = drawn[:, lower] * (1 - t) + drawn[:, upper] * t
        coherent = torch.sinc(spread[:, lower] * (1 - t) / math.pi) * torch.sinc(
            spread[:, upper] * t / math.pi
        )
        power = coherent.square() + (1 - coherent.square()) * self.spread_rho
        return torch.polar(power.rsqrt(), phase)

    def _source_phase(self, f0):
        # Each bin turns by the fixed phase of the harmonic of f0 nearest to it,
        # so every harmonic turns whole and keeps its phase as it moves
        harmonic = torch.round(self.source_bins / f0).long()
        harmonic = harmonic.clamp(max=len(self.source_phases) - 1)
        phase = self.source_phases[harmonic] * self.source_ramp
        return torch.polar(torch.ones_like(phase), phase)

    def _noise_phase(self, noise):
        # Unit phasors of the noise's spectrogram over the harmonic frames, from
        # the phase head's first bin: a real signal's, so coherent across
        # overlapping frames
        n_fft, hop = self.harmonic_fft, self.synthesis_hop
        padded = F.pad(noise.unsqueeze(1), (n_fft // 2, n_fft // 2), mode="reflect")
        padded = padded.squeeze(1)
        size = self.harmonic_window.shape[0]
        left = (n_fft - size) // 2
        window = F.pad(self.harmonic_window, (left, n_fft - size - left))

        def phase(start, end):
            chunk = padded[:, start * hop : (end - 1) * hop + n_fft]
            spectrum = torch.stft(
                chunk, n_fft, hop, window=window, center=False, return_complex=True
            )
            spectrum = spectrum[:, self.phase_first_bin :]
            return spectrum / (spectrum.abs() + 1e-8)

        return phase

    def _render(self, signal, n_fft, window, gain, steps, samples):
        # The iSTFT of the STFT of `signal` times `gain(start, end)`, a block of
        # frames at a time; overlap-add sums over frames, so the blocks add up exactly
        hop = self.synthesis_hop
        padded = F.pad(signal.unsqueeze(1), (n_fft // 2, n_fft // 2), mode="reflect")
        padded = padded.squeeze(1)
        left = (n_fft - window.shape[0]) // 2
        window = F.pad(window, (left, n_fft - window.shape[0] - left))
        length = (steps - 1) * hop + n_fft
        out = signal.new_zeros(signal.shape[0], length)
        for start in range(0, steps, self.block_steps):
            end = min(steps, start + self.block_steps)
            chunk = padded[:, start * hop : (end - 1) * hop + n_fft]
            spectrum = torch.stft(
                chunk, n_fft, hop, window=window, center=False, return_complex=True
            )
            frames = torch.fft.irfft(spectrum * gain(start, end), n=n_fft, dim=1)
            frames = frames * window.view(1, -1, 1)
            span = (end - start - 1) * hop + n_fft
            added = F.fold(frames, (1, span), (1, n_fft), stride=(1, hop)).flatten(1)
            positions = torch.arange(start * hop, start * hop + span, device=out.device)
            out = out.index_add(1, positions, added)
        envelope = F.fold(
            window.square().view(1, -1, 1).expand(1, -1, steps).contiguous(),
            (1, length),
            (1, n_fft),
            stride=(1, hop),
        ).flatten()
        out = out / envelope.clamp(min=1e-8)
        return out[:, n_fft // 2 : n_fft // 2 + samples]

    def forward(self, mel, f0, contrast=None):
        """
        Args:
            mel (torch.Tensor): Normalised log mel of shape (batch, num_mels, frames).
            f0 (torch.Tensor): Pitch of shape (batch, frames), 0 where unvoiced.
            contrast (torch.Tensor, optional): Band contrast from `known_contrast`, shape (batch, bands + 1, frames). Defaults to None, unknown.
        """
        f0 = f0.float()
        frames = f0.shape[-1]
        samples = frames * self.hop_length
        steps = samples // self.synthesis_hop + 1
        voiced = f0 > 0
        cf0 = continuous_f0(f0)
        controls, turns = self._controls(mel, cf0, voiced, contrast)

        with torch.autocast(mel.device.type, enabled=False):
            tract, glottal, noise, residual, curve, scalars, spread = controls.split(
                self.splits, dim=1
            )
            log_tract = self._per_step(tract, self.harmonic_substeps, steps)
            log_tract = log_tract.clamp(*LOG_GAIN_RANGE)
            log_glottal = None
            if self.glottal_bands:
                log_glottal = self._per_step(glottal, 1, steps).clamp(*LOG_GAIN_RANGE)
            log_noise = self._per_step(noise, self.noise_substeps, steps)
            if self.curve_knots:
                # Voiced noise follows the harmonic envelope times a smooth curve
                harmonic_level = log_tract
                if log_glottal is not None:
                    harmonic_level = harmonic_level + torch.einsum(
                        "bcs,cf->bfs", log_glottal, self.glottal_at_bands
                    )
                curve = torch.einsum(
                    "bks,kf->bfs", self._per_step(curve, 1, steps), self.curve_bands
                )
                voicing = to_steps(
                    voiced.float().unsqueeze(1),
                    steps,
                    self.synthesis_hop,
                    self.hop_length,
                    self.hop_length // 2,
                )
                log_noise = (
                    voicing * (harmonic_level + curve) + (1 - voicing) * log_noise
                )
            log_noise = log_noise.clamp(*LOG_GAIN_RANGE)
            if self.splits[3]:
                residual = torch.tanh(self._per_step(residual, 1, steps))
            else:
                residual = None
            if self.splits[6]:
                spread = torch.sigmoid(spread) * self.max_phase_spread
                spread = self._per_step(spread, 1, steps)
            if turns is not None:
                turns = self._per_step(turns, self.harmonic_substeps, steps)
                parts = turns.chunk(self.phase_outputs, dim=1)
                angle = torch.atan2(parts[1], parts[0]) * self.phase_ramp
                turns = torch.polar(torch.ones_like(angle), angle)
                if self.phase_noise_mix:
                    mix = torch.sigmoid(parts[2]) * self.phase_ramp
                    noise = torch.randn(f0.shape[0], samples, device=f0.device)
                    noise_phase = self._noise_phase(noise)
            frame_center = self.hop_length // 2
            aspiration, scalars = scalars.split([scalars.shape[1] - 3, 3], dim=1)
            jitter, shimmer = torch.sigmoid(scalars[:, :2]).unbind(1)
            correction = torch.tanh(scalars[:, 2]) * self.max_f0_correction
            gates = torch.cat(
                (
                    torch.sigmoid(aspiration),
                    torch.stack(
                        (
                            jitter * self.max_jitter,
                            shimmer * self.max_shimmer,
                            correction,
                            voiced.float(),
                            cf0,
                        ),
                        dim=1,
                    ),
                ),
                dim=1,
            )
            gates = to_steps(gates, samples, 1, self.hop_length, frame_center)
            aspiration = gates[:, :-5]
            jitter, shimmer, correction, gate, pitch = gates[:, -5:].unbind(1)

            # Jitter and shimmer as random walks smooth over a synthesis step
            wobble = to_steps(
                torch.randn(f0.shape[0], 2, steps, device=f0.device),
                samples,
                1,
                self.synthesis_hop,
                0,
            )
            pitch = pitch * torch.exp2(correction) * (1 + jitter * wobble[:, 0])
            pulse, theta = self._pulse(pitch, gate, shimmer * wobble[:, 1])
            if self.source_phase_hz is not None:
                # The pulse's own f0 at each frame centre
                centres = torch.arange(steps, device=pitch.device) * self.synthesis_hop
                step_f0 = pitch[:, centres.clamp(max=samples - 1)]
                step_f0 = step_f0.clamp(min=SOURCE_F0_FLOOR).unsqueeze(1)

            def harmonic_gain(start, end):
                log_gain = torch.einsum(
                    "bcs,cf->bfs",
                    log_tract[..., start:end].to(torch.complex64),
                    self.tract_log,
                )
                if log_glottal is not None:
                    log_gain = log_gain + torch.einsum(
                        "bcs,cf->bfs",
                        log_glottal[..., start:end].to(torch.complex64),
                        self.glottal_log,
                    )
                gain = torch.exp(log_gain)
                if residual is not None:
                    real, imag = residual[..., start:end].chunk(2, dim=1)
                    real = torch.einsum("bcs,cf->bfs", real, self.tract_bands)
                    imag = torch.einsum("bcs,cf->bfs", imag, self.tract_bands)
                    gain = gain * (1 + self.residual_bound * torch.complex(real, imag))
                if self.splits[6]:
                    gain = gain * self._phase_spread(spread[..., start:end])
                if turns is not None:
                    first = self.phase_first_bin
                    turn = turns[..., start:end]
                    if self.phase_noise_mix:
                        # Unit mean power whatever the share, the two being independent
                        share = mix[..., start:end]
                        blend = (1 - share) * turn + share * noise_phase(start, end)
                        turn = blend * ((1 - share).square() + share.square()).rsqrt()
                    turned = gain[:, first:] * turn
                    gain = torch.cat((gain[:, :first], turned), dim=1)
                if self.source_phase_hz is not None:
                    gain = gain * self._source_phase(step_f0[..., start:end])
                return gain

            def noise_gain(start, end):
                return torch.exp(
                    torch.einsum(
                        "bcs,cf->bfs", log_noise[..., start:end], self.noise_bands
                    )
                )

            voice = self._render(
                pulse,
                self.harmonic_fft,
                self.harmonic_window,
                harmonic_gain,
                steps,
                samples,
            )
            # Aspiration follows the glottal cycle where voiced; plain noise elsewhere
            if not self.aspiration_regions:
                envelope = 1 + aspiration[:, 0] * gate * torch.cos(
                    theta - self.aspiration_phase
                )
                breath = self._render(
                    torch.randn_like(pulse) * envelope,
                    self.noise_fft,
                    self.noise_window,
                    noise_gain,
                    steps,
                    samples,
                )
                return (voice + breath).unsqueeze(1)

            regions = self.aspiration_regions
            depth = aspiration[:, :regions] * gate.unsqueeze(1)
            sharpness = aspiration[:, regions:] * self.max_aspiration_sharpness
            # A von Mises pulse of mean 1 per region
            cycle = (
                torch.cos(theta.unsqueeze(1) - self.aspiration_phase.view(1, -1, 1)) - 1
            )
            envelope = 1 + depth * (
                torch.exp(sharpness * cycle) / torch.special.i0e(sharpness) - 1
            )

            def region_gain(start, end):
                gain = noise_gain(start, end).unsqueeze(1)
                return (gain * self.aspiration_masks.unsqueeze(-1)).flatten(0, 1)

            breath = self._render(
                (torch.randn_like(envelope) * envelope).flatten(0, 1),
                self.noise_fft,
                self.noise_window,
                region_gain,
                steps,
                samples,
            )
            breath = breath.view(-1, regions, samples).sum(1)
            return (voice + breath).unsqueeze(1)
