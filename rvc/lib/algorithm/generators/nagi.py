import math

import torch
from librosa import mel_frequencies
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "nagi"
SOURCE_F0_FLOOR = 50.0
# Scale of the band contrast input, from nepers to about [0, 1.5]
CONTRAST_SCALE = 4.0
LOG_GAIN_RANGE = (-14.0, 5.0)
SCATTER_SEED = 0
# Knots per second of the drift of the scattered phases
SCATTER_DRIFT_HZ = 8.0
# The crossfade window spans this many control hops
OVERLAP = 4
# Frames each side over which the envelopes of the free branch lose their mean
FREE_DETREND_FRAMES = 1
# Nepers the pulses start under the mel outside voiced frames
UNVOICED_DROP = 6.0
PULSE_GAIN_BIAS = -1.5


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


def continuous_f0(f0):
    """
    The pitch with its unvoiced zeros filled linearly between voiced frames and
    held past the first and last; all-unvoiced rows stay zero.

    Args:
        f0 (torch.Tensor): Pitch of shape (batch, frames), 0 where unvoiced.
    """
    frames = f0.shape[-1]
    voiced = f0 > 0
    index = torch.arange(frames, device=f0.device).expand_as(f0)
    last = torch.where(voiced, index, -1).cummax(-1).values
    following = (
        torch.where(voiced, index, frames).flip(-1).cummin(-1).values.flip(-1)
    )
    f_last = f0.gather(-1, last.clamp(min=0))
    f_next = f0.gather(-1, following.clamp(max=frames - 1))
    weight = (index - last) / (following - last).clamp(min=1)
    filled = torch.where(
        (last >= 0) & (following < frames),
        f_last + (f_next - f_last) * weight,
        torch.where(
            last >= 0,
            f_last,
            torch.where(following < frames, f_next, torch.zeros_like(f0)),
        ),
    )
    return torch.where(voiced, f0, filled)


def band_centers(sample_rate, min_hz, ratio):
    """
    Band centres from 0 to Nyquist, `min_hz` apart at the bottom and a `ratio`
    of their frequency apart above, which keeps the impulse response of every
    gain short.

    Args:
        sample_rate (int): Sampling rate.
        min_hz (float): Smallest spacing between bands, in Hz.
        ratio (float): Spacing above, as a share of the frequency.
    """
    nyquist = sample_rate / 2
    centers = [0.0]
    while centers[-1] < nyquist:
        centers.append(centers[-1] + max(min_hz, ratio * centers[-1]))
    centers[-1] = nyquist
    return torch.tensor(centers, dtype=torch.float64)


def interpolation_matrix(centers, freqs):
    """
    Linear interpolation of values at `centers` to `freqs`, held flat past both
    ends, of shape (len(centers), len(freqs)).

    Args:
        centers (torch.Tensor): Increasing frequencies the values are at.
        freqs (torch.Tensor): Frequencies to interpolate to.
    """
    freqs = freqs.clamp(centers[0], centers[-1])
    upper = torch.searchsorted(centers, freqs, right=True).clamp(1, len(centers) - 1)
    lower = upper - 1
    t = (freqs - centers[lower]) / (centers[upper] - centers[lower])
    matrix = torch.zeros(len(centers), len(freqs), dtype=torch.float64)
    columns = torch.arange(len(freqs))
    matrix[lower, columns] = 1 - t
    matrix[upper, columns] += t
    return matrix


def band_matrix(centers, n_fft, sample_rate):
    """
    Linear interpolation of band values to the bins of an FFT.

    Args:
        centers (torch.Tensor): Band centres, in Hz.
        n_fft (int): Size of the FFT.
        sample_rate (int): Sampling rate.
    """
    bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64)
    return interpolation_matrix(centers, bins)


def minimum_phase_basis(matrix, n_fft):
    """
    Each row of a log-magnitude basis as the complex log spectrum of its
    minimum-phase version. Folding the real cepstrum is linear, so a weighted
    sum of these rows is the minimum-phase log spectrum of the same sum of log
    magnitudes.

    Args:
        matrix (torch.Tensor): Basis of shape (bands, bins).
        n_fft (int): Size of the FFT.
    """
    cepstrum = torch.fft.irfft(matrix, n=n_fft, dim=-1)
    fold = torch.zeros(n_fft, dtype=cepstrum.dtype)
    fold[0] = 1.0
    fold[1 : n_fft // 2] = 2.0
    fold[n_fft // 2] = 1.0
    return torch.fft.rfft(cepstrum * fold, n=n_fft, dim=-1)


def to_steps(x, steps, step, frame, offset, mode="linear"):
    """
    Values at the centres `n * frame + offset` interpolated to `steps` points
    `step` apart, linearly or, with `mode` "cubic", by Catmull-Rom, whose slope
    has no kink at the centres to modulate the gains at the frame rate.

    Args:
        x (torch.Tensor): Values of shape (batch, channels, frames).
        steps (int): Number of points.
        step (int): Samples between points.
        frame (int): Samples between frames.
        offset (int): Sample of the first frame's centre.
        mode (str, optional): "linear" or "cubic". Defaults to "linear".
    """
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
        * (
            (2 * before - 5 * a + 4 * b - after)
            + weight * (3 * (a - b) + after - before)
        )
    )


def cosine_sum(x, phase, first, last):
    """
    `sum(cos(k * x + phase) for k in first..last)` elementwise, in closed form;
    zero where `last < first`.
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


def class_pulse(turns, f0, phases, start, taper):
    """
    `sum_k w_k cos(k theta + phases[k % classes])` over the harmonics of f0
    under `start + taper` Hz, faded by a raised cosine over the last `taper`.
    Each phase class is an arithmetic progression of harmonics, so it has the
    closed form of `cosine_sum`, and so has its faded part.

    Args:
        turns (torch.Tensor): Theta in cycles, in double, shape (batch, samples).
        f0 (torch.Tensor): Pitch per sample, shape (batch, samples).
        phases (torch.Tensor): Phase per class, (classes,) or (batch, classes, samples).
        start (float): Where the fade starts, in Hz.
        taper (float): Width of the fade, in Hz.
    """
    if phases.dim() == 1:
        phases = phases.view(1, -1, 1)
    classes = phases.shape[1]
    index = torch.arange(classes, device=f0.device).view(1, -1, 1)
    turns = turns.unsqueeze(1)
    x = (2 * math.pi * torch.remainder(classes * turns, 1.0)).float()
    phase = (2 * math.pi * torch.remainder(index * turns, 1.0)).float() + phases
    f0, index = f0.unsqueeze(1), index.float()
    # Class 0 starts at its second member: the first is DC
    first = (index == 0).float()
    flat = torch.floor((start / f0 - index) / classes)
    top = torch.ceil(((start + taper) / f0 - index) / classes) - 1
    slope = math.pi * classes * f0 / taper
    offset = math.pi * (index * f0 - start) / taper
    fade_first = torch.maximum(flat + 1, first)
    faded = (
        0.5 * cosine_sum(x, phase, fade_first, top)
        + 0.25 * cosine_sum(x + slope, phase + offset, fade_first, top)
        + 0.25 * cosine_sum(x - slope, phase - offset, fade_first, top)
    )
    return (cosine_sum(x, phase, first.expand_as(flat), flat) + faded).sum(1)


def free_band_centers(min_hz, ratio, max_hz, top):
    """
    Band centres of the free branch up to `top`: as `band_centers`, but none at
    0 Hz and never more than `max_hz` apart.
    """
    centers = [float(min_hz)]
    while True:
        following = centers[-1] + min(max(min_hz, ratio * centers[-1]), max_hz)
        if following > top:
            return torch.tensor(centers, dtype=torch.float64)
        centers.append(following)


def free_atoms(centers, sample_rate, hop, min_hz, ratio, max_hz):
    """
    Each band's Hann-windowed cosine, then its sine, of shape (2 * bands, 1,
    taps): the window two band spacings wide in frequency and at least
    `OVERLAP` hops long, scaled so that hop-spaced copies add up to the sinusoid.
    """
    lengths = []
    for hz in centers.tolist():
        spacing = min(max(min_hz, ratio * hz), max_hz)
        lengths.append(max(int(2 * sample_rate / spacing), OVERLAP * hop) | 1)
    taps = max(lengths)
    atoms = torch.zeros(2, len(lengths), taps, dtype=torch.float64)
    for band, (hz, length) in enumerate(zip(centers.tolist(), lengths)):
        position = torch.arange(length, dtype=torch.float64) - (length - 1) / 2
        window = torch.hann_window(length, periodic=False, dtype=torch.float64)
        window = window * (2 * hop / length)
        angle = 2 * math.pi * hz * position / sample_rate
        begin = (taps - length) // 2
        atoms[0, band, begin : begin + length] = window * torch.cos(angle)
        atoms[1, band, begin : begin + length] = window * torch.sin(angle)
    return atoms.view(-1, 1, taps).float()


class ConvNeXtBlock(nn.Module):
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


class Nagi(nn.Module):
    """
    Nagi vocoder from ShiroRVC, inference only. Band-limited sources (a pulse
    train with every harmonic in phase, one with a scattered, slowly drifting
    phase per harmonic, and white noise, plain and modulated by the glottal
    cycle) go through time-varying FIR filters applied in blocks crossfaded
    under a Hann window. A network at the frame rate, then at `control_ratio`
    times it, predicts the band gains and phases the taps are designed from.
    No nonlinearity touches the audio, so nothing folds or mirrors. The free
    branch writes band envelopes a fixed bank of sinusoids turns into audio,
    for what filtered sources cannot make.

    Args:
        sample_rate (int): Sampling rate.
        num_mels (int): Number of mel bins.
        hop_length (int): Samples per mel frame.
        dim (int): Channels of the frame-rate network.
        intermediate_dim (int): Hidden channels of its blocks.
        num_blocks (int): Blocks of the frame-rate network.
        kernel_size (int): Kernel size of the blocks.
        control_ratio (int): Control steps per frame.
        control_dim (int): Channels at the control rate.
        control_blocks (int): Blocks at the control rate.
        band_min_hz (float): Smallest spacing of the pulse bands.
        band_ratio (float): Spacing of the pulse bands above, as a share of their frequency.
        noise_band_min_hz (float): Smallest spacing of the noise bands.
        noise_band_ratio (float): Spacing of the noise bands above.
        pulse_taps (int): Length of the pulse filters.
        noise_taps (int): Length of the noise filters.
        noise_harmonics (int): Modulations of the noise by the glottal cycle.
        noise_control_ratio (int): Noise filter steps per control step.
        top_margin_hz (float): Top of the band where no harmonic is made.
        taper_hz (float): Width of the fade of the harmonics under it.
        phase_classes (int): Phase classes of the scattered pulse.
        scatter_drift (float): Radians the scattered phases wander, 0 holds them.
        free_branch (bool): Whether the free branch is built.
        free_dim (int): Channels of the free branch.
        free_blocks (int): Blocks of the free branch.
        free_bound (float): Scale of the ceiling of the free branch.
        contrast_band_edges (list, optional): Edges of the band contrast input, in Hz.
        mel_fmin (float): Lowest mel frequency.
        mel_fmax (float, optional): Highest mel frequency, Nyquist when None.
        mel_mean (float): Mean of the mel normalisation.
        mel_std (float): Standard deviation of the mel normalisation.
    """

    def __init__(
        self,
        sample_rate,
        num_mels,
        hop_length,
        dim=256,
        intermediate_dim=768,
        num_blocks=6,
        kernel_size=7,
        control_ratio=8,
        control_dim=128,
        control_blocks=3,
        band_min_hz=100.0,
        band_ratio=0.03,
        noise_band_min_hz=300.0,
        noise_band_ratio=0.05,
        pulse_taps=1024,
        noise_taps=256,
        noise_harmonics=6,
        noise_control_ratio=2,
        top_margin_hz=1000.0,
        taper_hz=1000.0,
        phase_classes=32,
        scatter_drift=1.0,
        free_branch=True,
        free_dim=512,
        free_blocks=8,
        free_bound=1.0,
        contrast_band_edges=None,
        mel_fmin=0.0,
        mel_fmax=None,
        mel_mean=0.0,
        mel_std=1.0,
    ):
        super().__init__()
        if hop_length % (control_ratio * noise_control_ratio):
            raise ValueError(
                f"control_ratio {control_ratio} times noise_control_ratio "
                f"{noise_control_ratio} must divide the hop {hop_length}."
            )
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.control_hop = self.hop_length // int(control_ratio)
        self.noise_ratio = int(noise_control_ratio)
        self.noise_hop = self.control_hop // self.noise_ratio
        self.scatter_drift = float(scatter_drift)
        self.mel_mean, self.mel_std = float(mel_mean), float(mel_std)
        self.pulse_taps, self.noise_taps = int(pulse_taps), int(noise_taps)
        self.pulse_fft, self.noise_fft = 2 * self.pulse_taps, 2 * self.noise_taps
        # The minimum-phase response is causal; the predicted turn may lead it
        self.pulse_pre = self.pulse_taps // 4
        self.noise_harmonics = int(noise_harmonics)
        self.taper_hz = float(taper_hz)
        self.pulse_top = self.sample_rate / 2 - float(top_margin_hz)

        self.contrast = None
        contrast_inputs = 0
        if contrast_band_edges:
            self.contrast = BandContrast(sample_rate, hop_length, contrast_band_edges)
            contrast_inputs = self.contrast.bands + 1

        centers = band_centers(sample_rate, band_min_hz, band_ratio)
        noise_centers = band_centers(sample_rate, noise_band_min_hz, noise_band_ratio)
        self.bands, self.noise_bands = len(centers), len(noise_centers)
        mel_centers = mel_frequencies(
            n_mels=num_mels + 2, fmin=mel_fmin, fmax=mel_fmax or sample_rate / 2
        )
        mel_centers = torch.from_numpy(mel_centers[1:-1]).double()
        to_bins = band_matrix(centers, self.pulse_fft, sample_rate)
        phase_bins = minimum_phase_basis(to_bins, self.pulse_fft).imag
        noise_bins = band_matrix(noise_centers, self.noise_fft, sample_rate)
        self.register_buffer("band_bins", to_bins.float(), persistent=False)
        self.register_buffer("phase_bins", phase_bins.float(), persistent=False)
        self.register_buffer("noise_bins", noise_bins.float(), persistent=False)
        shortcuts = [
            interpolation_matrix(mel_centers, centers),
            interpolation_matrix(mel_centers, noise_centers),
        ]

        post = self.pulse_taps - self.pulse_pre
        pulse_fade = torch.cat(
            (
                torch.hann_window(2 * self.pulse_pre + 1, periodic=False)[
                    : self.pulse_pre
                ],
                torch.hann_window(2 * post + 1, periodic=False)[post : 2 * post],
            )
        )
        self.register_buffer("pulse_fade", pulse_fade, persistent=False)
        self.register_buffer(
            "noise_fade", torch.hann_window(self.noise_taps), persistent=False
        )
        # Hop-spaced copies add up to one
        self.register_buffer(
            "window",
            torch.hann_window(OVERLAP * self.control_hop) * 2 / OVERLAP,
            persistent=False,
        )
        self.register_buffer(
            "noise_window",
            torch.hann_window(OVERLAP * self.noise_hop) * 2 / OVERLAP,
            persistent=False,
        )
        generator = torch.Generator().manual_seed(SCATTER_SEED)
        scatter_phases = 2 * math.pi * torch.rand(int(phase_classes), generator=generator)
        self.register_buffer("scatter_phases", scatter_phases, persistent=False)
        self.register_buffer("aligned_phases", torch.zeros(1), persistent=False)

        self.embed = nn.Conv1d(
            num_mels + 2 + contrast_inputs,
            dim,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim, kernel_size, 1 / num_blocks)
            for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.project = nn.Linear(dim, control_dim)
        self.control_blocks = nn.ModuleList(
            ConvNeXtBlock(control_dim, 3 * control_dim, kernel_size, 1 / control_blocks)
            for _ in range(control_blocks)
        )
        self.control_norm = nn.LayerNorm(control_dim, eps=1e-6)
        # Per pulse band a gain, a scattered share and a turn; per noise band a
        # gain and its modulations', each once per noise step
        noise_outputs = self.noise_ratio * self.noise_bands
        self.splits = [self.bands] * 3 + [
            noise_outputs,
            2 * self.noise_harmonics * noise_outputs,
        ]
        self.head = nn.Linear(control_dim, sum(self.splits))

        # Control steps computed past each end: the crossfade needs one, the
        # free branch's atoms half their length
        self.margin = 1
        self.free_head = None
        if free_branch:
            max_hz = self.sample_rate / (2 * self.control_hop)
            free_centers = free_band_centers(
                band_min_hz, band_ratio, max_hz, self.pulse_top
            )
            atoms = free_atoms(
                free_centers,
                sample_rate,
                self.control_hop,
                band_min_hz,
                band_ratio,
                max_hz,
            )
            self.free_bands = len(free_centers)
            self.free_log_bound = math.log(float(free_bound)) + PULSE_GAIN_BIAS
            self.margin = math.ceil((atoms.shape[-1] - 1) / 2 / self.control_hop) + 1
            self.free_detrend = 2 * FREE_DETREND_FRAMES * int(control_ratio) + 1
            atom_norm = atoms.square().sum(-1).sqrt().clamp(min=1e-8).view(1, -1, 1)
            self.register_buffer("atoms", atoms, persistent=False)
            self.register_buffer("atom_norm", atom_norm, persistent=False)
            self.register_buffer("free_centers", free_centers, persistent=False)
            shortcuts.append(interpolation_matrix(mel_centers, free_centers))
            # The noise's envelopes and the filtered sources'
            self.free_input = nn.Conv1d(4 * self.free_bands, free_dim, 1)
            self.free_project = nn.Linear(control_dim, free_dim)
            self.free_blocks = nn.ModuleList(
                ConvNeXtBlock(free_dim, 3 * free_dim, kernel_size, 1 / free_blocks)
                for _ in range(free_blocks)
            )
            self.free_norm = nn.LayerNorm(free_dim, eps=1e-6)
            self.free_head = nn.Linear(free_dim, 2 * self.free_bands)
        self.register_buffer(
            "mel_shortcut", torch.cat(shortcuts, dim=1).float(), persistent=False
        )

    def _sources(self, cf0, samples):
        """
        The pulses (batch, 2, samples) and the noises (batch, 1 + 2 *
        noise_harmonics, samples): white, then its products with the cosine and
        sine of each harmonic of the glottal cycle.
        """
        f0 = F.interpolate(
            cf0.unsqueeze(1), size=samples, mode="linear", align_corners=False
        ).squeeze(1)
        f0 = f0.clamp(min=SOURCE_F0_FLOOR)
        increment = f0.double() / self.sample_rate
        turns = torch.remainder(torch.cumsum(increment, dim=-1), 1.0)
        start = self.pulse_top - self.taper_hz
        scatter = self.scatter_phases
        if self.scatter_drift > 0:
            # Held, the scattered phases are one frozen pattern
            knots = math.ceil(samples * SCATTER_DRIFT_HZ / self.sample_rate) + 2
            drift = torch.randn(f0.shape[0], len(scatter), knots, device=f0.device)
            drift = F.interpolate(drift, size=samples, mode="linear", align_corners=True)
            scatter = scatter.view(1, -1, 1) + self.scatter_drift * drift
        pulses = torch.stack(
            (
                class_pulse(turns, f0, self.aligned_phases, start, self.taper_hz),
                class_pulse(turns, f0, scatter, start, self.taper_hz),
            ),
            dim=1,
        )
        white = torch.randn_like(f0)
        noises = [white]
        for harmonic in range(1, self.noise_harmonics + 1):
            theta = (2 * math.pi * torch.remainder(harmonic * turns, 1.0)).float()
            noises += [white * torch.cos(theta), white * torch.sin(theta)]
        return pulses, torch.stack(noises, dim=1)

    def _analysis(self, audio):
        """Audio (batch, samples) on the free branch's atoms at every control step."""
        pad = self.margin * self.control_hop + (self.atoms.shape[-1] - 1) // 2
        padded = F.pad(audio.unsqueeze(1), (pad, pad + 1))
        return F.conv1d(padded, self.atoms, stride=self.control_hop)

    def _carriers(self, steps):
        """Cosine and sine of each band's carrier, run from sample 0, at the centres of the control steps."""
        centre = torch.arange(steps, device=self.atoms.device, dtype=torch.float64)
        centre = (centre - self.margin) * self.control_hop
        turn = torch.outer(self.free_centers.double(), centre) / self.sample_rate
        turn = 2 * math.pi * torch.remainder(turn, 1.0)
        return torch.cos(turn).float(), torch.sin(turn).float()

    def _heard(self, audio, log_bound, carriers):
        """
        What the filtered sources put in each band, as the envelopes `_free`
        writes: over the band's bound, and compressed so that none passes one.
        """
        cos, sin = carriers
        # A sinusoid at a band's centre reads half a hop of its amplitude
        projected = self._analysis(audio) * (2 / self.control_hop)
        a, b = projected.reshape(audio.shape[0], 2, self.free_bands, -1).unbind(1)
        envelope = torch.stack((a * cos - b * sin, b * cos + a * sin), dim=1)
        envelope = envelope * torch.exp(-log_bound).unsqueeze(1)
        norm = 1 + envelope.square().sum(1, keepdim=True).sqrt()
        return (envelope / norm).flatten(1, 2)

    def _controls(self, mel, cf0, voiced, contrast, steps):
        """The head's outputs at the control steps, in FP32, and the features they are read from."""
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
        x = self.project(self.output_norm(x.transpose(1, 2))).transpose(1, 2)
        # Interpolated, not repeated, which would modulate the gains at the frame rate
        x = self._to_steps(x, steps, "cubic")
        for block in self.control_blocks:
            x = block(x)
        out = self.head(self.control_norm(x.transpose(1, 2))).transpose(1, 2)
        return out.float(), x

    def _free_controls(self, features, analysis):
        """The free branch's envelopes at the control steps, in FP32."""
        y = self.free_project(features.transpose(1, 2)).transpose(1, 2)
        y = y + self.free_input(analysis.to(features.dtype))
        for block in self.free_blocks:
            y = block(y)
        out = self.free_head(self.free_norm(y.transpose(1, 2))).transpose(1, 2)
        return out.float()

    def _to_steps(self, x, steps, mode="linear", hop=None):
        """Frame values at `steps` points `hop` apart, the first `margin` control hops before sample 0."""
        offset = self.hop_length // 2 + self.margin * self.control_hop
        return to_steps(x, steps, hop or self.control_hop, self.hop_length, offset, mode)

    def _noise_steps(self, x):
        """The head's value per noise step inside each control step, laid out along time."""
        batch, _, steps = x.shape
        x = x.reshape(batch, self.noise_ratio, -1, steps).permute(0, 2, 3, 1)
        return x.reshape(batch, -1, steps * self.noise_ratio)

    def _pulse_kernels(self, log_gain, scatter, turn):
        """Band values (batch, bands, blocks) to the taps of the aligned and the scattered pulse."""
        magnitude = torch.exp(torch.einsum("brl,rf->blf", log_gain, self.band_bins))
        phase = torch.einsum("brl,rf->blf", log_gain, self.phase_bins)
        # An angle, so the powers of the two shares add up to one
        share = torch.einsum("brl,rf->blf", torch.sigmoid(scatter), self.band_bins)
        angle = 0.5 * math.pi * share
        turn = torch.einsum("brl,rf->blf", turn, self.band_bins)
        response = torch.stack(
            (
                torch.polar(magnitude * torch.cos(angle), phase + turn),
                torch.polar(magnitude * torch.sin(angle), phase + turn),
            ),
            dim=2,
        )
        response = torch.fft.irfft(response, n=self.pulse_fft)
        taps = torch.cat(
            (
                response[..., -self.pulse_pre :],
                response[..., : self.pulse_taps - self.pulse_pre],
            ),
            dim=-1,
        )
        return taps * self.pulse_fade

    def _noise_kernels(self, log_gain, modulation):
        """Band gains and the modulations' to zero-phase, centred noise taps."""
        level = torch.exp(torch.einsum("brl,rf->blf", log_gain, self.noise_bins))
        level = level.unsqueeze(2)
        depth = torch.einsum("bnrl,rf->blnf", torch.tanh(modulation), self.noise_bins)
        response = torch.fft.irfft(
            torch.cat((level, level * depth), dim=2), n=self.noise_fft
        )
        half = self.noise_taps // 2
        taps = torch.cat(
            (response[..., -half:], response[..., : self.noise_taps - half]), dim=-1
        )
        return taps * self.noise_fade

    def _filter(self, sources, kernels, pre, hop):
        """
        Sources (batch, channels, samples) through kernels (batch, blocks,
        channels, taps) whose first tap is lag `-pre`: block l is the `OVERLAP *
        hop` samples centred on sample `(l - 1) * hop`, summed over the channels.
        """
        batch, blocks, channels, taps = kernels.shape
        window = OVERLAP * hop
        reach = (OVERLAP // 2 + 1) * hop
        padded = F.pad(sources, (reach + taps - pre - 1, reach + pre))
        segments = padded.unfold(-1, window + taps - 1, hop).transpose(1, 2)
        groups = batch * blocks * channels
        # One group per block and source: each has its own kernel
        out = F.conv1d(
            segments.reshape(1, groups, -1),
            kernels.flip(-1).reshape(groups, 1, taps),
            groups=groups,
        )
        return out.view(batch, blocks, channels, window).sum(2)

    @staticmethod
    def _overlap_add(rendered, window, hop, samples):
        """Blocks (batch, blocks, OVERLAP * hop) crossfaded under `window`."""
        rendered = (rendered * window).transpose(1, 2)
        length = (rendered.shape[-1] + OVERLAP - 1) * hop
        audio = F.fold(rendered, (1, length), (1, OVERLAP * hop), stride=(1, hop))
        start = (OVERLAP // 2 + 1) * hop
        return audio[:, 0, 0, start : start + samples]

    def _free(self, raw, log_bound, carriers, samples):
        """The head's outputs as envelopes of the bands' carriers, each under `exp(log_bound)`."""
        steps = raw.shape[-1]
        envelope = torch.tanh(raw).reshape(raw.shape[0], 2, self.free_bands, steps)
        envelope = envelope * torch.exp(log_bound).unsqueeze(1)
        # Less its local mean: held steady it would draw a line at every band's centre
        flat = envelope.flatten(1, 2)
        mean = F.avg_pool1d(
            flat,
            self.free_detrend,
            1,
            self.free_detrend // 2,
            count_include_pad=False,
        )
        envelope = (flat - mean).reshape(envelope.shape)
        # The atoms' carriers start at each atom's centre, turned to the one that runs from sample 0
        cos, sin = carriers
        coefficients = torch.cat(
            (
                envelope[:, 0] * cos + envelope[:, 1] * sin,
                envelope[:, 1] * cos - envelope[:, 0] * sin,
            ),
            dim=1,
        )
        audio = F.conv_transpose1d(coefficients, self.atoms, stride=self.control_hop)
        start = self.margin * self.control_hop + (self.atoms.shape[-1] - 1) // 2
        return audio[:, 0, start : start + samples]

    def forward(self, mel, f0, contrast=None):
        """
        Args:
            mel (torch.Tensor): Normalised log mel of shape (batch, n_mels, frames).
            f0 (torch.Tensor): Pitch of shape (batch, frames), 0 where unvoiced.
            contrast (torch.Tensor, optional): Band contrast from `known_contrast`, shape (batch, bands + 1, frames).
        """
        f0 = f0.float()
        samples = f0.shape[-1] * self.hop_length
        hop, margin = self.control_hop, self.margin
        steps = samples // hop + 2 * margin + 1
        voiced = f0 > 0
        cf0 = continuous_f0(f0)
        device = mel.device.type
        with torch.no_grad(), torch.autocast(device, enabled=False):
            pulses, noises = self._sources(cf0, samples)
        controls, features = self._controls(mel, cf0, voiced, contrast, steps)
        # The filters in FP32: their taps span more than BF16 resolves
        with torch.autocast(device, enabled=False):
            level = mel.float() * self.mel_std + self.mel_mean
            frames = torch.cat(
                (
                    torch.einsum("bmt,mr->brt", level, self.mel_shortcut),
                    voiced.unsqueeze(1).float(),
                ),
                dim=1,
            )
            shortcut = self._to_steps(frames, steps)
            tract, scatter, turn, noise, modulation = controls.split(self.splits, dim=1)
            log_pulse = shortcut[:, : self.bands] + tract
            log_pulse = log_pulse - UNVOICED_DROP * (1 - shortcut[:, -1:])
            log_pulse = log_pulse.clamp(*LOG_GAIN_RANGE)
            # The crossfade's blocks: one step past each end
            blocks = slice(margin - 1, steps - margin + 1)
            pulse_kernels = self._pulse_kernels(
                log_pulse[..., blocks], scatter[..., blocks], turn[..., blocks]
            )
            ratio, noise_hop = self.noise_ratio, self.noise_hop
            noise_level = self._to_steps(
                frames[:, self.bands : self.bands + self.noise_bands],
                steps * ratio,
                hop=noise_hop,
            )
            log_noise = noise_level + self._noise_steps(noise)
            log_noise = log_noise.clamp(*LOG_GAIN_RANGE)
            modulation = self._noise_steps(modulation).reshape(
                mel.shape[0], 2 * self.noise_harmonics, self.noise_bands, steps * ratio
            )
            noise_blocks = slice(
                margin * ratio - 1, margin * ratio + samples // noise_hop + 2
            )
            noise_kernels = self._noise_kernels(
                log_noise[..., noise_blocks], modulation[..., noise_blocks]
            )
            harmonic = self._overlap_add(
                self._filter(pulses, pulse_kernels, self.pulse_pre, hop),
                self.window,
                hop,
                samples,
            )
            audio = harmonic + self._overlap_add(
                self._filter(noises, noise_kernels, self.noise_taps // 2, noise_hop),
                self.noise_window,
                noise_hop,
                samples,
            )
        if self.free_head is not None:
            with torch.no_grad(), torch.autocast(device, enabled=False):
                bound = shortcut[:, self.bands + self.noise_bands : -1]
                log_bound = bound + self.free_log_bound
                carriers = self._carriers(steps)
                analysis = torch.cat(
                    (
                        self._analysis(noises[:, 0]) / self.atom_norm,
                        self._heard(audio, log_bound, carriers),
                    ),
                    dim=1,
                )
            free = self._free_controls(features, analysis)
            with torch.autocast(device, enabled=False):
                audio = audio + self._free(free, log_bound, carriers, samples)
        return audio.unsqueeze(1)
