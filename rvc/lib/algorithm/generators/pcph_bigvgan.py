import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import (
    register_parametrization,
    remove_parametrizations,
)

from rvc.lib.algorithm.commons import get_padding, init_weights

# Upsampler filters of the four and of the five stage layouts.
UPSAMPLE_FILTERS = {
    4: dict(
        filter_width=(12, 24, 32, 48),
        rolloff=(0.84, 0.92, 0.94, 0.94),
        filter_beta=(6.0, 6.0, 6.0, 9.0),
    ),
    5: dict(
        filter_width=(12, 24, 32, 48, 48),
        rolloff=(0.84, 0.92, 0.94, 0.94, 0.94),
        filter_beta=(6.0, 6.0, 6.0, 9.0, 9.0),
    ),
}
# Decimation filter that brings the excitation down to each stage's rate.
SOURCE_DECIMATION = dict(width=24, rolloff=0.88, filter_beta=8.0)
# Channels of the rectified source branch at the output rate, doubled per stage.
SOURCE_BRANCH_CHANNELS = 16
SOURCE_BRANCH_SLOPE = 0.1
DEEP_SOURCE_ACTIVATION = dict(filter_width=64, rolloff=0.97)
PRENET_EXPANSION = 3
# SnakeBeta's oversampling round trip.
SNAKE_ACTIVATION = dict(filter_width=16, rolloff=0.89, filter_beta=5.0)
# The output stage may fold into its top band while that stays above this.
AUDIBLE_LIMIT = 20000.0
MAX_OUTPUT_ROLLOFF = 0.99
# Moving-average window the output's DC is taken from, in seconds.
DC_WINDOW_SECONDS = 0.1
# Fraction of Nyquist over which a harmonic fades out instead of switching off.
NYQUIST_TAPER = 0.1


def lowpass_kernel(factor, width, rolloff, filter_beta):
    """
    Kaiser-windowed sinc lowpass at `rolloff` of the Nyquist of the lower rate.

    Args:
        factor (int): Ratio between the two rates.
        width (int): Half-length of the kernel, in samples of the lower rate.
        rolloff (float): Fraction of the Nyquist the filter keeps.
        filter_beta (float): Beta of the Kaiser window.
    """
    half = max(1, int(width) * max(1, int(factor)))
    positions = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = 0.5 * float(rolloff) / max(1, int(factor))
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)
    kernel = kernel * torch.kaiser_window(
        kernel.numel(), periodic=False, beta=float(filter_beta), dtype=kernel.dtype
    )
    return (kernel / kernel.sum()).view(1, 1, -1)


def output_design(sample_rate, filter_width=16, filter_beta=6.0):
    """
    Design of a 2x round trip at the output rate whose folds stay above
    AUDIBLE_LIMIT, or fold nowhere when the rate leaves no room.

    Args:
        sample_rate (int): Sampling rate of the audio.
        filter_width (int, optional): Half-length of the kernel. Defaults to 16.
        filter_beta (float, optional): Beta of the Kaiser window. Defaults to 6.0.
    """
    attenuation = filter_beta / 0.1102 + 8.7
    half_band = (attenuation - 8.0) / (28.72 * filter_width)
    audible = 2.0 - AUDIBLE_LIMIT / (sample_rate / 2.0) - half_band
    rolloff = min(MAX_OUTPUT_ROLLOFF, max(1.0 - half_band, audible))
    return dict(
        filter_width=filter_width, rolloff=round(rolloff, 3), filter_beta=filter_beta
    )


def expand_f0(f0, length):
    """
    Interpolate the f0 from the frame rate to the output rate, in log Hz and
    only over the voiced frames.

    Args:
        f0 (torch.Tensor): Pitch in Hz, shape (batch, 1, frames).
        length (int): Number of output samples.
    """
    voiced = (f0 > 0).to(f0.dtype)
    log_f0 = torch.log(f0.clamp_min(1.0)) * voiced
    weight = F.interpolate(voiced, size=length, mode="linear", align_corners=False)
    log_f0 = F.interpolate(log_f0, size=length, mode="linear", align_corners=False)
    log_f0 = log_f0 / weight.clamp_min(1e-6)
    gate = F.interpolate(voiced, size=length, mode="nearest")
    return torch.exp(log_f0) * gate


def remove_dc(x, window):
    """
    Subtract the centred moving average of the signal.

    Args:
        x (torch.Tensor): Signal, shape (..., samples).
        window (int): Length of the moving average.
    """
    length = x.shape[-1]
    half = window // 2
    total = F.pad(x.double().cumsum(-1), (1, 0))
    index = torch.arange(length, device=x.device)
    low = (index - half).clamp(min=0)
    high = (index + half + 1).clamp(max=length)
    mean = (total[..., high] - total[..., low]) / (high - low)
    return x - mean.to(x.dtype)


class UnitNorm(nn.Module):
    """
    Weight norm without the gain: every output channel's filter at unit norm.
    """

    def forward(self, weight):
        return weight / torch.linalg.vector_norm(
            weight, dim=tuple(range(1, weight.dim())), keepdim=True
        )


class FixedLowPass1d(nn.Module):
    """
    Fixed windowed-sinc lowpass, optionally decimating.

    Args:
        factor (int): Ratio between the two rates.
        width (int): Half-length of the kernel, in samples of the lower rate.
        rolloff (float): Fraction of the Nyquist the filter keeps.
        filter_beta (float): Beta of the Kaiser window.
        stride (int, optional): Decimation factor. Defaults to 1.
    """

    def __init__(self, factor, width, rolloff, filter_beta, stride=1):
        super().__init__()
        self.stride = int(stride)
        self.register_buffer(
            "kernel",
            lowpass_kernel(factor, width, rolloff, filter_beta),
            persistent=False,
        )

    def forward(self, x):
        channels = x.shape[1]
        kernel = self.kernel.to(x.dtype).expand(channels, -1, -1)
        padding = (kernel.shape[-1] - 1) // 2
        mode = "reflect" if x.shape[-1] > padding else "replicate"
        x = F.pad(x, (padding, padding), mode=mode)
        return F.conv1d(x, kernel, stride=self.stride, groups=channels)


class AntiAliasedUpsample1d(nn.Module):
    """
    Fixed windowed-sinc upsampler, computed as one convolution per output phase.

    Args:
        factor (int): Upsampling factor.
        filter_width (int): Half-length of the kernel, in input samples.
        rolloff (float): Fraction of the input Nyquist the filter keeps.
        filter_beta (float): Beta of the Kaiser window.
    """

    def __init__(self, factor, filter_width, rolloff, filter_beta):
        super().__init__()
        self.factor = int(factor)
        kernel = lowpass_kernel(self.factor, filter_width, rolloff, filter_beta)
        kernel = kernel[0, 0] * self.factor
        kernel_size = int(kernel.shape[-1])
        taps = -(-kernel_size // self.factor)

        # Each phase flipped for conv1d, a shorter one right-aligned.
        weight = kernel.new_zeros(self.factor, taps)
        for phase in range(self.factor):
            part = kernel[phase :: self.factor]
            weight[phase, taps - part.numel() :] = part.flip(-1)
        self.register_buffer("phase_weight", weight, persistent=False)

        pad = kernel_size // self.factor - 1
        pad_left = pad * self.factor + (kernel_size - 1) // 2
        left = pad - pad_left // self.factor + taps - 1
        self.phase_pad = (left, taps - 1 - left)

    def forward(self, x):
        if self.factor == 1:
            return x
        batch, channels, length = x.shape
        weight = self.phase_weight.to(x.dtype)[:, None].repeat(channels, 1, 1)
        padded = F.pad(x, self.phase_pad, mode="replicate")
        phases = F.conv1d(padded, weight, groups=channels)
        return (
            phases.view(batch, channels, self.factor, length)
            .transpose(2, 3)
            .reshape(batch, channels, length * self.factor)
        )


class AntiAliasedActivation(nn.Module):
    """
    A pointwise nonlinearity evaluated at twice the rate, then filtered back,
    so the harmonics it creates do not fold.

    Args:
        activation (torch.nn.Module, optional): The nonlinearity, LeakyReLU when None.
        leaky_relu_slope (float, optional): Slope of the LeakyReLU. Defaults to 0.2.
        factor (int, optional): Oversampling factor. Defaults to 2.
        filter_width (int, optional): Half-length of the kernels. Defaults to 16.
        rolloff (float, optional): Fraction of the Nyquist the filters keep. Defaults to 0.99.
        filter_beta (float, optional): Beta of the Kaiser window. Defaults to 6.0.
    """

    def __init__(
        self,
        activation=None,
        leaky_relu_slope=0.2,
        factor=2,
        filter_width=16,
        rolloff=0.99,
        filter_beta=6.0,
    ):
        super().__init__()
        self.activation = (
            nn.LeakyReLU(leaky_relu_slope) if activation is None else activation
        )
        self.up = AntiAliasedUpsample1d(factor, filter_width, rolloff, filter_beta)
        self.down = FixedLowPass1d(
            factor, filter_width, rolloff, filter_beta, stride=factor
        )

    def forward(self, x):
        length = x.shape[-1]
        x = self.down(self.activation(self.up(x)))
        if x.shape[-1] > length:
            return x[..., :length]
        if x.shape[-1] < length:
            return F.pad(x, (0, length - x.shape[-1]), mode="replicate")
        return x


class SnakeBeta(nn.Module):
    """
    BigVGAN's `x + sin^2(alpha * x) / beta`, per channel, with both in log scale.

    Args:
        channels (int): Number of channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.exp()[:, None]
        inv_beta = torch.exp(-self.beta)[:, None]
        return torch.addcmul(x, torch.sin(x * alpha).square(), inv_beta)


class StageRateActivation(nn.Module):
    """
    SnakeBeta at the stage's own rate, with the keys of the oversampled one.

    Args:
        activation (torch.nn.Module): The nonlinearity.
    """

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)


def snake_activation(channels, antialias=True, design=SNAKE_ACTIVATION):
    if antialias:
        return AntiAliasedActivation(SnakeBeta(channels), **design)
    return StageRateActivation(SnakeBeta(channels))


def amp_conv(channels, kernel_size, dilation=1):
    conv = weight_norm(
        nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
    )
    conv.apply(init_weights)
    return conv


class AMPBlock(nn.Module):
    """
    BigVGAN's AMPBlock: dilated convolutions behind SnakeBeta activations.

    Args:
        channels (int): Number of channels.
        kernel_size (int): Kernel size of the convolutions.
        dilation (tuple[int]): Dilation rates.
        antialias (bool, optional): Oversample the activations. Defaults to True.
        pairs (bool, optional): AMPBlock1, with two convolutions per dilation, instead of AMPBlock2. Defaults to True.
        design (dict, optional): Filter design of the oversampled activations.
    """

    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        antialias=True,
        pairs=True,
        design=SNAKE_ACTIVATION,
    ):
        super().__init__()
        self.pairs = pairs
        self.convs1 = nn.ModuleList(
            [amp_conv(channels, kernel_size, d) for d in dilation]
        )
        self.acts1 = nn.ModuleList(
            [snake_activation(channels, antialias, design) for _ in dilation]
        )
        if pairs:
            self.convs2 = nn.ModuleList(
                [amp_conv(channels, kernel_size) for _ in dilation]
            )
            self.acts2 = nn.ModuleList(
                [snake_activation(channels, antialias, design) for _ in dilation]
            )

    def forward(self, x):
        for index, (c1, a1) in enumerate(zip(self.convs1, self.acts1)):
            xt = c1(a1(x))
            if self.pairs:
                xt = self.convs2[index](self.acts2[index](xt))
            x = xt + x
        return x


class PrenetBlock(nn.Module):
    """
    ConvNeXt block at the mel frame rate.

    Args:
        channels (int): Number of channels.
        layer_scale (float): Initial scale of the branch.
    """

    def __init__(self, channels, layer_scale):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.up = nn.Linear(channels, channels * PRENET_EXPANSION)
        self.down = nn.Linear(channels * PRENET_EXPANSION, channels)
        self.gamma = nn.Parameter(torch.full((channels,), float(layer_scale)))

    def forward(self, x):
        y = self.depthwise(x).transpose(1, 2)
        y = self.down(F.gelu(self.up(self.norm(y)))) * self.gamma
        return x + y.transpose(1, 2)


def source_conv(in_channels, channels, deep):
    """
    The convolution that adds the excitation of a stage, three of them with
    oversampled rectifiers between when `deep`.
    """
    if not deep:
        return nn.Conv1d(in_channels, channels, 7, 1, padding=3)
    return nn.Sequential(
        nn.Conv1d(in_channels, channels, 7, 1, padding=3),
        AntiAliasedActivation(
            leaky_relu_slope=SOURCE_BRANCH_SLOPE, **DEEP_SOURCE_ACTIVATION
        ),
        nn.Conv1d(channels, channels, 7, 1, padding=3),
        AntiAliasedActivation(
            leaky_relu_slope=SOURCE_BRANCH_SLOPE, **DEEP_SOURCE_ACTIVATION
        ),
        nn.Conv1d(channels, channels, 7, 1, padding=3),
    )


class PCPHSource(nn.Module):
    """
    Pseudo-constant-power harmonic excitation (Wavehax's PCPH prior): every
    harmonic up to Nyquist at a multiple of the fundamental's phase, so they
    add up to one band-limited pulse per period, at a power that does not
    depend on f0.

    Args:
        sample_rate (int): Sampling rate of the audio.
        sine_amp (float, optional): Amplitude of the excitation. Defaults to 0.1.
        noise_std (float, optional): Noise added to the voiced samples. Defaults to 0.003.
        noise_eq (list, optional): `(hz, db)` points that shape the spectrum of the voiced noise.
    """

    def __init__(self, sample_rate, sine_amp=0.1, noise_std=0.003, noise_eq=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.noise_eq = None
        if noise_eq:
            points = sorted((float(hz), float(db)) for hz, db in noise_eq)
            self.noise_eq = (
                np.array([p[0] for p in points]),
                np.array([p[1] for p in points]),
            )

    def equalize(self, noise):
        hz, db = self.noise_eq
        freqs = np.fft.rfftfreq(noise.shape[-1], 1.0 / self.sample_rate)
        gain = torch.from_numpy(10.0 ** (np.interp(freqs, hz, db) / 20.0))
        gain = gain.to(noise.device, torch.float32)
        spectrum = torch.fft.rfft(noise.float(), dim=-1) * gain
        return torch.fft.irfft(spectrum, n=noise.shape[-1])

    @staticmethod
    def sine_sum(phase, count):
        # sum_{k=1..count} sin(2 pi k phase), by the Dirichlet identity; the
        # angles are reduced in float64 since count reaches hundreds.
        half = phase * 0.5
        numerator = (
            torch.sin(2 * np.pi * ((count * half) % 1.0)).float()
            * torch.sin(2 * np.pi * (((count + 1) * half) % 1.0)).float()
        )
        denominator = torch.sin(np.pi * phase).float()
        safe = denominator.abs() > 1e-6
        return torch.where(safe, numerator / torch.where(safe, denominator, 1.0), 0.0)

    @torch.no_grad()
    def forward(self, f0):
        """
        Args:
            f0 (torch.Tensor): Pitch in Hz at the output rate, shape (batch, samples).
        """
        uv = (f0 > 0).float()
        nyquist = self.sample_rate / 2.0
        taper = nyquist * NYQUIST_TAPER
        safe_f0 = torch.where(uv > 0, f0, nyquist).double()
        phase = torch.cumsum(
            torch.where(uv > 0, f0, 0.0).double() / self.sample_rate, dim=-1
        )
        phase = phase % 1.0

        full = torch.floor((nyquist - taper) / safe_f0)
        harmonics = self.sine_sum(phase, full)
        power = full.float()
        # Harmonics in the top band, faded by their own frequency.
        voiced_f0 = f0[uv > 0]
        fading = 0
        if voiced_f0.numel():
            fading = int(np.ceil(taper / voiced_f0.min().item())) + 1
        for offset in range(1, fading + 1):
            order = full + offset
            fade = ((nyquist - order * safe_f0) / taper).clamp(0.0, 1.0).float()
            sine = torch.sin(2 * np.pi * ((order * phase) % 1.0)).float()
            harmonics = harmonics + fade * sine
            power = power + fade.square()

        amplitude = self.sine_amp * torch.sqrt(2.0 / power.clamp(min=1.0))
        harmonics = harmonics * amplitude * uv
        noise = torch.randn_like(uv)
        voiced_noise = self.equalize(noise) if self.noise_eq is not None else noise
        unvoiced_noise = (1 - uv) * self.sine_amp / 3 * noise
        return harmonics + uv * self.noise_std * voiced_noise + unvoiced_noise


class NoiseBranch(nn.Module):
    """
    Filtered noise added at the output and gains on the excitation, both per
    frame and per band, read from the mel by one small head.

    Args:
        num_mels (int): Number of mel bins.
        bands (int): Number of bands, spaced on the mel scale up to Nyquist.
        sample_rate (int): Sampling rate of the audio.
        hop (int): Hop size of the mel.
    """

    def __init__(self, num_mels, bands, sample_rate, hop):
        super().__init__()
        self.bands, self.hop, self.n_fft = int(bands), int(hop), 4 * int(hop)
        self.head = nn.Sequential(
            nn.Conv1d(num_mels, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 2 * self.bands, 3, padding=1),
        )

        top = 2595.0 * np.log10(1.0 + sample_rate / 2.0 / 700.0)
        centres = 700.0 * (10.0 ** (np.linspace(0.0, top, self.bands) / 2595.0) - 1.0)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        interp = np.stack(
            [np.interp(freqs, centres, row) for row in np.eye(self.bands)], axis=1
        )
        self.register_buffer(
            "interp", torch.from_numpy(interp).float(), persistent=False
        )
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def gains(self, mel):
        """
        Get the noise and the excitation gains, each (batch, bands, frames).
        """
        noise, excitation = self.head(mel).float().chunk(2, dim=1)
        noise = 2.0 * torch.sigmoid(noise) ** np.log(10.0) + 1e-7
        return noise, 2.0 * torch.sigmoid(excitation)

    def shape(self, signal, gains):
        """
        Filter a signal by the band gains.

        Args:
            signal (torch.Tensor): Signal, shape (batch, samples).
            gains (torch.Tensor): Band gains, shape (batch, bands, frames).
        """
        spectrum = torch.stft(
            signal.float(),
            self.n_fft,
            self.hop,
            window=self.window,
            return_complex=True,
        )
        frames = spectrum.shape[-1]
        if gains.shape[-1] < frames:
            gains = F.pad(gains, (0, frames - gains.shape[-1]), mode="replicate")
        per_bin = torch.einsum("fk,bkt->bft", self.interp, gains[..., :frames].float())
        return torch.istft(
            spectrum * per_bin,
            self.n_fft,
            self.hop,
            window=self.window,
            length=signal.shape[-1],
        )


class PCPHBigVGANGenerator(nn.Module):
    """
    PCPH-BigVGAN generator: BigVGAN v2 at voice scale driven by a PCPH
    excitation. Each stage is a channel projection, a fixed windowed-sinc
    upsampler, the excitation band-limited to the stage's rate and AMP blocks
    with anti-aliased SnakeBeta activations.

    Args:
        sample_rate (int): Sampling rate of the audio.
        num_mels (int): Number of mel bins.
        upsample_rates (tuple[int]): Upsampling rates.
        upsample_initial_channel (int, optional): Number of channels before the upsampling. Defaults to 512.
        resblock_kernel_sizes (tuple[int], optional): Kernel sizes of the AMP blocks. Defaults to (3, 7, 11).
        resblock_dilation_sizes (tuple[tuple[int]], optional): Dilation rates of the AMP blocks.
        resblock (str, optional): "1" for AMPBlock1, "2" for AMPBlock2. Defaults to "1".
        antialias (bool or tuple[bool], optional): Oversample the activations, per stage. Defaults to True.
        filter_width (tuple[int], optional): Half-lengths of the upsampler kernels, per stage.
        rolloff (tuple[float], optional): Rolloffs of the upsampler kernels, per stage.
        filter_beta (tuple[float], optional): Kaiser betas of the upsampler kernels, per stage.
        source_noise_std (float, optional): Noise of the voiced excitation. Defaults to 0.003.
        source_branch (str, optional): "linear" adds the excitation through one convolution, "rectified" through a learned multi-channel pyramid. Defaults to "linear".
        source_noise_eq (list, optional): `(hz, db)` points that shape the noise of the voiced excitation.
        noise_branch_bands (int, optional): Bands of the noise branch, 0 for none. Defaults to 0.
        output_gain (bool, optional): Whether the output has a learned level. Defaults to False.
        stage_channels (tuple[int], optional): Width of each stage, halved at every stage when None.
        prenet_blocks (int, optional): ConvNeXt blocks at the mel frame rate. Defaults to 0.
        deep_source_stages (int, optional): Number of the last stages that add the excitation through three convolutions. Defaults to 0.
    """

    def __init__(
        self,
        sample_rate,
        num_mels,
        upsample_rates,
        upsample_initial_channel=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        resblock="1",
        antialias=True,
        filter_width=None,
        rolloff=None,
        filter_beta=None,
        source_noise_std=0.003,
        source_branch="linear",
        source_noise_eq=None,
        noise_branch_bands=0,
        output_gain=False,
        stage_channels=None,
        prenet_blocks=0,
        deep_source_stages=0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.dc_window = int(DC_WINDOW_SECONDS * self.sample_rate) | 1
        self.upsample_rates = tuple(int(rate) for rate in upsample_rates)
        self.upp = int(np.prod(self.upsample_rates))
        self.num_kernels = len(resblock_kernel_sizes)
        self.source_branch = source_branch

        count = len(self.upsample_rates)
        if stage_channels is None:
            stage_channels = [
                int(upsample_initial_channel) // 2 ** (stage + 1)
                for stage in range(count)
            ]
        if isinstance(antialias, bool):
            antialias = (antialias,) * count
        filters = UPSAMPLE_FILTERS.get(count, {})

        def per_stage(value, name):
            value = filters[name] if value is None else value
            if isinstance(value, (int, float)):
                return (value,) * count
            return tuple(value)

        filter_width = per_stage(filter_width, "filter_width")
        rolloff = per_stage(rolloff, "rolloff")
        filter_beta = per_stage(filter_beta, "filter_beta")

        self.m_source = PCPHSource(
            self.sample_rate, noise_std=source_noise_std, noise_eq=source_noise_eq
        )
        # Output-rate excitation -> each earlier stage's rate, last stage first.
        self.source_downs = nn.ModuleList(
            [
                FixedLowPass1d(rate, stride=rate, **SOURCE_DECIMATION)
                for rate in reversed(self.upsample_rates[1:])
            ]
        )
        if source_branch == "rectified":
            source_channels = [
                SOURCE_BRANCH_CHANNELS * 2**index for index in range(count)
            ][::-1]
            self.source_pre = weight_norm(
                nn.Conv1d(1, SOURCE_BRANCH_CHANNELS, 7, 1, padding=3)
            )
            self.source_act = AntiAliasedActivation(
                leaky_relu_slope=SOURCE_BRANCH_SLOPE,
                **output_design(self.sample_rate),
            )
            self.source_blocks = nn.ModuleList(
                [
                    weight_norm(nn.Conv1d(channels, channels * 2, 7, 1, padding=3))
                    for channels in reversed(source_channels[1:])
                ]
            )
        else:
            source_channels = [1] * count

        self.noise_branch = None
        if noise_branch_bands > 0:
            self.noise_branch = NoiseBranch(
                num_mels, noise_branch_bands, self.sample_rate, self.upp
            )

        channels = int(upsample_initial_channel)
        self.conv_pre = weight_norm(nn.Conv1d(num_mels, channels, 7, 1, padding=3))
        self.prenet = None
        if prenet_blocks > 0:
            self.prenet = nn.Sequential(
                *[
                    PrenetBlock(channels, 1.0 / prenet_blocks)
                    for _ in range(prenet_blocks)
                ]
            )

        output_snake = output_design(self.sample_rate)
        self.projections = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.source_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for stage, rate in enumerate(self.upsample_rates):
            new_channels = int(stage_channels[stage])
            self.projections.append(
                weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3))
            )
            self.ups.append(
                AntiAliasedUpsample1d(
                    rate,
                    filter_width=filter_width[stage],
                    rolloff=rolloff[stage],
                    filter_beta=filter_beta[stage],
                )
            )
            self.source_convs.append(
                source_conv(
                    source_channels[stage],
                    new_channels,
                    deep=stage >= count - deep_source_stages,
                )
            )
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    AMPBlock(
                        new_channels,
                        kernel,
                        dilation,
                        antialias=bool(antialias[stage]),
                        pairs=str(resblock) == "1",
                        design=output_snake if stage == count - 1 else SNAKE_ACTIVATION,
                    )
                )
            channels = new_channels

        # No skip around these two, so their lowpass is the output's band.
        final = output_design(self.sample_rate, filter_width=64, filter_beta=8.0)
        self.activation_post = snake_activation(channels, bool(antialias[-1]), final)
        self.output_act = AntiAliasedActivation(nn.Tanh(), **final)
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        register_parametrization(self.conv_post, "weight", UnitNorm())
        self.output_log_gain = nn.Parameter(torch.zeros(())) if output_gain else None

    def excitation(self, frames, f0, band_gains=None):
        """
        Get the excitation at the output rate of every stage, first stage first.

        Args:
            frames (int): Number of mel frames.
            f0 (torch.Tensor): Pitch in Hz, shape (batch, 1, frames).
            band_gains (torch.Tensor, optional): Gains of the noise branch on the excitation.
        """
        f0 = expand_f0(f0, frames * self.upp)
        source = self.m_source(f0[:, 0])
        if band_gains is not None:
            source = self.noise_branch.shape(source, band_gains)
        source = source.unsqueeze(1)
        if self.source_branch == "rectified":
            sources = [self.source_act(self.source_pre(source))]
            for down, block in zip(self.source_downs, self.source_blocks):
                sources.append(block(down(sources[-1])))
        else:
            sources = [source]
            for down in self.source_downs:
                sources.append(down(sources[-1]))
        return sources[::-1]

    def forward(self, x, f0):
        """
        Args:
            x (torch.Tensor): Normalised log mel, shape (batch, num_mels, frames).
            f0 (torch.Tensor): Pitch in Hz, shape (batch, frames).
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        noise_gains = excitation_gains = None
        if self.noise_branch is not None:
            noise_gains, excitation_gains = self.noise_branch.gains(x)
        sources = self.excitation(x.shape[-1], f0.float(), excitation_gains)

        x = self.conv_pre(x)
        if self.prenet is not None:
            x = self.prenet(x)

        for stage in range(len(self.upsample_rates)):
            x = self.ups[stage](self.projections[stage](x))
            x = x + self.source_convs[stage](sources[stage])
            blocks = self.resblocks[
                stage * self.num_kernels : (stage + 1) * self.num_kernels
            ]
            x = sum(block(x) for block in blocks) / self.num_kernels

        x = self.conv_post(self.activation_post(x))
        if self.output_log_gain is not None:
            x = x * self.output_log_gain.exp()
        if noise_gains is not None:
            noise = torch.randn(x.shape[0], x.shape[-1], device=x.device)
            x = x + self.noise_branch.shape(noise, noise_gains).unsqueeze(1)
        x = remove_dc(x, self.dc_window)
        return self.output_act(x)

    def remove_weight_norm(self):
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
