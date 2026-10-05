"""V3's separate universal F0-conditioned complex-spectral vocoder.

Physical log-mel [B,M,T] and F0/voicing [B,T] become waveform [B,T*hop]. A
harmonic/noise prior is analyzed into learned lower-rate streams, processed in
complex frequency space, reconstructed by overlap-add and merged by a learned
synthesis filter. No target speaker embedding is used here; target timbre is
carried by mel acoustics. Critics are used only during vocoder training.

References: Wavehax (2411.06807), multistream synthesis (2506.03554). This is an
original native Torch implementation, not an official checkpoint loader.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from rvc.configs.neural import VocoderConfig
from rvc.lib.algorithm.acoustic.spectral import SameSTFT


def harmonic_prior(f0, voiced, hop, sample_rate, phase=None, noise=None):
    """Band-limited sine sum, with caller-controlled phase and noise."""
    if f0.ndim != 2 or voiced.shape != f0.shape:
        raise ValueError("F0 and voicing must have shape [batch, frames]")
    if not torch.isfinite(f0).all() or torch.any(f0 < 0):
        raise ValueError("F0 must be finite and nonnegative")
    # Hold interpolation avoids dependence on a future, unfinalized pitch frame.
    frequency = f0.float().repeat_interleave(hop, dim=-1)
    gate = voiced.float().repeat_interleave(hop, dim=-1)
    if phase is None:
        phase = torch.zeros(f0.shape[0], device=f0.device, dtype=torch.float64)
    if phase.shape != (f0.shape[0],):
        raise ValueError("Oscillator phase must have one value per stream")
    theta = phase[:, None] + torch.cumsum(
        frequency.double() * (2 * math.pi / sample_rate), dim=-1
    )
    end_phase = theta[:, -1].remainder(2 * math.pi)
    theta = theta.remainder(2 * math.pi).float()
    count = torch.floor((sample_rate / 2) / frequency.clamp_min(1))
    count = torch.where(frequency > 0, count, torch.zeros_like(count))
    denominator = 2 * torch.sin(theta / 2)
    safe = denominator.abs() > 1e-5
    numerator = torch.cos(theta / 2) - torch.cos((count + 0.5) * theta)
    signal = torch.where(
        safe,
        numerator / torch.where(safe, denominator, torch.ones_like(denominator)),
        torch.zeros_like(theta),
    )
    signal = signal * torch.sqrt(2 / count.clamp_min(1)) * 0.1 * gate
    if noise is None:
        noise = torch.randn_like(signal)
    if noise.shape != signal.shape:
        raise ValueError("Excitation noise length must equal frames times hop")
    return signal + 0.003 * noise, end_phase


class SpectralBlock(nn.Module):
    """Frequency/time residual block with per-frame channel/frequency normalization."""

    def __init__(self, channels, kernel, causal):
        super().__init__()
        self.kernel, self.causal = kernel, causal
        self.depthwise = nn.Conv2d(channels, channels, kernel, groups=channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.project = nn.Conv2d(2 * channels, channels, 1)
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))
        self.norm_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.norm_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Normalize over channels/frequency independently for each frame.
        mean = x.float().mean((1, 2), keepdim=True)
        variance = x.float().var((1, 2), keepdim=True, unbiased=False)
        y = ((x.float() - mean) * torch.rsqrt(variance + 1e-6)).to(x.dtype)
        y = y * self.norm_weight + self.norm_bias
        half = self.kernel // 2
        temporal = (self.kernel - 1, 0) if self.causal else (half, half)
        y = F.pad(y, (*temporal, half, half))
        return x + self.project(F.gelu(self.expand(self.depthwise(y)))) * self.gain


class SpectralVocoder(nn.Module):
    """Predict complex substream spectra, then synthesize full-rate PCM.

    streams divides the full-rate hop. Learned analysis/synthesis filters define
    the waveform decomposition; the internal FFT differs from the mel-target FFT.
    Causal spectral blocks still require the filter/overlap halo managed by
    VocoderStream: causal blocks alone do not imply zero algorithmic latency.
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config
        c = config
        self.bins = c.n_fft // 2 + 1
        self.analysis = nn.Conv1d(
            1,
            c.streams,
            c.filter_taps,
            stride=c.streams,
            padding=c.filter_taps // 2,
            bias=False,
        )
        self.synthesis = nn.ConvTranspose1d(
            c.streams,
            1,
            c.filter_taps,
            stride=c.streams,
            padding=c.filter_taps // 2,
            output_padding=c.streams - 1,
            bias=False,
        )
        self.stft = SameSTFT(c.n_fft, c.hop_length // c.streams)
        self.mel_projection = nn.Conv1d(c.mel_dim, c.streams * self.bins, 1)
        self.input_projection = nn.Conv2d(3 * c.streams, c.channels, 1)
        self.blocks = nn.ModuleList(
            [SpectralBlock(c.channels, c.kernel_size, c.causal) for _ in range(c.depth)]
        )
        self.output_projection = nn.Conv2d(c.channels, 2 * c.streams, 1)

    def forward(self, mel, f0, voiced=None, phase=None, noise=None, return_phase=False):
        """Render exactly T*hop samples, optionally returning oscillator end phase.

        Passing phase and absolute-position noise lets overlapping streaming windows
        reproduce full-recording excitation. Without them, this call starts a fresh
        oscillator and samples new noise.
        """

        c = self.config
        if (
            mel.ndim != 3
            or mel.shape[1] != c.mel_dim
            or f0.shape != (mel.shape[0], mel.shape[-1])
        ):
            raise ValueError("Vocoder requires matching mel [B,M,T] and F0 [B,T]")
        if voiced is None:
            voiced = (f0 > 0).to(f0.dtype)
        prior, phase = harmonic_prior(
            f0, voiced, c.hop_length, c.sample_rate, phase, noise
        )
        streams = self.analysis(prior[:, None]).float()
        batch, count, length = streams.shape
        spectrum = self.stft(streams.reshape(batch * count, length))
        spectrum = spectrum.reshape(batch, count, self.bins, mel.shape[-1])
        condition = self.mel_projection(mel).reshape(
            batch, count, self.bins, mel.shape[-1]
        )
        x = self.input_projection(
            torch.cat([spectrum.real, spectrum.imag, condition], dim=1)
        )
        for block in self.blocks:
            x = (
                checkpoint(block, x, use_reentrant=False)
                if c.checkpoint_blocks and x.requires_grad
                else block(x)
            )
        parts = (
            self.output_projection(x)
            .float()
            .reshape(batch, 2, count, self.bins, mel.shape[-1])
        )
        output_spectrum = torch.complex(parts[:, 0], parts[:, 1]).reshape(
            batch * count, self.bins, mel.shape[-1]
        )
        output_streams = self.stft.inverse(output_spectrum).reshape(batch, count, -1)
        output = self.synthesis(output_streams)[:, 0]
        if output.shape[-1] != mel.shape[-1] * c.hop_length:
            raise RuntimeError("Vocoder violated its frame/sample contract")
        return (output, phase) if return_phase else output


class PeriodCritic(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        channels = [1, 16, 64, 128, 256]
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(a, b, (5, 1), (3, 1), (2, 0))
                for a, b in zip(channels[:-1], channels[1:])
            ]
        )
        self.output = nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0))

    def forward(self, audio):
        pad = (-audio.shape[-1]) % self.period
        x = F.pad(audio[:, None], (0, pad)).reshape(audio.shape[0], 1, -1, self.period)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        x = self.output(x)
        return x, features


class WaveformCritics(nn.Module):
    """Five periodic waveform critics, omitted from inference exports."""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.critics = nn.ModuleList([PeriodCritic(p) for p in periods])
        self.spectral = nn.ModuleList([SpectralCritic(n) for n in (512, 1024, 2048)])

    def forward(self, audio):
        return [critic(audio) for critic in self.critics] + [
            critic(audio) for critic in self.spectral
        ]


class SpectralCritic(nn.Module):
    """Training-only critic over real/imaginary STFT channels at one resolution."""

    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(n_fft))
        channels = (2, 16, 32, 64)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(a, b, (3, 5), stride=(1, 2), padding=(1, 2))
                for a, b in zip(channels[:-1], channels[1:])
            ]
        )
        self.output = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, audio):
        with torch.autocast(device_type=audio.device.type, enabled=False):
            spectrum = torch.stft(
                audio.float(),
                self.n_fft,
                self.n_fft // 4,
                window=self.window.float(),
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            x = torch.stack([spectrum.real, spectrum.imag], dim=1)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


def discriminator_loss(real, fake):
    return sum(
        (r[0].float() - 1).square().mean() + f[0].float().square().mean()
        for r, f in zip(real, fake, strict=True)
    ) / len(real)


def generator_loss(real, fake):
    adversarial = sum((f[0].float() - 1).square().mean() for f in fake) / len(fake)
    matching = sum(
        sum(
            (a.detach().float() - b.float()).abs().mean()
            for a, b in zip(r[1], f[1], strict=True)
        )
        / len(r[1])
        for r, f in zip(real, fake, strict=True)
    ) / len(fake)
    return adversarial, matching
