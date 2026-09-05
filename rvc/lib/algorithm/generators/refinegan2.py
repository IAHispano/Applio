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
    AntiAliasedUpsample1d,
    filter_schedule,
)


# Stage rates per sample rate. Not the config's ascending ``upsample_rates``:
# this decoder wants them descending, so the last residual block has the least
# left to synthesise from scratch. They leave no trace in the state dict, which
# is why they are a table here and not a config key.
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


# Interpolation filter for the trunk's upsamplers, one entry per stage.
# Zero-stuffing copies the input spectrum to every multiple of the input rate;
# what the filter leaves of those copies is an image, and an image at
# ``k*R_in - j*f0`` moves against f0 exactly like a fold does. Only the last
# stage gets a long kernel: its image is the loudest at the output, and the
# early stages are short enough that a long kernel reads more of the padding
# than of the signal.
DEFAULT_UPSAMPLE_WIDTH = (12, 24, 32, 48)
DEFAULT_UPSAMPLE_ROLLOFF = (0.90, 0.95, 0.97, 0.99)
DEFAULT_UPSAMPLE_BETA = (6.0, 6.0, 6.0, 9.0)

# The excitation gain is one channel, so its upsample chain is free whatever
# the kernel length -- and it is the one path where an image is multiplied onto
# every harmonic as a sideband. It gets the longest design at every stage.
SOURCE_GAIN_WIDTH = 48
SOURCE_GAIN_ROLLOFF = 0.99
SOURCE_GAIN_BETA = 9.0


class ResBlock(nn.Module):
    """
    Residual block with multiple dilated convolutions.

    Args:
        channels (int): Number of channels.
        kernel_size (int, optional): Kernel size for the convolutional layers. Defaults to 7.
        dilation (tuple[int], optional): Dilation rates for the convolutional layers. Defaults to (1, 3, 5).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope

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

    def forward(self, x: torch.Tensor):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x

        return x


class AdaIN(nn.Module):
    """
    Noise-regularised activation, wrapped either side of every ResBlock.

    The noise is a training-time regulariser only; in eval this is a plain
    Leaky ReLU.

    Args:
        channels (int): Number of channels.
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        # safe to use in-place as it is used on a new x+gaussian tensor
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor):
        # skipped in eval: it is a regulariser, and it is 25% of the forward
        if not self.training:
            return self.activation(x)

        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    """
    Runs several ResBlocks with different kernel sizes in parallel and averages them.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_sizes (tuple[int], optional): Kernel size of each parallel block. Defaults to (3, 7, 11).
        dilation (tuple[int], optional): Dilation rates inside each block. Defaults to (1, 3, 5).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

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
                    ),
                    ResBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(
                        channels=out_channels,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)


class BlitGenerator(nn.Module):
    """
    Band-limited impulse train excitation, replacing the sine source.

    A sine carries one partial, so the trunk has to manufacture every harmonic
    above it out of activation products. A BLIT hands the trunk every harmonic
    under Nyquist already, and it is band-limited by construction: it is the
    Dirichlet kernel ``sin(pi*M*phi) / (M*sin(pi*phi))``, which is a sum of M
    cosines and nothing else. The harmonic count follows f0 per sample and is
    fractional, so a moving pitch does not step it discontinuously.

    Args:
        samp_rate (int): Output sample rate in Hz.
        wave_amp (float, optional): Excitation level. Under ``normalize`` it is the
            amplitude of the equivalent sine; otherwise it is the pulse's peak. Defaults to 0.1.
        noise_std (float, optional): Gaussian noise std in voiced regions. Defaults to 0.003.
        voiced_threshold (float, optional): f0 above which a frame counts as voiced. Defaults to 0.0.
        bandwidth (float, optional): Fraction of Nyquist to fill, in (0, 1]. 1.0 is the
            true BLIT and the only control this decoder has over activation fold;
            lowering it delivers less detail but less intermodulation downstream. Defaults to 1.0.
        learn_gain (bool, optional): A single learned scalar on the excitation level. Defaults to True.
        normalize (bool, optional): Hold the excitation's energy constant across the
            pitch range instead of its peak, which is worth 13-23 dB of level
            depending on the note. Defaults to True.
    """

    def __init__(
        self,
        samp_rate: int,
        wave_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        bandwidth: float = 1.0,
        learn_gain: bool = True,
        normalize: bool = True,
    ):
        super().__init__()

        self.sampling_rate = int(samp_rate)
        self.wave_amp = float(wave_amp)
        self.noise_std = float(noise_std)
        self.voiced_threshold = float(voiced_threshold)
        self.bandwidth = float(bandwidth)
        self.normalize = bool(normalize)

        # One scalar, with grad, so the excitation level is learned while the
        # waveform itself stays a pure function of f0.
        self.gain = (
            nn.Parameter(torch.ones(1)) if learn_gain else None
        )

    # Inductor cannot compile the phase cumsum; it is a pure function of f0
    # under no_grad, so keeping it out of the graph costs no fusion.
    @torch.compiler.disable
    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """f0: (batch, 1, samples) at the output rate.  Returns the same shape."""

        with torch.no_grad():
            uv = (f0 > self.voiced_threshold).to(f0.dtype)
            f0_safe = f0.clamp_min(1.0)

            # float64: in float32 the running sum reaches ~1e4 cycles within a
            # second of audio, which is audible phase jitter on every harmonic
            phase = torch.cumsum(f0_safe.double() / self.sampling_rate, dim=-1)
            phase = phase - torch.floor(phase)

            # Fractional harmonic count. An integer count evaluated per sample
            # steps whenever f0 crosses ``limit / N``, and every step is a
            # discontinuity whose rate tracks the pitch -- which reads as a
            # fold in a spectrogram. Weighting the top conjugate pair by the
            # fraction makes the count continuous in f0, for one extra cosine
            # (5-cent vibrato: -67.1 -> -106.7 dB in the 20 Hz .. f0/2 band).
            #
            # The ``- f0_safe`` is a one-partial margin: whenever f0 divides
            # sr/2 the top partial would land on Nyquist, where its conjugate
            # image doubles it, 6 dB above every other partial.
            limit = self.bandwidth * self.sampling_rate / 2.0 - f0_safe.double()
            kmax = (limit / f0_safe.double()).clamp_min(1.0)
            n_har = torch.floor(kmax)
            w = kmax - n_har
            m = 2.0 * n_har + 1.0

            denominator = torch.sin(np.pi * phase)
            # phi -> 0 is the removable singularity where every cosine is in
            # phase and the unnormalised kernel equals M.
            singular = denominator.abs() < 1e-12
            core = torch.where(
                singular,
                m,
                torch.sin(np.pi * m * phase)
                / torch.where(singular, torch.ones_like(denominator), denominator),
            )
            core = core + 2.0 * w * torch.cos(2.0 * np.pi * (n_har + 1.0) * phase)
            # The value at phi = 0, so the kernel keeps its unit peak. The
            # normalisation below divides by this one too, or the integer step
            # would go straight back into the level.
            weight = m + 2.0 * w

            blit = core / weight
            # The kernel is normalised to a unit *peak*, so each of its M
            # harmonics carries 1/M and the excitation gets quieter the lower
            # the note -- a 20 dB tilt across a singer's range that nothing
            # downstream knows about, and it inverted the voiced/unvoiced
            # balance too. ``sqrt(M/2)`` puts the RMS at ``wave_amp / sqrt(2)``
            # at every pitch, which is exactly a sine of amplitude ``wave_amp``.
            if self.normalize:
                blit = blit * torch.sqrt(weight / 2.0)
            blit = blit.to(f0.dtype) * self.wave_amp

            # Unvoiced regions are noise; voiced ones get a small dither. Both
            # sides are RMS since the normalisation above, so the voiced /
            # unvoiced ratio is 2.12, as it was under the sine source.
            noise_amp = uv * self.noise_std + (1.0 - uv) * self.wave_amp / 3.0
            excitation = blit * uv + noise_amp * torch.randn_like(blit)

        if self.gain is not None:
            excitation = excitation * self.gain

        return excitation


class RefineGAN2Generator(nn.Module):
    """
    RefineGAN2 generator for audio synthesis.

    Downsamples and upchannels an excitation, fuses it with the latent, and
    upsamples through parallel residual blocks. Against the original: a
    band-limited impulse train instead of the sine, descending stage rates, a
    windowed-sinc interpolation filter that crops its own group delay, an
    excitation gain projected from the conditioning, and f0 interpolated in log
    with a hard voiced/unvoiced gate. Every pointwise nonlinearity is a plain
    Leaky ReLU at its own rate.

    Args:
        sample_rate (int, optional): Sampling rate of the audio. Defaults to 32000.
        upsample_rates (tuple[int], optional): Upsampling rate of each stage, descending. Defaults to (8, 8, 2, 2).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
        num_mels (int, optional): Number of channels in the conditioning. Defaults to 128.
        start_channels (int, optional): Channels in the first downsampling block. Defaults to 16.
        gin_channels (int, optional): Channels for the global conditioning input. Defaults to 256.
        checkpointing (bool, optional): Whether to use checkpointing for memory efficiency. Defaults to False.
        upsample_initial_channel (int, optional): Channels at the top of the trunk. Defaults to 512.
        filter_width (int | Sequence[int], optional): Interpolation filter length, scalar or one per stage.
        rolloff (float | Sequence[float], optional): Fraction of the stage's Nyquist the filter keeps.
        filter_beta (float | Sequence[float], optional): Kaiser beta for the interpolation filter.
        source_gain (bool, optional): Scale the excitation by an intensity envelope
            projected from the conditioning, as RefineGAN's paper does with the mel. Defaults to False.
        source_bandwidth (float, optional): Fraction of Nyquist the BLIT fills. Defaults to 1.0.
        source_normalize (bool, optional): Hold the excitation's energy constant across
            the pitch range instead of its peak. Defaults to True.
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
        source_gain: bool = False,
        source_bandwidth: float = 1.0,
        source_normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing

        # Scalar or one per stage, normalised in one place. The down path
        # doubles start_channels per stage and the up path expects the skip to
        # be a quarter of the trunk, so the two only meet at one value.
        count = len(upsample_rates)
        self.filter_width = filter_schedule(filter_width, count)
        self.rolloff = filter_schedule(rolloff, count)
        self.filter_beta = filter_schedule(filter_beta, count)

        # ``int``, not the np.int64 np.prod returns: Dynamo wraps a numpy
        # scalar as a CPU tensor, and one CPU node makes Inductor emit a C++
        # kernel, which on Windows needs cl.exe and fails the whole compile.
        self.upp = int(np.prod(upsample_rates))

        # The excitation. Neither of these appears in the state dict --
        # BlitGenerator owns one scalar parameter whatever they are -- so a
        # checkpoint trained against one source loads into another silently.
        self.source_type = "blit"
        self.source_bandwidth = float(source_bandwidth)
        self.source_normalize = bool(source_normalize)
        self.m_source = BlitGenerator(
            sample_rate,
            bandwidth=source_bandwidth,
            normalize=source_normalize,
        )

        # ``start_channels``, not a literal 16.  It was hardcoded here while
        # the down path below was built from ``start_channels``, so any value
        # but 16 produced a channel mismatch at ``downsample_blocks[0]`` -- a
        # config knob that could only take one value, which is worse than no
        # knob.
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
            self.cond = nn.Conv1d(gin_channels, channels // 2, 1)

        # The paper scales its template by intensity values read off the mel;
        # this decoder is handed z, from which a least-squares fit recovers the
        # log intensity at r = 0.996. The BLIT carries no envelope of its own,
        # so this is worth having: held-out multi-scale mel on a fixed trunk
        # improves 1.97 -> 1.74, for 193 parameters.
        self.has_source_gain = bool(source_gain)
        if self.has_source_gain:
            self.source_gain = nn.Conv1d(num_mels, 1, 1)
            # Identity at initialisation (softplus(0.5413) = 1.0 with zero
            # weights), so switching this on starts from exactly the excitation
            # the run had before and the projection earns every departure.
            nn.init.zeros_(self.source_gain.weight)
            nn.init.constant_(self.source_gain.bias, 0.5413248546129181)

            # The gain multiplies the excitation, so a residual image in it
            # stamps a sideband onto every harmonic. This chain runs on
            # (B, 1, T), where taps are free, so every stage gets the longest
            # design rather than the trunk's schedule.
            self.source_gain_ups = nn.ModuleList(
                [
                    AntiAliasedUpsample1d(
                        rate,
                        filter_width=SOURCE_GAIN_WIDTH,
                        rolloff=SOURCE_GAIN_ROLLOFF,
                        filter_beta=SOURCE_GAIN_BETA,
                    )
                    for rate in upsample_rates
                ]
            )

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for stage, rate in enumerate(upsample_rates):
            new_channels = channels // 2

            # Was nn.Upsample(mode="linear"), whose triangular kernel rejects
            # the first image by only 1.7-9.6 dB, stamping the frame grid into
            # the waveform as a mirrored partial either side of every harmonic.
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
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        )
        self.conv_post.apply(init_weights)

        self.out_tanh = nn.Tanh()

    # torchaudio builds its sinc kernel from Python ints on every call, which
    # Inductor compiles to a CPU kernel and fails on Windows without cl.exe.
    # Kept out of the graph rather than replaced: this filter is what keeps
    # each decimation from folding the harmonics it discards.
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

    @staticmethod
    def _expand_f0(f0: torch.Tensor, length: int) -> torch.Tensor:
        """
        f0 at the frame rate -> f0 at the output rate, (batch, 1, length).

        Interpolating in Hz makes the frame-rate ripple a constant absolute
        wobble, whose sidebands grow with the harmonic number; in log it is
        constant in cents instead. And interpolating across a voiced/unvoiced
        boundary ramps f0 toward zero while the gate stays open, which chirps
        every harmonic at once, so the gate is interpolated separately.
        """

        voiced = (f0 > 0).to(f0.dtype)
        # Interpolate the *pitch*, in log Hz, and the gate separately.
        log_f0 = torch.log(f0.clamp_min(1.0))
        log_f0 = F.interpolate(log_f0, size=length, mode="linear", align_corners=False)
        voiced = F.interpolate(voiced, size=length, mode="nearest")
        return torch.exp(log_f0) * voiced

    def _apply_source_gain(self, har_source: torch.Tensor, mel: torch.Tensor):
        """
        Scale the excitation by an intensity envelope read off the
        conditioning, which arrives at the frame rate as ``mel``.
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
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        f0 = self._expand_f0(f0, f0_size * self.upp)
        har_source = self.m_source(f0)
        har_source = self._apply_source_gain(har_source, mel)
        x = self.pre_conv(har_source)
        downs = []
        for block, (old_size, new_size) in zip(self.downsample_blocks, self.df0):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)
            x = self._decimate(x, int(f0_size * old_size), int(f0_size * new_size))
            x = block(x)

        mel = self.mel_conv(mel)
        if g is not None:
            mel = mel + self.cond(g)

        x = torch.cat([mel, x], dim=1)

        for ups, res, down in zip(
            self.upsample_blocks,
            self.upsample_conv_blocks,
            reversed(downs),
        ):
            x = F.leaky_relu(x, self.leaky_relu_slope)

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
        x = self.out_tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        """
        Fold every weight norm back into its weight, by walking the modules
        rather than listing them by name.
        """

        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)