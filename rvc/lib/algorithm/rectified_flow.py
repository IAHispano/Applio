import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F

# Centre and spread of log f0, so the normalised pitch sits roughly in [-2, 2].
LOG_F0_CENTER = math.log(200.0)
LOG_F0_SCALE = 0.7
SAMPLERS = ("euler", "heun")
SCHEDULES = ("uniform", "sway", "logit-normal")
RESCALE_MODES = ("global", "frame")
# How training draws its times: logit-normal hardly trains the ends of the
# range, where the sampling of a shallow flow starts.
TIME_SAMPLINGS = ("uniform", "logit-normal")
# Share of the frames that get the second time under dual timestep.
DUAL_TIMESTEP_SHARE = 0.25
# Inputs that start at zero, so a checkpoint from before them loads unchanged.
ZERO_INPUTS = ("encoder.tension.",)


class Conditioning(NamedTuple):
    """
    The inputs the flow is conditioned on. The optional ones are ignored by a
    model that does not take them.

    Args:
        content (torch.Tensor): Content features, shape (batch, frames, channels).
        f0 (torch.Tensor): Pitch in Hz, shape (batch, frames).
        energy (torch.Tensor): Loudness curve, shape (batch, frames).
        speaker (torch.Tensor): Speaker ids, shape (batch,).
        mask (torch.Tensor): Frame mask, shape (batch, 1, frames).
        breathiness (torch.Tensor, optional): Aperiodicity curve, fully aperiodic when None.
        key_shift (torch.Tensor, optional): Formant shift in semitones, 0 when None.
        speed (torch.Tensor, optional): Time stretch, 1 when None.
        tension (torch.Tensor, optional): Tension curve, 0 when None.
    """

    content: torch.Tensor
    f0: torch.Tensor
    energy: torch.Tensor
    speaker: torch.Tensor
    mask: torch.Tensor
    breathiness: Optional[torch.Tensor] = None
    key_shift: Optional[torch.Tensor] = None
    speed: Optional[torch.Tensor] = None
    tension: Optional[torch.Tensor] = None

    def map(self, function):
        """
        Applies a function to every input that is not None.

        Args:
            function (Callable): The function to apply.
        """
        return Conditioning(
            *(None if value is None else function(value) for value in self)
        )

    def to(self, device, non_blocking: bool = False):
        """
        Moves every input to a device.

        Args:
            device (torch.device): The target device.
            non_blocking (bool, optional): Whether to copy asynchronously. Defaults to False.
        """
        return self.map(lambda value: value.to(device, non_blocking=non_blocking))

    def crop(self, start: int, stop: int):
        """
        Slices the per-frame inputs.

        Args:
            start (int): First frame.
            stop (int): Frame after the last one.
        """

        def cut(value):
            return None if value is None else value[:, start:stop]

        return self._replace(
            content=cut(self.content),
            f0=cut(self.f0),
            energy=cut(self.energy),
            mask=self.mask[..., start:stop],
            breathiness=cut(self.breathiness),
            tension=cut(self.tension),
        )


def time_grid(schedule: str, steps: int, start: float, device):
    """
    Sampling times from `start` to 1.

    Args:
        schedule (str): One of SCHEDULES.
        steps (int): Number of steps; the grid has `steps + 1` times.
        start (float): First time of the grid.
        device (torch.device): Device of the returned tensor.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, not {schedule!r}.")
    u = torch.linspace(0.0, 1.0, steps + 1, device=device)
    if schedule == "sway":
        g = 1.0 - torch.cos(0.5 * math.pi * u)
    elif schedule == "logit-normal":
        g = torch.sigmoid(
            math.sqrt(2.0) * torch.erfinv((2.0 * u - 1.0).clamp(-1.0, 1.0))
        )
    else:
        g = u
    g[0], g[-1] = 0.0, 1.0
    return start + (1.0 - start) * g


def pitch_features(f0: torch.Tensor, fourier: int = 0):
    """
    Normalised log f0, the voiced flag and sines/cosines of the log f0 at
    octave-spaced frequencies, all 0 where unvoiced.

    Args:
        f0 (torch.Tensor): Pitch in Hz, shape (batch, frames).
        fourier (int, optional): Number of sine/cosine pairs. Defaults to 0.
    """
    voiced = (f0 > 0).float()
    log_f0 = (torch.log(f0.clamp_min(1.0)) - LOG_F0_CENTER) / LOG_F0_SCALE
    features = [log_f0 * voiced, voiced]
    for index in range(fourier):
        angle = (2.0**index * math.pi) * log_f0
        features += [torch.sin(angle) * voiced, torch.cos(angle) * voiced]
    return torch.stack(features, dim=1)


def timestep_embedding(t: torch.Tensor, channels: int, scale: float = 1000.0):
    """
    Sinusoidal embedding of the flow time.

    Args:
        t (torch.Tensor): Times, shape (batch,).
        channels (int): Size of the embedding.
        scale (float, optional): Multiplier of the time. Defaults to 1000.0.
    """
    half = channels // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    angles = (t.float() * scale)[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class HarmonicPrior(nn.Module):
    """
    Where the harmonics of f0 fall in the mel, 0 where unvoiced.

    Args:
        sample_rate (int): Sampling rate of the audio.
        n_fft (int): FFT size of the mel.
        n_mels (int): Number of mel bins.
        fmin (float): Lowest frequency of the mel.
        fmax (float): Highest frequency of the mel.
    """

    def __init__(self, sample_rate, n_fft, n_mels, fmin, fmax):
        super().__init__()
        basis = torch.from_numpy(
            librosa_mel_fn(
                sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
            )
        ).float()
        basis = basis / basis.sum(1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer(
            "freqs",
            torch.fft.rfftfreq(n_fft, 1.0 / sample_rate).float(),
            persistent=False,
        )
        self.sigma = sample_rate / n_fft

    def forward(self, f0: torch.Tensor):
        f0 = f0.float()
        ratio = self.freqs[None, :, None] / f0.clamp_min(1.0)[:, None, :]
        nearest = ratio.round()
        distance = (ratio - nearest).abs() * f0[:, None, :]
        comb = torch.exp(-0.5 * (distance / self.sigma).square()) * (nearest >= 1)
        with torch.autocast(f0.device.type, enabled=False):
            prior = torch.matmul(self.basis, comb)
        return prior * (f0 > 0).float()[:, None, :]


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block over (batch, channels, frames).

    Args:
        channels (int): Number of channels.
        layer_scale (float, optional): Initial scale of the branch, 0 for none. Defaults to 0.0.
        dropout (float, optional): Dropout of the branch. Defaults to 0.0.
        speaker_channels (int, optional): Size of the speaker embedding that modulates the block, 0 for none. Defaults to 0.
    """

    def __init__(
        self,
        channels: int,
        layer_scale: float = 0.0,
        dropout: float = 0.0,
        speaker_channels: int = 0,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.up = nn.Linear(channels, channels * 4)
        self.down = nn.Linear(channels * 4, channels)
        self.gamma = (
            nn.Parameter(torch.full((channels,), layer_scale))
            if layer_scale > 0
            else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.speaker = None
        if speaker_channels > 0:
            self.speaker = nn.Linear(speaker_channels, channels * 2)
            nn.init.zeros_(self.speaker.weight)
            nn.init.zeros_(self.speaker.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        voice: Optional[torch.Tensor] = None,
    ):
        y = self.norm(self.depthwise(x * mask).transpose(1, 2))
        if self.speaker is not None:
            shift, scale = self.speaker(voice)[:, None, :].chunk(2, dim=-1)
            # Not `1 + scale`: BF16 rounds small modulations to nothing.
            y = y + y * scale + shift
        y = self.down(F.gelu(self.up(y)))
        if self.gamma is not None:
            y = y * self.gamma
        return (x + self.dropout(y.transpose(1, 2))) * mask


class ConditionEncoder(nn.Module):
    """
    Content, pitch, loudness, breathiness, tension, key shift, speed and speaker
    to the per-frame conditioning. Speaker row `speaker_count` is the null
    speaker used for classifier-free guidance.

    Args:
        content_channels (int): Number of content channels.
        hidden_channels (int): Number of conditioning channels.
        speaker_count (int): Number of speakers.
        speaker_channels (int): Size of the speaker embedding.
        layers (int): Number of ConvNeXt blocks.
        pitch_fourier (int, optional): Sine/cosine pairs of the pitch. Defaults to 0.
        harmonic_prior (HarmonicPrior, optional): Harmonic prior of the pitch. Defaults to None.
        breathiness (bool, optional): Whether to take the breathiness curve. Defaults to False.
        key_shift (bool, optional): Whether to take the formant shift. Defaults to False.
        speed (bool, optional): Whether to take the time stretch. Defaults to False.
        tension (bool, optional): Whether to take the tension curve. Defaults to False.
    """

    def __init__(
        self,
        content_channels: int,
        hidden_channels: int,
        speaker_count: int,
        speaker_channels: int,
        layers: int,
        pitch_fourier: int = 0,
        harmonic_prior: Optional[HarmonicPrior] = None,
        breathiness: bool = False,
        key_shift: bool = False,
        speed: bool = False,
        tension: bool = False,
    ):
        super().__init__()
        self.speaker_count = int(speaker_count)
        self.content = nn.Linear(content_channels, hidden_channels)
        self.pitch_fourier = int(pitch_fourier)
        self.pitch = nn.Conv1d(
            2 + 2 * self.pitch_fourier, hidden_channels, 3, padding=1
        )
        self.harmonic_prior = harmonic_prior
        if harmonic_prior is not None:
            self.harmonics = nn.Conv1d(
                harmonic_prior.basis.shape[0], hidden_channels, 1
            )
        self.energy = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.breathiness = (
            nn.Conv1d(1, hidden_channels, 3, padding=1) if breathiness else None
        )
        self.tension = None
        if tension:
            self.tension = nn.Conv1d(1, hidden_channels, 3, padding=1)
            nn.init.zeros_(self.tension.weight)
            nn.init.zeros_(self.tension.bias)
        self.key_shift = nn.Linear(1, hidden_channels) if key_shift else None
        self.speed = nn.Linear(1, hidden_channels) if speed else None
        self.speaker = nn.Embedding(self.speaker_count + 1, speaker_channels)
        self.speaker_proj = nn.Linear(speaker_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(hidden_channels) for _ in range(layers)]
        )

    def voice(self, speaker: torch.Tensor):
        return self.speaker_proj(self.speaker(speaker))

    def forward(self, inputs: Conditioning):
        """
        Returns the per-frame conditioning, shape (batch, hidden_channels, frames).

        Args:
            inputs (Conditioning): The inputs of the flow.
        """
        (
            content,
            f0,
            energy,
            speaker,
            mask,
            breathiness,
            key_shift,
            speed,
            tension,
        ) = inputs
        x = self.content(content).transpose(1, 2)
        x = x + self.pitch(pitch_features(f0, self.pitch_fourier))
        if self.harmonic_prior is not None:
            x = x + self.harmonics(self.harmonic_prior(f0))
        x = x + self.energy(energy.unsqueeze(1))
        if self.breathiness is not None:
            if breathiness is None:
                breathiness = torch.ones_like(energy)
            x = x + self.breathiness(breathiness.unsqueeze(1))
        if self.tension is not None:
            if tension is None:
                tension = torch.zeros_like(energy)
            x = x + self.tension(tension.unsqueeze(1))
        if self.key_shift is not None:
            if key_shift is None:
                key_shift = torch.zeros(content.shape[0], device=content.device)
            x = x + self.key_shift(key_shift.float().view(-1, 1) / 12.0).unsqueeze(-1)
        if self.speed is not None:
            if speed is None:
                speed = torch.ones(content.shape[0], device=content.device)
            x = x + self.speed(speed.float().view(-1, 1)).unsqueeze(-1)
        x = x + self.voice(speaker).unsqueeze(-1)
        x = x * mask
        for block in self.blocks:
            x = block(x, mask)
        return x


class _ATanGLU(torch.autograd.Function):
    """`out * atan(gate)`, keeping two tensors for backward instead of three."""

    @staticmethod
    def forward(ctx, out, gate):
        atan_gate = torch.atan(gate)
        ctx.save_for_backward(out / gate.square().add(1.0), atan_gate)
        return out * atan_gate

    @staticmethod
    def backward(ctx, grad):
        decay_out, atan_gate = ctx.saved_tensors
        return grad * atan_gate, grad * decay_out


def atan_glu(x: torch.Tensor):
    """
    ATanGLU activation.

    Args:
        x (torch.Tensor): Input, split in two along the last dimension.
    """
    out, gate = x.chunk(2, dim=-1)
    if torch.is_grad_enabled():
        return _ATanGLU.apply(out, gate)
    return out * torch.atan(gate)


class LYNXNet2Block(nn.Module):
    """
    Depthwise conv, then two ATanGLU projections, pre-norm and residual.

    Args:
        channels (int): Number of channels.
        expansion (float): Expansion of the inner projections.
        kernel_size (int): Kernel size of the depthwise conv.
        adaln (bool, optional): Modulate the block by the time and speaker embedding (adaLN-Zero). Defaults to False.
    """

    def __init__(self, channels, expansion, kernel_size, adaln=False):
        super().__init__()
        inner = int(channels * expansion)
        self.norm = nn.LayerNorm(channels, elementwise_affine=not adaln)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.up = nn.Linear(channels, inner * 2)
        self.mid = nn.Linear(inner, inner * 2)
        self.down = nn.Linear(inner, channels)
        self.modulation = None
        if adaln:
            self.modulation = nn.Linear(channels, channels * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(self, x, mask, embedding=None):
        y = self.norm(x)
        gate = None
        if self.modulation is not None:
            shift, scale, gate = self.modulation(F.silu(embedding)).chunk(3, dim=-1)
            # Not `1 + scale`: BF16 rounds small modulations to nothing.
            y = y + y * scale + shift
        y = self.depthwise((y * mask).transpose(1, 2)).transpose(1, 2)
        y = self.down(atan_glu(self.mid(atan_glu(self.up(y)))))
        if gate is not None:
            y = y + gate * y
        return (x + y) * mask


class LYNXNet2Backbone(nn.Module):
    """
    DiffSinger's LYNXNet2: condition and time added at the input, then
    depthwise-separable gated blocks.

    Args:
        n_mels (int): Number of mel bins.
        cond_channels (int): Number of conditioning channels.
        channels (int, optional): Number of channels. Defaults to 1024.
        layers (int, optional): Number of blocks. Defaults to 6.
        expansion (float, optional): Expansion of the blocks. Defaults to 1.
        kernel_size (int, optional): Kernel size of the blocks. Defaults to 31.
        adaln (bool, optional): Modulate every block by time and speaker. Defaults to False.
        time_scale (float, optional): Multiplier of the flow time before its sinusoids. Defaults to 1000.0.
    """

    def __init__(
        self,
        n_mels,
        cond_channels,
        channels=1024,
        layers=6,
        expansion=1,
        kernel_size=31,
        adaln=False,
        time_scale=1000.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.time_scale = float(time_scale)
        self.input = nn.Linear(n_mels, channels)
        self.input_cond = nn.Conv1d(cond_channels, channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.layers = nn.ModuleList(
            [
                LYNXNet2Block(channels, expansion, kernel_size, adaln)
                for _ in range(layers)
            ]
        )
        self.voice = nn.Linear(cond_channels, channels) if adaln else None
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, n_mels)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.input_cond.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, cond, mask, voice=None):
        """
        Args:
            x (torch.Tensor): Noisy mel, shape (batch, n_mels, frames).
            t (torch.Tensor): Flow time, shape (batch,) or (batch, frames).
            cond (torch.Tensor): Conditioning, shape (batch, cond_channels, frames).
            mask (torch.Tensor): Frame mask, shape (batch, 1, frames).
            voice (torch.Tensor, optional): Speaker embedding, shape (batch, cond_channels).
        """
        time = self.time_mlp(
            timestep_embedding(t.reshape(-1), self.channels, self.time_scale)
        )
        time = time.view(t.shape[0], -1, self.channels)
        frame_mask = mask.transpose(1, 2)
        # Full precision in: at late t the leftover noise is under BF16's step.
        with torch.autocast(x.device.type, enabled=False):
            h = self.input(x.transpose(1, 2).to(self.input.weight.dtype))
        h = h + self.input_cond(cond).transpose(1, 2) + time
        h = h * frame_mask
        embedding = None
        if self.voice is not None:
            embedding = time + self.voice(voice)[:, None, :]
        for layer in self.layers:
            h = layer(h, frame_mask, embedding)
        h = self.norm(h)
        return (self.output(h) * frame_mask).transpose(1, 2)


class AuxDecoder(nn.Module):
    """
    DiffSinger's ConvNeXt aux decoder: a deterministic mel from the
    conditioning, where shallow sampling starts.

    Args:
        cond_channels (int): Number of conditioning channels.
        n_mels (int): Number of mel bins.
        channels (int, optional): Number of channels. Defaults to 512.
        layers (int, optional): Number of blocks. Defaults to 6.
        dropout (float, optional): Dropout of the blocks. Defaults to 0.1.
        speaker (bool, optional): Modulate every block by the speaker embedding. Defaults to False.
    """

    def __init__(
        self, cond_channels, n_mels, channels=512, layers=6, dropout=0.1, speaker=False
    ):
        super().__init__()
        self.input = nn.Conv1d(cond_channels, channels, 7, padding=3)
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    channels,
                    layer_scale=1e-6,
                    dropout=dropout,
                    speaker_channels=cond_channels if speaker else 0,
                )
                for _ in range(layers)
            ]
        )
        self.output = nn.Conv1d(channels, n_mels, 7, padding=3)
        self.output.use_adamw = True

    def forward(self, cond, mask, voice=None):
        x = self.input(cond) * mask
        for block in self.blocks:
            x = block(x, mask, voice)
        return self.output(x) * mask


@dataclass(frozen=True)
class _Guidance:
    """
    The guidance of the sampler and the velocity it makes of the passes.

    Args:
        cfg_scale (float, optional): Guidance away from the null speaker, 1 is off. Defaults to 1.0.
        content_guidance (float, optional): Guidance away from the blurred content, 0 is off. Defaults to 0.0.
        rescale (float, optional): Pull of the guided velocity's spread back to the unguided one's. Defaults to 0.0.
        rescale_mode (str, optional): One of RESCALE_MODES. Defaults to "global".
        interval (tuple, optional): Flow time the guidances apply in. Defaults to (0.0, 1.0).
    """

    cfg_scale: float = 1.0
    content_guidance: float = 0.0
    rescale: float = 0.0
    rescale_mode: str = "global"
    interval: tuple = (0.0, 1.0)

    def variants(self, inputs: Conditioning, null_speaker: int):
        """
        Returns the (content, speaker) pair of each pass: the plain one, then
        the null speaker and the blurred content when their guidance is on.

        Args:
            inputs (Conditioning): The inputs of the flow.
            null_speaker (int): Id of the null speaker.
        """
        variants = [(inputs.content, inputs.speaker)]
        if self.cfg_scale != 1.0:
            null = torch.full_like(inputs.speaker, null_speaker)
            variants.append((inputs.content, null))
        if self.content_guidance > 0:
            frames = inputs.content.shape[1]
            blurred = F.interpolate(
                inputs.content.transpose(1, 2), size=max(1, frames // 4), mode="linear"
            )
            blurred = F.interpolate(blurred, size=frames, mode="linear").transpose(
                1, 2
            )
            variants.append((blurred, inputs.speaker))
        return variants

    def active(self, now: float):
        """
        Whether the guidances apply at a flow time.

        Args:
            now (float): The flow time.
        """
        start, until = self.interval
        # An interval reaching 1 includes it, where Heun's last evaluation lands.
        return start <= now and (now < until or until >= 1.0)

    def combine(self, passes, mask: torch.Tensor):
        """
        Returns the guided velocity.

        Args:
            passes (tuple): Velocity of each pass, in the order of `variants`.
            mask (torch.Tensor): Frame mask, shape (batch, 1, frames).
        """
        plain = passes[0]
        guided, index = plain, 1
        if self.cfg_scale != 1.0:
            guided = guided + (self.cfg_scale - 1.0) * (plain - passes[index])
            index += 1
        if self.content_guidance > 0:
            guided = guided + self.content_guidance * (plain - passes[index])
        if self.rescale > 0:
            spread = self.spread(guided, mask).clamp_min(1e-6)
            rescaled = guided * self.spread(plain, mask) / spread
            guided = self.rescale * rescaled + (1.0 - self.rescale) * guided
        return guided

    def spread(self, velocity: torch.Tensor, mask: torch.Tensor):
        """
        Returns the RMS of a velocity, per frame or over each item.

        Args:
            velocity (torch.Tensor): Velocity, shape (batch, n_mels, frames).
            mask (torch.Tensor): Frame mask, shape (batch, 1, frames).
        """
        if self.rescale_mode == "frame":
            return velocity.square().mean(1, keepdim=True).sqrt()
        count = mask.sum((1, 2)).clamp_min(1.0) * velocity.shape[1]
        rms = ((velocity.square() * mask).sum((1, 2)) / count).sqrt()
        return rms[:, None, None]


class RectifiedFlow(nn.Module):
    """
    Velocity field from Gaussian noise (t = 0) to the normalised log mel
    (t = 1), conditioned per frame.

    Args:
        n_mels (int): Number of mel bins.
        speaker_count (int): Number of speakers.
        content_channels (int, optional): Number of content channels. Defaults to 768.
        hidden_channels (int, optional): Number of conditioning channels. Defaults to 384.
        encoder_layers (int, optional): Number of blocks of the condition encoder. Defaults to 4.
        speaker_channels (int, optional): Size of the speaker embedding. Defaults to 256.
        pitch_fourier (int, optional): Sine/cosine pairs of the pitch. Defaults to 0.
        harmonic_prior (dict, optional): `sample_rate`, `n_fft`, `fmin` and `fmax` of the mel, None for no prior.
        breathiness (bool, optional): Whether to take the breathiness curve. Defaults to False.
        key_shift (bool, optional): Whether to take the formant shift. Defaults to False.
        speed (bool, optional): Whether to take the time stretch. Defaults to False.
        backbone (str, optional): Name of the backbone, only "lynxnet2". Defaults to "lynxnet2".
        backbone_args (dict, optional): Arguments of the backbone.
        aux_decoder (dict, optional): Arguments of the aux decoder, None for a flow from pure noise.
        t_start (float, optional): Time the shallow flow starts at, with an aux decoder. Defaults to 0.0.
        aux_grad (float, optional): Scale of the aux decoder's gradient into the encoder and the speaker table. Defaults to 0.1.
        tension (bool, optional): Whether to take the tension curve. Defaults to False.
        dual_timestep (bool, optional): Train a share of the frames at a second time. Defaults to False.
        time_sampling (str, optional): One of TIME_SAMPLINGS. Defaults to "logit-normal".
    """

    def __init__(
        self,
        n_mels: int,
        speaker_count: int,
        content_channels: int = 768,
        hidden_channels: int = 384,
        encoder_layers: int = 4,
        speaker_channels: int = 256,
        pitch_fourier: int = 0,
        harmonic_prior: Optional[dict] = None,
        breathiness: bool = False,
        key_shift: bool = False,
        speed: bool = False,
        backbone: str = "lynxnet2",
        backbone_args: Optional[dict] = None,
        aux_decoder: Optional[dict] = None,
        t_start: float = 0.0,
        aux_grad: float = 0.1,
        tension: bool = False,
        dual_timestep: bool = False,
        time_sampling: str = "logit-normal",
    ):
        super().__init__()
        if backbone != "lynxnet2":
            raise ValueError(
                f"Only the lynxnet2 backbone is supported, not {backbone!r}."
            )
        if time_sampling not in TIME_SAMPLINGS:
            raise ValueError(
                f"time_sampling must be one of {TIME_SAMPLINGS}, not {time_sampling!r}."
            )
        self.time_sampling = time_sampling
        self.n_mels = int(n_mels)
        self.hidden_channels = int(hidden_channels)
        self.encoder = ConditionEncoder(
            content_channels,
            hidden_channels,
            speaker_count,
            speaker_channels,
            encoder_layers,
            pitch_fourier,
            HarmonicPrior(n_mels=n_mels, **harmonic_prior) if harmonic_prior else None,
            breathiness,
            key_shift,
            speed,
            tension,
        )
        self.backbone = LYNXNet2Backbone(
            n_mels, hidden_channels, **(backbone_args or {})
        )
        self.aux = (
            AuxDecoder(hidden_channels, n_mels, **aux_decoder) if aux_decoder else None
        )
        # Time and speaker paths stay on AdamW, where gradient clipping bounds the step.
        conditioning = [self.backbone.time_mlp, *self.speaker_layers()]
        for module in filter(None, conditioning):
            for child in module.modules():
                child.use_adamw = True
        self.t_start = float(t_start) if self.aux is not None else 0.0
        self.aux_grad = float(aux_grad)
        self.dual_timestep = bool(dual_timestep)

    @property
    def speaker_count(self):
        return self.encoder.speaker_count

    def speaker_layers(self):
        """
        The layers the speaker embedding reaches the network through, None for
        one the model does not have.
        """
        layers = [self.encoder.speaker_proj, self.backbone.voice]
        layers += [layer.modulation for layer in self.backbone.layers]
        if self.aux is not None:
            layers += [block.speaker for block in self.aux.blocks]
        return layers

    def _drop_speakers(self, speaker, speaker_dropout):
        if speaker_dropout <= 0:
            return speaker
        dropped = torch.rand(speaker.shape, device=speaker.device) < speaker_dropout
        return torch.where(
            dropped, torch.full_like(speaker, self.speaker_count), speaker
        )

    def _times(self, batch, device):
        # Times over the trained range, stratified across the batch.
        u = (
            torch.arange(batch, device=device) + torch.rand(batch, device=device)
        ) / batch
        u = u[torch.randperm(batch, device=device)]
        if self.time_sampling == "logit-normal":
            u = u.clamp(1e-6, 1.0 - 1e-6)
            u = torch.sigmoid(math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0))
        return self.t_start + (1.0 - self.t_start) * u

    def _flow_error(self, mel, cond, voice, mask, t, noise):
        mix = t[:, None, None] if t.dim() == 1 else t[:, None, :]
        x_t = (1.0 - mix) * noise + mix * mel
        prediction = self.backbone(x_t, t, cond, mask, voice)
        error = (prediction.float() - (mel - noise).float()).square() * mask
        return error.sum((1, 2)) / (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)

    def _aux_loss(self, mel, cond, voice, mask):
        if self.aux is None:
            return None
        cond = cond * self.aux_grad + cond.detach() * (1.0 - self.aux_grad)
        voice = voice * self.aux_grad + voice.detach() * (1.0 - self.aux_grad)
        error = (self.aux(cond, mask, voice).float() - mel.float()).abs() * mask
        return error.sum() / (mask.sum() * self.n_mels).clamp_min(1.0)

    def _train_times(self, batch, frames, device):
        # A time per item; under dual timestep a share of the frames of each
        # item gets a second one.
        t = self._times(batch, device)
        if not self.dual_timestep:
            return t
        other = self._times(batch, device)
        swap = torch.rand(batch, frames, device=device) < DUAL_TIMESTEP_SHARE
        return torch.where(swap, other[:, None], t[:, None])

    def forward(
        self,
        mel,
        inputs: Conditioning,
        speaker_dropout=0.0,
        tension_dropout=0.0,
    ):
        """
        Returns the flow-matching loss and the aux decoder's L1 (None without one).

        Args:
            mel (torch.Tensor): Normalised mel, shape (batch, n_mels, frames).
            inputs (Conditioning): The inputs of the flow.
            speaker_dropout (float, optional): Share of the items trained on the null speaker.
            tension_dropout (float, optional): Share of the items trained with a flat tension.
        """
        speaker = self._drop_speakers(inputs.speaker, speaker_dropout)
        tension = inputs.tension
        if tension is not None and tension_dropout > 0:
            kept = (
                torch.rand(tension.shape[0], 1, device=tension.device)
                >= tension_dropout
            )
            tension = tension * kept
        mask = inputs.mask
        cond = self.encoder(inputs._replace(speaker=speaker, tension=tension))
        voice = self.encoder.voice(speaker)
        noise = torch.randn_like(mel)

        t = self._train_times(mel.shape[0], mel.shape[-1], mel.device)
        error = self._flow_error(mel, cond, voice, mask, t, noise)
        frames = mask.sum((1, 2))
        flow = (error * frames).sum() / frames.sum().clamp_min(1.0)
        return flow, self._aux_loss(mel, cond, voice, mask)

    @torch.no_grad()
    def validation_losses(self, mel, inputs: Conditioning, noise, fractions):
        """
        Flow loss at each of `fractions` of the trained time range from fixed
        `noise`, and the aux decoder's L1 (None without one).

        Args:
            mel (torch.Tensor): Normalised mel, shape (batch, n_mels, frames).
            inputs (Conditioning): The inputs of the flow.
            noise (torch.Tensor): Noise the flow starts from, shape of `mel`.
            fractions (tuple): Where in the trained time range the loss is taken.
        """
        mask = inputs.mask
        cond = self.encoder(inputs)
        voice = self.encoder.voice(inputs.speaker)
        frames = mask.sum((1, 2))
        losses = []
        for fraction in fractions:
            t = torch.full(
                (mel.shape[0],),
                self.t_start + (1.0 - self.t_start) * fraction,
                device=mel.device,
            )
            error = self._flow_error(mel, cond, voice, mask, t, noise)
            losses.append((error * frames).sum() / frames.sum().clamp_min(1.0))
        return torch.stack(losses), self._aux_loss(mel, cond, voice, mask)

    @torch.no_grad()
    def aux_mel(self, inputs: Conditioning):
        """
        The normalised mel of the aux decoder, where the sampling of a shallow
        flow starts, shape (batch, n_mels, frames).

        Args:
            inputs (Conditioning): The inputs of the flow.
        """
        voice = self.encoder.voice(inputs.speaker)
        return self.aux(self.encoder(inputs), inputs.mask, voice)

    def _start(self, noise, mel, mask, start):
        # Where sampling begins: the state and its flow time.
        if self.t_start <= 0:
            return noise * mask, 0.0
        t0 = self.t_start
        if start is not None:
            t0 = min(max(self.t_start, float(start)), 0.99)
        return ((1.0 - t0) * noise + t0 * mel) * mask, t0

    @staticmethod
    def _renoise(x, now, back, fresh, temperature):
        # Takes x from `now` back to `back`, on the path (1 - t) noise + t mel.
        scale = back / now
        top_up = math.sqrt(max((1.0 - back) ** 2 - (scale * (1.0 - now)) ** 2, 0.0))
        return scale * x + temperature * top_up * fresh

    @torch.no_grad()
    def sample(
        self,
        inputs: Conditioning,
        steps: int = 16,
        method: str = "euler",
        cfg_scale: float = 1.0,
        noise: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[], None]] = None,
        content_guidance: float = 0.0,
        guidance_rescale: float = 0.0,
        temperature: float = 1.0,
        start: Optional[float] = None,
        guidance_interval: tuple = (0.0, 1.0),
        rescale_mode: str = "global",
        schedule: str = "uniform",
        churn: float = 0.0,
        churn_noise: Optional[Callable[[int], torch.Tensor]] = None,
        start_mel: Optional[torch.Tensor] = None,
    ):
        """
        Integrate the ODE from noise to a normalised mel, shape (batch, n_mels, frames).

        Args:
            inputs (Conditioning): The inputs of the flow.
            steps (int, optional): Number of steps. Defaults to 16.
            method (str, optional): One of SAMPLERS. Defaults to "euler".
            cfg_scale (float, optional): Guidance away from the null speaker, 1 is off. Defaults to 1.0.
            noise (torch.Tensor, optional): Starting noise, drawn when None.
            callback (Callable, optional): Called after every step.
            content_guidance (float, optional): Guidance away from the blurred content, 0 is off. Defaults to 0.0.
            guidance_rescale (float, optional): Pull of the guided velocity's spread back to the unguided one's. Defaults to 0.0.
            temperature (float, optional): Scale of the starting noise. Defaults to 1.0.
            start (float, optional): Time sampling begins at with an aux decoder, `t_start` when None.
            guidance_interval (tuple, optional): Flow time the guidances apply in. Defaults to (0.0, 1.0).
            rescale_mode (str, optional): One of RESCALE_MODES. Defaults to "global".
            schedule (str, optional): One of SCHEDULES. Defaults to "uniform".
            churn (float, optional): Share of each step re-noised before it, 0 is the plain ODE. Defaults to 0.0.
            churn_noise (Callable, optional): Gives the fresh noise of a step, drawn when None.
            start_mel (torch.Tensor, optional): Mel a shallow flow starts from, the aux decoder's when None.
        """
        if method not in SAMPLERS:
            raise ValueError(f"method must be one of {SAMPLERS}, not {method!r}.")
        if rescale_mode not in RESCALE_MODES:
            raise ValueError(
                f"rescale_mode must be one of {RESCALE_MODES}, not {rescale_mode!r}."
            )
        guidance = _Guidance(
            cfg_scale,
            content_guidance,
            guidance_rescale,
            rescale_mode,
            tuple(float(value) for value in guidance_interval),
        )
        batch, frames = inputs.content.shape[:2]
        mask = inputs.mask

        # Each guided variant rides along in the batch, after the plain pass.
        variants = guidance.variants(inputs, self.speaker_count)
        count = len(variants)

        def repeat(value):
            if value is None:
                return None
            return value.repeat(count, *([1] * (value.dim() - 1)))

        stacked = inputs.map(repeat)._replace(
            content=torch.cat([content for content, _ in variants]),
            speaker=torch.cat([speaker for _, speaker in variants]),
        )
        cond = self.encoder(stacked)
        voice = self.encoder.voice(stacked.speaker)

        def velocity(x, t):
            if count == 1 or not guidance.active(float(t[0])):
                return self.backbone(x, t, cond[:batch], mask, voice[:batch])
            passes = self.backbone(repeat(x), repeat(t), cond, stacked.mask, voice)
            return guidance.combine(passes.chunk(count), mask)

        if noise is None:
            shape = (batch, self.n_mels, frames)
            noise = torch.randn(shape, device=inputs.content.device)
        noise = noise * float(temperature)
        if start_mel is None and self.t_start > 0:
            start_mel = self.aux(cond[:batch], mask, voice[:batch])
        x, t0 = self._start(noise, start_mel, mask, start)
        times = time_grid(schedule, max(1, int(steps)), t0, x.device)
        for index in range(times.shape[0] - 1):
            now = float(times[index])
            back = max(
                self.t_start,
                now - float(churn) * float(times[index + 1] - times[index]),
            )
            if churn > 0 and 0 < back < now:
                if churn_noise is None:
                    fresh = torch.randn_like(x)
                else:
                    fresh = churn_noise(index)
                x = self._renoise(x, now, back, fresh, float(temperature)) * mask
                now = back
            t = torch.full((batch,), now, device=x.device, dtype=times.dtype)
            dt = times[index + 1] - now
            v = velocity(x, t)
            if method == "heun":
                v_next = velocity(x + dt * v, times[index + 1].expand(batch))
                v = 0.5 * (v + v_next)
            x = x + dt * v
            if callback is not None:
                callback()
        return x * mask


def resize_speakers(state_dict: dict, speaker_count: int):
    """
    Fit a checkpoint's speaker table to `speaker_count` speakers: every row
    starts at the mean of the trained speakers and the null row is kept.

    Args:
        state_dict (dict): Weights of a flow model.
        speaker_count (int): Number of speakers of the new model.
    """
    key = "encoder.speaker.weight"
    table = state_dict[key]
    if table.shape[0] == speaker_count + 1:
        return state_dict
    trained, null = table[:-1], table[-1:]
    rows = trained.mean(0, keepdim=True).expand(speaker_count, -1).clone()
    state_dict = dict(state_dict)
    state_dict[key] = torch.cat((rows, null), dim=0)
    return state_dict


def match_inputs(state_dict: dict, model: RectifiedFlow):
    """
    Fit a pretrain's weights to the optional inputs of `model`: the ones it
    lacks keep the model's zero start, and the span input of a model trained
    with Mean Flow is dropped, which leaves its plain flow.

    Args:
        state_dict (dict): Weights of a flow model.
        model (RectifiedFlow): The model the weights are loaded into.
    """
    own = model.state_dict()
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("backbone.span_mlp.")
    }
    for key, value in own.items():
        if key not in state_dict and key.startswith(ZERO_INPUTS):
            state_dict[key] = value
    return state_dict


def build_flow(config: dict, speaker_count: int):
    """
    Build the flow model of a rectified flow config.

    Args:
        config (dict): The config, with its `data` and `flow` sections.
        speaker_count (int): Number of speakers.
    """
    model = dict(config["flow"]["model"])
    # named by the config of a model trained with Mean Flow
    model.pop("mean_flow", None)
    data = config["data"]
    if model.pop("harmonic_prior", False):
        model["harmonic_prior"] = dict(
            sample_rate=data["sample_rate"],
            n_fft=data["n_fft"],
            fmin=data["mel_fmin"],
            fmax=data["mel_fmax"],
        )
    return RectifiedFlow(n_mels=data["n_mels"], speaker_count=speaker_count, **model)
