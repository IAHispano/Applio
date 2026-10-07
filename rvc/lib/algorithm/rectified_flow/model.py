import math
from typing import Callable, Optional

import torch
from torch import nn

from rvc.lib.algorithm.rectified_flow.backbone import AuxDecoder, LYNXNet2Backbone
from rvc.lib.algorithm.rectified_flow.conditioning import (
    ConditionEncoder,
    Conditioning,
    HarmonicPrior,
)
from rvc.lib.algorithm.rectified_flow.sampling import (
    RESCALE_MODES,
    SAMPLERS,
    Guidance,
    time_grid,
)

# Share of the frames that get the second time under dual timestep.
DUAL_TIMESTEP_SHARE = 0.25
# Inputs that start at zero, so a checkpoint from before them loads unchanged.
ZERO_INPUTS = ("encoder.tension.", "backbone.step.")
# What a model without the input leaves out of the weights it loads.
DROPPED_INPUTS = ("backbone.span_mlp.", "backbone.step.")
# The per-bin mel statistics: a checkpoint from before them keeps the identity.
MEL_STATS = ("mel_shift", "mel_scale")
# Floor of the spread of a bin, so one that hardly moves is not blown up into noise.
MIN_MEL_SCALE = 0.1


class RectifiedFlow(nn.Module):
    """
    Velocity field from Gaussian noise (t = 0) to the normalised log mel
    (t = 1), conditioned per frame. Inside, each mel bin is normalised again
    by the statistics of `set_mel_stats`.

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
        shortcut (bool, optional): Give the backbone a jump length to read, so it also samples in few steps (Frans et al., "One Step Diffusion via Shortcut Models"). Defaults to False.
        shortcut_steps (int, optional): Shortest jump, as a share of the trained time range; a power of two. Defaults to 128.
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
        shortcut: bool = False,
        shortcut_steps: int = 128,
    ):
        super().__init__()
        if backbone != "lynxnet2":
            raise ValueError(
                f"Only the lynxnet2 backbone is supported, not {backbone!r}."
            )
        if shortcut and (shortcut_steps < 2 or shortcut_steps & (shortcut_steps - 1)):
            raise ValueError(
                f"shortcut_steps must be a power of two of 2 or more, not {shortcut_steps}."
            )
        # Jump lengths the backbone knows, 0 for a plain flow.
        self.shortcut_levels = int(math.log2(shortcut_steps)) if shortcut else 0
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
            n_mels,
            hidden_channels,
            step_levels=self.shortcut_levels,
            **(backbone_args or {}),
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
        self.register_buffer("mel_shift", torch.zeros(self.n_mels, 1))
        self.register_buffer("mel_scale", torch.ones(self.n_mels, 1))

    @property
    def speaker_count(self):
        return self.encoder.speaker_count

    @property
    def starts_from_aux(self):
        # Whether sampling starts from the mel of the aux decoder.
        return self.t_start > 0

    @torch.no_grad()
    def set_mel_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Normalise each mel bin inside the model by its mean and spread over the
        training set.

        Args:
            mean (torch.Tensor): Mean of each bin of the normalised mel, shape (n_mels,).
            std (torch.Tensor): Spread of each bin of the normalised mel, shape (n_mels,).
        """
        self.mel_shift.copy_(mean.view(-1, 1))
        self.mel_scale.copy_(std.view(-1, 1).clamp_min(MIN_MEL_SCALE))

    def _encode(self, mel):
        return (mel - self.mel_shift) / self.mel_scale

    def _decode(self, mel):
        return mel * self.mel_scale + self.mel_shift

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
        return self.t_start + (1.0 - self.t_start) * u

    def _flow_error(self, mel, noise, cond, voice, mask, t, jump=None):
        # Flow loss per item on the path from `noise` to `mel`. The items of
        # `jump`, from `_jump_targets`, stand in for the first ones of the batch.
        mix = t[:, None, None] if t.dim() == 1 else t[:, None, :]
        x_t = (1.0 - mix) * noise + mix * mel
        target = mel - noise
        if jump is None:
            prediction = self.backbone(x_t, t, cond, mask, voice)
        else:
            count, x_jump, t_jump, level, target_jump = jump
            if t.dim() == 2:
                t_jump = t_jump[:, None].expand(-1, t.shape[1])
            x_t = torch.cat((x_jump.to(x_t.dtype), x_t[count:]))
            target = torch.cat((target_jump.to(target.dtype), target[count:]))
            t = torch.cat((t_jump, t[count:]))
            # the rest of the batch trains the plain flow, the finest level
            plain = level.new_full((mel.shape[0] - count,), self.shortcut_levels)
            level = torch.cat((level, plain))
            prediction = self.backbone(x_t, t, cond, mask, voice, level)
        error = (prediction.float() - target.float()).square() * mask
        return error.sum((1, 2)) / (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)

    @torch.no_grad()
    def _jump_targets(self, teacher, mel, noise, inputs: Conditioning, share: float):
        """
        Self-consistency targets for the first items of the batch: a jump of
        one of the lengths, from a time on the grid of that length, has to
        land where two jumps of the teacher half as long do. Returns the
        number of items, their state, time, level and target, or None when
        the draw gives no item.

        Args:
            teacher (RectifiedFlow): Copy of the model that lags it.
            mel (torch.Tensor): Per-bin normalised mel, shape (batch, n_mels, frames).
            noise (torch.Tensor): Noise the flow starts from, shape of `mel`.
            inputs (Conditioning): The inputs of the flow.
            share (float): Share of the batch that trains the jumps.
        """
        device = mel.device
        draw = torch.rand((), device=device).item()
        count = min(mel.shape[0], int(share * mel.shape[0] + draw))
        if count < 1:
            return None
        mel, noise = mel[:count], noise[:count]
        inputs = inputs.map(lambda value: value[:count])
        # Per item: spread over the batch instead, a small one trains few lengths.
        level = torch.randint(0, self.shortcut_levels, (count,), device=device)
        sections = 2.0**level
        span = 1.0 - self.t_start
        slot = torch.rand(count, device=device) * sections
        t = self.t_start + span * torch.floor(slot) / sections
        half = 0.5 * span / sections
        mix = t[:, None, None]
        x_t = (1.0 - mix) * noise + mix * mel
        cond = teacher.encoder(inputs)
        voice = teacher.encoder.voice(inputs.speaker)
        finer = level + 1
        first = teacher.backbone(x_t, t, cond, inputs.mask, voice, finer).float()
        middle = x_t + half[:, None, None] * first
        second = teacher.backbone(middle, t + half, cond, inputs.mask, voice, finer)
        return count, x_t, t, level, 0.5 * (first + second.float())

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
        teacher=None,
        shortcut_share=0.0,
    ):
        """
        Returns the flow-matching loss and the aux decoder's L1 (None without
        one), both on the per-bin normalised mel.

        Args:
            mel (torch.Tensor): Normalised mel, shape (batch, n_mels, frames).
            inputs (Conditioning): The inputs of the flow.
            speaker_dropout (float, optional): Share of the items trained on the null speaker.
            tension_dropout (float, optional): Share of the items trained with a flat tension.
            teacher (RectifiedFlow, optional): Copy of a shortcut model that lags it, which its jumps are trained against.
            shortcut_share (float, optional): Share of the batch that trains the jumps of a shortcut model.
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
        mel = self._encode(mel) * mask
        inputs = inputs._replace(speaker=speaker, tension=tension)
        cond = self.encoder(inputs)
        voice = self.encoder.voice(speaker)
        noise = torch.randn_like(mel)

        t = self._train_times(mel.shape[0], mel.shape[-1], mel.device)
        jump = None
        if teacher is not None and self.shortcut_levels and shortcut_share > 0:
            jump = self._jump_targets(teacher, mel, noise, inputs, shortcut_share)
        error = self._flow_error(mel, noise, cond, voice, mask, t, jump)
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
        mel = self._encode(mel) * mask
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
            error = self._flow_error(mel, noise, cond, voice, mask, t)
            losses.append((error * frames).sum() / frames.sum().clamp_min(1.0))
        return torch.stack(losses), self._aux_loss(mel, cond, voice, mask)

    @torch.no_grad()
    def aux_mel(self, inputs: Conditioning):
        """
        The normalised mel of the aux decoder, where sampling starts under
        `t_start`, shape (batch, n_mels, frames).

        Args:
            inputs (Conditioning): The inputs of the flow.
        """
        voice = self.encoder.voice(inputs.speaker)
        mel = self.aux(self.encoder(inputs), inputs.mask, voice)
        return self._decode(mel) * inputs.mask

    def _start(self, noise, mel, mask, start):
        # Where sampling begins: the state and its flow time.
        if self.t_start <= 0:
            return noise * mask, 0.0
        t0 = self.t_start
        if start is not None:
            t0 = min(max(self.t_start, float(start)), 0.99)
        return ((1.0 - t0) * noise + t0 * mel) * mask, t0

    @staticmethod
    def _renoise(x, now, back, fresh):
        # Takes x from `now` back to `back`, on the path (1 - t) noise + t mel.
        scale = back / now
        top_up = math.sqrt(max((1.0 - back) ** 2 - (scale * (1.0 - now)) ** 2, 0.0))
        return scale * x + top_up * fresh

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
            start (float, optional): Time sampling begins at with an aux decoder, the start of the model when None.
            guidance_interval (tuple, optional): Flow time the guidances apply in. Defaults to (0.0, 1.0).
            rescale_mode (str, optional): One of RESCALE_MODES. Defaults to "global".
            schedule (str, optional): One of SCHEDULES. Defaults to "uniform".
            churn (float, optional): Share of each step re-noised before it, 0 is the plain ODE. Defaults to 0.0.
            churn_noise (Callable, optional): Gives the fresh noise of a step, drawn when None.
            start_mel (torch.Tensor, optional): Mel sampling starts from, the aux decoder's when None.
        """
        if method not in SAMPLERS:
            raise ValueError(f"method must be one of {SAMPLERS}, not {method!r}.")
        if rescale_mode not in RESCALE_MODES:
            raise ValueError(
                f"rescale_mode must be one of {RESCALE_MODES}, not {rescale_mode!r}."
            )
        guidance = Guidance(
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

        level = None

        def velocity(x, t):
            if count == 1 or not guidance.active(float(t[0])):
                return self.backbone(x, t, cond[:batch], mask, voice[:batch], level)
            passes = self.backbone(
                repeat(x), repeat(t), cond, stacked.mask, voice, repeat(level)
            )
            return guidance.combine(passes.chunk(count), mask)

        if noise is None:
            shape = (batch, self.n_mels, frames)
            noise = torch.randn(shape, device=inputs.content.device)
        noise = noise * float(temperature)
        if start_mel is not None:
            start_mel = self._encode(start_mel)
        elif self.starts_from_aux:
            start_mel = self.aux(cond[:batch], mask, voice[:batch])
        x, t0 = self._start(noise, start_mel, mask, start)
        steps = max(1, int(steps))
        times = time_grid(schedule, steps, t0, x.device)
        # A shortcut model takes its trained jumps on the grid it learned them
        # on; any other sampling integrates the plain flow.
        on_grid = (
            method == "euler"
            and schedule == "uniform"
            and churn <= 0
            and t0 == self.t_start
        )
        if on_grid and steps & (steps - 1) == 0 and steps < 2**self.shortcut_levels:
            level = torch.full((batch,), int(math.log2(steps)), device=x.device)
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
                x = self._renoise(x, now, back, float(temperature) * fresh) * mask
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
        return self._decode(x) * mask


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
    Fit other weights to `model`: the optional inputs and the mel statistics
    they lack keep the model's own, and the span input of a model trained with
    Mean Flow and the jump lengths of a shortcut one are dropped from a model
    without them, which leaves the plain flow.

    Args:
        state_dict (dict): Weights of a flow model.
        model (RectifiedFlow): The model the weights are loaded into.
    """
    own = model.state_dict()
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in own or not key.startswith(DROPPED_INPUTS)
    }
    for key, value in own.items():
        if key not in state_dict and key.startswith(ZERO_INPUTS + MEL_STATS):
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
    # named by the config of a model from when these were options
    for name in ("mean_flow", "prior_noise", "prior_degrade", "time_sampling"):
        model.pop(name, None)
    data = config["data"]
    if model.pop("harmonic_prior", False):
        model["harmonic_prior"] = dict(
            sample_rate=data["sample_rate"],
            n_fft=data["n_fft"],
            fmin=data["mel_fmin"],
            fmax=data["mel_fmax"],
        )
    return RectifiedFlow(n_mels=data["n_mels"], speaker_count=speaker_count, **model)
