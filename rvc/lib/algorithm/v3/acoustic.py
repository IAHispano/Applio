"""V3 acoustics: content and target controls -> normalized mel -> refinement.

Content enters as [batch, frames, content_channels]; convolutions use
[batch, channels, frames]. F0/voicing/energy use [batch, frames], and padding
masks use [batch, 1, frames]. The predictor learns deterministic mel structure;
the refiner learns a distribution around it. Neither branch generates PCM or
performs the invertible VITS latent flow used by classic RVC.

Reading order: condition_input -> predict -> flow_loss -> sample. The trainer
selects trainable branches and fits statistics. The separate universal vocoder
consumes denormalized log-mel. See docs/README.md for the complete architecture.
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from rvc.configs.v3 import AcousticConfig


def masked_mean(value, mask):
    """Normalize by actual valid elements, including broadcast feature channels."""
    mask = torch.broadcast_to(mask, value.shape).to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


class FrameNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GatedBlock(nn.Module):
    """Residual frame block with reusable causal convolution history.

    Layer normalization uses channels within each frame, so future packets cannot
    change previous normalization. Only the depthwise convolution needs kernel-1
    past frames; the pointwise projections have no temporal history.
    """

    def __init__(self, width, kernel, expansion=2, causal=True):
        super().__init__()
        self.causal, self.kernel = causal, kernel
        self.norm = FrameNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel, groups=width)
        self.expand = nn.Conv1d(width, 2 * width * expansion, 1)
        self.project = nn.Conv1d(width * expansion, width, 1)
        self.scale = nn.Parameter(torch.full((1, width, 1), 0.01))

    def forward(self, x, mask=None):
        y = self.norm(x)
        pad = (self.kernel - 1, 0) if self.causal else (self.kernel // 2,) * 2
        y = self.depthwise(F.pad(y, pad))
        a, b = self.expand(y).chunk(2, dim=1)
        out = x + self.project(F.silu(a) * b) * self.scale
        return out if mask is None else out * mask

    def stream(self, x, history=None):
        if not self.causal:
            raise ValueError("Streaming requires causal acoustic weights")
        y = self.norm(x)
        if history is None:
            history = y.new_zeros(y.shape[0], y.shape[1], self.kernel - 1)
        joined = torch.cat([history, y], dim=-1)
        a, b = self.expand(self.depthwise(joined)).chunk(2, dim=1)
        return x + self.project(F.silu(a) * b) * self.scale, joined[
            ..., -(self.kernel - 1) :
        ].clone() if self.kernel > 1 else joined[..., :0]


def run_blocks(blocks, x, mask, use_checkpoint=False):
    for block in blocks:
        if use_checkpoint and torch.is_grad_enabled() and x.requires_grad:
            x = checkpoint(block, x, mask, use_reentrant=False)
        else:
            x = block(x, mask)
    return x


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        frequencies = torch.exp(
            -math.log(10000) * torch.arange((width + 1) // 2) / max(width // 2 - 1, 1)
        )
        self.register_buffer("frequencies", frequencies)
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )

    def forward(self, t):
        phase = t.float()[:, None] * 1000 * self.frequencies[None]
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)[:, : self.width]
        return self.net(embedding)


class AcousticModel(nn.Module):
    """Shared conditioning, deterministic mel prediction and residual velocity.

    mel_mean/mel_std are fitted on training mel. residual_scale is per-band RMS
    around the frozen predictor, fitted before flow training. These checkpoint
    buffers must remain fixed in inference and adaptation. Capability flags record
    completed optimizer updates, not convergence or acceptable audio quality.
    """

    def __init__(self, config: AcousticConfig):
        super().__init__()
        self.config = config
        cw, pw, rw = (
            config.condition_width,
            config.predictor_width,
            config.refiner_width,
        )
        self.content_norm = nn.LayerNorm(config.content_dim)
        self.content_projection = nn.Conv1d(config.content_dim, cw, 1)
        self.control_projection = nn.Conv1d(6, cw, 1)
        self.speaker = nn.Embedding(config.speakers, cw)
        self.condition_blocks = nn.ModuleList(
            [
                GatedBlock(cw, config.kernel_size, config.expansion, config.causal)
                for _ in range(2)
            ]
        )
        self.predictor_in = nn.Conv1d(cw, pw, 1)
        self.predictor_blocks = nn.ModuleList(
            [
                GatedBlock(pw, config.kernel_size, config.expansion, config.causal)
                for _ in range(config.predictor_depth)
            ]
        )
        self.predictor_out = nn.Conv1d(pw, config.mel_dim, 1)
        self.refiner_in = nn.Conv1d(2 * config.mel_dim, rw, 1)
        self.refiner_condition = nn.Conv1d(cw, rw, 1)
        self.time_embedding, self.step_embedding = TimeEmbedding(rw), TimeEmbedding(rw)
        self.refiner_blocks = nn.ModuleList(
            [
                GatedBlock(rw, config.kernel_size, config.expansion, config.causal)
                for _ in range(config.refiner_depth)
            ]
        )
        self.refiner_out = nn.Conv1d(rw, config.mel_dim, 1)
        nn.init.zeros_(self.refiner_out.weight)
        nn.init.zeros_(self.refiner_out.bias)
        self.register_buffer("mel_mean", torch.zeros(1, config.mel_dim, 1))
        self.register_buffer("mel_std", torch.ones(1, config.mel_dim, 1))
        self.register_buffer("residual_scale", torch.ones(1, config.mel_dim, 1))
        self.register_buffer("shortcut_trained", torch.tensor(False))
        self.register_buffer("flow_trained", torch.tensor(False))
        self.register_buffer("predictor_trained", torch.tensor(False))
        self.register_buffer("statistics_fitted", torch.tensor(False))

    def normalize(self, mel):
        return (mel - self.mel_mean) / self.mel_std

    def denormalize(self, mel):
        return mel * self.mel_std + self.mel_mean

    def condition_input(
        self,
        content,
        f0,
        voiced,
        energy,
        speaker,
        confidence=None,
        confidence_valid=None,
    ):
        """Combine frame-aligned content, six scalar controls and target identity.

        F0 is represented relative to 220 Hz in octaves and gated by voicing.
        Confidence has a separate validity channel: unavailable confidence must not
        look like an actual low-confidence measurement. The target speaker embedding
        broadcasts across time; source content and pitch retain their original timing.
        """

        if content.ndim != 3 or content.shape[-1] != self.config.content_dim:
            raise ValueError(
                "Content must have shape [batch, frames, configured content width]"
            )
        if torch.any(speaker < 0) or torch.any(speaker >= self.config.speakers):
            raise ValueError("Unknown target speaker ID")
        if confidence is None:
            confidence = torch.zeros_like(f0)
        if confidence_valid is None:
            confidence_valid = torch.zeros_like(f0)
        log_f0 = torch.log2(f0.clamp_min(1) / 220) * voiced
        controls = torch.stack(
            [
                log_f0 / 3,
                voiced,
                confidence,
                confidence_valid,
                energy.clamp(-12, 2) / 6,
                voiced * log_f0.sin(),
            ],
            dim=1,
        )
        x = self.content_projection(self.content_norm(content).transpose(1, 2))
        x = x + self.control_projection(controls) + self.speaker(speaker)[:, :, None]
        return x

    def condition(
        self,
        content,
        f0,
        voiced,
        energy,
        speaker,
        confidence=None,
        confidence_valid=None,
        mask=None,
    ):
        x = self.condition_input(
            content, f0, voiced, energy, speaker, confidence, confidence_valid
        )
        if mask is not None:
            x = x * mask
        return run_blocks(self.condition_blocks, x, mask, self.config.checkpoint_blocks)

    def predict(self, condition, mask=None):
        """Return normalized mel [B, mel_bands, T]; budget zero uses this path."""

        x = self.predictor_in(condition)
        x = run_blocks(self.predictor_blocks, x, mask, self.config.checkpoint_blocks)
        out = self.predictor_out(x)
        return out if mask is None else out * mask

    def refine(self, z, t, step, condition, base, mask=None):
        """Predict residual velocity for state z, time t and integration interval step.

        Zero step denotes the ordinary flow objective. Shortcut training supplies a
        nonzero finite interval. During refiner training, base is the detached predictor
        output in normalized-mel coordinates.
        """

        x = self.refiner_in(torch.cat([z, base], dim=1)) + self.refiner_condition(
            condition
        )
        x = x + (self.time_embedding(t) + self.step_embedding(step))[:, :, None]
        if mask is not None:
            x = x * mask
        x = run_blocks(self.refiner_blocks, x, mask, self.config.checkpoint_blocks)
        out = self.refiner_out(x)
        return out if mask is None else out * mask

    @torch.no_grad()
    def sample(self, condition, steps=4, noise=None, seed=0, mask=None, ordinary=False):
        """Integrate noise from t=0 to t=1 and return physical log-mel values.

        Centered mode returns denormalize(base + residual_scale * z); the direct-mel
        ablation denormalizes z instead. Budget zero bypasses noise and refinement.
        Small budgets require shortcut weights unless ordinary Euler is requested.
        """

        if steps not in {0, 1, 2, 4, 8, 16, 32}:
            raise ValueError("Refinement budget must be 0, 1, 2, 4, 8, 16 or 32")
        if steps and not bool(self.flow_trained):
            raise ValueError(
                "Residual inference requires an ordinary-flow training checkpoint"
            )
        if steps and steps < 8 and not ordinary and not bool(self.shortcut_trained):
            raise ValueError(
                "Few-step inference requires a checkpoint trained with shortcut targets"
            )
        base = self.predict(condition, mask)
        if not steps:
            return self.denormalize(base)
        if noise is None:
            generator = torch.Generator(device=base.device).manual_seed(seed)
            noise = torch.randn(
                base.shape, device=base.device, dtype=base.dtype, generator=generator
            )
        if noise.shape != base.shape:
            raise ValueError("Residual noise shape differs from acoustic shape")
        z = noise.clone()
        dt = 1 / steps
        for i in range(steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            d = torch.full_like(
                t, 0 if ordinary or not bool(self.shortcut_trained) else dt
            )
            z = z + dt * self.refine(z, t, d, condition, base, mask)
        return self.denormalize(
            base + self.residual_scale * z if self.config.prediction_centered else z
        )

    def predictor_loss(self, condition, mel, mask):
        """Masked normalized-mel L1 plus 0.05 times adjacent-frame difference L1."""

        base, target = self.predict(condition, mask), self.normalize(mel)
        loss = masked_mean((base - target).abs(), mask)
        adjacent = mask[..., 1:] * mask[..., :-1]
        temporal = masked_mean(
            (
                (base[..., 1:] - base[..., :-1]) - (target[..., 1:] - target[..., :-1])
            ).abs(),
            adjacent,
        )
        return loss + 0.05 * temporal

    def flow_loss(
        self,
        condition,
        mel,
        mask,
        teacher=None,
        bootstrap_fraction=0.125,
        bootstrap_select=None,
    ):
        """Learn noise-to-target velocity, optionally using EMA shortcut targets.

        Centered target = (normalized_mel - detached_base) / residual_scale. Ordinary
        examples interpolate noise and target at t, with velocity target-noise.
        Selected shortcut examples learn one interval d from two detached EMA half-
        steps; their mean velocity reaches the composed endpoint. Unselected examples
        retain the ordinary-flow anchor so bootstrapping does not replace it.
        """

        base = self.predict(condition, mask).detach()
        target = (
            (self.normalize(mel) - base) / self.residual_scale
            if self.config.prediction_centered
            else self.normalize(mel)
        )
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device)
        step = torch.zeros_like(t)
        z = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        velocity = target - noise
        if teacher is not None:
            if not 0 < bootstrap_fraction <= 1:
                raise ValueError("Bootstrap fraction must lie in (0, 1]")
            # Batch size one still learns both the anchor and bootstrap in expectation.
            select = (
                torch.rand_like(t) < bootstrap_fraction
                if bootstrap_select is None
                else bootstrap_select
            )
            if select.shape != t.shape or select.dtype != torch.bool:
                raise ValueError("Bootstrap selection must be a boolean batch mask")
            if select.any():
                levels = torch.randint(0, 5, (int(select.sum()),), device=z.device)
                d = 2.0 ** (-levels.float())
                grid = torch.floor(torch.rand_like(d) / d) * d
                step[select], t[select] = d, grid
                z[select] = (1 - grid[:, None, None]) * noise[select] + grid[
                    :, None, None
                ] * target[select]
                with torch.no_grad():
                    # Same condition/base/scale coordinates for both teacher calls.
                    first = teacher.refine(
                        z[select],
                        grid,
                        d / 2,
                        condition[select].detach(),
                        base[select],
                        mask[select],
                    )
                    midpoint = z[select] + d[:, None, None] * first / 2
                    second = teacher.refine(
                        midpoint,
                        grid + d / 2,
                        d / 2,
                        condition[select].detach(),
                        base[select],
                        mask[select],
                    )
                    velocity[select] = (first + second) / 2
        prediction = self.refine(z, t, step, condition, base, mask)
        return masked_mean((prediction.float() - velocity.float()).square(), mask)


class EMA:
    """Detached moving-average teacher/export model with copied contract buffers.

    Parameters are averaged. Statistics and capability buffers are copied exactly
    so student and teacher use the same acoustic coordinate system.
    """

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for target, value in zip(
            self.model.parameters(), model.parameters(), strict=True
        ):
            target.lerp_(value.detach(), 1 - self.decay)
        for target, value in zip(self.model.buffers(), model.buffers(), strict=True):
            target.copy_(value)
