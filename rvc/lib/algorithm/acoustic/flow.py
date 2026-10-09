"""Joint shallow rectified flow in physical-mel package coordinates.

The auxiliary mel and velocity field are optimized together. Training samples
the straight noise-to-data path at t >= t_start; inference starts from a mixture
of Gaussian noise and auxiliary prediction at t_start. This avoids freezing a
deterministic error distribution as the only training target. The formulation
is inspired by DiffSinger, but this native graph is not their checkpoint
loader. Statistics, feature contracts and target speaker IDs remain Applio's.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.acoustic.model import AcousticModel, FrameNorm
from rvc.lib.algorithm.acoustic.losses import masked_mean


class ModulatedVelocityBlock(nn.Module):
    """Time/condition modulation in each block; symmetric context is file-only."""

    def __init__(self, width, kernel, expansion):
        super().__init__()
        self.norm = FrameNorm(width)
        self.modulation = nn.Conv1d(width, width * 3, 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.depthwise = nn.Conv1d(
            width, width, kernel, padding=kernel // 2, groups=width
        )
        self.expand = nn.Conv1d(width, width * expansion * 2, 1)
        self.project = nn.Conv1d(width * expansion, width, 1)

    def forward(self, x, embedding, mask):
        shift, scale, gate = self.modulation(F.silu(embedding)).chunk(3, dim=1)
        y = self.norm(x)
        y = (y + y * scale + shift) * mask
        a, b = self.expand(self.depthwise(y)).chunk(2, dim=1)
        y = self.project(torch.atan(a) * b)
        return (x + y + gate * y) * mask


class ShallowFlowModel(AcousticModel):
    """Shared condition, auxiliary predictor and generative direct-mel velocity.

    Budget zero is auxiliary-only, not the final model. Ordinary Euler budgets
    integrate the trained [t_start,1] interval; shortcut capability is deliberately
    absent. Both branches train in one predictor/adaptation stage. No streaming
    or compatibility with old residual-refiner weights is implied.
    """

    def __init__(self, config, mel=None):
        if config.family != "shallow-flow":
            raise ValueError("ShallowFlowModel requires its serialized family")
        super().__init__(config, mel)
        width = config.refiner_width
        self.refiner_in = nn.Conv1d(config.mel_dim, width, 1)
        self.refiner_blocks = nn.ModuleList(
            [
                ModulatedVelocityBlock(width, config.kernel_size, config.expansion)
                for _ in range(config.refiner_depth)
            ]
        )
        del self.step_embedding

    def refine(self, z, t, step, condition, base, mask=None):
        if mask is None:
            mask = torch.ones_like(z[:, :1])
        embedding = (
            self.refiner_condition(condition) + self.time_embedding(t)[:, :, None]
        )
        x = (self.refiner_in(z.float()) + embedding) * mask
        for block in self.refiner_blocks:
            if self.config.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(block, x, embedding, mask, use_reentrant=False)
            else:
                x = block(x, embedding, mask)
        return self.refiner_out(x) * mask

    def flow_loss(self, condition, mel, mask, teacher=None, **kwargs):
        if teacher is not None:
            raise ValueError("Shallow-flow shortcut targets are not implemented")
        target = self.normalize(mel).float()
        noise = torch.randn_like(target) * mask
        t = self.config.shallow_start + (1 - self.config.shallow_start) * torch.rand(
            target.shape[0], device=target.device
        )
        z = ((1 - t[:, None, None]) * noise + t[:, None, None] * target) * mask
        velocity = self.refine(z, t, torch.zeros_like(t), condition, None, mask)
        return masked_mean((velocity.float() - (target - noise)).square(), mask)

    def predictor_loss(self, condition, mel, mask, detail_weight=0.0, prediction=None):
        if detail_weight:
            raise ValueError(
                "Evaluate the joint shallow-flow baseline before adding detail objectives"
            )
        auxiliary = super().predictor_loss(condition, mel, mask, prediction=prediction)
        return (
            self.flow_loss(condition, mel, mask)
            + self.config.auxiliary_weight * auxiliary
        )

    @torch.no_grad()
    def sample(
        self,
        condition,
        steps=8,
        noise=None,
        seed=0,
        mask=None,
        ordinary=False,
        start_mel=None,
    ):
        if steps not in {0, 1, 2, 4, 8, 16, 32}:
            raise ValueError("Unsupported shallow-flow budget")
        base = self.predict(condition, mask).float()
        if not steps:
            return self.denormalize(base)
        if not bool(self.flow_trained):
            raise ValueError("Shallow-flow inference requires joint velocity training")
        if noise is None:
            generator = torch.Generator(device=base.device).manual_seed(seed)
            noise = torch.randn(base.shape, device=base.device, generator=generator)
        if noise.shape != base.shape:
            raise ValueError("Shallow-flow noise shape mismatch")
        if mask is None:
            mask = torch.ones_like(base[:, :1])
        auxiliary = base if start_mel is None else self.normalize(start_mel)
        if auxiliary.shape != base.shape:
            raise ValueError("Oracle-start mel shape mismatch")
        start = self.config.shallow_start
        z = ((1 - start) * noise.float() + start * auxiliary.float()) * mask
        dt = (1 - start) / steps
        for i in range(steps):
            t = torch.full((z.shape[0],), start + i * dt, device=z.device)
            z = (
                z + dt * self.refine(z, t, torch.zeros_like(t), condition, None, mask)
            ) * mask
        return self.denormalize(z)
