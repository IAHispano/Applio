import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rvc.lib.algorithm.rectified_flow.conditioning import Conditioning

SAMPLERS = ("euler", "heun")
SCHEDULES = ("uniform", "sway", "logit-normal")
RESCALE_MODES = ("global", "frame")


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


def blur_content(content: torch.Tensor):
    """
    Returns the content blurred to a quarter of its frame rate: what the
    content guidance pushes away from.

    Args:
        content (torch.Tensor): Content features, shape (batch, frames, channels).
    """
    frames = content.shape[1]
    blurred = F.interpolate(
        content.transpose(1, 2), size=max(1, frames // 4), mode="linear"
    )
    return F.interpolate(blurred, size=frames, mode="linear").transpose(1, 2)


@dataclass(frozen=True)
class Guidance:
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
            variants.append((blur_content(inputs.content), inputs.speaker))
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
