"""Per-step diagnostics: the numbers the loop logs but never optimises.

Everything here is measurement -- latent gaps, per-head separation, gradient
norms, rolling means.  None of it feeds a backward pass, which is why it can
live outside the training loop.
"""

import math

import torch

from torch.nn.utils import clip_grad_norm_


def branch_separation(disc_real_outputs, disc_generated_outputs):
    """Per-head ``mean(real logit) - mean(fake logit)``, detached.

    Under SAN a head returns ``(function, direction)`` rather than one tensor;
    the function output is the one the generator is scored by, so it is the one
    whose separation means anything.
    """

    def logits(output):
        return (output[0] if isinstance(output, (list, tuple)) else output).detach()

    return torch.stack(
        [
            logits(dr).float().mean() - logits(dg).float().mean()
            for dr, dg in zip(disc_real_outputs, disc_generated_outputs)
        ]
    )


def clip_or_sample_grad_norm(
    parameters,
    max_norm,
    step,
    sample_interval,
):
    max_norm = float(max_norm)
    should_measure = math.isfinite(max_norm) or step % max(1, sample_interval) == 0
    if not should_measure:
        return None
    return clip_grad_norm_(parameters, max_norm=max_norm)
