"""Exponential moving average of the generator weights.

A GAN vocoder's weights oscillate by construction (generator and
discriminator chasing each other), so averaging over the recent trajectory
is normally better than any single step. Held as a shadow copy rather than
folded back into the live weights, since feeding it back would make this a
different optimiser rather than an average of this one.
"""

from __future__ import annotations

import contextlib

import torch


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class WeightEMA:
    """Shadow copy of ``model``'s state, updated toward it every step.

    The checkpoint format is unchanged.  ``save_checkpoint`` writes the shadow
    under its own ``ema`` key, and a checkpoint without one -- every checkpoint
    written before this existed, including the existing HiFi-GAN pretrains --
    seeds the shadow from whatever weights it does carry.
    """

    #: Horizon is ``1 / (1 - decay)`` steps, and the choice is a real tradeoff
    #: rather than "longer is smoother".  Averaging only helps up to the point
    #: where it has cancelled the oscillation; past that it is pure lag, and
    #: lag is what the overtrain detector pays for, because a lagging average
    #: reaches its own minimum well after the model did.  A 1000-step horizon
    #: wins at every oscillation level tested; 10000 (decay 0.9999) is an order
    #: of magnitude worse, with its minimum landing thousands of steps late.
    #: Do not raise it on the intuition that longer is smoother.
    DEFAULT_DECAY = 0.999

    def __init__(self, model, decay: float = DEFAULT_DECAY, warmup: bool = True):
        self.decay = min(1.0, max(0.0, float(decay)))
        self.warmup = bool(warmup)
        self.updates = 0

        state = _unwrap(model).state_dict()
        self.shadow = {key: value.detach().clone() for key, value in state.items()}

        # Cached so the per-step cost is one foreach call and no dict work.
        # ``state_dict`` hands back tensors that share storage with the live
        # parameters, and ``load_state_dict`` copies *into* those tensors rather
        # than rebinding them, so these references stay valid for the life of
        # the model -- including across the preview weight swaps.
        self._live_float = []
        self._shadow_float = []
        self._other_keys = []
        for key, value in state.items():
            if value.is_floating_point():
                self._live_float.append(value)
                self._shadow_float.append(self.shadow[key])
            else:
                # Integer buffers (step counters, ``num_batches_tracked``) have
                # no meaningful average; they track the live value instead.
                self._other_keys.append(key)

    def current_decay(self) -> float:
        """Decay for the update currently being applied.

        Until the exponential horizon is longer than the run so far, the plain
        running mean is the better estimator of the same quantity, and using it
        removes the startup bias without needing a correction term.

        ``1 - 1/updates`` with ``updates`` already incremented is exactly that
        running mean: the first update takes decay 0 and adopts the live
        weights outright, the second averages two, the n-th averages n.  The
        off-by-one matters -- ``1 - 1/(updates + 1)`` would leave the first
        update half-weighted on the initialisation and never fully shake it
        off, which is precisely the startup bias this is here to avoid.
        """
        if not self.warmup or self.updates < 1:
            return self.decay
        return min(self.decay, 1.0 - 1.0 / self.updates)

    @torch.no_grad()
    def update(self, model) -> None:
        """Advance the shadow toward the live weights.

        Must only be called while ``model`` holds live training weights -- not
        inside :meth:`applied`, and not during a preview swap.
        """
        self.updates += 1
        decay = self.current_decay()
        # lerp(shadow, live, 1 - decay) == decay * shadow + (1 - decay) * live
        torch._foreach_lerp_(self._shadow_float, self._live_float, 1.0 - decay)
        if self._other_keys:
            state = _unwrap(model).state_dict()
            for key in self._other_keys:
                self.shadow[key].copy_(state[key])

    @contextlib.contextmanager
    def applied(self, model):
        """Temporarily swap the averaged weights into ``model``.

        The backup goes to the CPU deliberately.  Cloning it in place would put
        a *third* copy of the generator on the device -- live, shadow, backup --
        at exactly the moment the holdout evaluation wants room for inference
        activations.  Two transfers of a few dozen megabytes cost single-digit
        milliseconds against an evaluation that runs a forward pass per slice.
        """
        module = _unwrap(model)
        backup = {
            key: value.detach().to("cpu", copy=True)
            for key, value in module.state_dict().items()
        }
        module.load_state_dict(self.shadow, strict=True)
        try:
            yield module
        finally:
            module.load_state_dict(backup, strict=True)

    def cpu_state_dict(self) -> dict:
        """The averaged weights, detached to CPU, shaped like a model state."""
        return {key: value.detach().to("cpu", copy=True) for key, value in self.shadow.items()}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    @torch.no_grad()
    def reseed(self, model) -> None:
        """Reset the shadow to ``model``'s current weights, average discarded."""
        state = _unwrap(model).state_dict()
        for key, value in self.shadow.items():
            value.copy_(state[key])
        self.updates = 0

    @torch.no_grad()
    def load_state_dict(self, data, model) -> bool:
        """Restore from a checkpoint, reseeding from ``model`` if it cannot.

        ``model`` is required rather than optional because the failure path is
        the common one -- no checkpoint written before this existed carries an
        EMA -- and a shadow left holding the weights it was *constructed* with
        would be the model's initialisation, which is worse than useless after
        a resume.  Returns whether an average was actually restored.
        """
        shadow = (data or {}).get("shadow")
        if not shadow or any(key not in shadow for key in self.shadow):
            # Missing, or a shape/architecture change.  Starting the average
            # again from the restored weights costs only the averaging already
            # done, and is correct from this step onward.
            self.reseed(model)
            return False
        for key, value in self.shadow.items():
            value.copy_(shadow[key])
        self.updates = int(data.get("updates", 0))
        return True
