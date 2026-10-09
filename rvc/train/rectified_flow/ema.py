import contextlib

import torch


class WeightEMA:
    """
    Exponential moving average of a model's weights, held as a shadow copy.
    The flow is sampled and exported from it.

    Args:
        model (torch.nn.Module): The model to average.
        decay (float, optional): Decay of the average. Defaults to 0.999.
    """

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.updates = 0
        state = model.state_dict()
        self.shadow = {key: value.detach().clone() for key, value in state.items()}
        # The state dict shares storage with the live parameters.
        self.live = list(state.values())
        self.averaged = [self.shadow[key] for key in state]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        # The running mean until the horizon is longer than the run so far.
        decay = min(self.decay, 1.0 - 1.0 / self.updates)
        torch._foreach_lerp_(self.averaged, self.live, 1.0 - decay)

    @contextlib.contextmanager
    def applied(self, model):
        """
        Temporarily swap the averaged weights into `model`.
        """
        backup = {
            key: value.detach().to("cpu", copy=True)
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)
        try:
            yield model
        finally:
            model.load_state_dict(backup)

    @torch.no_grad()
    def reseed(self, model):
        """
        Reset the shadow to the current weights of `model`.
        """
        state = model.state_dict()
        for key, value in self.shadow.items():
            value.copy_(state[key])
        self.updates = 0

    def cpu_state_dict(self):
        return {
            key: value.detach().to("cpu", copy=True)
            for key, value in self.shadow.items()
        }

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    @torch.no_grad()
    def load_state_dict(self, data, model):
        """
        Restore the average of a checkpoint, or reseed from `model` when it has none.
        """
        shadow = (data or {}).get("shadow")
        if not shadow or any(key not in shadow for key in self.shadow):
            self.reseed(model)
            return
        for key, value in self.shadow.items():
            value.copy_(shadow[key])
        self.updates = int(data.get("updates", 0))
