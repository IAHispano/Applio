"""Small device adaptations shared by standalone inference and training."""

import torch


class CPUSpectralTransform(torch.nn.Module):
    """Compute unsupported complex/FFT operations on CPU; keep neural models on MPS."""

    def __init__(self, transform):
        super().__init__()
        self.transform = transform.cpu().float()

    def forward(self, audio, *args, **kwargs):
        device = audio.device
        return self.transform(audio.to("cpu", dtype=torch.float32), *args, **kwargs).to(
            device
        )


def is_mps_error(error):
    return isinstance(error, (RuntimeError, NotImplementedError, TypeError)) and any(
        key in str(error).lower()
        for key in ("mps", "metal", "placeholder storage has not been allocated")
    )
