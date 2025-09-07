import os
import torch

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from torchfcpe import spawn_bundled_infer_model
import torchcrepe
from swift_f0 import SwiftF0
import numpy as np


class RMVPE:
    def __init__(self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03):
        f0 = self.model.infer_from_audio(x, thred=filter_radius)
        return f0


class CREPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, model="full"):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        batch_size = 512

        f0, pd = torchcrepe.predict(
            x.float().to(self.device).unsqueeze(dim=0),
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            model=model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()

        return f0


class FCPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = spawn_bundled_infer_model(self.device)

    def get_f0(self, x, p_len=None, filter_radius=0.006):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        f0 = (
            self.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return f0


class SWIFT:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = "cpu"
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, confidence_threshold=0.9):
        if torch.is_tensor(x):
            x = x.cpu().numpy()

        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        f0_min = max(f0_min, 46.875)
        f0_max = min(f0_max, 2093.75)

        detector = SwiftF0(
            fmin=f0_min, fmax=f0_max, confidence_threshold=confidence_threshold
        )
        result = detector.detect_from_array(x, self.sample_rate)
        if len(result.timestamps) == 0:
            return np.zeros(p_len)
        target_time = (
            np.arange(p_len) * self.hop_size + self.hop_size / 2
        ) / self.sample_rate
        pitch = np.nan_to_num(result.pitch_hz, nan=0.0)
        pitch[~result.voicing] = 0.0
        f0 = np.interp(target_time, result.timestamps, pitch, left=0.0, right=0.0)

        return f0
