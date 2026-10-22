import dataclasses
import pathlib
import librosa
import numpy as np
import resampy
import torch
import torchfcpe
import torchcrepe

from rvc.lib.predictors.chunking import extract_pitch
from rvc.lib.predictors.f0 import RMVPE, SWIFT
from rvc.configs.config import Config

config = Config()


@dataclasses.dataclass
class F0Extractor:
    wav_path: pathlib.Path
    sample_rate: int = 44100
    hop_length: int = 512
    f0_min: int = 50
    f0_max: int = 1600
    method: str = "rmvpe"
    x: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.x, self.sample_rate = librosa.load(self.wav_path, sr=self.sample_rate)

    @property
    def hop_size(self):
        return self.hop_length / self.sample_rate

    @property
    def wav16k(self):
        return resampy.resample(self.x, self.sample_rate, 16000)

    def extract_f0(self):
        """Extract a pitch curve without sending the whole recording to a model."""
        f0 = None
        method = self.method
        if method == "crepe":
            device = "cpu" if str(config.device) == "mps" else config.device
            f0 = extract_pitch(
                self.wav16k, lambda window: self._crepe_window(window, device)
            )
        elif method == "fcpe":
            model = torchfcpe.spawn_bundled_infer_model(device=config.device)
            f0 = extract_pitch(
                self.x,
                lambda window: self._fcpe_window(model, window),
                self.sample_rate,
                self.hop_length,
            )
        elif method == "rmvpe":
            predictor = RMVPE(device=config.device, high_register={"enabled": False})
            f0 = predictor.get_f0(self.wav16k)
        elif method in ("swift", "swiftf0", "swift-f0"):
            predictor = SWIFT(device=config.device)
            f0 = predictor.get_f0(self.wav16k, f0_min=self.f0_min, f0_max=self.f0_max)

        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self.hz_to_cents(f0, librosa.midi_to_hz(0))

    @torch.inference_mode()
    def _crepe_window(self, audio, device):
        """Retain the curve utility's unsmoothed CREPE output in bounded windows."""
        return (
            torchcrepe.predict(
                torch.from_numpy(audio).float().unsqueeze(0).to(device),
                sample_rate=16000,
                hop_length=160,
                batch_size=512,
                fmin=self.f0_min,
                fmax=self.f0_max,
                device=device,
            )[0]
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def _fcpe_window(self, model, audio):
        """Preserve FCPE curve limits and output hop while bounding model input."""
        audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1)
        return (
            model.infer(
                audio.to(config.device),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                interp_uv=False,
                output_interp_target_length=len(audio[0]) // self.hop_length + 1,
            )
            .reshape(-1)
            .cpu()
            .numpy()
        )

    def plot_f0(self, f0):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(f0)
        plt.title(self.method)
        plt.xlabel("Time (frames)")
        plt.ylabel("F0 (cents)")
        plt.show()

    @staticmethod
    def hz_to_cents(F, F_ref=55.0):
        F_temp = np.array(F).astype(float)
        F_temp[F_temp == 0] = np.nan
        F_cents = 1200 * np.log2(F_temp / F_ref)
        return F_cents
