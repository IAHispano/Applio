import dataclasses
import pathlib

import audioflux
import libf0
import librosa
import numpy as np
import parselmouth
import pyworld as pw
import resampy
import torch
import torchcrepe
import torchfcpe

# from tools.anyf0.rmvpe import RMVPE
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.configs.config import Config
config = Config()

@dataclasses.dataclass
class F0Extractor:
    wav_path: pathlib.Path
    sample_rate: int = 44100
    hop_length: int = 512
    f0_min: int = 50
    f0_max: int = 1600
    method: str = "praat_ac"
    x: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.x, self.sample_rate = librosa.load(self.wav_path, sr=self.sample_rate)

    @property
    def hop_size(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def wav16k(self) -> np.ndarray:
        return resampy.resample(self.x, self.sample_rate, 16000)

    def extract_f0(self) -> np.ndarray:
        f0 = None
        method = self.method
        if method == "dio":
            _f0, t = pw.dio(
                self.x.astype("double"),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                channels_in_octave=2,
                frame_period=(1000 * self.hop_size),
            )
            f0 = pw.stonemask(self.x.astype("double"), _f0, t, self.sample_rate)
            f0 = f0.astype("float")
        elif method == "harvest":
            f0, _ = pw.harvest(
                self.x.astype("double"),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=(1000 * self.hop_size),
            )
            f0 = f0.astype("float")
        elif method == "pyin":
            f0, _, _ = librosa.pyin(
                y=self.wav16k,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=16000,
                hop_length=80,
            )
        elif method == "piptrack":
            pitches, magnitudes = librosa.piptrack(
                y=self.wav16k,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=16000,
                hop_length=80,
            )
            max_indexes = np.argmax(magnitudes, axis=0)
            f0 = pitches[max_indexes, range(magnitudes.shape[1])]
        elif method in ["cep", "hps", "lhs", "ncf", "pef"]:
            f0 = {
                "cep": audioflux.PitchCEP,
                "hps": audioflux.PitchHPS,
                "lhs": audioflux.PitchLHS,
                "ncf": audioflux.PitchNCF,
                "pef": audioflux.PitchPEF,
            }[self.method](
                16000,
                low_fre=self.f0_min,
                high_fre=self.f0_max,
                slide_length=80,
            ).pitch(
                np.pad(self.wav16k, (2048, 2048))
            )
        elif method == "stft":
            f0, _ = audioflux.PitchSTFT(
                16000,
                low_fre=self.f0_min,
                high_fre=self.f0_max,
                slide_length=80,
            ).pitch(np.pad(self.wav16k, (2048, 2048)))
        elif method == "yin":
            f0, _, _ = audioflux.PitchYIN(
                16000,
                low_fre=self.f0_min,
                high_fre=self.f0_max,
                slide_length=80,
            ).pitch(np.pad(self.wav16k, (2048, 2048)))
        elif method in ["swipe", "salience"]:
            f0, _, _ = {
                "swipe": libf0.swipe,
                "salience": libf0.salience,
            }[self.method](
                self.x,
                Fs=self.sample_rate,
                H=self.hop_length,
                F_min=self.f0_min,
                F_max=self.f0_max,
            )
        elif method == "torchcrepe":
            wav16k_torch = torch.FloatTensor(self.wav16k).unsqueeze(0).to(config.device)
            f0 = torchcrepe.predict(
                wav16k_torch,
                sample_rate=16000,
                hop_length=160,
                batch_size=512,
                fmin=self.f0_min,
                fmax=self.f0_max,
                device=config.device,
            )
            f0 = f0[0].cpu().numpy()
        elif method == "torchfcpe":
            audio = librosa.to_mono(self.x)
            audio_length = len(audio)
            f0_target_length = (audio_length // self.hop_length) + 1
            audio = (
                torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(config.device)
            )
            model = torchfcpe.spawn_bundled_infer_model(device=config.device)

            f0 = model.infer(
                audio,
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                interp_uv=False,
                output_interp_target_length=f0_target_length,
            )
            f0 = f0.squeeze().cpu().numpy()
        elif method == "rmvpe":
            model_rmvpe = RMVPE0Predictor(
                "rmvpe.pt",
                is_half=config.is_half,
                device=config.device,
                # hop_length=80
            )
            f0 = model_rmvpe.infer_from_audio(self.wav16k, thred=0.03)
        elif method in ["praat_ac", "praat_cc"]:
            l_pad = int(np.ceil(1.5 / self.f0_min * self.sample_rate))
            r_pad = int(
                self.hop_size * ((len(self.x) - 1) // self.hop_size + 1)
                - len(self.x)
                + l_pad
                + 1
            )
            f0 = getattr(
                parselmouth.Sound(np.pad(self.x, (l_pad, r_pad)), self.sample_rate),
                "to_pitch_" + self.method.rpartition("_")[-1],
            )(
                time_step=self.hop_size,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            ).selected_array[
                "frequency"
            ]
        elif method == "praat_shs":
            l_pad = int(np.ceil(1.5 / self.f0_min * self.sample_rate))
            r_pad = int(
                self.hop_size * ((len(self.x) - 1) // self.hop_size + 1)
                - len(self.x)
                + l_pad
                + 1
            )
            f0 = (
                parselmouth.Sound(np.pad(self.x, (l_pad, r_pad)), self.sample_rate)
                .to_pitch_shs(
                    time_step=self.hop_size,
                    minimum_pitch=self.f0_min,
                    maximum_frequency_component=self.f0_max,
                )
                .selected_array["frequency"]
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return libf0.hz_to_cents(f0, librosa.midi_to_hz(0))

    def plot_f0(self, f0):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(f0)
        plt.title(self.method)
        plt.xlabel("Time (frames)")
        plt.ylabel("F0 (cents)")
        plt.show()
