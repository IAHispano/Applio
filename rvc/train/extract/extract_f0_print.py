import os
import sys
import time
import tqdm
import torch
import pyworld
import torchcrepe
import parselmouth
import numpy as np
from multiprocessing import Pool

current_directory = os.getcwd()
sys.path.append(current_directory)

from rvc.lib.utils import load_audio
from rvc.lib.predictors.RMVPE import RMVPE0Predictor


exp_dir = sys.argv[1]
f0_method = sys.argv[2]
hop_length = int(sys.argv[3])
num_processes = int(sys.argv[4])


# Define a class for f0 extraction
class FeatureInput:
    def __init__(self, sample_rate=16000, hop_size=160):
        self.fs = sample_rate
        self.hop = hop_size

        self.f0_method_dict = self.get_f0_method_dict()

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    # Define a function to extract f0 using various methods
    def compute_f0(self, path, f0_method, hop_length):
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop

        if f0_method == "crepe":
            f0 = self.mncrepe(f0_method, x, p_len, hop_length)
        elif f0_method in self.f0_method_dict:
            f0 = (
                self.f0_method_dict[f0_method](x, p_len)
                if f0_method == "pm"
                else self.f0_method_dict[f0_method](x)
            )
        return f0

    # Define a function to extract f0 using CREPE
    def mncrepe(self, method, x, p_len, hop_length):
        audio = torch.from_numpy(x.astype(np.float32)).to(self.device, copy=True)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = torch.unsqueeze(audio, dim=0)

        pitch = torchcrepe.predict(
            audio,
            self.fs,
            hop_length,
            self.f0_min,
            self.f0_max,
            "full",
            batch_size=hop_length * 2,
            device=self.device,
            pad=True,
        )

        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        return np.nan_to_num(target)

    # Define functions for different f0 extraction methods
    def get_pm(self, x, p_len):
        f0 = (
            parselmouth.Sound(x, self.fs)
            .to_pitch_ac(
                time_step=160 / 16000,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            .selected_array["frequency"]
        )
        return np.pad(
            f0,
            [
                [
                    max(0, (p_len - len(f0) + 1) // 2),
                    max(0, p_len - len(f0) - (p_len - len(f0) + 1) // 2),
                ]
            ],
            mode="constant",
        )

    def get_harvest(self, x):
        f0_spectral = pyworld.harvest(
            x.astype(np.double),
            fs=self.fs,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=1000 * self.hop / self.fs,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.fs)

    def get_dio(self, x):
        f0_spectral = pyworld.dio(
            x.astype(np.double),
            fs=self.fs,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=1000 * self.hop / self.fs,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.fs)

    def get_rmvpe(self, x):
        model_rmvpe = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
            is_half=False,
            device="cpu",
        )
        return model_rmvpe.infer_from_audio(x, thred=0.03)

    # Helper function to get f0 method dictionary
    def get_f0_method_dict(self):
        return {
            "pm": self.get_pm,
            "harvest": self.get_harvest,
            "dio": self.get_dio,
            "rmvpe": self.get_rmvpe,
        }

    # Define a function to convert f0 to coarse f0
    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    # Define a function to process a single audio file
    def process_file(self, file_info):
        inp_path, opt_path1, opt_path2 = file_info

        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
            return

        try:
            feature_pit = self.compute_f0(inp_path, f0_method, hop_length)
            np.save(opt_path2, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(f"f0fail-{inp_path}-{error}")


# Define the main function
if __name__ == "__main__":
    feature_input = FeatureInput()
    paths = []
    input_root = f"{exp_dir}/sliced_audios_16k"
    output_root1 = f"{exp_dir}/f0"
    output_root2 = f"{exp_dir}/f0_voiced"

    os.makedirs(output_root1, exist_ok=True)
    os.makedirs(output_root2, exist_ok=True)

    for name in sorted(list(os.listdir(input_root))):
        input_path = f"{input_root}/{name}"
        if "spec" in input_path:
            continue
        output_path1 = f"{output_root1}/{name}"
        output_path2 = f"{output_root2}/{name}"
        paths.append([input_path, output_path1, output_path2])

    print(f"Starting extraction with {num_processes} cores and {f0_method}...")

    start_time = time.time()

    # Use multiprocessing Pool for parallel processing with progress bar
    with tqdm.tqdm(total=len(paths), desc="F0 Extraction") as pbar:
        with Pool(processes=num_processes) as pool:
            for _ in pool.imap_unordered(feature_input.process_file, paths):
                pbar.update()

    elapsed_time = time.time() - start_time
    print(f"F0 extraction completed in {elapsed_time:.2f} seconds.")
