import os
import sys
import time
import tqdm
import torch
import torchcrepe
import numpy as np
from multiprocessing import Pool

current_directory = os.getcwd()
sys.path.append(current_directory)

from rvc.lib.utils import load_audio
from rvc.lib.predictors.RMVPE import RMVPE0Predictor

# Parse command line arguments
exp_dir = str(sys.argv[1])
f0_method = str(sys.argv[2])
hop_length = int(sys.argv[3])
num_processes = int(sys.argv[4])


class FeatureInput:
    """Class for F0 extraction."""

    def __init__(self, sample_rate=16000, hop_size=160):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.model_rmvpe = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
            is_half=False,
            device="cpu",
        )

    def compute_f0(self, np_arr, f0_method, hop_length):
        """Extract F0 using the specified method."""
        p_len = np_arr.shape[0] // self.hop

        if f0_method == "crepe":
            f0 = self.get_crepe(np_arr, p_len, hop_length)
        elif f0_method == "rmvpe":
            f0 = self.model_rmvpe.infer_from_audio(np_arr, thred=0.03)
        else:
            raise ValueError(f"Unknown F0 method: {f0_method}")

        return f0

    def get_crepe(self, x, p_len, hop_length):
        """Extract F0 using CREPE."""
        audio = torch.from_numpy(x.astype(np.float32)).to("cpu")
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
            device="cpu",
            pad=True,
        )

        source = pitch.squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        return np.nan_to_num(target)

    def coarse_f0(self, f0):
        """Convert F0 to coarse F0."""
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

    def process_file(self, file_info):
        """Process a single audio file for F0 extraction."""
        inp_path, opt_path1, opt_path2, np_arr = file_info

        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
            return

        try:
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path2, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(f"F0 extraction failed for {inp_path}: {error}")


if __name__ == "__main__":
    feature_input = FeatureInput()
    paths = []
    input_root = os.path.join(exp_dir, "sliced_audios_16k")
    output_root1 = os.path.join(exp_dir, "f0")
    output_root2 = os.path.join(exp_dir, "f0_voiced")

    os.makedirs(output_root1, exist_ok=True)
    os.makedirs(output_root2, exist_ok=True)

    for name in sorted(os.listdir(input_root)):
        if "spec" in name:
            continue
        input_path = os.path.join(input_root, name)
        output_path1 = os.path.join(output_root1, name)
        output_path2 = os.path.join(output_root2, name)
        np_arr = load_audio(input_path, 16000) #self.fs?
        paths.append([input_path, output_path1, output_path2, np_arr])

    print(f"Starting extraction with {num_processes} cores and {f0_method}...")

    start_time = time.time()

    # Use multiprocessing Pool for parallel processing with progress bar
    with tqdm.tqdm(total=len(paths), desc="F0 Extraction") as pbar:
        with Pool(processes=num_processes) as pool:
            for _ in pool.imap_unordered(feature_input.process_file, paths):
                pbar.update()

    elapsed_time = time.time() - start_time
    print(f"F0 extraction completed in {elapsed_time:.2f} seconds.")
