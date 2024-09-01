import os, glob
import sys
import time
import tqdm
import torch

# Zluda
if torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]"):
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
import torchcrepe
import numpy as np
import concurrent.futures

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from rvc.lib.utils import load_audio, load_embedding
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.configs.config import Config

# Load config
config = Config()


class FeatureInput:
    """Class for F0 extraction."""

    def __init__(self, sample_rate=16000, hop_size=160, device="cpu"):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.model_rmvpe = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
            is_half=False,
            device=device,
        )

    def compute_f0(self, np_arr, f0_method, hop_length):
        """Extract F0 using the specified method."""
        if f0_method == "crepe":
            return self.get_crepe(np_arr, hop_length)
        elif f0_method == "rmvpe":
            return self.model_rmvpe.infer_from_audio(np_arr, thred=0.03)
        else:
            raise ValueError(f"Unknown F0 method: {f0_method}")

    def get_crepe(self, x, hop_length):
        """Extract F0 using CREPE."""
        audio = torch.from_numpy(x.astype(np.float32)).to(self.device)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = audio.unsqueeze(0)

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

        source = pitch.squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * (x.size // self.hop), len(source))
            / (x.size // self.hop),
            np.arange(0, len(source)),
            source,
        )
        return np.nan_to_num(target)

    def coarse_f0(self, f0):
        """Convert F0 to coarse F0."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel = np.clip(
            (f0_mel - self.f0_mel_min)
            * (self.f0_bin - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1,
            1,
            self.f0_bin - 1,
        )
        return np.rint(f0_mel).astype(int)

    def process_file(self, file_info, f0_method, hop_length):
        """Process a single audio file for F0 extraction."""
        inp_path, opt_path1, opt_path2, _ = file_info
        # print(f"Process file {inp_path}. Class on {self.device}, model is on {self.model_rmvpe.device}")

        if os.path.exists(opt_path1) and os.path.exists(opt_path2):
            return

        try:
            np_arr = load_audio(inp_path, 16000)
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path2, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            )

    def process_files(self, files, f0_method, hop_length, pbar):
        """Process multiple files."""
        for file_info in files:
            self.process_file(file_info, f0_method, hop_length)
            pbar.update(1)


def run_pitch_extraction(files, devices, f0_method, hop_length, num_processes):
    print(f"Starting pitch extraction with {num_processes} cores and {f0_method}...")
    start_time = time.time()

    pbar = tqdm.tqdm(total=len(files), desc="Pitch Extraction")
    num_gpus = len(devices)
    process_partials = []
    for idx, gpu in enumerate(devices):
        device = torch.device(gpu)
        feature_input = FeatureInput(device=device)
        part_paths = files[idx::num_gpus]
        process_partials.append((feature_input, part_paths))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                FeatureInput.process_files,
                feature_input,
                part_paths,
                f0_method,
                hop_length,
                pbar,
            )
            for feature_input, part_paths in process_partials
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    pbar.close()

    elapsed_time = time.time() - start_time
    print(f"Pitch extraction completed in {elapsed_time:.2f} seconds.")


def process_file_embedding(file_info, model, device):
    """Process a single audio file for embedding extraction."""
    wav_file_path, _, _, out_file_path = file_info

    if os.path.exists(out_file_path):
        return
    dtype = torch.float16 if config.is_half and "cuda" in device else torch.float32
    model = model.to(dtype).to(device)
    feats = torch.from_numpy(load_audio(wav_file_path, 16000)).to(dtype).to(device)
    feats = feats.view(1, -1)

    with torch.no_grad():
        feats = model(feats)["last_hidden_state"]
        feats = model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats

    feats = feats.squeeze(0).float().cpu().numpy()
    if not np.isnan(feats).any():
        np.save(out_file_path, feats, allow_pickle=False)
    else:
        print(f"{file} contains NaN values and will be skipped.")


def run_embedding_extraction(files, devices, embedder_model, embedder_model_custom):
    """Main function to orchestrate the embedding extraction process."""
    print("Starting embedding extraction...")
    start_time = time.time()
    model = load_embedding(embedder_model, embedder_model_custom)

    pbar = tqdm.tqdm(total=len(files), desc="Embedding Extraction")

    # add multi-threading here?
    for i, file_info in enumerate(files):
        device = devices[i % len(devices)]
        try:
            process_file_embedding(file_info, model, device)
        except Exception as error:
            print(f"An error occurred processing {file_info[0]}: {error}")
        pbar.update(1)

    pbar.close()
    elapsed_time = time.time() - start_time
    print(f"Embedding extraction completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":

    exp_dir = sys.argv[1]
    f0_method = sys.argv[2]
    hop_length = int(sys.argv[3])
    num_processes = int(sys.argv[4])
    gpus = sys.argv[5]
    version = sys.argv[6]
    pitch_guidance = sys.argv[7]
    sample_rate = sys.argv[8]
    embedder_model = sys.argv[9]
    embedder_model_custom = sys.argv[10] if len(sys.argv) > 10 else None

    # prep
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, version + "_extracted"), exist_ok=True)

    files = []
    for file in glob.glob(os.path.join(wav_path, "*.wav")):
        file_name = os.path.basename(file)
        file_info = [
            file,  # full path to sliced 16k wav
            os.path.join(exp_dir, "f0", file_name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
            os.path.join(
                exp_dir, version + "_extracted", file_name.replace("wav", "npy")
            ),
        ]
        files.append(file_info)

    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]

    # Run Pitch Extraction
    run_pitch_extraction(files, devices, f0_method, hop_length, num_processes)

    # Run Embedding Extraction
    run_embedding_extraction(files, devices, embedder_model, embedder_model_custom)

    # Run Preparing Files
    generate_config(version, sample_rate, exp_dir)
    generate_filelist(pitch_guidance, exp_dir, version, sample_rate)
