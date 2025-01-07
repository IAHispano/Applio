import os
import sys
import glob
import time
import tqdm
import torch
import torchcrepe
import numpy as np
import concurrent.futures
import multiprocessing as mp
import json

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda

from rvc.lib.utils import load_audio, load_embedding
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.configs.config import Config

# Load config
config = Config()
mp.set_start_method("spawn", force=True)


class FeatureInput:
    def __init__(self, sample_rate=16000, hop_size=160, device="cpu"):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.model_rmvpe = None

    def compute_f0(self, audio_array, method, hop_length):
        if method == "crepe":
            return self._get_crepe(audio_array, hop_length, type="full")
        elif method == "crepe-tiny":
            return self._get_crepe(audio_array, hop_length, type="tiny")
        elif method == "rmvpe":
            return self.model_rmvpe.infer_from_audio(audio_array, thred=0.03)

    def _get_crepe(self, x, hop_length, type):
        audio = torch.from_numpy(x.astype(np.float32)).to(self.device)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = audio.unsqueeze(0)
        pitch = torchcrepe.predict(
            audio,
            self.fs,
            hop_length,
            self.f0_min,
            self.f0_max,
            type,
            batch_size=hop_length * 2,
            device=audio.device,
            pad=True,
        )
        source = pitch.squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan
        return np.nan_to_num(
            np.interp(
                np.arange(0, len(source) * (x.size // self.hop), len(source))
                / (x.size // self.hop),
                np.arange(0, len(source)),
                source,
            )
        )

    def coarse_f0(self, f0):
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
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
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return

        try:
            np_arr = load_audio(inp_path, self.fs)
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            )

    def process_files(self, files, f0_method, hop_length, device, threads):
        self.device = device
        if f0_method == "rmvpe":
            self.model_rmvpe = RMVPE0Predictor(
                os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                device=device,
            )

        def worker(file_info):
            self.process_file(file_info, f0_method, hop_length)

        with tqdm.tqdm(total=len(files), leave=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                futures = [executor.submit(worker, f) for f in files]
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)


def run_pitch_extraction(files, devices, f0_method, hop_length, threads):
    devices_str = ", ".join(devices)
    print(
        f"Starting pitch extraction with {num_processes} cores on {devices_str} using {f0_method}..."
    )
    start_time = time.time()
    fe = FeatureInput()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                fe.process_files,
                files[i :: len(devices)],
                f0_method,
                hop_length,
                devices[i],
                threads // len(devices),
            )
            for i in range(len(devices))
        ]
        concurrent.futures.wait(tasks)

    print(f"Pitch extraction completed in {time.time() - start_time:.2f} seconds.")


def process_file_embedding(
    files, embedder_model, embedder_model_custom, device_num, device, n_threads
):
    model = load_embedding(embedder_model, embedder_model_custom).to(device).float()
    model.eval()
    n_threads = max(1, n_threads)

    def worker(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio(wav_file_path, 16000)).to(device).float()
        feats = feats.view(1, -1)
        with torch.no_grad():
            result = model(feats)["last_hidden_state"]
        feats_out = result.squeeze(0).float().cpu().numpy()
        if not np.isnan(feats_out).any():
            np.save(out_file_path, feats_out, allow_pickle=False)
        else:
            print(f"{wav_file_path} produced NaN values; skipping.")

    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, f) for f in files]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def run_embedding_extraction(
    files, devices, embedder_model, embedder_model_custom, threads
):
    devices_str = ", ".join(devices)
    print(
        f"Starting embedding extraction with {num_processes} cores on {devices_str}..."
    )
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_file_embedding,
                files[i :: len(devices)],
                embedder_model,
                embedder_model_custom,
                i,
                devices[i],
                threads // len(devices),
            )
            for i in range(len(devices))
        ]
        concurrent.futures.wait(tasks)

    print(f"Embedding extraction completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    exp_dir = sys.argv[1]
    f0_method = sys.argv[2]
    hop_length = int(sys.argv[3])
    num_processes = int(sys.argv[4])
    gpus = sys.argv[5]
    sample_rate = sys.argv[6]
    embedder_model = sys.argv[7]
    embedder_model_custom = sys.argv[8] if len(sys.argv) > 8 else None
    include_mutes = int(sys.argv[9]) if len(sys.argv) > 9 else 2

    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

    chosen_embedder_model = (
        embedder_model_custom if embedder_model == "custom" else embedder_model
    )
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data["embedder_model"] = chosen_embedder_model
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    files = []
    for file in glob.glob(os.path.join(wav_path, "*.wav")):
        file_name = os.path.basename(file)
        file_info = [
            file,
            os.path.join(exp_dir, "f0", file_name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
            os.path.join(exp_dir, "extracted", file_name.replace("wav", "npy")),
        ]
        files.append(file_info)

    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]

    run_pitch_extraction(files, devices, f0_method, hop_length, num_processes)

    run_embedding_extraction(
        files, devices, embedder_model, embedder_model_custom, num_processes
    )

    generate_config(sample_rate, exp_dir)
    generate_filelist(exp_dir, sample_rate, include_mutes)
