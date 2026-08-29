import concurrent.futures
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import torch
import tqdm

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda
from rvc.configs.config import Config
from rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE, load_high_register_settings
from rvc.lib.utils import load_audio_16k, load_embedding
from rvc.train.extract.preparing_files import generate_config, generate_filelist

# Load config
config = Config()
mp.set_start_method("spawn", force=True)

# formats the preprocessor may have written the slices in
AUDIO_EXTENSIONS = (".wav", ".flac")


def strtobool(value: str) -> bool:
    return str(value).lower() in ("yes", "true", "t", "y", "1")


class FeatureInput:
    def __init__(self, f0_method="rmvpe", device="cpu", compact_f0=False):
        self.hop_size = 160  # default
        self.sample_rate = 16000  # default
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        # coarse pitch is a bucket index clipped to [1, 255], so uint8 holds it
        # exactly at an eighth of the int64 size
        self.coarse_dtype = np.uint8 if compact_f0 else int
        if f0_method in ("crepe", "crepe-tiny"):
            self.model = CREPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "rmvpe":
            # Training labels must be the TRUE pitch, never fold-mode values
            # (fold is an inference-side trick for models trained on stock
            # octave-folded labels). Only relevant when the corrector is
            # enabled in assets/config.json.
            high_register = load_high_register_settings()
            high_register["mode"] = "true_pitch"
            self.model = RMVPE(
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.hop_size,
                high_register=high_register,
            )
        elif f0_method == "fcpe":
            self.model = FCPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        self.f0_method = f0_method

    def compute_f0(self, x, p_len=None):
        if self.f0_method == "crepe":
            f0 = self.model.get_f0(x, self.f0_min, self.f0_max, p_len, "full")
        elif self.f0_method == "crepe-tiny":
            f0 = self.model.get_f0(x, self.f0_min, self.f0_max, p_len, "tiny")
        elif self.f0_method == "rmvpe":
            f0 = self.model.get_f0(x, filter_radius=0.03)
        elif self.f0_method == "fcpe":
            f0 = self.model.get_f0(x, p_len, filter_radius=0.006)
        return f0

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
        return np.rint(f0_mel).astype(self.coarse_dtype)

    def process_file(self, file_info):
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return

        try:
            np_arr = load_audio_16k(inp_path)
            feature_pit = self.compute_f0(np_arr)
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            )


def process_files(files, f0_method, device, threads, compact_f0=False):
    if device == "cpu":
        torch.set_num_threads(max(1, threads))
    fe = FeatureInput(f0_method=f0_method, device=device, compact_f0=compact_f0)
    with tqdm.tqdm(total=len(files), leave=True) as pbar:
        for file_info in files:
            fe.process_file(file_info)
            pbar.update(1)


def run_pitch_extraction(files, devices, f0_method, threads, compact_f0=False):
    devices_str = ", ".join(devices)
    print(f"Starting pitch extraction on {devices_str} using {f0_method}...")
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_files,
                files[i :: len(devices)],
                f0_method,
                devices[i],
                threads // len(devices),
                compact_f0,
            )
            for i in range(len(devices))
        ]
        # .result() is what makes a worker crash reach this process, otherwise
        # extraction reports success over a half-built dataset
        for task in concurrent.futures.as_completed(tasks):
            task.result()

    print(f"Pitch extraction completed in {time.time() - start_time:.2f} seconds.")


def process_file_embedding(
    files,
    embedder_model,
    embedder_model_custom,
    device_num,
    device,
    n_threads,
    fp16_embeddings=False,
):
    model = load_embedding(embedder_model, embedder_model_custom).to(device).float()
    model.eval()
    n_threads = max(1, n_threads)
    if device == "cpu":
        torch.set_num_threads(n_threads)
    # half precision on the forward pass only, the result is stored as float32
    use_amp = fp16_embeddings and str(device).startswith("cuda")

    def worker(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio_16k(wav_file_path)).to(device).float()
        feats = feats.view(1, -1)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            result = model(feats)["last_hidden_state"]
        feats_out = result.squeeze(0).float().cpu().numpy()
        # inf is as unusable as nan downstream, and an isnan check misses it
        if np.isfinite(feats_out).all():
            np.save(out_file_path, feats_out, allow_pickle=False)
        else:
            print(f"{wav_file_path} produced non-finite values; skipping.")

    # one file at a time: threads here shared a single CUDA model, contending
    # for the GIL while each in-flight file added its own VRAM peak
    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        with torch.no_grad():
            for file_info in files:
                worker(file_info)
                pbar.update(1)


def run_embedding_extraction(
    files,
    devices,
    embedder_model,
    embedder_model_custom,
    threads,
    fp16_embeddings=False,
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
                fp16_embeddings,
            )
            for i in range(len(devices))
        ]
        for task in concurrent.futures.as_completed(tasks):
            task.result()

    print(f"Embedding extraction completed in {time.time() - start_time:.2f} seconds.")


def discard_16k_slices(wav_path):
    """
    Deletes the 16 kHz slices once the features derived from them exist.

    They feed pitch and embedder extraction only, so training never reads them.
    Re-extracting afterwards means preprocessing the dataset again.

    Args:
        wav_path (str): Path to the sliced_audios_16k directory.
    """
    if os.path.basename(os.path.normpath(wav_path)) != "sliced_audios_16k":
        return
    if not os.path.isdir(wav_path):
        return

    # the mute folders are shared by every model that includes silent files
    experiment_name = os.path.basename(os.path.dirname(os.path.normpath(wav_path)))
    if experiment_name.lower().startswith("mute"):
        print(f"Keeping 16 kHz slices: '{experiment_name}' is a shared mute asset.")
        return

    # only discard once the artifacts that replace them are on disk
    exp_dir = os.path.dirname(os.path.normpath(wav_path))
    if not os.path.isfile(os.path.join(exp_dir, "filelist.txt")):
        print(
            "Keeping 16 kHz slices: filelist.txt is missing, so extraction looks incomplete."
        )
        return

    freed = 0
    count = 0
    for root, _, names in os.walk(wav_path):
        for name in names:
            try:
                freed += os.path.getsize(os.path.join(root, name))
                count += 1
            except OSError:
                pass

    try:
        shutil.rmtree(wav_path)
    except OSError as error:
        print(f"Could not remove the 16 kHz slices: {error}")
        return

    print(f"Removed {count} 16 kHz slices, freeing {freed / (1024 ** 3):.2f} GB.")


if __name__ == "__main__":
    exp_dir = sys.argv[1]
    f0_method = sys.argv[2]
    num_processes = int(sys.argv[3])
    gpus = sys.argv[4]
    sample_rate = sys.argv[5]
    embedder_model = sys.argv[6]
    embedder_model_custom = sys.argv[7] if len(sys.argv) > 7 else None
    include_mutes = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    remove_16k_slices = strtobool(sys.argv[9]) if len(sys.argv) > 9 else False
    compact_f0 = strtobool(sys.argv[10]) if len(sys.argv) > 10 else False
    fp16_embeddings = strtobool(sys.argv[11]) if len(sys.argv) > 11 else False

    wav_path = os.path.join(exp_dir, "sliced_audios_16k")

    if not os.path.exists(wav_path):
        print(
            f"Folder for feature extraction not found at {wav_path}. Did you run the preprocessing step?"
        )
        sys.exit(1)

    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

    chosen_embedder_model = (
        embedder_model_custom if embedder_model == "custom" else embedder_model
    )
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data["embedder_model"] = chosen_embedder_model
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    audio_files = [
        path
        for path in glob.glob(os.path.join(wav_path, "*"))
        if os.path.isfile(path) and path.lower().endswith(AUDIO_EXTENSIONS)
    ]
    # longest first, so the strided split leaves every device a similar total
    audio_files.sort(key=os.path.getsize, reverse=True)

    files = []
    for file in audio_files:
        file_name = os.path.basename(file)
        file_info = [
            file,
            os.path.join(exp_dir, "f0", file_name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
            os.path.join(exp_dir, "extracted", os.path.splitext(file_name)[0] + ".npy"),
        ]
        files.append(file_info)

    if not files:
        print(
            f"Sliced audios not found at {wav_path}. Did you run the preprocessing step?"
        )
        sys.exit(1)

    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]

    run_pitch_extraction(files, devices, f0_method, num_processes, compact_f0)

    run_embedding_extraction(
        files,
        devices,
        embedder_model,
        embedder_model_custom,
        num_processes,
        fp16_embeddings,
    )

    generate_config(sample_rate, exp_dir)
    generate_filelist(exp_dir, sample_rate, include_mutes)

    if remove_16k_slices:
        discard_16k_slices(wav_path)
