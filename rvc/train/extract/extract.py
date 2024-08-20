import os
import sys
import time
import tqdm
import torch
import torchcrepe
import numpy as np
import soundfile as sf
from multiprocessing import Pool
from functools import partial
import concurrent.futures
import torch.nn.functional as F

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from rvc.lib.utils import load_audio, load_embedding
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.configs.config import Config

# Load config
config = Config()


def setup_paths(exp_dir: str, version: str = None):
    """Set up input and output paths."""
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    if version:
        out_path = os.path.join(
            exp_dir, "v1_extracted" if version == "v1" else "v2_extracted"
        )
        os.makedirs(out_path, exist_ok=True)
        return wav_path, out_path
    else:
        output_root1 = os.path.join(exp_dir, "f0")
        output_root2 = os.path.join(exp_dir, "f0_voiced")
        os.makedirs(output_root1, exist_ok=True)
        os.makedirs(output_root2, exist_ok=True)
        return wav_path, output_root1, output_root2


def read_wave(wav_path: str, normalize: bool = False):
    """Read a wave file and return its features."""
    wav, sr = sf.read(wav_path)
    assert sr == 16000, "Sample rate must be 16000"

    feats = torch.from_numpy(wav).float()
    if config.is_half:
        feats = feats.half()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    feats = feats.view(1, -1)

    if normalize:
        feats = F.layer_norm(feats, feats.shape)
    return feats


def get_device(gpu_index):
    """Get the appropriate device based on GPU availability."""
    if gpu_index == "cpu":
        return "cpu"
    try:
        index = int(gpu_index)
        if index < torch.cuda.device_count():
            return f"cuda:{index}"
        else:
            print("Invalid GPU index. Switching to CPU.")
    except ValueError:
        print("Invalid GPU index format. Switching to CPU.")
    return "cpu"


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
        inp_path, opt_path1, opt_path2, np_arr = file_info

        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
            return

        try:
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path2, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(f"An error occurred extracting file {inp_path}: {error}")

    def process_files(self, files, f0_method, hop_length, pbar):
        """Process multiple files."""
        for file_info in files:
            self.process_file(file_info, f0_method, hop_length)
            pbar.update()


def run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, gpus):
    input_root, *output_roots = setup_paths(exp_dir)

    if len(output_roots) == 2:
        output_root1, output_root2 = output_roots
    else:
        output_root1 = output_roots[0]
        output_root2 = None

    paths = [
        (
            os.path.join(input_root, name),
            os.path.join(output_root1, name) if output_root1 else None,
            os.path.join(output_root2, name) if output_root2 else None,
            load_audio(os.path.join(input_root, name), 16000),
        )
        for name in sorted(os.listdir(input_root))
        if "spec" not in name
    ]

    print(f"Starting pitch extraction with {num_processes} cores and {f0_method}...")
    start_time = time.time()

    if gpus != "-":
        gpus = gpus.split("-")
        num_gpus = len(gpus)
        process_partials = []
        pbar = tqdm.tqdm(total=len(paths), desc="Pitch Extraction")

        for idx, gpu in enumerate(gpus):
            device = get_device(gpu)
            feature_input = FeatureInput(device=device)
            part_paths = paths[idx::num_gpus]
            process_partials.append((feature_input, part_paths))

        with concurrent.futures.ThreadPoolExecutor() as executor:
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

    else:
        feature_input = FeatureInput(device="cpu")
        with tqdm.tqdm(total=len(paths), desc="Pitch Extraction") as pbar:
            with Pool(processes=num_processes) as pool:
                process_file_partial = partial(
                    feature_input.process_file,
                    f0_method=f0_method,
                    hop_length=hop_length,
                )
                for _ in pool.imap_unordered(process_file_partial, paths):
                    pbar.update()

    elapsed_time = time.time() - start_time
    print(f"Pitch extraction completed in {elapsed_time:.2f} seconds.")


def process_file_embedding(file, wav_path, out_path, model, device, version, saved_cfg):
    """Process a single audio file for embedding extraction."""
    wav_file_path = os.path.join(wav_path, file)
    out_file_path = os.path.join(out_path, file.replace("wav", "npy"))

    if os.path.exists(out_file_path):
        return

    feats = read_wave(wav_file_path, normalize=saved_cfg.task.normalize)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    feats = feats.to(dtype).to(device)

    padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(dtype).to(device)
    inputs = {
        "source": feats,
        "padding_mask": padding_mask,
        "output_layer": 9 if version == "v1" else 12,
    }

    with torch.no_grad():
        model = model.to(device).to(dtype)
        logits = model.extract_features(**inputs)
        feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

    feats = feats.squeeze(0).float().cpu().numpy()
    if not np.isnan(feats).any():
        np.save(out_file_path, feats, allow_pickle=False)
    else:
        print(f"{file} contains NaN values and will be skipped.")


def run_embedding_extraction(
    exp_dir, version, gpus, embedder_model, embedder_model_custom
):
    """Main function to orchestrate the embedding extraction process."""
    wav_path, out_path = setup_paths(exp_dir, version)

    print("Starting embedding extraction...")
    start_time = time.time()

    models, saved_cfg, _ = load_embedding(embedder_model, embedder_model_custom)
    model = models[0]
    devices = [get_device(gpu) for gpu in (gpus.split("-") if gpus != "-" else ["cpu"])]

    paths = sorted([file for file in os.listdir(wav_path) if file.endswith(".wav")])
    if not paths:
        print("No audio files found. Make sure you have provided the audios correctly.")
        sys.exit(1)

    pbar = tqdm.tqdm(total=len(paths) * len(devices), desc="Embedding Extraction")

    tasks = [
        (file, wav_path, out_path, model, device, version, saved_cfg)
        for file in paths
        for device in devices
    ]

    for task in tasks:
        try:
            process_file_embedding(*task)
        except Exception as error:
            print(f"An error occurred processing {task[0]}: {error}")
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

    # Run Pitch Extraction
    run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, gpus)

    # Run Embedding Extraction
    run_embedding_extraction(
        exp_dir, version, gpus, embedder_model, embedder_model_custom
    )

    # Run Preparing Files
    generate_config(version, sample_rate, exp_dir)
    generate_filelist(pitch_guidance, exp_dir, version, sample_rate)
