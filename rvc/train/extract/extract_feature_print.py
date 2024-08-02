import os
import sys
import tqdm
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import time

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.utils import load_embedding
from rvc.configs.config import Config

config = Config()


def setup_paths(exp_dir: str, version: str):
    """Set up input and output paths."""
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    out_path = os.path.join(
        exp_dir, "v1_extracted" if version == "v1" else "v2_extracted"
    )
    os.makedirs(out_path, exist_ok=True)
    return wav_path, out_path


def read_wave(wav_path: str, normalize: bool = False):
    """Read a wave file and return its features."""
    wav, sr = sf.read(wav_path)
    assert sr == 16000, "Sample rate must be 16000"

    feats = torch.from_numpy(wav)
    feats = feats.half() if config.is_half else feats.float()
    feats = feats.mean(-1) if feats.dim() == 2 else feats
    feats = feats.view(1, -1)

    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats


def process_file(
    file: str,
    wav_path: str,
    out_path: str,
    model: torch.nn.Module,
    device: str,
    version: str,
    saved_cfg: Config,
):
    """Process a single audio file."""
    wav_file_path = os.path.join(wav_path, file)
    out_file_path = os.path.join(out_path, file.replace("wav", "npy"))

    if os.path.exists(out_file_path):
        return

    # Load and prepare features
    feats = read_wave(wav_file_path, normalize=saved_cfg.task.normalize)

    # Adjust dtype based on the device
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


def main():
    """Main function to orchestrate the feature extraction process."""
    try:
        exp_dir = str(sys.argv[1])
        version = str(sys.argv[2])
        gpus = str(sys.argv[3])
        embedder_model = str(sys.argv[4])
        embedder_model_custom = str(sys.argv[5]) if len(sys.argv) > 5 else None

        os.environ["CUDA_VISIBLE_DEVICES"] = gpus.replace("-", ",")
    except IndexError:
        print("Invalid arguments provided.")
        sys.exit(1)

    wav_path, out_path = setup_paths(exp_dir, version)

    print("Starting feature extraction...")
    start_time = time.time()

    models, saved_cfg, task = load_embedding(embedder_model, embedder_model_custom)
    model = models[0]

    gpus = gpus.split("-") if gpus != "-" else ["cpu"]

    devices = []
    for gpu in gpus:
        try:
            if gpu != "cpu":
                index = int(gpu)
                if index < torch.cuda.device_count():
                    devices.append(f"cuda:{index}")
                else:
                    print(
                        f"Oops, there was an issue initializing GPU. Maybe you don't have a GPU? No worries, switching to CPU for now."
                    )
                    devices.append("cpu")
            else:
                devices.append("cpu")
        except ValueError:
            f"Oops, there was an issue initializing GPU. Maybe you don't have a GPU? No worries, switching to CPU for now."
            devices.append("cpu")

    paths = sorted(os.listdir(wav_path))
    if not paths:
        print("No audio files found. Make sure you have provided the audios correctly.")
        sys.exit(1)

    pbar = tqdm.tqdm(total=len(paths), desc="Feature Extraction")

    # Create a list of tasks to be processed
    tasks = [
        (
            file,
            wav_path,
            out_path,
            model,
            device,
            version,
            saved_cfg,
        )
        for file in paths
        if file.endswith(".wav")
        for device in devices
    ]

    # Process files
    for task in tasks:
        try:
            process_file(*task)
        except Exception as error:
            print(f"An error occurred processing {task[0]}: {error}")
        pbar.update(1)

    pbar.close()
    elapsed_time = time.time() - start_time
    print(f"Feature extraction completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
