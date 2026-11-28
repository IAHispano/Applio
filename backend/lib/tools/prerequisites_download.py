import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
from backend.lib.tools.http_download import download_http


def migrate_legacy_models(base_dir: str = "."):
    """Seamlessly migrate weights from legacy rvc/models/ into models/ so users never re-download."""
    legacy_root = os.path.join(base_dir, "rvc", "models")
    target_root = os.path.join(base_dir, "models")
    if not os.path.isdir(legacy_root):
        return
    moved = 0
    for root, _, files in os.walk(legacy_root):
        for fn in files:
            if fn == ".gitkeep":
                continue
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, legacy_root).replace("\\", "/")
            if rel.startswith("pretraineds/hifi-gan/"):
                rel = "pretraineds/rvc/" + rel[len("pretraineds/") :]
            elif rel.startswith("pretraineds/refinegan/"):
                rel = "pretraineds/rvc/" + rel[len("pretraineds/") :]
            dst = os.path.join(target_root, rel)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.move(src, dst)
                    moved += 1
                except OSError:
                    pass
    if moved > 0:
        print(f"Migrated {moved} existing model file(s) from legacy 'rvc/models/' to 'models/'.", flush=True)

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

pretraineds_hifigan_list = [
    (
        "pretrained_v2/",
        [
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    ),
]
pretraineds_refinegan_list = [
    (
        "refinegan/",
        [
            "f0D24k.pth",
            "f0G24k.pth",
            "f0D32k.pth",
            "f0G32k.pth",
        ],
    ),
]
pretraineds_hsvc_list = [
    (
        "hsvc/",
        [
            "hsvc_base_48k.pth",
            "hsvc_base_48k_d.pth",
        ],
    ),
]
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt", "swift.onnx"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": "models/pretraineds/rvc/hifi-gan/",
    "refinegan/": "models/pretraineds/rvc/refinegan/",
    "hsvc/": "models/pretraineds/hsvc/",
    "embedders/contentvec/": "models/embedders/contentvec/",
    "predictors/": "models/predictors/",
    "formant/": "models/formant/",
}


def is_downloaded(file):
    return os.path.isfile(file) and os.path.getsize(file) > 0


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not is_downloaded(destination_path):
                url = f"{url_base}/{remote_folder}{file}"
                # Size discovery only controls progress; it must never prevent
                # downloading when HEAD is unsupported or has no Content-Length.
                try:
                    with requests.head(
                        url, allow_redirects=True, timeout=30
                    ) as response:
                        response.raise_for_status()
                        total_size += int(response.headers.get("content-length", 0))
                except (requests.RequestException, ValueError):
                    pass
    return total_size


def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """

    previous = 0

    def progress(received, total):
        nonlocal previous
        global_bar.update(received - previous)
        previous = received

    download_http(url, destination_path, progress=progress)


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not is_downloaded(destination_path):
                    url = f"{url_base}/{remote_folder}{file}"
                    futures.append(
                        executor.submit(
                            download_file, url, destination_path, global_bar
                        )
                    )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    total_size += get_file_size_if_missing(pretraineds_hifigan)
    if pretraineds_hifigan:
        total_size += get_file_size_if_missing(pretraineds_refinegan_list)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Manage the download pipeline for different categories of files.
    """
    migrate_legacy_models()
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
    )

    with tqdm(
        total=total_size or None,
        unit="iB",
        unit_scale=True,
        desc="Downloading all files",
    ) as global_bar:
        if models:
            download_mapping_files(models_list, global_bar)
            download_mapping_files(embedders_list, global_bar)
        if exe:
            if os.name == "nt":
                download_mapping_files(executables_list, global_bar)
            else:
                print("No executables needed")
        if pretraineds_hifigan:
            download_mapping_files(pretraineds_hifigan_list, global_bar)
            download_mapping_files(pretraineds_refinegan_list, global_bar)


def ensure_pretrained(vocoder, sample_rate):
    """Download only the default G/D pair needed for this training run."""
    migrate_legacy_models()
    if str(vocoder).upper() == "HSVC":
        sr_int = int(sample_rate)
        if sr_int < 16000 or sr_int > 48000:
            raise ValueError(
                f"HSVC universal pretrained supports 16000-48000 Hz (got {sample_rate} Hz)"
            )
        remote = "hsvc/"
        files = ["hsvc_base_48k.pth", "hsvc_base_48k_d.pth"]
        local_dir = os.path.abspath(folder_mapping_list[remote])
        os.makedirs(local_dir, exist_ok=True)
        paths = [os.path.join(local_dir, f) for f in files]
        if not all(is_downloaded(file) for file in paths):
            hsvc_base_url = os.environ.get(
                "APPLIO_HSVC_PRETRAIN_URL", f"{url_base}/hsvc"
            ).rstrip("/")
            print(
                f"Downloading universal HSVC pretrained (48 kHz -> adapts to {sr_int} Hz) from {hsvc_base_url}...",
                flush=True,
            )
            with tqdm(
                total=None, unit="iB", unit_scale=True, desc="Downloading HSVC pretrains"
            ) as bar:
                for fname, dst in zip(files, paths):
                    if not is_downloaded(dst):
                        try:
                            download_file(f"{hsvc_base_url}/{fname}", dst, bar)
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to download HSVC pretrained '{fname}' from {hsvc_base_url}/{fname}: {e}\n"
                                f"Place hsvc_base_48k.pth and hsvc_base_48k_d.pth in '{local_dir}' "
                                f"(or set APPLIO_HSVC_PRETRAIN_URL)."
                            ) from e
        if not all(is_downloaded(file) for file in paths):
            raise RuntimeError(
                f"Default HSVC pretrains are missing in '{local_dir}'; training cannot start."
            )
        return tuple(paths)
    rates = {"HiFi-GAN": (32000, 40000, 48000), "RefineGAN": (24000, 32000)}
    if vocoder not in rates or int(sample_rate) not in rates[vocoder]:
        raise ValueError(f"Unsupported pretrained: {vocoder} at {sample_rate} Hz")
    remote = "pretrained_v2/" if vocoder == "HiFi-GAN" else "refinegan/"
    files = [f"f0{kind}{int(sample_rate) // 1000}k.pth" for kind in ("G", "D")]
    mapping = [(remote, files)]
    paths = [
        os.path.abspath(os.path.join(folder_mapping_list[remote], file))
        for file in files
    ]
    if not all(is_downloaded(file) for file in paths):
        print(
            f"Downloading default {vocoder} pretrains ({sample_rate} Hz)...", flush=True
        )
        with tqdm(
            total=None, unit="iB", unit_scale=True, desc="Downloading pretrains"
        ) as bar:
            download_mapping_files(mapping, bar)
    if not all(is_downloaded(file) for file in paths):
        raise RuntimeError("Default pretrains are missing; training cannot start.")
    return tuple(paths)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Applio Prerequisites Downloader")
    parser.add_argument("--pretraineds-hifigan", action="store_true", default=False)
    parser.add_argument("--models", action="store_true", default=False)
    parser.add_argument("--exe", action="store_true", default=False)
    parser.add_argument("--vocoder", choices=["HiFi-GAN", "RefineGAN", "HSVC"])
    parser.add_argument("--sample-rate", type=int)
    args = parser.parse_args()

    if args.vocoder:
        if not args.sample_rate:
            parser.error("--sample-rate is required with --vocoder")
        ensure_pretrained(args.vocoder, args.sample_rate)
    else:
        prequisites_download_pipeline(args.pretraineds_hifigan, args.models, args.exe)
    print("Prerequisites installed successfully.")
