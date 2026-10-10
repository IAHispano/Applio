import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

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
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt", "swift.onnx"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": "rvc/models/pretraineds/hifi-gan/",
    "refinegan/": "rvc/models/pretraineds/refinegan/",
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
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

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    temporary_path = None
    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            expected = int(response.headers.get("content-length", 0))
            downloaded = 0
            with tempfile.NamedTemporaryFile(
                dir=dir_name or ".", prefix=".applio-download-", delete=False
            ) as file:
                temporary_path = file.name
                for data in response.iter_content(1024 * 1024):
                    if data:
                        file.write(data)
                        downloaded += len(data)
                        global_bar.update(len(data))
            if not downloaded or (expected and downloaded != expected):
                raise RuntimeError(f"Incomplete download: {url}")
            # Only complete, successful responses become reusable model files.
            os.replace(temporary_path, destination_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


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
    parser.add_argument("--vocoder", choices=["HiFi-GAN", "RefineGAN"])
    parser.add_argument("--sample-rate", type=int)
    args = parser.parse_args()

    if args.vocoder:
        if not args.sample_rate:
            parser.error("--sample-rate is required with --vocoder")
        ensure_pretrained(args.vocoder, args.sample_rate)
    else:
        prequisites_download_pipeline(args.pretraineds_hifigan, args.models, args.exe)
    print("Prerequisites installed successfully.")
