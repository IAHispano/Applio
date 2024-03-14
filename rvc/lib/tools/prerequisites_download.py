import os
import wget
import argparse

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"
pretraineds_v1 = [
    (
        "pretrained_v1/",
        [
            "D32k.pth",
            "D40k.pth",
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    ),
]
pretraineds_v2 = [
    (
        "pretrained_v2/",
        [
            "D32k.pth",
            "D40k.pth",
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    ),
]

models = [
    "hubert_base.pt",
    "rmvpe.pt",
    "fcpe.pt",
    # "rmvpe.onnx"
]

executables = ["ffmpeg.exe", "ffprobe.exe"]

folder_mapping = {
    "pretrained_v1/": "rvc/pretraineds/pretrained_v1/",
    "pretrained_v2/": "rvc/pretraineds/pretrained_v2/",
}

parser = argparse.ArgumentParser(description="Download files from a URL.")
parser.add_argument(
    "--pretraineds_v1", type=str, default="False", help="Download pretrained_v1 files"
)
parser.add_argument(
    "--pretraineds_v2", type=str, default="False", help="Download pretrained_v2 files"
)
parser.add_argument("--models", type=str, default="False", help="Download model files")
parser.add_argument(
    "--exe", type=str, default="False", help="Download executable files"
)

args = parser.parse_args()


def download_files(file_list):
    for file_name in file_list:
        destination_path = os.path.join(file_name)
        url = f"{url_base}/{file_name}"
        if not os.path.exists(destination_path):
            os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
            print(f"\nDownloading {url} to {destination_path}...")
            wget.download(url, out=destination_path)


if args.models == "True":
    download_files(models)

if args.exe == "True" and os.name == "nt":
    download_files(executables)

if args.pretraineds_v1 == "True":
    for remote_folder, file_list in pretraineds_v1:
        local_folder = folder_mapping.get(remote_folder, "")
        for file in file_list:
            destination_path = os.path.join(local_folder, file)
            url = f"{url_base}/{remote_folder}{file}"
            if not os.path.exists(destination_path):
                os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
                print(f"\nDownloading {url} to {destination_path}...")
                wget.download(url, out=destination_path)

if args.pretraineds_v2 == "True":
    for remote_folder, file_list in pretraineds_v2:
        local_folder = folder_mapping.get(remote_folder, "")
        for file in file_list:
            destination_path = os.path.join(local_folder, file)
            url = f"{url_base}/{remote_folder}{file}"
            if not os.path.exists(destination_path):
                os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
                print(f"\nDownloading {url} to {destination_path}...")
                wget.download(url, out=destination_path)
