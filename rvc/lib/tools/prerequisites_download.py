import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"
pretraineds_v1_list = [
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
pretraineds_v2_list = [
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

models_list = [
    (
        "predictors/",
        [
            "rmvpe.pt",
            "fcpe.pt",
        ],
    ),
]

embedders_list = [
    (
        "embedders/",
        [
            "contentvec_base.pt",
        ],
    ),
]


executables_list = ["ffmpeg.exe", "ffprobe.exe"]

folder_mapping_list = {
    "pretrained_v1/": "rvc/models/pretraineds/pretrained_v1/",
    "pretrained_v2/": "rvc/models/pretraineds/pretrained_v2/",
    "embedders/": "rvc/models/embedders/",
    "predictors/": "rvc/models/predictors/",
}


def download_file(url, destination_path, desc):
    if not os.path.exists(destination_path):
        os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=desc)
        with open(destination_path, "wb") as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR: Something went wrong during the download")


def download_files(file_list):
    with ThreadPoolExecutor() as executor:
        futures = []
        for file_name in file_list:
            destination_path = os.path.join(file_name)
            url = f"{url_base}/{file_name}"
            futures.append(
                executor.submit(download_file, url, destination_path, file_name)
            )
        for future in futures:
            future.result()


def download_mapping_files(list):
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                url = f"{url_base}/{remote_folder}{file}"
                futures.append(
                    executor.submit(
                        download_file, url, destination_path, f"{remote_folder}{file}"
                    )
                )
        for future in futures:
            future.result()


def prequisites_download_pipeline(pretraineds_v1, pretraineds_v2, models, exe):
    if models == "True":
        download_mapping_files(models_list)
        download_mapping_files(embedders_list)

    if exe == "True" and os.name == "nt":
        download_files(executables_list)

    if pretraineds_v1 == "True":
        download_mapping_files(pretraineds_v1_list)

    if pretraineds_v2 == "True":
        download_mapping_files(pretraineds_v2_list)

    clear_console()  # Clear the console after all downloads are completed


def clear_console():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")
