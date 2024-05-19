import os
import wget

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
    "rmvpe.pt",
    "fcpe.pt",
    # "rmvpe.onnx"
]

embedders_list = [
    (
        "embedders/",
        [
            "hubert_base.pt",
            "contentvec_base.pt",
        ],
    ),
]

executables_list = ["ffmpeg.exe", "ffprobe.exe"]

folder_mapping_list = {
    "pretrained_v1/": "rvc/pretraineds/pretrained_v1/",
    "pretrained_v2/": "rvc/pretraineds/pretrained_v2/",
    "embedders/": "rvc/embedders/",
}


def prequisites_download_pipeline(pretraineds_v1, pretraineds_v2, models, exe):
    def download_files(file_list):
        for file_name in file_list:
            destination_path = os.path.join(file_name)
            url = f"{url_base}/{file_name}"
            if not os.path.exists(destination_path):
                os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
                print(f"\nDownloading {url} to {destination_path}...")
                wget.download(url, out=destination_path)

    def download_mapping_files(list):
        for remote_folder, file_list in list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                url = f"{url_base}/{remote_folder}{file}"
                if not os.path.exists(destination_path):
                    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
                    print(f"\nDownloading {url} to {destination_path}...")
                    wget.download(url, out=destination_path)

    if models == "True":
        download_files(models_list)
        download_mapping_files(embedders_list)

    if exe == "True" and os.name == "nt":
        download_files(executables_list)

    if pretraineds_v1 == "True":
        download_mapping_files(pretraineds_v1_list)

    if pretraineds_v2 == "True":
        download_mapping_files(pretraineds_v2_list)
