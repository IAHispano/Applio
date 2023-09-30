import subprocess
import os
import sys

import errno
import shutil
import yt_dlp
import datetime
import torch
import glob
import gradio as gr
import traceback
import lib.infer.infer_libs.uvr5_pack.mdx as mdx
from lib.infer.modules.uvr5.mdxprocess import (
    get_model_list,
    id_to_ptm,
    prepare_mdx,
    run_mdx,
)
import requests
import wget
import ffmpeg
import hashlib
current_script_path = os.path.abspath(__file__)
script_parent_directory = os.path.dirname(current_script_path)
now_dir = os.path.dirname(script_parent_directory)
sys.path.append(now_dir)
import re
from lib.infer.modules.vc.pipeline import Pipeline

VC = Pipeline
from lib.infer.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

from assets.configs.config import Config
from lib.infer.modules.uvr5.mdxnet import MDXNetDereverb
from lib.infer.modules.uvr5.preprocess import AudioPre, AudioPreDeEcho
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
config = Config()

weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
audio_root = "assets/audios"
names = [
    os.path.join(root, file)
    for root, _, files in os.walk(weight_root)
    for file in files
    if file.endswith((".pth", ".onnx"))
]

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}
audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_root
]


uvr5_names = [
    name.replace(".pth", "")
    for name in os.listdir(weight_uvr5_root)
    if name.endswith(".pth") or "onnx" in name
]


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
import unicodedata

def format_title(title):
    formatted_title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8')
    formatted_title = re.sub(r'[\u2500-\u257F]+', '', title)
    formatted_title = re.sub(r'[^\w\s-]', '', title)
    formatted_title = re.sub(r'\s+', '_', formatted_title)
    return formatted_title


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def get_md5(temp_folder):
    for root, subfolders, files in os.walk(temp_folder):
        for file in files:
            if (
                not file.startswith("G_")
                and not file.startswith("D_")
                and file.endswith(".pth")
                and not "_G_" in file
                and not "_D_" in file
            ):
                md5_hash = calculate_md5(os.path.join(root, file))
                return md5_hash

    return None


def find_parent(search_dir, file_name):
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if file_name in filenames:
            return os.path.abspath(dirpath)
    return None


def find_folder_parent(search_dir, folder_name):
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if folder_name in dirnames:
            return os.path.abspath(dirpath)
    return None

file_path = find_folder_parent(now_dir, "assets")
tmp = os.path.join(file_path, "temp")
shutil.rmtree(tmp, ignore_errors=True)
os.environ["temp"] = tmp

def get_mediafire_download_link(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    download_button = soup.find('a', {'class': 'input popsok', 'aria-label': 'Download file'})
    if download_button:
        download_link = download_button.get('href')
        return download_link
    else:
        return None

def download_from_url(url):
    file_path = find_folder_parent(now_dir, "assets")
    print(file_path)
    zips_path = os.path.join(file_path, "assets", "zips")
    print(zips_path)
    os.makedirs(zips_path, exist_ok=True)
    if url != "":
        print(i18n("Downloading the file: ") + f"{url}")
        if "drive.google.com" in url:
            if "file/d/" in url:
                file_id = url.split("file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                return None

            if file_id:
                os.chdir(zips_path)
                result = subprocess.run(
                    ["gdown", f"https://drive.google.com/uc?id={file_id}", "--fuzzy"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                if (
                    "Too many users have viewed or downloaded this file recently"
                    in str(result.stderr)
                ):
                    return "too much use"
                if "Cannot retrieve the public link of the file." in str(result.stderr):
                    return "private link"
                print(result.stderr)

        elif "/blob/" in url or "/resolve/" in url:
            os.chdir(zips_path)
            if "/blob/" in url:
                url = url.replace("/blob/", "/resolve/")
            
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_name = url.split("/")[-1]
                file_name = file_name.replace("%20", "_")
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar_length = 50
                progress = 0
                with open(os.path.join(zips_path, file_name), 'wb') as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        progress += len(data)
                        progress_percent = int((progress / total_size_in_bytes) * 100)
                        num_dots = int((progress / total_size_in_bytes) * progress_bar_length)
                        progress_bar = "[" + "." * num_dots + " " * (progress_bar_length - num_dots) + "]"
                        print(f"{progress_percent}% {progress_bar} {progress}/{total_size_in_bytes}  ", end="\r")
                        if progress_percent == 100:
                            print("\n")
            else:
                os.chdir(file_path)
                return None
        elif "mega.nz" in url:
            if "#!" in url:
                file_id = url.split("#!")[1].split("!")[0]
            elif "file/" in url:
                file_id = url.split("file/")[1].split("/")[0]
            else:
                return None
            if file_id:
                print("Mega.nz is unsupported due mega.py deprecation")
        elif "/tree/main" in url:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            temp_url = ""
            for link in soup.find_all("a", href=True):
                if link["href"].endswith(".zip"):
                    temp_url = link["href"]
                    break
            if temp_url:
                url = temp_url
                url = url.replace("blob", "resolve")
                if "huggingface.co" not in url:
                    url = "https://huggingface.co" + url

                    wget.download(url)
            else:
                print("No .zip file found on the page.")
        elif "cdn.discordapp.com" in url:
            file = requests.get(url)
            os.chdir("./assets/zips")
            if file.status_code == 200:
                name = url.split("/")
                with open(
                    os.path.join(name[-1]), "wb"
                ) as newfile:
                    newfile.write(file.content)
            else:
                return None
        elif "pixeldrain.com" in url:
            try:
                file_id = url.split("pixeldrain.com/u/")[1]
                os.chdir(zips_path)
                print(file_id)
                response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
                if response.status_code == 200:
                    file_name = (
                        response.headers.get("Content-Disposition")
                        .split("filename=")[-1]
                        .strip('";')
                    )
                    os.makedirs(zips_path, exist_ok=True)
                    with open(os.path.join(zips_path, file_name), "wb") as newfile:
                        newfile.write(response.content)
                        os.chdir(file_path)
                        return "downloaded"
                else:
                    os.chdir(file_path)
                    return None
            except Exception as e:
                print(e)
                os.chdir(file_path)
                return None
        elif "mediafire.com" in url:
            download_link = get_mediafire_download_link(url)
            if download_link:
                os.chdir(zips_path)
                wget.download(download_link)
            else:
                return None
        elif "www.weights.gg" in url:
            #Pls weights creator dont fix this because yes. c:
            url_parts = url.split("/")
            weights_gg_index = url_parts.index("www.weights.gg")
            if weights_gg_index != -1 and weights_gg_index < len(url_parts) - 1:
                model_part = "/".join(url_parts[weights_gg_index + 1:])
                if "models" in model_part:
                    model_part = model_part.split("models/")[-1]
                    print(model_part)
                    if model_part:
                        download_url = f"https://www.weights.gg/es/models/{model_part}"
                        response = requests.get(download_url)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, "html.parser")
                            button_link = soup.find("a", class_="bg-black text-white px-3 py-2 rounded-lg flex items-center gap-1")
                            if button_link:
                                download_link = button_link["href"]
                                result = download_from_url(download_link)
                                if result == "downloaded":
                                    return "downloaded"
                                else:
                                    return None
                            else:
                                return None
                        else:
                            return None
                    else:
                        return None
                else:
                    return None
            else:
                return None
        else:
            os.chdir(zips_path)
            wget.download(url)

        # Fix points in the zips
        for currentPath, _, zipFiles in os.walk(zips_path):
            for Files in zipFiles:
                filePart = Files.split(".")
                extensionFile = filePart[len(filePart) - 1]
                filePart.pop()
                nameFile = "_".join(filePart)
                realPath = os.path.join(currentPath, Files)
                os.rename(realPath, nameFile + "." + extensionFile)

        os.chdir(file_path)
        print(i18n("Full download"))
        return "downloaded"
    else:
        return None


class error_message(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje
        super().__init__(mensaje)


def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return (
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )
import zipfile
from tqdm import tqdm

def extract_and_show_progress(zipfile_path, unzips_path):
    try:
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            total_files = len(zip_ref.infolist())
            with tqdm(total=total_files, unit='files', ncols= 100, colour= 'green') as pbar:
                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info, unzips_path)
                    pbar.update(1)
        return True
    except Exception as e:
        print(f"Error al descomprimir {zipfile_path}: {e}")
        return False
    

def load_downloaded_model(url):
    parent_path = find_folder_parent(now_dir, "assets")
    try:
        infos = []
        zips_path = os.path.join(parent_path, "assets", "zips")
        unzips_path = os.path.join(parent_path, "assets", "unzips")
        weights_path = os.path.join(parent_path, "logs", "weights")
        logs_dir = ""

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)

        os.mkdir(zips_path)
        os.mkdir(unzips_path)

        download_file = download_from_url(url)
        if not download_file:
            print(i18n("The file could not be downloaded."))
            infos.append(i18n("The file could not be downloaded."))
            yield "\n".join(infos)
        elif download_file == "downloaded":
            print(i18n("It has been downloaded successfully."))
            infos.append(i18n("It has been downloaded successfully."))
            yield "\n".join(infos)
        elif download_file == "too much use":
            raise Exception(
                i18n("Too many users have recently viewed or downloaded this file")
            )
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))

        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path, filename)
                print(i18n("Proceeding with the extraction..."))
                infos.append(i18n("Proceeding with the extraction..."))
                #shutil.unpack_archive(zipfile_path, unzips_path, "zip")
                model_name = os.path.basename(zipfile_path)
                logs_dir = os.path.join(
                    parent_path,
                    "logs",
                    os.path.normpath(str(model_name).replace(".zip", "")),
                )
                
                yield "\n".join(infos)
                success = extract_and_show_progress(zipfile_path, unzips_path)
                if success:
                    yield f"Extracción exitosa: {model_name}"
                else:
                    yield f"Fallo en la extracción: {model_name}"
                yield "\n".join(infos)
            else:
                print(i18n("Unzip error."))
                infos.append(i18n("Unzip error."))
                yield "\n".join(infos)
                return ""

        index_file = False
        model_file = False

        for path, subdirs, files in os.walk(unzips_path):
            for item in files:
                item_path = os.path.join(path, item)
                if not "G_" in item and not "D_" in item and item.endswith(".pth"):
                    model_file = True
                    model_name = item.replace(".pth", "")
                    logs_dir = os.path.join(parent_path, "logs", model_name)
                    if os.path.exists(logs_dir):
                        shutil.rmtree(logs_dir)
                    os.mkdir(logs_dir)
                    if not os.path.exists(weights_path):
                        os.mkdir(weights_path)
                    if os.path.exists(os.path.join(weights_path, item)):
                        os.remove(os.path.join(weights_path, item))
                    if os.path.exists(item_path):
                        shutil.move(item_path, weights_path)

        if not model_file and not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        for path, subdirs, files in os.walk(unzips_path):
            for item in files:
                item_path = os.path.join(path, item)
                if item.startswith("added_") and item.endswith(".index"):
                    index_file = True
                    if os.path.exists(item_path):
                        if os.path.exists(os.path.join(logs_dir, item)):
                            os.remove(os.path.join(logs_dir, item))
                        shutil.move(item_path, logs_dir)
                if item.startswith("total_fea.npy") or item.startswith("events."):
                    if os.path.exists(item_path):
                        if os.path.exists(os.path.join(logs_dir, item)):
                            os.remove(os.path.join(logs_dir, item))
                        shutil.move(item_path, logs_dir)

        result = ""
        if model_file:
            if index_file:
                print(i18n("The model works for inference, and has the .index file."))
                infos.append(
                    "\n"
                    + i18n("The model works for inference, and has the .index file.")
                )
                yield "\n".join(infos)
            else:
                print(
                    i18n(
                        "The model works for inference, but it doesn't have the .index file."
                    )
                )
                infos.append(
                    "\n"
                    + i18n(
                        "The model works for inference, but it doesn't have the .index file."
                    )
                )
                yield "\n".join(infos)

        if not index_file and not model_file:
            print(i18n("No relevant file was found to upload."))
            infos.append(i18n("No relevant file was found to upload."))
            yield "\n".join(infos)
        

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)
        os.chdir(parent_path)
        return result
    except Exception as e:
        os.chdir(parent_path)
        if "too much use" in str(e):
            print(i18n("Too many users have recently viewed or downloaded this file"))
            yield i18n("Too many users have recently viewed or downloaded this file")
        elif "private link" in str(e):
            print(i18n("Cannot get file from this private link"))
            yield i18n("Cannot get file from this private link")
        else:
            print(e)
            yield i18n("An error occurred downloading")
    finally:
        os.chdir(parent_path)


def load_dowloaded_dataset(url):
    parent_path = find_folder_parent(now_dir, "assets")
    infos = []
    try:
        zips_path = os.path.join(parent_path, "assets", "zips")
        unzips_path = os.path.join(parent_path, "assets", "unzips")
        datasets_path = os.path.join(parent_path, "datasets")
        audio_extenions = [
            "wav",
            "mp3",
            "flac",
            "ogg",
            "opus",
            "m4a",
            "mp4",
            "aac",
            "alac",
            "wma",
            "aiff",
            "webm",
            "ac3",
        ]

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)

        if not os.path.exists(datasets_path):
            os.mkdir(datasets_path)

        os.mkdir(zips_path)
        os.mkdir(unzips_path)

        download_file = download_from_url(url)

        if not download_file:
            print(i18n("An error occurred downloading"))
            infos.append(i18n("An error occurred downloading"))
            yield "\n".join(infos)
            raise Exception(i18n("An error occurred downloading"))
        elif download_file == "downloaded":
            print(i18n("It has been downloaded successfully."))
            infos.append(i18n("It has been downloaded successfully."))
            yield "\n".join(infos)
        elif download_file == "too much use":
            raise Exception(
                i18n("Too many users have recently viewed or downloaded this file")
            )
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))

        zip_path = os.listdir(zips_path)
        foldername = ""
        for file in zip_path:
            if file.endswith(".zip"):
                file_path = os.path.join(zips_path, file)
                print("....")
                foldername = file.replace(".zip", "").replace(" ", "").replace("-", "_")
                dataset_path = os.path.join(datasets_path, foldername)
                print(i18n("Proceeding with the extraction..."))
                infos.append(i18n("Proceeding with the extraction..."))
                yield "\n".join(infos)
                shutil.unpack_archive(file_path, unzips_path, "zip")
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)

                os.mkdir(dataset_path)

                for root, subfolders, songs in os.walk(unzips_path):
                    for song in songs:
                        song_path = os.path.join(root, song)
                        if song.endswith(tuple(audio_extenions)):
                            formatted_song_name = format_title(
                                os.path.splitext(song)[0]
                            )
                            extension = os.path.splitext(song)[1]
                            new_song_path = os.path.join(
                                dataset_path, f"{formatted_song_name}{extension}"
                            )
                            shutil.move(song_path, new_song_path)
            else:
                print(i18n("Unzip error."))
                infos.append(i18n("Unzip error."))
                yield "\n".join(infos)

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)

        print(i18n("The Dataset has been loaded successfully."))
        infos.append(i18n("The Dataset has been loaded successfully."))
        yield "\n".join(infos)
    except Exception as e:
        os.chdir(parent_path)
        if "too much use" in str(e):
            print(i18n("Too many users have recently viewed or downloaded this file"))
            yield i18n("Too many users have recently viewed or downloaded this file")
        elif "private link" in str(e):
            print(i18n("Cannot get file from this private link"))
            yield i18n("Cannot get file from this private link")
        else:
            print(e)
            yield i18n("An error occurred downloading")
    finally:
        os.chdir(parent_path)


SAVE_ACTION_CONFIG = {
    i18n("Save all"): {
        'destination_folder': "manual_backup",
        'copy_files': True,  # "Save all" Copy all files and folders
        'include_weights': False
    },
    i18n("Save D and G"): {
        'destination_folder': "manual_backup",
        'copy_files': False,  # "Save D and G" Do not copy everything, only specific files
        'files_to_copy': ["D_*.pth", "G_*.pth", "added_*.index"],
        'include_weights': True,
    },
    i18n("Save voice"): {
        'destination_folder': "finished",
        'copy_files': False,  # "Save voice" Do not copy everything, only specific files
        'files_to_copy': ["added_*.index"],
        'include_weights': True,
    },
}

import os
import shutil
import zipfile
import glob
import fnmatch

import os
import shutil
import zipfile
import glob

import os
import shutil
import zipfile


def save_model(modelname, save_action):
    try:
        # Define paths
        parent_path = find_folder_parent(now_dir, "assets")
        weight_path = os.path.join(parent_path, "logs", "weights", f"{modelname}.pth")
        logs_path = os.path.join(parent_path, "logs", modelname)
        save_folder = SAVE_ACTION_CONFIG[save_action]['destination_folder']
        save_path = os.path.join(parent_path, "logs", save_folder, modelname + ".zip")
        infos = []

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Comprobar si el directorio de logs existe
        if not os.path.exists(logs_path):
            raise Exception(f"The logs directory '{logs_path}' does not exist.")

        # Realizar las acciones según la opción seleccionada
        if SAVE_ACTION_CONFIG[save_action]['copy_files']:
            # Option: Copy all files and folders
            infos.append(save_action)
            print(save_action)
            yield "\n".join(infos)
            with zipfile.ZipFile(save_path, 'w') as zipf:
                for root, _, files in os.walk(logs_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Create a folder with the model name in the ZIP
                        model_folder = os.path.join(modelname, "")
                        zipf.write(file_path, os.path.join(model_folder, os.path.relpath(file_path, logs_path)))
                zipf.write(weight_path, os.path.join(model_folder, os.path.basename(weight_path)))
        
        else:
            # Option: Copy specific files
            infos.append(save_action)
            print(save_action)
            yield "\n".join(infos)
            files_to_copy = SAVE_ACTION_CONFIG[save_action]['files_to_copy']
            with zipfile.ZipFile(save_path, 'w') as zipf:
                for root, _, files in os.walk(logs_path):
                    for file in files:
                        for pattern in files_to_copy:
                            if fnmatch.fnmatch(file, pattern):
                                file_path = os.path.join(root, file)
                                zipf.write(file_path, os.path.relpath(file_path, logs_path))

        if SAVE_ACTION_CONFIG[save_action]['include_weights']:
            # Include the weight file in the ZIP
            with zipfile.ZipFile(save_path, 'a') as zipf:
                zipf.write(weight_path, os.path.basename(weight_path))

        infos.append(i18n("The model has been saved successfully."))
        yield "\n".join(infos)

    except Exception as e:
        # Handle exceptions and print error messages
        error_message = str(e)
        print(f"Error: {error_message}")
        yield error_message

def load_downloaded_backup(url):
    parent_path = find_folder_parent(now_dir, "assets")
    try:
        infos = []
        logs_folders = [
            "0_gt_wavs",
            "1_16k_wavs",
            "2a_f0",
            "2b-f0nsf",
            "3_feature256",
            "3_feature768",
        ]
        zips_path = os.path.join(parent_path, "assets", "zips")
        unzips_path = os.path.join(parent_path, "assets", "unzips")
        weights_path = os.path.join(parent_path, "assets", "logs", "weights")
        logs_dir = os.path.join(parent_path, "logs")

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)

        os.mkdir(zips_path)
        os.mkdir(unzips_path)

        download_file = download_from_url(url)
        if not download_file:
            print(i18n("The file could not be downloaded."))
            infos.append(i18n("The file could not be downloaded."))
            yield "\n".join(infos)
        elif download_file == "downloaded":
            print(i18n("It has been downloaded successfully."))
            infos.append(i18n("It has been downloaded successfully."))
            yield "\n".join(infos)
        elif download_file == "too much use":
            raise Exception(
                i18n("Too many users have recently viewed or downloaded this file")
            )
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))

        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path, filename)
                zip_dir_name = os.path.splitext(filename)[0]
                unzip_dir = unzips_path
                print(i18n("Proceeding with the extraction..."))
                infos.append(i18n("Proceeding with the extraction..."))
                shutil.unpack_archive(zipfile_path, unzip_dir, "zip")

                if os.path.exists(os.path.join(unzip_dir, zip_dir_name)):
                    shutil.move(os.path.join(unzip_dir, zip_dir_name), logs_dir)
                else:
                    new_folder_path = os.path.join(logs_dir, zip_dir_name)
                    os.mkdir(new_folder_path)
                    for item_name in os.listdir(unzip_dir):
                        item_path = os.path.join(unzip_dir, item_name)
                        if os.path.isfile(item_path):
                            shutil.move(item_path, new_folder_path)
                        elif os.path.isdir(item_path):
                            shutil.move(item_path, new_folder_path)

                yield "\n".join(infos)
            else:
                print(i18n("Unzip error."))
                infos.append(i18n("Unzip error."))
                yield "\n".join(infos)

        result = ""

        for filename in os.listdir(unzips_path):
            if filename.endswith(".zip"):
                silentremove(filename)

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(os.path.join(parent_path, "assets", "unzips")):
            shutil.rmtree(os.path.join(parent_path, "assets", "unzips"))
        print(i18n("The Backup has been uploaded successfully."))
        infos.append("\n" + i18n("The Backup has been uploaded successfully."))
        yield "\n".join(infos)
        os.chdir(parent_path)
        return result
    except Exception as e:
        os.chdir(parent_path)
        if "too much use" in str(e):
            print(i18n("Too many users have recently viewed or downloaded this file"))
            yield i18n("Too many users have recently viewed or downloaded this file")
        elif "private link" in str(e):
            print(i18n("Cannot get file from this private link"))
            yield i18n("Cannot get file from this private link")
        else:
            print(e)
            yield i18n("An error occurred downloading")
    finally:
        os.chdir(parent_path)


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        new_path = ".assets/audios/" + new_name
        shutil.move(path_to_file, new_path)
        return new_name


def change_choices2():
    audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_root
    ]
    return {"choices": sorted(audio_paths), "__type__": "update"}, {
        "__type__": "update"
    }


def uvr(
    input_url,
    output_path,
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    agg,
    format0,
    architecture,
):
    carpeta_a_eliminar = "yt_downloads"
    if os.path.exists(carpeta_a_eliminar) and os.path.isdir(carpeta_a_eliminar):
        for archivo in os.listdir(carpeta_a_eliminar):
            ruta_archivo = os.path.join(carpeta_a_eliminar, archivo)
            if os.path.isfile(ruta_archivo):
                os.remove(ruta_archivo)
            elif os.path.isdir(ruta_archivo):
                shutil.rmtree(ruta_archivo)

    ydl_opts = {
        "no-windows-filenames": True,
        "restrict-filenames": True,
        "extract_audio": True,
        "format": "bestaudio",
        "quiet": True,
        "no-warnings": True,
    }

    try:
        print(i18n("Downloading audio from the video..."))
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(input_url, download=False)
            formatted_title = format_title(info_dict.get("title", "default_title"))
            formatted_outtmpl = output_path + "/" + formatted_title + ".wav"
            ydl_opts["outtmpl"] = formatted_outtmpl
            ydl = yt_dlp.YoutubeDL(ydl_opts)
            ydl.download([input_url])
        print(i18n("Audio downloaded!"))
    except Exception as error:
        print(i18n("An error occurred:"), error)

    actual_directory = os.path.dirname(__file__)
    actual_directory = os.path.abspath(os.path.join(actual_directory, ".."))

    vocal_directory = os.path.join(actual_directory, save_root_vocal)
    instrumental_directory = os.path.join(actual_directory, save_root_ins)

    vocal_formatted = f"vocal_{formatted_title}.wav.reformatted.wav_10.wav"
    instrumental_formatted = f"instrument_{formatted_title}.wav.reformatted.wav_10.wav"

    vocal_audio_path = os.path.join(vocal_directory, vocal_formatted)
    instrumental_audio_path = os.path.join(
        instrumental_directory, instrumental_formatted
    )

    vocal_formatted_mdx = f"{formatted_title}_vocal_.wav"
    instrumental_formatted_mdx = f"{formatted_title}_instrument_.wav"

    vocal_audio_path_mdx = os.path.join(vocal_directory, vocal_formatted_mdx)
    instrumental_audio_path_mdx = os.path.join(
        instrumental_directory, instrumental_formatted_mdx
    )

    if architecture == "VR":
        try:
            print(i18n("Starting audio conversion... (This might take a moment)"))
            inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            save_root_vocal = (
                save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )
            save_root_ins = (
                save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )
            usable_files = [
                os.path.join(inp_root, file)
                for file in os.listdir(inp_root)
                if file.endswith(tuple(sup_audioext))
            ]
            if model_name == "onnx_dereverb_By_FoxJoy":
                pre_fun = MDXNetDereverb(15, config.device)
            else:
                func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
                pre_fun = func(
                    agg=int(agg),
                    model_path=os.path.join(
                        os.getenv("weight_uvr5_root"), model_name + ".pth"
                    ),
                    device=config.device,
                    is_half=config.is_half,
                )
            if inp_root != "":
                paths = usable_files
            else:
                paths = [path.name for path in paths]
            for path in paths:
                inp_path = os.path.join(inp_root, path)
                need_reformat = 1
                done = 0
                try:
                    info = ffmpeg.probe(inp_path, cmd="ffprobe")
                    if (
                        info["streams"][0]["channels"] == 2
                        and info["streams"][0]["sample_rate"] == "44100"
                    ):
                        need_reformat = 0
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                        done = 1
                except:
                    need_reformat = 1
                    traceback.print_exc()
                if need_reformat == 1:
                    tmp_path = "%s/%s.reformatted.wav" % (
                        os.path.join(os.environ["temp"]),
                        os.path.basename(inp_path),
                    )
                    os.system(
                        "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                        % (inp_path, tmp_path)
                    )
                    inp_path = tmp_path
                try:
                    if done == 0:
                        pre_fun.path_audio(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                    print("%s->Success" % (os.path.basename(inp_path)))
                except:
                    try:
                        if done == 0:
                            pre_fun._path_audio_(
                                inp_path, save_root_ins, save_root_vocal, format0
                            )
                        print("%s->Success" % (os.path.basename(inp_path)))
                    except:
                        print(
                            "%s->%s"
                            % (os.path.basename(inp_path), traceback.format_exc())
                        )
        except:
            print(traceback.format_exc())
        finally:
            try:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del pre_fun.pred.model
                    del pre_fun.pred.model_
                else:
                    del pre_fun.model
                    del pre_fun
                return i18n("Finished"), vocal_audio_path, instrumental_audio_path
            except:
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Executed torch.cuda.empty_cache()")
    elif architecture == "MDX":
        try:
            print(i18n("Starting audio conversion... (This might take a moment)"))
            inp_root, save_root_vocal, save_root_ins = [
                x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
                for x in [inp_root, save_root_vocal, save_root_ins]
            ]

            usable_files = [
                os.path.join(inp_root, file)
                for file in os.listdir(inp_root)
                if file.endswith(tuple(sup_audioext))
            ]
            try:
                if paths != None:
                    paths = [path.name for path in paths]
                else:
                    paths = usable_files

            except:
                traceback.print_exc()
                paths = usable_files
            print(paths)
            invert = True
            denoise = True
            use_custom_parameter = True
            dim_f = 2048
            dim_t = 256
            n_fft = 7680
            use_custom_compensation = True
            compensation = 1.025
            suffix = "vocal_"  # @param ["Vocals", "Drums", "Bass", "Other"]{allow-input: true}
            suffix_invert = "instrument_"  # @param ["Instrumental", "Drumless", "Bassless", "Instruments"]{allow-input: true}
            print_settings = True  # @param{type:"boolean"}
            onnx = id_to_ptm(model_name)
            compensation = (
                compensation
                if use_custom_compensation or use_custom_parameter
                else None
            )
            mdx_model = prepare_mdx(
                onnx,
                use_custom_parameter,
                dim_f,
                dim_t,
                n_fft,
                compensation=compensation,
            )

            for path in paths:
                # inp_path = os.path.join(inp_root, path)
                suffix_naming = suffix if use_custom_parameter else None
                diff_suffix_naming = suffix_invert if use_custom_parameter else None
                run_mdx(
                    onnx,
                    mdx_model,
                    path,
                    format0,
                    diff=invert,
                    suffix=suffix_naming,
                    diff_suffix=diff_suffix_naming,
                    denoise=denoise,
                )

            if print_settings:
                print()
                print("[MDX-Net_Colab settings used]")
                print(f"Model used: {onnx}")
                print(f"Model MD5: {mdx.MDX.get_hash(onnx)}")
                print(f"Model parameters:")
                print(f"    -dim_f: {mdx_model.dim_f}")
                print(f"    -dim_t: {mdx_model.dim_t}")
                print(f"    -n_fft: {mdx_model.n_fft}")
                print(f"    -compensation: {mdx_model.compensation}")
                print()
                print("[Input file]")
                print("filename(s): ")
                for filename in paths:
                    print(f"    -{filename}")
                    print(f"{os.path.basename(filename)}->Success")
        except:
            traceback.print_exc()
        finally:
            try:
                del mdx_model
                return (
                    i18n("Finished"),
                    vocal_audio_path_mdx,
                    instrumental_audio_path_mdx,
                )
            except:
                traceback.print_exc()

            print("clean_empty_cache")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_downloaded_audio(url):
    parent_path = find_folder_parent(now_dir, "assets")
    try:
        infos = []
        audios_path = os.path.join(parent_path, "assets", "audios")
        zips_path = os.path.join(parent_path, "assets", "zips")

        if not os.path.exists(audios_path):
            os.mkdir(audios_path)

        download_file = download_from_url(url)
        if not download_file:
            print(i18n("The file could not be downloaded."))
            infos.append(i18n("The file could not be downloaded."))
            yield "\n".join(infos)
        elif download_file == "downloaded":
            print(i18n("It has been downloaded successfully."))
            infos.append(i18n("It has been downloaded successfully."))
            yield "\n".join(infos)
        elif download_file == "too much use":
            raise Exception(
                i18n("Too many users have recently viewed or downloaded this file")
            )
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))

        for filename in os.listdir(zips_path):
            item_path = os.path.join(zips_path, filename)
            if item_path.split(".")[-1] in sup_audioext:
                if os.path.exists(item_path):
                    shutil.move(item_path, audios_path)

        result = ""
        print(i18n("Audio files have been moved to the 'audios' folder."))
        infos.append(i18n("Audio files have been moved to the 'audios' folder."))
        yield "\n".join(infos)

        os.chdir(parent_path)
        return result
    except Exception as e:
        os.chdir(parent_path)
        if "too much use" in str(e):
            print(i18n("Too many users have recently viewed or downloaded this file"))
            yield i18n("Too many users have recently viewed or downloaded this file")
        elif "private link" in str(e):
            print(i18n("Cannot get file from this private link"))
            yield i18n("Cannot get file from this private link")
        else:
            print(e)
            yield i18n("An error occurred downloading")
    finally:
        os.chdir(parent_path)


class error_message(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje
        super().__init__(mensaje)


def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return (
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )


def update_model_choices(select_value):
    model_ids = get_model_list()
    model_ids_list = list(model_ids)
    if select_value == "VR":
        return {"choices": uvr5_names, "__type__": "update"}
    elif select_value == "MDX":
        return {"choices": model_ids_list, "__type__": "update"}


def save_drop_model_pth(dropbox):
    file_path = dropbox.name 
    file_name = os.path.basename(file_path)
    target_path = os.path.join("logs", "weights", os.path.basename(file_path))
    
    if not file_name.endswith('.pth'):
        print(i18n("The file does not have the .pth extension. Please upload the correct file."))
        return None
    
    shutil.move(file_path, target_path)
    return target_path

def extract_folder_name(file_name):
    match = re.search(r'nprobe_(.*?)\.index', file_name)
    
    if match:
        return match.group(1)
    else:
        return

def save_drop_model_index(dropbox):
    file_path = dropbox.name
    file_name = os.path.basename(file_path)
    folder_name = extract_folder_name(file_name)

    if not file_name.endswith('.index'):
        print(i18n("The file does not have the .index extension. Please upload the correct file."))
        return None

    out_path = os.path.join("logs", folder_name)
    os.mkdir(out_path)

    target_path = os.path.join(out_path, os.path.basename(file_path))

    shutil.move(file_path, target_path)
    return target_path


def download_model():
    gr.Markdown(value="# " + i18n("Download Model"))
    gr.Markdown(value=i18n("It is used to download your inference models."))
    with gr.Row():
        model_url = gr.Textbox(label=i18n("Url:"))
    with gr.Row():
        download_model_status_bar = gr.Textbox(label=i18n("Status:"))
    with gr.Row():
        download_button = gr.Button(i18n("Download"))
        download_button.click(
            fn=load_downloaded_model,
            inputs=[model_url],
            outputs=[download_model_status_bar],
        )
    gr.Markdown(value=i18n("You can also drop your files to load your model."))    
    with gr.Row():
        dropbox_pth = gr.File(label=i18n("Drag your .pth file here:"))
        dropbox_index = gr.File(label=i18n("Drag your .index file here:"))

    dropbox_pth.upload(
        fn=save_drop_model_pth,
        inputs=[dropbox_pth],
    )
    dropbox_index.upload(
        fn=save_drop_model_index,
        inputs=[dropbox_index],
    )


def download_backup():
    gr.Markdown(value="# " + i18n("Download Backup"))
    gr.Markdown(value=i18n("It is used to download your training backups."))
    with gr.Row():
        model_url = gr.Textbox(label=i18n("Url:"))
    with gr.Row():
        download_model_status_bar = gr.Textbox(label=i18n("Status:"))
    with gr.Row():
        download_button = gr.Button(i18n("Download"))
        download_button.click(
            fn=load_downloaded_backup,
            inputs=[model_url],
            outputs=[download_model_status_bar],
        )


def update_dataset_list(name):
    new_datasets = []
    file_path = find_folder_parent(now_dir, "assets")
    for foldername in os.listdir("./datasets"):
        if "." not in foldername:
            new_datasets.append(
                os.path.join(
                    file_path, "datasets", foldername
                )
            )
    return gr.Dropdown.update(choices=new_datasets)


def download_dataset(trainset_dir4):
    gr.Markdown(value="# " + i18n("Download Dataset"))
    gr.Markdown(
        value=i18n(
            "Download the dataset with the audios in a compatible format (.wav/.flac) to train your model."
        )
    )
    with gr.Row():
        dataset_url = gr.Textbox(label=i18n("Url:"))
    with gr.Row():
        load_dataset_status_bar = gr.Textbox(label=i18n("Status:"))
    with gr.Row():
        load_dataset_button = gr.Button(i18n("Download"))
        load_dataset_button.click(
            fn=load_dowloaded_dataset,
            inputs=[dataset_url],
            outputs=[load_dataset_status_bar],
        )
        load_dataset_status_bar.change(update_dataset_list, dataset_url, trainset_dir4)


def download_audio():
    gr.Markdown(value="# " + i18n("Download Audio"))
    gr.Markdown(
        value=i18n(
            "Download audios of any format for use in inference (recommended for mobile users)."
        )
    )
    with gr.Row():
        audio_url = gr.Textbox(label=i18n("Url:"))
    with gr.Row():
        download_audio_status_bar = gr.Textbox(label=i18n("Status:"))
    with gr.Row():
        download_button2 = gr.Button(i18n("Download"))
        download_button2.click(
            fn=load_downloaded_audio,
            inputs=[audio_url],
            outputs=[download_audio_status_bar],
        )


def youtube_separator():
    gr.Markdown(value="# " + i18n("Separate YouTube tracks"))
    gr.Markdown(
        value=i18n(
            "Download audio from a YouTube video and automatically separate the vocal and instrumental tracks"
        )
    )
    with gr.Row():
        input_url = gr.inputs.Textbox(label=i18n("Enter the YouTube link:"))
        output_path = gr.Textbox(
            label=i18n(
                "Enter the path of the audio folder to be processed (copy it from the address bar of the file manager):"
            ),
            value=os.path.abspath(os.getcwd()).replace("\\", "/") + "/yt_downloads",
            visible=False,
        )
        advanced_settings_checkbox = gr.Checkbox(
            value=False,
            label=i18n("Advanced Settings"),
            interactive=True,
        )
    with gr.Row(
        label=i18n("Advanced Settings"), visible=False, variant="compact"
    ) as advanced_settings:
        with gr.Column():
            model_select = gr.Radio(
                label=i18n("Model Architecture:"),
                choices=["VR", "MDX"],
                value="VR",
                interactive=True,
            )
            model_choose = gr.Dropdown(
                label=i18n(
                    "Model: (Be aware that in some models the named vocal will be the instrumental)"
                ),
                choices=uvr5_names,
                value="HP5_only_main_vocal",
            )
            with gr.Row():
                agg = gr.Slider(
                    minimum=0,
                    maximum=20,
                    step=1,
                    label=i18n("Vocal Extraction Aggressive"),
                    value=10,
                    interactive=True,
                )
            with gr.Row():
                opt_vocal_root = gr.Textbox(
                    label=i18n("Specify the output folder for vocals:"),
                    value=((os.getcwd()).replace("\\", "/") + "/assets/audios"),
                )
            opt_ins_root = gr.Textbox(
                label=i18n("Specify the output folder for accompaniment:"),
                value=((os.getcwd()).replace("\\", "/") + "/assets/audios/audio-others"),
            )
            dir_wav_input = gr.Textbox(
                label=i18n("Enter the path of the audio folder to be processed:"),
                value=((os.getcwd()).replace("\\", "/") + "/yt_downloads"),
                visible=False,
            )
            format0 = gr.Radio(
                label=i18n("Export file format"),
                choices=["wav", "flac", "mp3", "m4a"],
                value="wav",
                visible=False,
                interactive=True,
            )
            wav_inputs = gr.File(
                file_count="multiple",
                label=i18n(
                    "You can also input audio files in batches. Choose one of the two options. Priority is given to reading from the folder."
                ),
                visible=False,
            )
        model_select.change(
            fn=update_model_choices,
            inputs=model_select,
            outputs=model_choose,
        )
    with gr.Row():
        vc_output4 = gr.Textbox(label=i18n("Status:"))
        vc_output5 = gr.Audio(label=i18n("Vocal"), type="filepath")
        vc_output6 = gr.Audio(label=i18n("Instrumental"), type="filepath")
    with gr.Row():
        but2 = gr.Button(i18n("Download and Separate"))
        but2.click(
            uvr,
            [
                input_url,
                output_path,
                model_choose,
                dir_wav_input,
                opt_vocal_root,
                wav_inputs,
                opt_ins_root,
                agg,
                format0,
                model_select,
            ],
            [vc_output4, vc_output5, vc_output6],
        )

    def toggle_advanced_settings(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    advanced_settings_checkbox.change(
        fn=toggle_advanced_settings,
        inputs=[advanced_settings_checkbox],
        outputs=[advanced_settings],
    )


