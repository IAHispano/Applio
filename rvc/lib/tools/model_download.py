import os
import re
import sys
import shutil
import zipfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse
from pathlib import Path
from email.message import Message
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.user_config import get_logs_dir
from rvc.lib.utils import format_title
from rvc.lib.tools import gdown
from rvc.lib.tools.http_download import download_http

file_path = get_logs_dir()
zips_path = os.path.join(file_path, "zips")
os.makedirs(zips_path, exist_ok=True)


def search_pth_index(folder):
    pth_paths = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file)) and file.endswith(".pth")
    ]
    index_paths = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file)) and file.endswith(".index")
    ]
    return pth_paths, index_paths


def download_from_url(url):
    os.chdir(zips_path)

    try:
        if "drive.google.com" in url:
            file_id = extract_google_drive_id(url)
            if file_id:
                gdown.download(
                    url=f"https://drive.google.com/uc?id={file_id}",
                    quiet=False,
                    fuzzy=True,
                    resume=True,
                )
        elif "/blob/" in url or "/resolve/" in url:
            download_blob_or_resolve(url)
        elif "/tree/main" in url:
            download_from_huggingface(url)
        else:
            download_file(url)

        rename_downloaded_files()
        return "downloaded"
    except Exception as error:
        print(f"An error occurred downloading the file: {error}")
        return None
    finally:
        os.chdir(now_dir)


def extract_google_drive_id(url):
    if "file/d/" in url:
        return url.split("file/d/")[1].split("/")[0]
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    return None


def download_blob_or_resolve(url):
    if "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
    download_file(url)


def save_response_content(response, url=None):
    message = Message()
    message["Content-Disposition"] = response.headers.get("Content-Disposition", "")
    url = url or (
        response.history[0].request.url if response.history else response.request.url
    )
    file_name = message.get_filename() or unquote(os.path.basename(urlparse(url).path))
    file_name = os.path.basename(file_name.replace("\\", "/")) or "downloaded_file.zip"
    response.close()
    with tqdm(unit="B", unit_scale=True, desc=file_name) as progress_bar:

        def progress(received, total):
            progress_bar.total = total or None
            progress_bar.n = received
            progress_bar.refresh()

        download_http(url, os.path.join(zips_path, file_name), progress=progress)


def download_from_huggingface(url):
    response = requests.get(url, timeout=(30, 60))
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    temp_url = next(
        (
            link["href"]
            for link in soup.find_all("a", href=True)
            if link["href"].endswith(".zip")
        ),
        None,
    )
    if temp_url:
        url = temp_url.replace("blob", "resolve")
        if "huggingface.co" not in url:
            url = "https://huggingface.co" + url
        download_file(url)
    else:
        raise ValueError("No zip file found in Huggingface URL")


def download_file(url):
    response = requests.get(
        url, stream=True, timeout=(30, 60), headers={"Accept-Encoding": "identity"}
    )
    if response.status_code == 200:
        save_response_content(response, url)
    else:
        raise ValueError(
            "Download failed with status code: " + str(response.status_code)
        )


def rename_downloaded_files():
    for currentPath, _, zipFiles in os.walk(zips_path):
        for file in zipFiles:
            file_name, extension = os.path.splitext(file)
            real_path = os.path.join(currentPath, file)
            os.rename(real_path, file_name.replace(os.path.sep, "_") + extension)


def extract(zipfile_path, unzips_path):
    try:
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            root = Path(unzips_path).resolve()
            for member in zip_ref.infolist():
                target = (root / member.filename.replace("\\", "/")).resolve()
                if os.path.commonpath([str(root), str(target)]) != str(root):
                    raise ValueError("Model archive contains an unsafe file path")
            damaged = zip_ref.testzip()
            if damaged:
                raise ValueError(f"Model archive checksum failed: {damaged}")
            zip_ref.extractall(unzips_path)
        os.remove(zipfile_path)
        return True
    except Exception as error:
        print(f"An error occurred extracting the zip file: {error}")
        return False


def unzip_file(zip_path, zip_file_name):
    zip_file_path = os.path.join(zip_path, zip_file_name + ".zip")
    extract_path = os.path.join(file_path, zip_file_name)
    if not extract(zip_file_path, extract_path):
        raise ValueError("Model archive could not be extracted")


def model_download_pipeline(url: str):
    try:
        result = download_from_url(url)
        if result == "downloaded":
            return handle_extraction_process()
        else:
            return "Error"
    except Exception as error:
        print(f"An unexpected error occurred: {error}")
        return "Error"


def handle_extraction_process():
    extract_folder_path = ""
    for filename in os.listdir(zips_path):
        if filename.endswith(".zip"):
            zipfile_path = os.path.join(zips_path, filename)
            model_name = format_title(os.path.basename(zipfile_path).split(".zip")[0])
            extract_folder_path = os.path.join(
                get_logs_dir(), os.path.normpath(model_name)
            )
            success = extract(zipfile_path, extract_folder_path)
            clean_extracted_files(extract_folder_path, model_name)

            if success:
                print(f"Model {model_name} downloaded!")
            else:
                print(f"Error downloading {model_name}")
                return "Error"
    if not extract_folder_path:
        print("Zip file was not found.")
        return "Error"
    return search_pth_index(extract_folder_path)


def clean_extracted_files(extract_folder_path, model_name):
    macosx_path = os.path.join(extract_folder_path, "__MACOSX")
    if os.path.exists(macosx_path):
        shutil.rmtree(macosx_path)

    subfolders = [
        f
        for f in os.listdir(extract_folder_path)
        if os.path.isdir(os.path.join(extract_folder_path, f))
    ]
    if len(subfolders) == 1:
        subfolder_path = os.path.join(extract_folder_path, subfolders[0])
        for item in os.listdir(subfolder_path):
            shutil.move(
                os.path.join(subfolder_path, item),
                os.path.join(extract_folder_path, item),
            )
        os.rmdir(subfolder_path)

    for item in os.listdir(extract_folder_path):
        source_path = os.path.join(extract_folder_path, item)
        if ".pth" in item:
            new_file_name = model_name + ".pth"
        elif ".index" in item:
            new_file_name = model_name + ".index"
        else:
            continue

        destination_path = os.path.join(extract_folder_path, new_file_name)
        if not os.path.exists(destination_path):
            os.rename(source_path, destination_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Applio Model Downloader")
    parser.add_argument(
        "model_link", nargs="?", default=None, help="URL to download the model from"
    )
    parser.add_argument(
        "--model-link",
        dest="opt_link",
        default=None,
        help="URL to download the model from",
    )
    args = parser.parse_args()

    link = args.model_link or args.opt_link
    if not link:
        print("Error: No model link provided.")
        sys.exit(1)

    result = model_download_pipeline(link)
    if result == "Error" or result is None:
        print(
            "An error occurred downloading the model. Please check the console logs for more details."
        )
        sys.exit(1)
    print("Model downloaded successfully.")
