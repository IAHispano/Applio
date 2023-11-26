from flask import Flask, request, Response
import os
import sys
import requests
import shutil
import subprocess
import wget
import signal
from bs4 import BeautifulSoup
import logging
import click


app = Flask(__name__)

# Disable flask starting message
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def secho(text, file=None, nl=None, err=None, color=None, **styles):
    pass

def echo(text, file=None, nl=None, err=None, color=None, **styles):
    pass

click.echo = echo
click.secho = secho

# Get the current directory path
now_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels in the directory hierarchy
for _ in range(2):
    now_dir = os.path.dirname(now_dir)

# Add now_dir to sys.path so Python can find modules in that location
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
i18n = I18nAuto()

# Use the code from the resources module but with some changes
def find_folder_parent(search_dir, folder_name):
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if folder_name in dirnames:
            return os.path.abspath(dirpath)
    return None

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

def extract_and_show_progress(zipfile_path, unzips_path):
    try:
        # Use shutil because zipfile not working
        shutil.unpack_archive(zipfile_path, unzips_path)
        return True
    except Exception as e:
        print(f"Error al descomprimir {zipfile_path}: {e}")
        return False


@app.route('/download/<path:url>', methods=['GET'])

def load_downloaded_model(url):
    parent_path = find_folder_parent(now_dir, "assets")
    response = requests.get(url)
    response.raise_for_status()
    try:
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
        elif download_file == "downloaded":
            print(i18n("It has been downloaded successfully."))
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
                model_name = os.path.basename(zipfile_path)
                logs_dir = os.path.join(
                    parent_path,
                    "logs",
                    os.path.normpath(str(model_name).replace(".zip", "")),
                )
                
                success = extract_and_show_progress(zipfile_path, unzips_path)
                if success:
                    print(f"Extracción exitosa: {model_name}")
                else:
                    print(f"Fallo en la extracción: {model_name}")
            else:
                print(i18n("Unzip error."))
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
                if item.startswith(("added_", "trained_")) and item.endswith(".index"):
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
            else:
                print(
                    i18n(
                        "The model works for inference, but it doesn't have the .index file."
                    )
                )

        if not index_file and not model_file:
            print(i18n("No relevant file was found to upload."))
        
        os.chdir(parent_path)
        if 'text/html' in request.headers.get('Accept', ''):
            return redirect("http://localhost:8000/downloaded", code=302)
        else:
            return ""
    except Exception as e:
        os.chdir(parent_path)
        if "too much use" in str(e):
            print(i18n("Too many users have recently viewed or downloaded this file"))
        elif "private link" in str(e):
            print(i18n("Cannot get file from this private link"))
        else:
            print(i18n("An error occurred downloading"))
            print(e)
    finally:
        os.chdir(parent_path)

@app.route('/downloaded', methods=['GET'])
def downloaded():
    return render_template('sucess.html')

@app.route('/shutdown', methods=['POST'])
def shoutdown():
    print("This flask server is shutting down... Please close the window!")   
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)

if __name__ == '__main__':
    app.run(host='localhost', port=8000)
