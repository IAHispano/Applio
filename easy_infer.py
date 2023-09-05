import subprocess
import os
import sys
import errno
import shutil
import yt_dlp
from mega import Mega
import datetime
import unicodedata
import torch
import glob
import gradio as gr
import gdown
import zipfile
import traceback
import json
import mdx
from mdx_processing_script import get_model_list,id_to_ptm,prepare_mdx,run_mdx
import requests
import wget
import ffmpeg
import hashlib
now_dir = os.getcwd()
sys.path.append(now_dir)
from unidecode import unidecode
import re
import time
from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from vc_infer_pipeline import VC
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from MDXNet import MDXNetDereverb
from config import Config
from infer_uvr5 import _audio_pre_, _audio_pre_new
from huggingface_hub import HfApi, list_models
from huggingface_hub import login
from i18n import I18nAuto
i18n = I18nAuto()
from bs4 import BeautifulSoup
from sklearn.cluster import MiniBatchKMeans

config = Config()
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.environ["TEMP"] = tmp
weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "./logs/"
audio_root = "audios"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []

global indexes_list
indexes_list = []

audio_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s\\%s" % (root, name))

for root, dirs, files in os.walk(audio_root, topdown=False):
    for name in files:
        audio_paths.append("%s/%s" % (root, name))

uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def format_title(title):
     formatted_title = re.sub(r'[^\w\s-]', '', title)
     formatted_title = formatted_title.replace(" ", "_")
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
      if not file.startswith("G_") and not file.startswith("D_") and file.endswith(".pth") and not "_G_" in file and not "_D_" in file:
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



def download_from_url(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    zips_path = os.path.join(parent_path, 'zips')
    
    if url != '':
        print(i18n("Downloading the file: ") + f"{url}")
        if "drive.google.com" in url:
            if "file/d/" in url:
                file_id = url.split("file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                return None
            
            if file_id:
                os.chdir('./zips')
                result = subprocess.run(["gdown", f"https://drive.google.com/uc?id={file_id}", "--fuzzy"], capture_output=True, text=True, encoding='utf-8')
                if "Too many users have viewed or downloaded this file recently" in str(result.stderr):
                    return "too much use"
                if "Cannot retrieve the public link of the file." in str(result.stderr):
                    return "private link"
                print(result.stderr)
                
        elif "/blob/" in url:
            os.chdir('./zips')
            url = url.replace("blob", "resolve")
            response = requests.get(url)
            if response.status_code == 200:
                file_name = url.split('/')[-1]
                with open(os.path.join(zips_path, file_name), "wb") as newfile:
                    newfile.write(response.content)
            else:
                    os.chdir(parent_path)
        elif "mega.nz" in url:
            if "#!" in url:
                file_id = url.split("#!")[1].split("!")[0]
            elif "file/" in url:
                file_id = url.split("file/")[1].split("/")[0]
            else:
                return None
            if file_id:
                m = Mega()
                m.download_url(url, zips_path)
        elif "/tree/main" in url:
           response = requests.get(url)
           soup = BeautifulSoup(response.content, 'html.parser')
           temp_url = ''
           for link in soup.find_all('a', href=True):
               if link['href'].endswith('.zip'):
                  temp_url = link['href']
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
            if file.status_code == 200:
                name = url.split('/')
                with open(os.path.join(zips_path, name[len(name)-1]), "wb") as newfile:
                    newfile.write(file.content)
            else:
                return None
        elif "pixeldrain.com" in url:
            try:
                file_id = url.split("pixeldrain.com/u/")[1]
                os.chdir('./zips')
                print(file_id)
                response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
                if response.status_code == 200:
                    file_name = response.headers.get("Content-Disposition").split('filename=')[-1].strip('";')
                    if not os.path.exists(zips_path):
                        os.makedirs(zips_path)
                    with open(os.path.join(zips_path, file_name), "wb") as newfile:
                        newfile.write(response.content)
                        os.chdir(parent_path)
                        return "downloaded"
                else:
                    os.chdir(parent_path)
                    return None
            except Exception as e:
                print(e)
                os.chdir(parent_path)
                return None
        else:
            os.chdir('./zips')
            wget.download(url)
            
        os.chdir(parent_path)
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
        
def load_downloaded_model(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    try:
        infos = []
        logs_folders = ['0_gt_wavs','1_16k_wavs','2a_f0','2b-f0nsf','3_feature256','3_feature768']
        zips_path = os.path.join(parent_path, 'zips')
        unzips_path = os.path.join(parent_path, 'unzips')
        weights_path = os.path.join(parent_path, 'weights')
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
            raise Exception(i18n("Too many users have recently viewed or downloaded this file"))
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))
        
        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path,filename)
                print(i18n("Proceeding with the extraction..."))
                infos.append(i18n("Proceeding with the extraction..."))
                shutil.unpack_archive(zipfile_path, unzips_path, 'zip')
                model_name = os.path.basename(zipfile_path)
                logs_dir = os.path.join(parent_path,'logs', os.path.normpath(str(model_name).replace(".zip","")))
                yield "\n".join(infos)
            else:
                print(i18n("Unzip error."))
                infos.append(i18n("Unzip error."))
                yield "\n".join(infos)
        
        index_file = False
        model_file = False
        D_file = False
        G_file = False
        
        for path, subdirs, files in os.walk(unzips_path):
            for item in files:
                item_path = os.path.join(path, item)
                if not 'G_' in item and not 'D_' in item and item.endswith('.pth'):
                    model_file = True
                    model_name = item.replace(".pth","")
                    logs_dir = os.path.join(parent_path,'logs', model_name)
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
                if item.startswith('added_') and item.endswith('.index'):
                    index_file = True
                    if os.path.exists(item_path):
                        if os.path.exists(os.path.join(logs_dir, item)):
                            os.remove(os.path.join(logs_dir, item))
                        shutil.move(item_path, logs_dir)
                if item.startswith('total_fea.npy') or item.startswith('events.'):
                    if os.path.exists(item_path):
                        if os.path.exists(os.path.join(logs_dir, item)):
                            os.remove(os.path.join(logs_dir, item))
                        shutil.move(item_path, logs_dir)
        
                
        result = ""
        if model_file:
            if index_file:
                print(i18n("The model works for inference, and has the .index file."))
                infos.append("\n" + i18n("The model works for inference, and has the .index file."))
                yield "\n".join(infos)
            else:
                print(i18n("The model works for inference, but it doesn't have the .index file."))
                infos.append("\n" + i18n("The model works for inference, but it doesn't have the .index file."))
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
    parent_path = find_folder_parent(".", "pretrained_v2")
    infos = []
    try:
        zips_path = os.path.join(parent_path, 'zips')
        unzips_path = os.path.join(parent_path, 'unzips')
        datasets_path = os.path.join(parent_path, 'datasets')
        audio_extenions =['wav', 'mp3', 'flac', 'ogg', 'opus',
                'm4a', 'mp4', 'aac', 'alac', 'wma',
                'aiff', 'webm', 'ac3']
        
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
            raise Exception(i18n("Too many users have recently viewed or downloaded this file"))
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))
  
        zip_path = os.listdir(zips_path)
        foldername = ""
        for file in zip_path:
            if file.endswith('.zip'):
                file_path = os.path.join(zips_path, file)
                print("....")
                foldername = file.replace(".zip","").replace(" ","").replace("-","_")
                dataset_path = os.path.join(datasets_path, foldername)
                print(i18n("Proceeding with the extraction..."))
                infos.append(i18n("Proceeding with the extraction..."))
                yield "\n".join(infos)
                shutil.unpack_archive(file_path, unzips_path, 'zip')
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)
                    
                os.mkdir(dataset_path)
                
                for root, subfolders, songs in os.walk(unzips_path):
                    for song in songs:
                        song_path = os.path.join(root, song)
                        if song.endswith(tuple(audio_extenions)):
                            formatted_song_name = format_title(os.path.splitext(song)[0])
                            extension = os.path.splitext(song)[1]
                            new_song_path = os.path.join(dataset_path, f"{formatted_song_name}{extension}")
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

def save_model(modelname, save_action):
       
    parent_path = find_folder_parent(".", "pretrained_v2")
    zips_path = os.path.join(parent_path, 'zips')
    dst = os.path.join(zips_path,modelname)
    logs_path = os.path.join(parent_path, 'logs', modelname)
    weights_path = os.path.join(parent_path, 'weights', f"{modelname}.pth")
    save_folder = parent_path
    infos = []    
    
    try:
        if not os.path.exists(logs_path):
            raise Exception("No model found.")
        
        if not 'content' in parent_path:
            save_folder = os.path.join(parent_path, 'RVC_Backup')
        else:
            save_folder = '/content/drive/MyDrive/RVC_Backup'
        
        infos.append(i18n("Save model"))
        yield "\n".join(infos)
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(os.path.join(save_folder, 'ManualTrainingBackup')):
            os.mkdir(os.path.join(save_folder, 'ManualTrainingBackup'))
        if not os.path.exists(os.path.join(save_folder, 'Finished')):
            os.mkdir(os.path.join(save_folder, 'Finished'))

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
            
        os.mkdir(zips_path)
        added_file = glob.glob(os.path.join(logs_path, "added_*.index"))
        d_file = glob.glob(os.path.join(logs_path, "D_*.pth"))
        g_file = glob.glob(os.path.join(logs_path, "G_*.pth"))
        
        if save_action == i18n("Choose the method"):
            raise Exception("No method choosen.")
        
        if save_action == i18n("Save all"):
            print(i18n("Save all"))
            save_folder = os.path.join(save_folder, 'ManualTrainingBackup')
            shutil.copytree(logs_path, dst)
        else:
            if not os.path.exists(dst):
                os.mkdir(dst)
            
        if save_action == i18n("Save D and G"):
            print(i18n("Save D and G"))
            save_folder = os.path.join(save_folder, 'ManualTrainingBackup')
            if len(d_file) > 0:
                shutil.copy(d_file[0], dst)
            if len(g_file) > 0:
                shutil.copy(g_file[0], dst)    
                
            if len(added_file) > 0:
                shutil.copy(added_file[0], dst)
            else:
                infos.append(i18n("Saved without index..."))
                
        if save_action == i18n("Save voice"):
            print(i18n("Save voice"))
            save_folder = os.path.join(save_folder, 'Finished')
            if len(added_file) > 0:
                shutil.copy(added_file[0], dst)
            else:
                infos.append(i18n("Saved without index..."))
        
        yield "\n".join(infos)
        if not os.path.exists(weights_path):
            infos.append(i18n("Saved without inference model..."))
        else:
            shutil.copy(weights_path, dst)
        
        yield "\n".join(infos)
        infos.append("\n" + i18n("This may take a few minutes, please wait..."))
        yield "\n".join(infos)
        
        shutil.make_archive(os.path.join(zips_path,f"{modelname}"), 'zip', zips_path)
        shutil.move(os.path.join(zips_path,f"{modelname}.zip"), os.path.join(save_folder, f'{modelname}.zip'))
        
        shutil.rmtree(zips_path)        
        infos.append("\n" + i18n("Model saved successfully"))
        yield "\n".join(infos)
        
    except Exception as e:
        print(e)
        if "No model found." in str(e):
            infos.append(i18n("The model you want to save does not exist, be sure to enter the correct name."))
        else:
            infos.append(i18n("An error occurred saving the model"))
            
        yield "\n".join(infos)
    
def load_downloaded_backup(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    try:
        infos = []
        logs_folders = ['0_gt_wavs','1_16k_wavs','2a_f0','2b-f0nsf','3_feature256','3_feature768']
        zips_path = os.path.join(parent_path, 'zips')
        unzips_path = os.path.join(parent_path, 'unzips')
        weights_path = os.path.join(parent_path, 'weights')
        logs_dir = os.path.join(parent_path, 'logs')
        
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
            raise Exception(i18n("Too many users have recently viewed or downloaded this file"))
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))
        
        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path,filename)
                zip_dir_name = os.path.splitext(filename)[0]
                unzip_dir = unzips_path
                print(i18n("Proceeding with the extraction..."))
                infos.append(i18n("Proceeding with the extraction..."))
                shutil.unpack_archive(zipfile_path, unzip_dir, 'zip')
                
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
        if os.path.exists(os.path.join(parent_path, 'unzips')):
            shutil.rmtree(os.path.join(parent_path, 'unzips'))
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
        path_to_file=record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path='./audios/'+new_name
        shutil.move(path_to_file,new_path)
        return new_name


def change_choices2():
    audio_paths=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('wav', 'mp3', 'flac', 'ogg', 'opus',
                'm4a', 'mp4', 'aac', 'alac', 'wma',
                'aiff', 'webm', 'ac3')):
            audio_paths.append(os.path.join('./audios',filename).replace('\\', '/'))
    return {"choices": sorted(audio_paths), "__type__": "update"}, {"__type__": "update"}





def uvr(input_url, output_path, model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0, architecture):
    carpeta_a_eliminar = "yt_downloads"
    if os.path.exists(carpeta_a_eliminar) and os.path.isdir(carpeta_a_eliminar):
        for archivo in os.listdir(carpeta_a_eliminar):
            ruta_archivo = os.path.join(carpeta_a_eliminar, archivo)
            if os.path.isfile(ruta_archivo):
                os.remove(ruta_archivo)
            elif os.path.isdir(ruta_archivo):
                shutil.rmtree(ruta_archivo) 
      
    

    ydl_opts = {
     'no-windows-filenames': True,
     'restrict-filenames': True,
     'extract_audio': True,
     'format': 'bestaudio',
     'quiet': True,
     'no-warnings': True,
     }
    
    try:
        print(i18n("Downloading audio from the video..."))
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
         info_dict = ydl.extract_info(input_url, download=False)
         formatted_title = format_title(info_dict.get('title', 'default_title'))
         formatted_outtmpl = output_path + '/' + formatted_title + '.wav'
         ydl_opts['outtmpl'] = formatted_outtmpl
         ydl = yt_dlp.YoutubeDL(ydl_opts)
         ydl.download([input_url])
        print(i18n("Audio downloaded!"))
    except Exception as error:
        print(i18n("An error occurred:"), error)

    actual_directory = os.path.dirname(__file__)
    
    vocal_directory = os.path.join(actual_directory, save_root_vocal)
    instrumental_directory = os.path.join(actual_directory, save_root_ins)
    
    vocal_formatted = f"vocal_{formatted_title}.wav.reformatted.wav_10.wav"
    instrumental_formatted = f"instrument_{formatted_title}.wav.reformatted.wav_10.wav"  
    
    vocal_audio_path = os.path.join(vocal_directory, vocal_formatted)
    instrumental_audio_path = os.path.join(instrumental_directory, instrumental_formatted)
    
    vocal_formatted_mdx = f"{formatted_title}_vocal_.wav"
    instrumental_formatted_mdx = f"{formatted_title}_instrument_.wav"
    
    vocal_audio_path_mdx = os.path.join(vocal_directory, vocal_formatted_mdx)
    instrumental_audio_path_mdx = os.path.join(instrumental_directory, instrumental_formatted_mdx)

    if architecture == "VR":
       try:
           print(i18n("Starting audio conversion... (This might take a moment)"))
           inp_root, save_root_vocal, save_root_ins = [x.strip(" ").strip('"').strip("\n").strip('"').strip(" ") for x in [inp_root, save_root_vocal, save_root_ins]]
           usable_files = [os.path.join(inp_root, file) 
                          for file in os.listdir(inp_root) 
                          if file.endswith(tuple(sup_audioext))]    
           
        
           pre_fun = MDXNetDereverb(15) if model_name == "onnx_dereverb_By_FoxJoy" else (_audio_pre_ if "DeEcho" not in model_name else _audio_pre_new)(
                       agg=int(agg),
                       model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                       device=config.device,
                       is_half=config.is_half,
                   )
                
           try:
              if paths != None:
                paths = [path.name for path in paths]
              else:
                paths = usable_files
                
           except:
                traceback.print_exc()
                paths = usable_files
           print(paths) 
           for path in paths:
               inp_path = os.path.join(inp_root, path)
               need_reformat, done = 1, 0

               try:
                   info = ffmpeg.probe(inp_path, cmd="ffprobe")
                   if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
                       need_reformat = 0
                       pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
                       done = 1
               except:
                   traceback.print_exc()

               if need_reformat:
                   tmp_path = f"{tmp}/{os.path.basename(inp_path)}.reformatted.wav"
                   os.system(f"ffmpeg -i {inp_path} -vn -acodec pcm_s16le -ac 2 -ar 44100 {tmp_path} -y")
                   inp_path = tmp_path

               try:
                   if not done:
                       pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
                   print(f"{os.path.basename(inp_path)}->Success")
               except:
                   print(f"{os.path.basename(inp_path)}->{traceback.format_exc()}")
       except:
           traceback.print_exc()
       finally:
           try:
               if model_name == "onnx_dereverb_By_FoxJoy":
                   del pre_fun.pred.model
                   del pre_fun.pred.model_
               else:
                   del pre_fun.model

               del pre_fun
               return i18n("Finished"), vocal_audio_path, instrumental_audio_path
           except: traceback.print_exc()

           if torch.cuda.is_available(): torch.cuda.empty_cache()

    elif architecture == "MDX":
       try:
           print(i18n("Starting audio conversion... (This might take a moment)"))
           inp_root, save_root_vocal, save_root_ins = [x.strip(" ").strip('"').strip("\n").strip('"').strip(" ") for x in [inp_root, save_root_vocal, save_root_ins]]
        
           usable_files = [os.path.join(inp_root, file) 
                          for file in os.listdir(inp_root) 
                          if file.endswith(tuple(sup_audioext))]    
           try:
              if paths != None:
                paths = [path.name for path in paths]
              else:
                paths = usable_files
                
           except:
                traceback.print_exc()
                paths = usable_files
           print(paths) 
           invert=True
           denoise=True
           use_custom_parameter=True
           dim_f=2048
           dim_t=256
           n_fft=7680
           use_custom_compensation=True
           compensation=1.025
           suffix = "vocal_" #@param ["Vocals", "Drums", "Bass", "Other"]{allow-input: true}
           suffix_invert = "instrument_" #@param ["Instrumental", "Drumless", "Bassless", "Instruments"]{allow-input: true}
           print_settings = True  # @param{type:"boolean"}
           onnx = id_to_ptm(model_name)
           compensation = compensation if use_custom_compensation or use_custom_parameter else None
           mdx_model = prepare_mdx(onnx,use_custom_parameter, dim_f, dim_t, n_fft, compensation=compensation)
           
       
           for path in paths:
               #inp_path = os.path.join(inp_root, path)
               suffix_naming = suffix if use_custom_parameter else None
               diff_suffix_naming = suffix_invert if use_custom_parameter else None
               run_mdx(onnx, mdx_model, path, format0, diff=invert,suffix=suffix_naming,diff_suffix=diff_suffix_naming,denoise=denoise)
    
           if print_settings:
               print()
               print('[MDX-Net_Colab settings used]')
               print(f'Model used: {onnx}')
               print(f'Model MD5: {mdx.MDX.get_hash(onnx)}')
               print(f'Model parameters:')
               print(f'    -dim_f: {mdx_model.dim_f}')
               print(f'    -dim_t: {mdx_model.dim_t}')
               print(f'    -n_fft: {mdx_model.n_fft}')
               print(f'    -compensation: {mdx_model.compensation}')
               print()
               print('[Input file]')
               print('filename(s): ')
               for filename in paths:
                   print(f'    -{filename}')
                   print(f"{os.path.basename(filename)}->Success")
       except:
           traceback.print_exc()
       finally:
           try:
               del mdx_model
               return i18n("Finished"), vocal_audio_path_mdx, instrumental_audio_path_mdx
           except: traceback.print_exc()

           print("clean_empty_cache")

           if torch.cuda.is_available(): torch.cuda.empty_cache()
sup_audioext = {'wav', 'mp3', 'flac', 'ogg', 'opus',
                'm4a', 'mp4', 'aac', 'alac', 'wma',
                'aiff', 'webm', 'ac3'}

def load_downloaded_audio(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    try:
        infos = []
        audios_path = os.path.join(parent_path, 'audios')
        zips_path = os.path.join(parent_path, 'zips')

        if not os.path.exists(audios_path):
            os.mkdir(audios_path)

        if not os.path.exists(zips_path):
            os.mkdir(zips_path)
        
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
            raise Exception(i18n("Too many users have recently viewed or downloaded this file"))
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))
        
        for filename in os.listdir(zips_path):
            item_path = os.path.join(zips_path, filename)
            if item_path.split('.')[-1] in sup_audioext:
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

def download_model():
    gr.Markdown(value="# " + i18n("Download Model"))
    gr.Markdown(value=i18n("It is used to download your inference models."))
    with gr.Row():
        model_url=gr.Textbox(label=i18n("Url:"))
    with gr.Row():
        download_model_status_bar=gr.Textbox(label=i18n("Status:"))
    with gr.Row():
        download_button=gr.Button(i18n("Download"))
        download_button.click(fn=load_downloaded_model, inputs=[model_url], outputs=[download_model_status_bar])

def download_backup():
    gr.Markdown(value="# " + i18n("Download Backup"))
    gr.Markdown(value=i18n("It is used to download your training backups."))
    with gr.Row():
        model_url=gr.Textbox(label=i18n("Url:"))
    with gr.Row():
        download_model_status_bar=gr.Textbox(label=i18n("Status:"))
    with gr.Row():
        download_button=gr.Button(i18n("Download"))
        download_button.click(fn=load_downloaded_backup, inputs=[model_url], outputs=[download_model_status_bar])

def update_dataset_list(name):
    new_datasets = []
    for foldername in os.listdir("./datasets"):
        if "." not in foldername:
            new_datasets.append(os.path.join(find_folder_parent(".","pretrained"),"datasets",foldername))
    return gr.Dropdown.update(choices=new_datasets)

def download_dataset(trainset_dir4):
    gr.Markdown(value="# " + i18n("Download Dataset"))
    gr.Markdown(value=i18n("Download the dataset with the audios in a compatible format (.wav/.flac) to train your model."))
    with gr.Row():
        dataset_url=gr.Textbox(label=i18n("Url:"))
    with gr.Row():
        load_dataset_status_bar=gr.Textbox(label=i18n("Status:"))
    with gr.Row():
        load_dataset_button=gr.Button(i18n("Download"))
        load_dataset_button.click(fn=load_dowloaded_dataset, inputs=[dataset_url], outputs=[load_dataset_status_bar])
        load_dataset_status_bar.change(update_dataset_list, dataset_url, trainset_dir4)

def download_audio():
    gr.Markdown(value="# " + i18n("Download Audio"))
    gr.Markdown(value=i18n("Download audios of any format for use in inference (recommended for mobile users)."))
    with gr.Row():
        audio_url=gr.Textbox(label=i18n("Url:"))
    with gr.Row():
        download_audio_status_bar=gr.Textbox(label=i18n("Status:"))
    with gr.Row():
        download_button2=gr.Button(i18n("Download"))
        download_button2.click(fn=load_downloaded_audio, inputs=[audio_url], outputs=[download_audio_status_bar])

def youtube_separator():
        gr.Markdown(value="# " + i18n("Separate YouTube tracks"))
        gr.Markdown(value=i18n("Download audio from a YouTube video and automatically separate the vocal and instrumental tracks"))
        with gr.Row():
            input_url = gr.inputs.Textbox(label=i18n("Enter the YouTube link:"))
            output_path = gr.Textbox(
                label=i18n("Enter the path of the audio folder to be processed (copy it from the address bar of the file manager):"),
                value=os.path.abspath(os.getcwd()).replace('\\', '/') + "/yt_downloads",
                visible=False,
                )
            advanced_settings_checkbox = gr.Checkbox(
                value=False,
                label=i18n("Advanced Settings"),
                interactive=True,
                )
        with gr.Row(label = i18n("Advanced Settings"), visible=False, variant='compact') as advanced_settings:
            with gr.Column(): 
                model_select = gr.Radio(
                    label=i18n("Model Architecture:"),
                    choices=["VR", "MDX"],
                    value="VR",
                    interactive=True,
                    )
                model_choose = gr.Dropdown(label=i18n("Model: (Be aware that in some models the named vocal will be the instrumental)"),                          
                    choices=uvr5_names,
                    value="HP5_only_main_vocal"   
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
                        label=i18n("Specify the output folder for vocals:"), value="audios",
                        )
                opt_ins_root = gr.Textbox(
                    label=i18n("Specify the output folder for accompaniment:"), value="audio-others",
                    ) 
                dir_wav_input = gr.Textbox(
                    label=i18n("Enter the path of the audio folder to be processed:"),
                    value=((os.getcwd()).replace('\\', '/') + "/yt_downloads"),
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
                    file_count="multiple", label=i18n("You can also input audio files in batches. Choose one of the two options. Priority is given to reading from the folder."),
                    visible=False,
                    )
            model_select.change(
                fn=update_model_choices,
                inputs=model_select,
                outputs=model_choose,
                )
        with gr.Row():
            vc_output4 = gr.Textbox(label=i18n("Status:"))
            vc_output5 = gr.Audio(label=i18n("Vocal"), type='filepath')
            vc_output6 = gr.Audio(label=i18n("Instrumental"), type='filepath')
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
                    model_select
                    ],
                    [vc_output4, vc_output5, vc_output6],
                )
        def toggle_advanced_settings(checkbox):
            return {"visible": checkbox, "__type__": "update"}
        
        advanced_settings_checkbox.change(
            fn=toggle_advanced_settings,
            inputs=[advanced_settings_checkbox],
            outputs=[advanced_settings]
            )


def get_bark_voice():
    mensaje = """
v2/en_speaker_0	English	Male
v2/en_speaker_1	English	Male
v2/en_speaker_2	English	Male
v2/en_speaker_3	English	Male
v2/en_speaker_4	English	Male
v2/en_speaker_5	English	Male
v2/en_speaker_6	English	Male
v2/en_speaker_7	English	Male
v2/en_speaker_8	English	Male
v2/en_speaker_9	English	Female
v2/zh_speaker_0	Chinese (Simplified)	Male
v2/zh_speaker_1	Chinese (Simplified)	Male
v2/zh_speaker_2	Chinese (Simplified)	Male
v2/zh_speaker_3	Chinese (Simplified)	Male
v2/zh_speaker_4	Chinese (Simplified)	Female
v2/zh_speaker_5	Chinese (Simplified)	Male
v2/zh_speaker_6	Chinese (Simplified)	Female
v2/zh_speaker_7	Chinese (Simplified)	Female
v2/zh_speaker_8	Chinese (Simplified)	Male
v2/zh_speaker_9	Chinese (Simplified)	Female
v2/fr_speaker_0	French	Male
v2/fr_speaker_1	French	Female
v2/fr_speaker_2	French	Female
v2/fr_speaker_3	French	Male
v2/fr_speaker_4	French	Male
v2/fr_speaker_5	French	Female
v2/fr_speaker_6	French	Male
v2/fr_speaker_7	French	Male
v2/fr_speaker_8	French	Male
v2/fr_speaker_9	French	Male
v2/de_speaker_0	German	Male
v2/de_speaker_1	German	Male
v2/de_speaker_2	German	Male
v2/de_speaker_3	German	Female
v2/de_speaker_4	German	Male
v2/de_speaker_5	German	Male
v2/de_speaker_6	German	Male
v2/de_speaker_7	German	Male
v2/de_speaker_8	German	Female
v2/de_speaker_9	German	Male
v2/hi_speaker_0	Hindi	Female
v2/hi_speaker_1	Hindi	Female
v2/hi_speaker_2	Hindi	Male
v2/hi_speaker_3	Hindi	Female
v2/hi_speaker_4	Hindi	Female
v2/hi_speaker_5	Hindi	Male
v2/hi_speaker_6	Hindi	Male
v2/hi_speaker_7	Hindi	Male
v2/hi_speaker_8	Hindi	Male
v2/hi_speaker_9	Hindi	Female
v2/it_speaker_0	Italian	Male
v2/it_speaker_1	Italian	Male
v2/it_speaker_2	Italian	Female
v2/it_speaker_3	Italian	Male
v2/it_speaker_4	Italian	Male
v2/it_speaker_5	Italian	Male
v2/it_speaker_6	Italian	Male
v2/it_speaker_7	Italian	Female
v2/it_speaker_8	Italian	Male
v2/it_speaker_9	Italian	Female
v2/ja_speaker_0	Japanese	Female
v2/ja_speaker_1	Japanese	Female
v2/ja_speaker_2	Japanese	Male
v2/ja_speaker_3	Japanese	Female
v2/ja_speaker_4	Japanese	Female
v2/ja_speaker_5	Japanese	Female
v2/ja_speaker_6	Japanese	Male
v2/ja_speaker_7	Japanese	Female
v2/ja_speaker_8	Japanese	Female
v2/ja_speaker_9	Japanese	Female
v2/ko_speaker_0	Korean	Female
v2/ko_speaker_1	Korean	Male
v2/ko_speaker_2	Korean	Male
v2/ko_speaker_3	Korean	Male
v2/ko_speaker_4	Korean	Male
v2/ko_speaker_5	Korean	Male
v2/ko_speaker_6	Korean	Male
v2/ko_speaker_7	Korean	Male
v2/ko_speaker_8	Korean	Male
v2/ko_speaker_9	Korean	Male
v2/pl_speaker_0	Polish	Male
v2/pl_speaker_1	Polish	Male
v2/pl_speaker_2	Polish	Male
v2/pl_speaker_3	Polish	Male
v2/pl_speaker_4	Polish	Female
v2/pl_speaker_5	Polish	Male
v2/pl_speaker_6	Polish	Female
v2/pl_speaker_7	Polish	Male
v2/pl_speaker_8	Polish	Male
v2/pl_speaker_9	Polish	Female
v2/pt_speaker_0	Portuguese	Male
v2/pt_speaker_1	Portuguese	Male
v2/pt_speaker_2	Portuguese	Male
v2/pt_speaker_3	Portuguese	Male
v2/pt_speaker_4	Portuguese	Male
v2/pt_speaker_5	Portuguese	Male
v2/pt_speaker_6	Portuguese	Male
v2/pt_speaker_7	Portuguese	Male
v2/pt_speaker_8	Portuguese	Male
v2/pt_speaker_9	Portuguese	Male
v2/ru_speaker_0	Russian	Male
v2/ru_speaker_1	Russian	Male
v2/ru_speaker_2	Russian	Male
v2/ru_speaker_3	Russian	Male
v2/ru_speaker_4	Russian	Male
v2/ru_speaker_5	Russian	Female
v2/ru_speaker_6	Russian	Female
v2/ru_speaker_7	Russian	Male
v2/ru_speaker_8	Russian	Male
v2/ru_speaker_9	Russian	Female
v2/es_speaker_0	Spanish	Male
v2/es_speaker_1	Spanish	Male
v2/es_speaker_2	Spanish	Male
v2/es_speaker_3	Spanish	Male
v2/es_speaker_4	Spanish	Male
v2/es_speaker_5	Spanish	Male
v2/es_speaker_6	Spanish	Male
v2/es_speaker_7	Spanish	Male
v2/es_speaker_8	Spanish	Female
v2/es_speaker_9	Spanish	Female
v2/tr_speaker_0	Turkish	Male
v2/tr_speaker_1	Turkish	Male
v2/tr_speaker_2	Turkish	Male
v2/tr_speaker_3	Turkish	Male
v2/tr_speaker_4	Turkish	Female
v2/tr_speaker_5	Turkish	Female
v2/tr_speaker_6	Turkish	Male
v2/tr_speaker_7	Turkish	Male
v2/tr_speaker_8	Turkish	Male
v2/tr_speaker_9	Turkish	Male
    """
# Dividir el mensaje en líneas
    lineas = mensaje.split("\n")
    datos_deseados = []
    for linea in lineas:
        partes = linea.split("\t")
        if len(partes) == 3:
            clave, _, genero = partes  
            datos_deseados.append(f"{clave}-{genero}")

    return datos_deseados

    
def get_edge_voice():
    completed_process = subprocess.run(['edge-tts',"-l"], capture_output=True, text=True)
    lines = completed_process.stdout.strip().split("\n")
    data = []
    current_entry = {}
    for line in lines:
        if line.startswith("Name: "):
            if current_entry:
                data.append(current_entry)
            current_entry = {"Name": line.split(": ")[1]}
        elif line.startswith("Gender: "):
            current_entry["Gender"] = line.split(": ")[1]
    if current_entry:
        data.append(current_entry)
    tts_voice = []
    for entry in data:
        name = entry["Name"]
        gender = entry["Gender"]
        formatted_entry = f'{name}-{gender}'
        tts_voice.append(formatted_entry)
    return tts_voice


#print(set_tts_voice)
