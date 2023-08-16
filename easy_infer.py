import subprocess
import os
import shutil
from mega import Mega
import datetime
import unicodedata
import glob
import gradio as gr
import gdown
import zipfile
import json
import requests
import wget
import hashlib
from unidecode import unidecode
import re
import time
from huggingface_hub import HfApi, list_models
from huggingface_hub import login
from i18n import I18nAuto
i18n = I18nAuto()
from bs4 import BeautifulSoup
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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

def get_drive_folder_id(url):
    if "drive.google.com" in url:
        if "file/d/" in url:
            file_id = url.split("file/d/")[1].split("/")[0]
        elif "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
        else:
            return None

def download_from_url(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    zips_path = os.path.join(parent_path, 'zips')
    
    if url != '':
        print(i18n("下载文件：") + f"{url}")
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
                    return "demasiado uso"
                if "Cannot retrieve the public link of the file." in str(result.stderr):
                    return "link privado"
                print(result.stderr)
                
        elif "/blob/" in url:
            os.chdir('./zips')
            url = url.replace("blob", "resolve")
          # print("Resolved URL:", url)  # Print the resolved URL
            wget.download(url)
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
          # print("Updated URL:", url)  # Print the updated URL
              url = url.replace("blob", "resolve")
          # print("Resolved URL:", url)  # Print the resolved URL

              if "huggingface.co" not in url:
                 url = "https://huggingface.co" + url

                 wget.download(url)
           else:
                 print("No .zip file found on the page.")
            # Handle the case when no .zip file is found
        else:
            os.chdir('./zips')
            wget.download(url)
            
        os.chdir(parent_path)
        print(i18n("完整下载"))
        return "downloaded"
    else:
        return None
                
class error_message(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje
        super().__init__(mensaje)
        
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
            print(i18n("无法下载模型。"))
            infos.append(i18n("无法下载模型。"))
            yield "\n".join(infos)
        elif download_file == "downloaded":
            print(i18n("模型下载成功。"))
            infos.append(i18n("模型下载成功。"))
            yield "\n".join(infos)
        elif download_file == "demasiado uso":
            raise Exception(i18n("最近查看或下载此文件的用户过多"))
        elif download_file == "link privado":
            raise Exception(i18n("无法从该私人链接获取文件"))
        
        # Descomprimir archivos descargados
        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path,filename)
                print(i18n("继续提取..."))
                infos.append(i18n("继续提取..."))
                shutil.unpack_archive(zipfile_path, unzips_path, 'zip')
                model_name = os.path.basename(zipfile_path)
                logs_dir = os.path.join(parent_path,'logs', os.path.normpath(str(model_name).replace(".zip","")))
                yield "\n".join(infos)
            else:
                print(i18n("解压缩出错。"))
                infos.append(i18n("解压缩出错。"))
                yield "\n".join(infos)
        
        index_file = False
        model_file = False
        D_file = False
        G_file = False
        
        # Copiar archivo pth
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
        # Copiar index
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
                print(i18n("该模型可用于推理，并有 .index 文件。"))
                infos.append("\n" + i18n("该模型可用于推理，并有 .index 文件。"))
                yield "\n".join(infos)
            else:
                print(i18n("该模型可用于推理，但没有 .index 文件。"))
                infos.append("\n" + i18n("该模型可用于推理，但没有 .index 文件。"))
                yield "\n".join(infos)
        
        if not index_file and not model_file:
            print(i18n("未找到可上传的相关文件"))
            infos.append(i18n("未找到可上传的相关文件"))
            yield "\n".join(infos)
        
        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)
        os.chdir(parent_path)    
        return result
    except Exception as e:
        os.chdir(parent_path)
        if "demasiado uso" in str(e):
            print(i18n("最近查看或下载此文件的用户过多"))
            yield i18n("最近查看或下载此文件的用户过多")
        elif "link privado" in str(e):
            print(i18n("无法从该私人链接获取文件"))
            yield i18n("无法从该私人链接获取文件")
        else:
            print(e)
            yield i18n("下载模型时发生错误。")
    finally:
        os.chdir(parent_path)
      
def load_dowloaded_dataset(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    infos = []
    try:
        zips_path = os.path.join(parent_path, 'zips')
        unzips_path = os.path.join(parent_path, 'unzips')
        datasets_path = os.path.join(parent_path, 'datasets')
        audio_extenions =["flac","wav"]
        
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
            print(i18n("下载模型时发生错误。"))
            infos.append(i18n("下载模型时发生错误。"))
            yield "\n".join(infos)
            raise Exception(i18n("下载模型时发生错误。"))
        elif download_file == "downloaded":
            print(i18n("模型下载成功。"))
            infos.append(i18n("模型下载成功。"))
            yield "\n".join(infos)
        elif download_file == "demasiado uso":
            raise Exception(i18n("最近查看或下载此文件的用户过多"))
        elif download_file == "link privado":
            raise Exception(i18n("无法从该私人链接获取文件"))
  
        zip_path = os.listdir(zips_path)
        foldername = ""
        for file in zip_path:
            if file.endswith('.zip'):
                file_path = os.path.join(zips_path, file)
                print("....")
                foldername = file.replace(".zip","").replace(" ","").replace("-","_")
                dataset_path = os.path.join(datasets_path, foldername)
                print(i18n("继续提取..."))
                infos.append(i18n("继续提取..."))
                yield "\n".join(infos)
                shutil.unpack_archive(file_path, unzips_path, 'zip')
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)
                    
                os.mkdir(dataset_path)
                
                for root, subfolders, songs in os.walk(unzips_path):
                    for song in songs:
                        song_path = os.path.join(root, song)
                        if song.endswith(tuple(audio_extenions)):
                            shutil.move(song_path, dataset_path)
            else:
                print(i18n("解压缩出错。"))
                infos.append(i18n("解压缩出错。"))
                yield "\n".join(infos)
                
                

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)
            
        print(i18n("数据集加载成功。"))
        infos.append(i18n("数据集加载成功。"))
        yield "\n".join(infos)
    except Exception as e:
        os.chdir(parent_path)
        if "demasiado uso" in str(e):
            print(i18n("最近查看或下载此文件的用户过多"))
            yield i18n("最近查看或下载此文件的用户过多")   
        elif "link privado" in str(e):
            print(i18n("无法从该私人链接获取文件"))
            yield i18n("无法从该私人链接获取文件")
        else:
            print(e)
            yield i18n("下载模型时发生错误。")
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
        
        infos.append(i18n("保存模型..."))
        yield "\n".join(infos)
        
        # Si no existe el folder RVC para guardar los modelos
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(os.path.join(save_folder, 'ManualTrainingBackup')):
            os.mkdir(os.path.join(save_folder, 'ManualTrainingBackup'))
        if not os.path.exists(os.path.join(save_folder, 'Finished')):
            os.mkdir(os.path.join(save_folder, 'Finished'))

        # Si ya existe el folders zips borro su contenido por si acaso
        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
            
        os.mkdir(zips_path)
        added_file = glob.glob(os.path.join(logs_path, "added_*.index"))
        d_file = glob.glob(os.path.join(logs_path, "D_*.pth"))
        g_file = glob.glob(os.path.join(logs_path, "G_*.pth"))
        
        if save_action == i18n("选择模型保存方法"):
            raise Exception("No method choosen.")
        
        if save_action == i18n("保存所有"):
            print(i18n("保存所有"))
            save_folder = os.path.join(save_folder, 'ManualTrainingBackup')
            shutil.copytree(logs_path, dst)
        else:
            # Si no existe el folder donde se va a comprimir el modelo
            if not os.path.exists(dst):
                os.mkdir(dst)
            
        if save_action == i18n("保存 D 和 G"):
            print(i18n("保存 D 和 G"))
            save_folder = os.path.join(save_folder, 'ManualTrainingBackup')
            if len(d_file) > 0:
                shutil.copy(d_file[0], dst)
            if len(g_file) > 0:
                shutil.copy(g_file[0], dst)    
                
            if len(added_file) > 0:
                shutil.copy(added_file[0], dst)
            else:
                infos.append(i18n("保存时未编制索引..."))
                
        if save_action == i18n("保存声音"):
            print(i18n("保存声音"))
            save_folder = os.path.join(save_folder, 'Finished')
            if len(added_file) > 0:
                shutil.copy(added_file[0], dst)
            else:
                infos.append(i18n("保存时未编制索引..."))
                #raise gr.Error("¡No ha generado el archivo added_*.index!")
        
        yield "\n".join(infos)
        # Si no existe el archivo del modelo no copiarlo
        if not os.path.exists(weights_path):
            infos.append(i18n("无模型保存（PTH）"))
            #raise gr.Error("¡No ha generado el modelo pequeño!")
        else:
            shutil.copy(weights_path, dst)
        
        yield "\n".join(infos)
        infos.append("\n" + i18n("这可能需要几分钟时间，请稍候..."))
        yield "\n".join(infos)
        
        shutil.make_archive(os.path.join(zips_path,f"{modelname}"), 'zip', zips_path)
        shutil.move(os.path.join(zips_path,f"{modelname}.zip"), os.path.join(save_folder, f'{modelname}.zip'))
        
        shutil.rmtree(zips_path)
        #shutil.rmtree(zips_path)
        
        infos.append("\n" + i18n("正确存储模型"))
        yield "\n".join(infos)
        
    except Exception as e:
        print(e)
        if "No model found." in str(e):
            infos.append(i18n("您要保存的模型不存在，请确保输入的名称正确。"))
        else:
            infos.append(i18n("保存模型时发生错误"))
            
        yield "\n".join(infos)
    
def load_downloaded_backup(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    try:
        infos = []
        logs_folders = ['0_gt_wavs','1_16k_wavs','2a_f0','2b-f0nsf','3_feature256','3_feature768']
        zips_path = os.path.join(parent_path, 'zips')
        unzips_path = os.path.join(parent_path, 'logs')
        weights_path = os.path.join(parent_path, 'weights')
        logs_dir = ""
        
        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)

        os.mkdir(zips_path)
        
        download_file = download_from_url(url)
        if not download_file:
            print(i18n("无法下载模型。"))
            infos.append(i18n("无法下载模型。"))
            yield "\n".join(infos)
        elif download_file == "downloaded":
            print(i18n("模型下载成功。"))
            infos.append(i18n("模型下载成功。"))
            yield "\n".join(infos)
        elif download_file == "demasiado uso":
            raise Exception(i18n("最近查看或下载此文件的用户过多"))
        elif download_file == "link privado":
            raise Exception(i18n("无法从该私人链接获取文件"))
        
        # Descomprimir archivos descargados
        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path,filename)
              # zip_dir_name = os.path.splitext(filename)[0]
              # unzip_dir = os.path.join(parent_path,'logs')
                print(i18n("继续提取..."))
                infos.append(i18n("继续提取..."))
                shutil.unpack_archive(zipfile_path, unzips_path, 'zip')
                yield "\n".join(infos)
            else:
                print(i18n("解压缩出错。"))
                infos.append(i18n("解压缩出错。"))
                yield "\n".join(infos)
                
        result = ""
        for filename in os.listdir(unzips_path):
            if filename.endswith(".zip"):
                os.remove(filename)
        
        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        print(i18n("备份已成功上传。"))
        infos.append("\n" + i18n("备份已成功上传。"))
        yield "\n".join(infos)
        os.chdir(parent_path)    
        return result
    except Exception as e:
        os.chdir(parent_path)
        if "demasiado uso" in str(e):
            print(i18n("最近查看或下载此文件的用户过多"))
            yield i18n("最近查看或下载此文件的用户过多")
        elif "link privado" in str(e):
            print(i18n("无法从该私人链接获取文件"))
            yield i18n("无法从该私人链接获取文件") 
        else:
            print(e)
            yield i18n("下载模型时发生错误。")
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

def save_to_wav2(dropbox):
    file_path = dropbox.name
    target_path = os.path.join('./audios', os.path.basename(file_path))

    if os.path.exists(target_path):
        os.remove(target_path)
      # print('Replacing old dropdown file...')

    shutil.move(file_path, target_path)
    return target_path

def change_choices2():
    audio_paths=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('wav', 'mp3', 'flac', 'ogg', 'opus',
                'm4a', 'mp4', 'aac', 'alac', 'wma',
                'aiff', 'webm', 'ac3')):
            audio_paths.append(os.path.join('./audios',filename).replace('\\', '/'))
    return {"choices": sorted(audio_paths), "__type__": "update"}, {"__type__": "update"}

def get_models_by_name(modelname):
    url = "https://script.google.com/macros/s/AKfycbzyrdLZzUww9qbjxnbnI08budD4yxbmRPHkWbp3UEJ9h3Id5cnNNVg0UtfFAnqqX5Rr/exec"
    
    response = requests.post(url, json={
        'type': 'search_by_filename',
        'filename': unidecode(modelname.strip().lower())
    })

    response_json = response.json()
    models = response_json['ocurrences']
    
    result = []
    message = "Busqueda realizada"
    if len(models) == 0:
        message = "No se han encontrado resultados."
    else:
        message = f"Se han encontrado {len(models)} resultados para {modelname}"
        
    for i in range(20):
        if i  < len(models):
            urls = models[i].get('url')
            url = eval(urls)[0]
            name = str(models[i].get('name'))
            filename = str(models[i].get('filename')) if not name or name.strip() == "" else name
            # Nombre
            result.append(
                {
                    "visible": True,
                    "value": str("### ") + filename,
                    "__type__": "update",
                })
            # Url
            result.append(
                {
                    "visible": False,
                    "value": url,
                    "__type__": "update",
                })
            # Boton
            result.append({
                    "visible": True,
                    "__type__": "update",
                })
            
            # Linea separadora
            if i == len(models) - 1:
                result.append({
                        "visible": False,
                        "__type__": "update",
                })
            else:
                result.append({
                    "visible": True,
                    "__type__": "update",
                })
                
            # Row
            result.append(
                {
                    "visible": True,
                    "__type__": "update",
                })
        else:
            # Nombre
            result.append(
                {
                    "visible": False,
                    "__type__": "update",
                })
            # Url
            result.append(
                {
                    "visible": False,
                    "value": False,
                    "__type__": "update",
                })
            # Boton
            result.append({
                    "visible": False,
                    "__type__": "update",
                })
            # Linea
            result.append({
                    "visible": False,
                    "__type__": "update",
                })
            # Row
            result.append(
                {
                    "visible": False,
                    "__type__": "update",
                })
     # Result
    result.append(
        {
            "value": message,
            "__type__": "update",
        }
    )
    
    return result

def search_model():
    gr.Markdown(value="# Buscar un modelo")
    with gr.Row():
        model_name = gr.inputs.Textbox(lines=1, label="Término de búsqueda")
        search_model_button=gr.Button("Buscar modelo")
        
    models = []
    results = gr.Textbox(label="Resultado", value="", max_lines=20)
    with gr.Row(visible=False) as row1:
        l1 = gr.Markdown(value="", visible=False)
        l1_url = gr.Textbox("Label 1", visible=False)
        b1 = gr.Button("Cargar modelo", visible=False)
    
    mk1 = gr.Markdown(value="---", visible=False)
    b1.click(fn=load_downloaded_model, inputs=l1_url, outputs=results)
    
    with gr.Row(visible=False) as row2:
        l2 = gr.Markdown(value="", visible=False)
        l2_url = gr.Textbox("Label 1", visible=False)
        b2 = gr.Button("Cargar modelo", visible=False)
    
    mk2 = gr.Markdown(value="---", visible=False)
    b2.click(fn=load_downloaded_model, inputs=l2_url, outputs=results)
    
    with gr.Row(visible=False) as row3:
        l3 = gr.Markdown(value="", visible=False)
        l3_url = gr.Textbox("Label 1", visible=False)
        b3 = gr.Button("Cargar modelo", visible=False)
    
    mk3 = gr.Markdown(value="---", visible=False)
    b3.click(fn=load_downloaded_model, inputs=l3_url, outputs=results)
        
    with gr.Row(visible=False) as row4:
        l4 = gr.Markdown(value="", visible=False)
        l4_url = gr.Textbox("Label 1", visible=False)
        b4 = gr.Button("Cargar modelo", visible=False)
    mk4 = gr.Markdown(value="---", visible=False)    
    b4.click(fn=load_downloaded_model, inputs=l4_url, outputs=results)
    
    with gr.Row(visible=False) as row5:
        l5 = gr.Markdown(value="", visible=False)
        l5_url = gr.Textbox("Label 1", visible=False)
        b5 = gr.Button("Cargar modelo", visible=False)
    
    mk5 = gr.Markdown(value="---", visible=False) 
    b5.click(fn=load_downloaded_model, inputs=l5_url, outputs=results)
        
    with gr.Row(visible=False) as row6:
        l6 = gr.Markdown(value="", visible=False)
        l6_url = gr.Textbox("Label 1", visible=False)
        b6 = gr.Button("Cargar modelo", visible=False)

    mk6 = gr.Markdown(value="---", visible=False)        
    b6.click(fn=load_downloaded_model, inputs=l6_url, outputs=results)
        
    with gr.Row(visible=False) as row7:
        l7 = gr.Markdown(value="", visible=False)
        l7_url = gr.Textbox("Label 1", visible=False)
        b7 = gr.Button("Cargar modelo", visible=False)
    
    mk7 = gr.Markdown(value="---", visible=False)
    b7.click(fn=load_downloaded_model, inputs=l7_url, outputs=results)
        
    with gr.Row(visible=False) as row8:
        l8 = gr.Markdown(value="", visible=False)
        l8_url = gr.Textbox("Label 1", visible=False)
        b8 = gr.Button("Cargar modelo", visible=False)
    
    mk8 = gr.Markdown(value="---", visible=False)
    b8.click(fn=load_downloaded_model, inputs=l8_url, outputs=results)
        
    with gr.Row(visible=False) as row9:
        l9 = gr.Markdown(value="", visible=False)
        l9_url = gr.Textbox("Label 1", visible=False)
        b9 = gr.Button("Cargar modelo", visible=False)
        
    mk9 = gr.Markdown(value="---", visible=False)
    b9.click(fn=load_downloaded_model, inputs=l9_url, outputs=results)
        
    with gr.Row(visible=False) as row10:
        l10 = gr.Markdown(value="", visible=False)
        l10_url = gr.Textbox("Label 1", visible=False)
        b10 = gr.Button("Cargar modelo", visible=False)
    
    mk10 = gr.Markdown(value="---", visible=False) 
    b10.click(fn=load_downloaded_model, inputs=l10_url, outputs=results)
        
    with gr.Row(visible=False) as row11:
        l11 = gr.Markdown(value="", visible=False)
        l11_url = gr.Textbox("Label 1", visible=False)
        b11 = gr.Button("Cargar modelo", visible=False)
        
    mk11 = gr.Markdown(value="---", visible=False)
    b11.click(fn=load_downloaded_model, inputs=l11_url, outputs=results)
        
    with gr.Row(visible=False) as row12:
        l12 = gr.Markdown(value="", visible=False)
        l12_url = gr.Textbox("Label 1", visible=False)
        b12 = gr.Button("Cargar modelo", visible=False)

    mk12 = gr.Markdown(value="---", visible=False)        
    b12.click(fn=load_downloaded_model, inputs=l12_url, outputs=results)
        
    with gr.Row(visible=False) as row13:
        l13 = gr.Markdown(value="", visible=False)
        l13_url = gr.Textbox("Label 1", visible=False)
        b13 = gr.Button("Cargar modelo", visible=False)
    
    mk13 = gr.Markdown(value="---", visible=False)
    b13.click(fn=load_downloaded_model, inputs=l13_url, outputs=results)
    
    with gr.Row(visible=False) as row14:
        l14 = gr.Markdown(value="", visible=False)
        l14_url = gr.Textbox("Label 1", visible=False)
        b14 = gr.Button("Cargar modelo", visible=False)
    
    mk14 = gr.Markdown(value="---", visible=False)
    b14.click(fn=load_downloaded_model, inputs=l14_url, outputs=results)
    
    with gr.Row(visible=False) as row15:
        l15 = gr.Markdown(value="", visible=False)
        l15_url = gr.Textbox("Label 1", visible=False)
        b15 = gr.Button("Cargar modelo", visible=False)

    mk15 = gr.Markdown(value="---", visible=False)        
    b15.click(fn=load_downloaded_model, inputs=l15_url, outputs=results)
        
    with gr.Row(visible=False) as row16:
        l16 = gr.Markdown(value="", visible=False)
        l16_url = gr.Textbox("Label 1", visible=False)
        b16 = gr.Button("Cargar modelo", visible=False)
        
    mk16 = gr.Markdown(value="---", visible=False)
    b16.click(fn=load_downloaded_model, inputs=l16_url, outputs=results)
        
    with gr.Row(visible=False) as row17:
        l17 = gr.Markdown(value="", visible=False)
        l17_url = gr.Textbox("Label 1", visible=False)
        b17 = gr.Button("Cargar modelo", visible=False)
    
    mk17 = gr.Markdown(value="---", visible=False)    
    b17.click(fn=load_downloaded_model, inputs=l17_url, outputs=results)
        
    with gr.Row(visible=False) as row18:
        l18 = gr.Markdown(value="", visible=False)
        l18_url = gr.Textbox("Label 1", visible=False)
        b18 = gr.Button("Cargar modelo", visible=False)
        
    mk18 = gr.Markdown(value="---", visible=False)
    b18.click(fn=load_downloaded_model, inputs=l18_url, outputs=results)
        
    with gr.Row(visible=False) as row19:
        l19 = gr.Markdown(value="", visible=False)
        l19_url = gr.Textbox("Label 1", visible=False)
        b19 = gr.Button("Cargar modelo", visible=False)
        
    mk19 = gr.Markdown(value="---", visible=False)
    b19.click(fn=load_downloaded_model, inputs=l19_url, outputs=results)
        
    with gr.Row(visible=False) as row20:
        l20 = gr.Markdown(value="", visible=False)
        l20_url = gr.Textbox("Label 1", visible=False)
        b20 = gr.Button("Cargar modelo", visible=False)
    
    mk20 = gr.Markdown(value="---", visible=False)
    b20.click(fn=load_downloaded_model, inputs=l20_url, outputs=results)
    
    #   to_return_protect1 = 

    search_model_button.click(fn=get_models_by_name, inputs=model_name, outputs=[l1,l1_url, b1, mk1, row1,
                                                                                 l2,l2_url, b2, mk2, row2,
                                                                                 l3,l3_url, b3, mk3, row3,
                                                                                 l4,l4_url, b4, mk4, row4,
                                                                                 l5,l5_url, b5, mk5, row5,
                                                                                 l6,l6_url, b6, mk6, row6,
                                                                                 l7,l7_url, b7, mk7, row7,
                                                                                 l8,l8_url, b8, mk8, row8,
                                                                                 l9,l9_url, b9, mk9, row9,
                                                                                 l10,l10_url, b10, mk10, row10,
                                                                                 l11,l11_url, b11, mk11, row11,
                                                                                 l12,l12_url, b12, mk12, row12,
                                                                                 l13,l13_url, b13, mk13, row13,
                                                                                 l14,l14_url, b14, mk14, row14,
                                                                                 l15,l15_url, b15, mk15, row15,
                                                                                 l16,l16_url, b16, mk16, row16,
                                                                                 l17,l17_url, b17, mk17, row17,
                                                                                 l18,l18_url, b18, mk18, row18,
                                                                                 l19,l19_url, b19, mk19, row19,
                                                                                 l20,l20_url, b20, mk20, row20,
                                                                                 results
                                                                                 ])
    

def descargar_desde_drive(url, name, output_file):

    print(f"Descargando {name} de drive")
    
    try:
        downloaded_file = gdown.download(url, output=output_file, fuzzy=True)
        return downloaded_file
    except:
        print("El intento de descargar con drive no funcionó")
        return None

def descargar_desde_mega(url, name):
  response = False
  try:
    file_id = None

    if "#!" in url:
      file_id = url.split("#!")[1].split("!")[0]
    elif "file/" in url:
      file_id = url.split("file/")[1].split("/")[0]
    else:
      file_id = None

    if file_id:
      mega = Mega()
      m = mega.login()

      print(f"Descargando {name} de mega")
      downloaded_file = m.download_url(url)

      return downloaded_file
    else:
      return None

  except Exception as e:
    print("Error**")
    print(e)
    return None

def descargar_desde_url_basica(url, name, output_file):
  try:
    print(f"Descargando {name} de URL BASICA")
    filename = wget.download(url=url, out=output_file)
    return filename
  except Exception as e:
     print(f"Error al descargar el archivo: {str(e)}")

def is_valid_model(name):
  parent_path = find_folder_parent(".", "pretrained_v2")
  unzips_path = os.path.join(parent_path, 'unzips')
  
  response = []
  file_path = os.path.join(unzips_path, name)

  has_model = False
  has_index = False

  for root, subfolders, files in os.walk(file_path):
    for file in files:
      current_file_path = os.path.join(root, file)
      if not file.startswith("G_") and not file.startswith("D_") and file.endswith(".pth") and not "_G_" in file and not "_D_" in file:
        has_model = True
      if file.startswith('added_') and file.endswith('.index'):
        has_index = True

  #if has_model and has_index:
  if has_index:
    response.append(".index")

  if has_model:
    response.append(".pth")

  return response


def create_zip(new_name):
    
    parent_path = find_folder_parent(".", "pretrained_v2")
    temp_folder_path = os.path.join(parent_path, 'temp_models')
    unzips_path = os.path.join(parent_path, 'unzips')
    zips_path = os.path.join(parent_path, 'zips')
  
    file_path = os.path.join(unzips_path, new_name)
    file_name = os.path.join(temp_folder_path, new_name)

    if not os.path.exists(zips_path):
        os.mkdir(zips_path)

    if os.path.exists(file_name):
        shutil.rmtree(file_name)

    os.mkdir(file_name)

    while not os.path.exists(file_name):
        time.sleep(1)

    for root, subfolders, files in os.walk(file_path):
        for file in files:
            current_file_path = os.path.join(root, file)
            if not file.startswith("G_") and not file.startswith("D_") and file.endswith(".pth") and not "_G_" in file and not "_D_" in file:
                print(f'Copiando {current_file_path} a {os.path.join(temp_folder_path, new_name)}')
                shutil.copy(current_file_path, file_name)
            if file.startswith('added_') and file.endswith('.index'):
                print(f'Copiando {current_file_path} a {os.path.join(temp_folder_path, new_name)}')
                shutil.copy(current_file_path, file_name)

    print("Comprimiendo modelo")
    zip_path =  os.path.join(zips_path, new_name)
    
    print(f"Comprimiendo {file_name} en {zip_path}")
    shutil.make_archive(zip_path, 'zip', file_name)
  
def upload_to_huggingface(file_path, new_filename):
    api = HfApi()
    login(token="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")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=new_filename,
        repo_id="juuxn/RVCModels",
        repo_type="model",
    )
    return f"https://huggingface.co/juuxn/RVCModels/resolve/main/{new_filename}"


def publish_model_clicked(model_name, model_url, model_version, model_creator):
    
    web_service_url = "https://script.google.com/macros/s/AKfycbzyrdLZzUww9qbjxnbnI08budD4yxbmRPHkWbp3UEJ9h3Id5cnNNVg0UtfFAnqqX5Rr/exec"
    name = unidecode(model_name)
    new_name = unidecode(name.strip().replace(" ","_").replace("'",""))
    
    downloaded_path = ""
    url = model_url
    version = model_version
    creator = model_creator
    parent_path = find_folder_parent(".", "pretrained_v2")
    output_folder = os.path.join(parent_path, 'archivos_descargados')
    output_file = os.path.join(output_folder, f'{new_name}.zip')
    unzips_path = os.path.join(parent_path, 'unzips')
    zips_path = os.path.join(parent_path, 'zips')
    temp_folder_path = os.path.join(parent_path, 'temp_models')
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    
    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    os.mkdir(temp_folder_path)
    
    
    if url and 'drive.google.com' in url:
    # Descargar el elemento si la URL es de Google Drive
        downloaded_path = descargar_desde_drive(url, new_name, output_file)
    elif url and 'mega.nz' in url:
        downloaded_path = descargar_desde_mega(url, new_name, output_file)
    elif url and 'pixeldrain' in url:
        print("No se puede descargar de pixeldrain")
    else:
        downloaded_path = descargar_desde_url_basica(url, new_name, output_file)
        
    if not downloaded_path:
        print(f"No se pudo descargar: {name}")
    else:
        filename = name.strip().replace(" ","_")
        dst =f'{filename}.zip'
        shutil.unpack_archive(downloaded_path, os.path.join(unzips_path, filename))
        md5_hash = get_md5(os.path.join(unzips_path, filename))

        if not md5_hash:
            print("No tiene modelo pequeño")
            return

        md5_response_raw = requests.post(web_service_url, json={
            'type': 'check_md5',
            'md5_hash': md5_hash
        })
        
        md5_response = md5_response_raw.json()
        ok = md5_response["ok"]
        exists = md5_response["exists"]
        message = md5_response["message"]

        is_valid = is_valid_model(filename)
        
        if md5_hash and exists:
            print(f"El archivo ya se ha publicado en spreadsheet con md5: {md5_hash}")
            return f"El archivo ya se ha publicado con md5: {md5_hash}"
        
        if ".pth" in is_valid and not exists:

            create_zip(filename)
            huggingface_url = upload_to_huggingface(os.path.join(zips_path,dst), dst)
                
            response = requests.post(web_service_url, json={
        'type': 'save_model',
        'elements': [{
                'name': name,
                'filename': filename,
                'url': [huggingface_url],
                'version': version,
                'creator': creator,
                'md5_hash': md5_hash,
                'content': is_valid
            }]})
            
            response_data = response.json()
            ok = response_data["ok"]
            message = response_data["message"]

            print({
                'name': name,
                'filename': filename,
                'url': [huggingface_url],
                'version': version,
                'creator': creator,
                'md5_hash': md5_hash,
                'content': is_valid
            })   
            
            if ok:
                return f"El archivo se ha publicado con md5: {md5_hash}"
            else:
                print(message)
                return message         
            
        # Eliminar folder donde se decarga el modelo zip
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            
        # Eliminar folder de zips, donde se descomprimio el modelo descargado
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)
            
        # Eliminar folder donde se copiaron los archivos indispensables del modelo
        if os.path.exists(temp_folder_path):
            shutil.rmtree(temp_folder_path)
        
        # Eliminar folder donde se comprimio el modelo para enviarse a huggingface
        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
            
def publish_models():
    with gr.Column():
        gr.Markdown("# Publicar un modelo en la comunidad")
        gr.Markdown("El modelo se va a verificar antes de publicarse. Importante que contenga el archivo **.pth** del modelo para que no sea rechazado.")
        
        model_name = gr.inputs.Textbox(lines=1, label="Nombre descriptivo del modelo Ej: (Ben 10 [Latino] - RVC V2 - 250 Epoch)")
        url = gr.inputs.Textbox(lines=1, label="Enlace del modelo")
        moder_version = gr.Radio(
            label="Versión",
            choices=["RVC v1", "RVC v2"],
            value="RVC v1",
            interactive=True,
        )
        model_creator = gr.inputs.Textbox(lines=1, label="ID de discord del creador del modelo Ej: <@123455656>")
        publish_model_button=gr.Button("Publicar modelo")
        results = gr.Textbox(label="Resultado", value="", max_lines=20)
        
        publish_model_button.click(fn=publish_model_clicked, inputs=[model_name, url, moder_version, model_creator], outputs=results)

def download_model():
    gr.Markdown(value="# " + i18n("下载模型"))
    gr.Markdown(value=i18n("它用于下载您的推理模型。"))
    with gr.Row():
        model_url=gr.Textbox(label=i18n("网址"))
    with gr.Row():
        download_model_status_bar=gr.Textbox(label=i18n("地位"))
    with gr.Row():
        download_button=gr.Button(i18n("下载"))
        download_button.click(fn=load_downloaded_model, inputs=[model_url], outputs=[download_model_status_bar])

def download_backup():
    gr.Markdown(value="# " + i18n("下载备份"))
    gr.Markdown(value=i18n("它用于下载您的训练备份。"))
    with gr.Row():
        model_url=gr.Textbox(label=i18n("网址"))
    with gr.Row():
        download_model_status_bar=gr.Textbox(label=i18n("地位"))
    with gr.Row():
        download_button=gr.Button(i18n("下载"))
        download_button.click(fn=load_downloaded_backup, inputs=[model_url], outputs=[download_model_status_bar])

def update_dataset_list(name):
    new_datasets = []
    for foldername in os.listdir("./datasets"):
        if "." not in foldername:
            new_datasets.append(os.path.join(find_folder_parent(".","pretrained"),"datasets",foldername))
    return gr.Dropdown.update(choices=new_datasets)

def download_dataset(trainset_dir4):
    gr.Markdown(value="# " + i18n("下载数据集"))
    gr.Markdown(value=i18n("它用于下载您的数据集。"))
    with gr.Row():
        dataset_url=gr.Textbox(label=i18n("网址"))
    with gr.Row():
        load_dataset_status_bar=gr.Textbox(label=i18n("地位"))
    with gr.Row():
        load_dataset_button=gr.Button(i18n("下载"))
        load_dataset_button.click(fn=load_dowloaded_dataset, inputs=[dataset_url], outputs=[load_dataset_status_bar])
        load_dataset_status_bar.change(update_dataset_list, dataset_url, trainset_dir4)