import gc
import requests
import subprocess
import sys
import os, warnings, librosa
import soundfile as sf
import numpy as np
import torch
import json

folder = os.path.dirname(os.path.abspath(__file__))
folder = os.path.dirname(folder)
folder = os.path.dirname(folder)
folder = os.path.dirname(folder)
now_dir = os.path.dirname(folder)

import sys
sys.path.append(now_dir)

import lib.infer.infer_libs.uvr5_pack.mdx as mdx
branch = "https://github.com/NaJeongMo/Colab-for-MDX_B"

model_params = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json"
_Models = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
# _models = "https://pastebin.com/raw/jBzYB8vz"
_models = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"


file_folder = "Colab-for-MDX_B"
model_request = requests.get(_models).json()
model_ids = model_request["mdx_download_list"].values()
demucs_download_list = model_request["demucs_download_list"]

# Iterate through the keys and get the model names
model_ids_demucs_inpure = [name.split(":")[1].strip() for name in demucs_download_list.keys()]

# Remove duplicates by converting the list to a set and then back to a list
model_ids_demucs = list(set(model_ids_demucs_inpure))

# Remove some not working models
demucs_ids_to_delete = ["tasnet_extra", "tasnet", "light_extra", "light", "demucs_extra", "demucs", "demucs_unittest", "demucs48_hq", "repro_mdx_a_hybrid_only", "repro_mdx_a_time_only", "repro_mdx_a", "UVR Model"]

# Add some models that are not in the list
demucs_ids_to_add = ["SIG"]

# Add the new ID to the model_ids_demucs list

for demucs_ids_to_add in demucs_ids_to_add:
    if demucs_ids_to_add not in model_ids_demucs:
        model_ids_demucs.append(demucs_ids_to_add)

# If the ID is in the list of IDs to delete, remove it from the list of model_ids_demucs
for demucs_ids_to_delete in demucs_ids_to_delete:
    if demucs_ids_to_delete in model_ids_demucs:
        model_ids_demucs.remove(demucs_ids_to_delete)
        
#print(model_ids)
model_params = requests.get(model_params).json()
#Remove request for stem_naming
stem_naming = {
    "Vocals": "Instrumental",
    "Other": "Instruments",
    "Instrumental": "Vocals",
    "Drums": "Drumless",
    "Bass": "Bassless"
}


os.makedirs(f"{now_dir}/assets/uvr5_weights/MDX", exist_ok=True)

warnings.filterwarnings("ignore")
cpu = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def get_model_list():
    return model_ids

def get_demucs_model_list():
    return model_ids_demucs

def id_to_ptm(mkey):
    if mkey in model_ids:
        #print(mkey)
        mpath = f"{now_dir}/assets/uvr5_weights/MDX/{mkey}"
        if not os.path.exists(f'{now_dir}/assets/uvr5_weights/MDX/{mkey}'):
            print('Downloading model...',end=' ')
            subprocess.run(
                ["python", "-m", "wget", "-o", mpath, _Models+mkey]
            )
            print(f'saved to {mpath}')
            return mpath
        else:
            return mpath
    else:
        mpath = f'{now_dir}/assets/uvr5_weights/{mkey}'
        return mpath

def prepare_mdx(onnx,custom_param=False, dim_f=None, dim_t=None, n_fft=None, stem_name=None, compensation=None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if custom_param:
        assert not (dim_f is None or dim_t is None or n_fft is None or compensation is None), 'Custom parameter selected, but incomplete parameters are provided.'
        mdx_model = mdx.MDX_Model(
            device,
            dim_f = dim_f,
            dim_t = dim_t,
            n_fft = n_fft,
            stem_name=stem_name,
            compensation=compensation
        )
    else:
        model_hash = mdx.MDX.get_hash(onnx)
        if model_hash in model_params:
            mp = model_params.get(model_hash)
            mdx_model = mdx.MDX_Model(
                device,
                dim_f = mp["mdx_dim_f_set"],
                dim_t = 2**mp["mdx_dim_t_set"],
                n_fft = mp["mdx_n_fft_scale_set"],
                stem_name=mp["primary_stem"],
                compensation=compensation if not custom_param and compensation is not None else mp["compensate"]
            )
    return mdx_model

def run_mdx(onnx, mdx_model,filename, output_format='wav',diff=False,suffix=None,diff_suffix=None, denoise=False, m_threads=2):
    mdx_sess = mdx.MDX(onnx,mdx_model)
    print(f"Processing: {filename}")
    if filename.lower().endswith('.wav'):
        wave, sr = librosa.load(filename, mono=False, sr=44100)
    else:
        temp_wav = 'temp_audio.wav'
        subprocess.run(['ffmpeg', '-i', filename, '-ar', '44100', '-ac', '2', temp_wav])  # Convert to WAV format
        wave, sr = librosa.load(temp_wav, mono=False, sr=44100)
        os.remove(temp_wav)
    
    #wave, sr = librosa.load(filename,mono=False, sr=44100)
    # normalizing input wave gives better output
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak
    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (mdx_sess.process_wave(wave, m_threads))
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)
    # return to previous peak
    wave_processed *= peak

    stem_name = mdx_model.stem_name if suffix is None else suffix # use suffix if provided
    save_path = os.path.basename(os.path.splitext(filename)[0])
    #vocals_save_path = os.path.join(vocals_folder, f"{save_path}_{stem_name}.{output_format}")
    #instrumental_save_path = os.path.join(instrumental_folder, f"{save_path}_{stem_name}.{output_format}")
    save_path = f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.{output_format}"
    save_path = os.path.join(
            'assets',
            'audios',
            save_path
        )
    sf.write(
        save_path,
        wave_processed.T,
        sr
    )

    print(f'done, saved to: {save_path}')

    if diff:
        diff_stem_name = stem_naming.get(stem_name) if diff_suffix is None else diff_suffix # use suffix if provided
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        save_path = f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.{output_format}"
        save_path = os.path.join(
                'assets',
                'audios',
                'audio-others',
                save_path
            )
        sf.write(
            save_path,
            (-wave_processed.T*mdx_model.compensation)+wave.T,
            sr
        )
        print(f'invert done, saved to: {save_path}')
    del mdx_sess, wave_processed, wave
    gc.collect()

if __name__ == "__main__":
    print()
