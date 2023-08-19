import sys
from shutil import rmtree
import shutil
import json # Mangio fork using json for preset saving
import datetime
import unicodedata
from glob import glob1
from signal import SIGTERM
import os
now_dir = os.getcwd()
sys.path.append(now_dir)
import lib.globals.globals as rvc_globals
from LazyImport import lazyload

math = lazyload('math')

import traceback
import warnings
tensorlowest = lazyload('tensorlowest')
import faiss
ffmpeg = lazyload('ffmpeg')

np = lazyload("numpy")
torch = lazyload('torch')
re = lazyload('regex')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
from random import shuffle
from subprocess import Popen
import easy_infer
gr = lazyload("gradio")
SF = lazyload("soundfile")
SFWrite = SF.write
from config import Config
from fairseq import checkpoint_utils
from i18n import I18nAuto
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from infer_uvr5 import _audio_pre_, _audio_pre_new
from MDXNet import MDXNetDereverb
from my_utils import load_audio
from train.process_ckpt import change_info, extract_small_model, merge, show_info
from vc_infer_pipeline import VC
from sklearn.cluster import MiniBatchKMeans

import time
import threading

from shlex import quote as SQuote

RQuote = lambda val: SQuote(str(val))

tmp = os.path.join(now_dir, "TEMP")
runtime_dir = os.path.join(now_dir, "runtime/Lib/site-packages")
directories = ['logs', 'audios', 'datasets', 'weights']

rmtree(tmp, ignore_errors=True)
rmtree(os.path.join(runtime_dir, "infer_pack"), ignore_errors=True)
rmtree(os.path.join(runtime_dir, "uvr5_pack"), ignore_errors=True)

os.makedirs(tmp, exist_ok=True)
for folder in directories:
    os.makedirs(os.path.join(now_dir, folder), exist_ok=True)

os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)
try:
    file = open('csvdb/stop.csv', 'x')
    file.close()
except FileExistsError: pass

global DoFormant, Quefrency, Timbre

DoFormant = rvc_globals.DoFormant
Quefrency = rvc_globals.Quefrency
Timbre = rvc_globals.Timbre

config = Config()
i18n = I18nAuto()
i18n.print()
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

keywords = ["10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", 
            "70", "80", "90", "M4", "T4", "TITAN"]

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i).upper()
        if any(keyword in gpu_name for keyword in keywords):
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1e9 + 0.4))

gpu_info = "\n".join(gpu_infos) if if_gpu_ok and gpu_infos else i18n("很遗憾您这没有能用的显卡来支持您训练")
default_batch_size = min(mem) // 2 if if_gpu_ok and gpu_infos else 1
gpus = "-".join(i[0] for i in gpu_infos)

hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")
    hubert_model = models[0].to(config.device)
    
    if config.is_half:
        hubert_model = hubert_model.half()

    hubert_model.eval()

datasets_root = "datasets/"
datasets_name = i18n("数据集名称")
weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
fshift_root = "formantshiftcfg"
audio_root = "audios"

sup_audioext = {'wav', 'mp3', 'flac', 'ogg', 'opus',
                'm4a', 'mp4', 'aac', 'alac', 'wma',
                'aiff', 'webm', 'ac3'}

names        = [os.path.join(root, file)
               for root, _, files in os.walk(weight_root)
               for file in files
               if file.endswith((".pth", ".onnx"))]

indexes_list = [os.path.join(root, name)
               for root, _, files in os.walk(index_root, topdown=False) 
               for name in files 
               if name.endswith(".index") and "trained" not in name]

audio_paths  = [os.path.join(root, name)
               for root, _, files in os.walk(audio_root, topdown=False) 
               for name in files
               if name.endswith(tuple(sup_audioext))]

uvr5_names  = [name.replace(".pth", "") 
              for name in os.listdir(weight_uvr5_root) 
              if name.endswith(".pth") or "onnx" in name]

check_for_name = lambda: sorted(names)[0] if names else ''

def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(index_root)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]
    
    return indexes_list if indexes_list else ''

def get_fshift_presets():
    fshift_presets_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(fshift_root)
        for filename in filenames
        if filename.endswith(".txt")
    ]
    
    return fshift_presets_list if fshift_presets_list else ''

def vc_single(
    sid:               str,
    input_audio_path0: str,
    input_audio_path1: str,
    f0_up_key:         int,
    f0_file:           str,
    f0_method:         str,
    file_index:        str,
    file_index2:       str,
    index_rate:        float,
    filter_radius:     int,
    resample_sr:       int,
    rms_mix_rate:      float,
    protect:           float,
    crepe_hop_length:  int,
    f0_min:            int,
    note_min:          str,
    f0_max:            int,
    note_max:          str,
):
    global total_time
    total_time = 0
    start_time = time.time()
    global tgt_sr, net_g, vc, hubert_model, version
    if not input_audio_path0 and not input_audio_path1:
        return "You need to upload an audio", None

    if (not os.path.exists(input_audio_path0)) and (not os.path.exists(os.path.join(now_dir, input_audio_path0))):
        return "Audio was not properly selected or doesn't exist", None
    
    # This might be jank, but I'm trying to make sure this gets the right file...
    input_audio_path1 = input_audio_path1 or input_audio_path0
    print(f"\nStarting inference for '{os.path.basename(input_audio_path1)}'")
    print("-------------------")

    f0_up_key = int(f0_up_key)
    
    if rvc_globals.NotesOrHertz and f0_method != 'rmvpe':
        f0_min = note_to_hz(note_min) if note_min else 50
        f0_max = note_to_hz(note_max) if note_max else 1100
        print(f"Converted min pitch freq - {f0_min}\n"
              f"Converted max pitch freq - {f0_max}")
    else:
        f0_min = f0_min or 50
        f0_max = f0_max or 1100
    try:
        print(f"Attempting to load {input_audio_path1}....")
        audio = load_audio(input_audio_path1,
                           16000,
                           DoFormant=rvc_globals.DoFormant,
                           Quefrency=rvc_globals.Quefrency,
                           Timbre=rvc_globals.Timbre)
        
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
            
        times = [0, 0, 0]
        if not hubert_model:
            print("Loading HuBERT for the first time...")
            load_hubert()
        
        try:
            if_f0 = cpt.get("f0", 1)
        except NameError:
            message = "Model was not properly selected"
            print(message)
            return message, None
        
        file_index = (
            file_index.strip(" ").strip('"').strip("\n").strip('"').strip(" ").replace("trained", "added")
        ) if file_index != "" else file_index2
        
        try:
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                sid,
                audio,
                input_audio_path1,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                if_f0,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                version,
                protect,
                crepe_hop_length,
                f0_file=f0_file,
                f0_min=f0_min,
                f0_max=f0_max
            )
        except AssertionError:
            message = "Mismatching index version detected (v1 with v2, or v2 with v1)."
            print(message)
            return message, None
        except NameError:
            message = "RVC libraries are still loading. Please try again in a few seconds."
            print(message)
            return message, None
        
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
            
        index_info = "Using index:%s." % file_index if os.path.exists(file_index) else "Index not used."

        end_time = time.time()
        total_time = end_time - start_time

        return f"Success.\n {index_info}\nTime:\n npy:{times[0]}, f0:{times[1]}, infer:{times[2]}\nTotal Time: {total_time} seconds", (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    format1,
    crepe_hop_length,
    f0_min,
    note_min,
    f0_max,
    note_max,
):
    if rvc_globals.NotesOrHertz and f0_method != 'rmvpe':
        f0_min = note_to_hz(note_min) if note_min else 50
        f0_max = note_to_hz(note_max) if note_max else 1100
        print(f"Converted min pitch freq - {f0_min}\n"
              f"Converted max pitch freq - {f0_max}")
    else:
        f0_min = f0_min or 50
        f0_max = f0_max or 1100

    try:
        dir_path, opt_root = [x.strip(" ").strip('"').strip("\n").strip('"').strip(" ") for x in [dir_path, opt_root]]
        os.makedirs(opt_root, exist_ok=True)
        
        paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)] if dir_path else [path.name for path in paths]
        infos = []

        for path in paths:
            info, opt = vc_single(sid, path, None, f0_up_key, None, f0_method, file_index, file_index2, index_rate, filter_radius,
                                  resample_sr, rms_mix_rate, protect, crepe_hop_length, f0_min, note_min, f0_max, note_max)

            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    #sys.stdout.write(f"\nTarget Sample Rate (tgt_sr): {tgt_sr}") # Debugging print
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    output_path = f"{opt_root}/{base_name}.{format1}"
                    path, extension = output_path, format1
                    path, extension = output_path if format1 in ["wav", "flac", "mp3", "ogg", "aac", "m4a"] else f"{output_path}.wav", format1
                    #sys.stdout.write(f"\nOutput Path: {path}") # Debugging print
                    #sys.stdout.write(f"\nFile Extension: {extension}") # Debugging print
                    SFWrite(path, audio_opt, tgt_sr)
                    #sys.stdout.write("\nFile Written Successfully with SFWrite") # Debugging print
                    if os.path.exists(path) and extension not in ["wav", "flac", "mp3", "ogg", "aac", "m4a"]:
                        sys.stdout.write(f"Running command: ffmpeg -i {RQuote(path)} -vn {RQuote(path[:-4] + '.' + extension)} -q:a 2 -y")
                        os.system(f"ffmpeg -i {RQuote(path)} -vn {RQuote(path[:-4] + '.' + extension)} -q:a 2 -y")
                        #print(f"\nFile Converted to {extension} using ffmpeg") # Debugging print
                except:
                    info += traceback.format_exc()
                    print(f"\nException encountered: {info}") # Debugging print
            infos.append(f"{os.path.basename(path)}->{info}")
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root, save_root_vocal, save_root_ins = [x.strip(" ").strip('"').strip("\n").strip('"').strip(" ") for x in [inp_root, save_root_vocal, save_root_ins]]
        
        pre_fun = MDXNetDereverb(15) if model_name == "onnx_dereverb_By_FoxJoy" else (_audio_pre_ if "DeEcho" not in model_name else _audio_pre_new)(
                    agg=int(agg),
                    model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                    device=config.device,
                    is_half=config.is_half,
                )
                
        paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)] if inp_root else [path.name for path in paths]

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
                tmp_path = f"{tmp}/{os.path.basename(RQuote(inp_path))}.reformatted.wav"
                os.system(f"ffmpeg -i {RQuote(inp_path)} -vn -acodec pcm_s16le -ac 2 -ar 44100 {RQuote(tmp_path)} -y")
                inp_path = tmp_path

            try:
                if not done:
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
                infos.append(f"{os.path.basename(inp_path)}->Success")
                yield "\n".join(infos)
            except:
                infos.append(f"{os.path.basename(inp_path)}->{traceback.format_exc()}")
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model

            del pre_fun
        except: traceback.print_exc()

        print("clean_empty_cache")

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    yield "\n".join(infos)

def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version, hubert_model
    if not sid:
        if hubert_model is not None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0, version = cpt.get("f0", 1), cpt.get("version", "v1")
            net_g = (SynthesizerTrnMs256NSFsid if version == "v1" else SynthesizerTrnMs768NSFsid)(
                *cpt["config"], is_half=config.is_half) if if_f0 == 1 else (SynthesizerTrnMs256NSFsid_nono if version == "v1" else SynthesizerTrnMs768NSFsid_nono)(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return ({"visible": False, "__type__": "update"},) * 3

    print(f"loading {sid}")
    cpt = torch.load(sid, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    if cpt.get("f0", 1) == 0:
        to_return_protect0 = to_return_protect1 = {"visible": False, "value": 0.5, "__type__": "update"}
    else:
        to_return_protect0 = {"visible": True, "value": to_return_protect0, "__type__": "update"}
        to_return_protect1 = {"visible": True, "value": to_return_protect1, "__type__": "update"}

    version = cpt.get("version", "v1")
    net_g = (SynthesizerTrnMs256NSFsid if version == "v1" else SynthesizerTrnMs768NSFsid)(
        *cpt["config"], is_half=config.is_half) if cpt.get("f0", 1) == 1 else (SynthesizerTrnMs256NSFsid_nono if version == "v1" else SynthesizerTrnMs768NSFsid_nono)(*cpt["config"])
    del net_g.enc_q

    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    return (
        {"visible": False, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1
    )


def change_choices():
    names        = [os.path.join(root, file)
                   for root, _, files in os.walk(weight_root)
                   for file in files
                   if file.endswith((".pth", ".onnx"))]
    indexes_list = [os.path.join(root, name) for root, _, files in os.walk(index_root, topdown=False) for name in files if name.endswith(".index") and "trained" not in name]
    audio_paths  = [os.path.join(audio_root, file) for file in os.listdir(os.path.join(now_dir, "audios"))]

    return (
        {"choices": sorted(names), "__type__": "update"}, 
        {"choices": sorted(indexes_list), "__type__": "update"}, 
        {"choices": sorted(audio_paths), "__type__": "update"}
    )

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def if_done(done, p):
    while p.poll() is None:
        time.sleep(0.5)

    done[0] = True

def if_done_multi(done, ps):
    while not all(p.poll() is not None for p in ps):
        time.sleep(0.5)
    done[0] = True

def formant_enabled(cbox, qfrency, tmbre):
    global DoFormant, Quefrency, Timbre

    DoFormant = cbox
    Quefrency = qfrency
    Timbre = tmbre

    rvc_globals.DoFormant = cbox
    rvc_globals.Quefrency = qfrency
    rvc_globals.Timbre = tmbre

    visibility_update = {"visible": DoFormant, "__type__": "update"}

    return (
        {"value": DoFormant, "__type__": "update"},
    ) + (visibility_update,) * 6
        

def formant_apply(qfrency, tmbre):
    global Quefrency, Timbre, DoFormant

    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True

    rvc_globals.DoFormant = True
    rvc_globals.Quefrency = qfrency
    rvc_globals.Timbre = tmbre

    return ({"value": Quefrency, "__type__": "update"}, {"value": Timbre, "__type__": "update"})

def update_fshift_presets(preset, qfrency, tmbre):

    if preset:  
        with open(preset, 'r') as p:
            content = p.readlines()
            qfrency, tmbre = content[0].strip(), content[1]
            
        formant_apply(qfrency, tmbre)
    else:
        qfrency, tmbre = preset_apply(preset, qfrency, tmbre)
        
    return (
        {"choices": get_fshift_presets(), "__type__": "update"},
        {"value": qfrency, "__type__": "update"},
        {"value": tmbre, "__type__": "update"},
    )

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    
    log_dir = os.path.join(now_dir, "logs", exp_dir)
    log_file = os.path.join(log_dir, "preprocess.log")
    
    os.makedirs(log_dir, exist_ok=True)

    with open(log_file, "w") as f: pass

    cmd = (
        f"{config.python_cmd} "
        "trainset_preprocess_pipeline_print.py "
        f"{trainset_dir} "
        f"{RQuote(sr)} "
        f"{RQuote(n_p)} "
        f"{log_dir} "
        f"{RQuote(config.noparallel)}"
    )
    print(cmd)

    p = Popen(cmd, shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done,p,)).start()

    while not done[0]:
        with open(log_file, "r") as f:
            yield f.read()
        time.sleep(1)
   
    with open(log_file, "r") as f:
        log = f.read()
    
    print(log)
    yield log

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, echl):
    gpus = gpus.split("-")
    log_dir = f"{now_dir}/logs/{exp_dir}"
    log_file = f"{log_dir}/extract_f0_feature.log"
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "w") as f: pass

    if if_f0:
        cmd = (
            f"{config.python_cmd} extract_f0_print.py {log_dir} " 
            f"{RQuote(n_p)} {RQuote(f0method)} {RQuote(echl)}"
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        done = [False]
        threading.Thread(target=if_done, args=(done, p)).start()

        while not done[0]:
            with open(log_file, "r") as f:
                yield f.read()
            time.sleep(1)

    leng = len(gpus)
    ps = []

    for idx, n_g in enumerate(gpus):
        cmd = (
            f"{config.python_cmd} extract_feature_print.py {RQuote(config.device)} "
            f"{RQuote(leng)} {RQuote(idx)} {RQuote(n_g)} {log_dir} {RQuote(version19)}"
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)

    done = [False]
    threading.Thread(target=if_done_multi, args=(done, ps)).start()

    while not done[0]:
        with open(log_file, "r") as f:
            yield f.read()
        time.sleep(1)
    
    with open(log_file, "r") as f:
        log = f.read()

    print(log)
    yield log

def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    model_paths = {"G": "", "D": ""}

    for model_type in model_paths:
        file_path = f"pretrained{path_str}/{f0_str}{model_type}{sr2}.pth"
        if os.access(file_path, os.F_OK):
            model_paths[model_type] = file_path
        else:
            print(f"{file_path} doesn't exist, will not use pretrained model.")
    
    return (model_paths["G"], model_paths["D"])


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    sr2 = "40k" if (sr2 == "32k" and version19 == "v1") else sr2
    choices_update = {
        "choices": ["40k", "48k"], "__type__": "update", "value": sr2
        } if version19 == "v1" else {
            "choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}

    f0_str = "f0" if if_f0_3 else ""
    model_paths = {"G": "", "D": ""}

    for model_type in model_paths:
        file_path = f"pretrained{path_str}/{f0_str}{model_type}{sr2}.pth"
        if os.access(file_path, os.F_OK):
            model_paths[model_type] = file_path
        else:
            print(f"{file_path} doesn't exist, will not use pretrained model.")

    return (model_paths["G"], model_paths["D"], choices_update)


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    
    pth_format = "pretrained%s/f0%s%s.pth"
    model_desc = { "G": "", "D": "" }
    
    for model_type in model_desc:
        file_path = pth_format % (path_str, model_type, sr2)
        if os.access(file_path, os.F_OK):
            model_desc[model_type] = file_path
        else:
            print(file_path, "doesn't exist, will not use pretrained model")

    return (
        {"visible": if_f0_3, "__type__": "update"},
        model_desc["G"],
        model_desc["D"],
        {"visible": if_f0_3, "__type__": "update"}
    )


global log_interval

def set_log_interval(exp_dir, batch_size12):
    log_interval = 1
    folder_path = os.path.join(exp_dir, "1_16k_wavs")

    if os.path.isdir(folder_path):
        wav_files_num = len(glob1(folder_path,"*.wav"))

        if wav_files_num > 0:
            log_interval = math.ceil(wav_files_num / batch_size12)
            if log_interval > 1:
                log_interval += 1

    return log_interval

global PID, PROCESS

def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    with open('csvdb/stop.csv', 'w+') as file: file.write("False")
    log_dir = os.path.join(now_dir, "logs", exp_dir1)
    
    os.makedirs(log_dir, exist_ok=True)

    gt_wavs_dir = os.path.join(log_dir, "0_gt_wavs")
    feature_dim = "256" if version19 == "v1" else "768"

    feature_dir = os.path.join(log_dir, f"3_feature{feature_dim}")

    log_interval = set_log_interval(log_dir, batch_size12)

    required_dirs = [gt_wavs_dir, feature_dir]
    
    if if_f0_3:
        f0_dir = f"{log_dir}/2a_f0"
        f0nsf_dir = f"{log_dir}/2b-f0nsf"
        required_dirs.extend([f0_dir, f0nsf_dir])

    names = set(name.split(".")[0] for directory in required_dirs for name in os.listdir(directory))

    def generate_paths(name):
        paths = [gt_wavs_dir, feature_dir]
        if if_f0_3:
            paths.extend([f0_dir, f0nsf_dir])
        return '|'.join([path.replace('\\', '\\\\') + '/' + name + ('.wav.npy' if path in [f0_dir, f0nsf_dir] else '.wav' if path == gt_wavs_dir else '.npy') for path in paths])

    opt = [f"{generate_paths(name)}|{spk_id5}" for name in names]
    mute_dir = f"{now_dir}/logs/mute"
    
    for _ in range(2):
        mute_string = f"{mute_dir}/0_gt_wavs/mute{sr2}.wav|{mute_dir}/3_feature{feature_dim}/mute.npy"
        if if_f0_3:
            mute_string += f"|{mute_dir}/2a_f0/mute.wav.npy|{mute_dir}/2b-f0nsf/mute.wav.npy"
        opt.append(mute_string+f"|{spk_id5}")

    shuffle(opt)
    with open(f"{log_dir}/filelist.txt", "w") as f:
        f.write("\n".join(opt))

    print("write filelist done")
    print("use gpus:", gpus16)

    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")

    G_train = f"-pg {pretrained_G14}" if pretrained_G14 else ""
    D_train = f"-pd {pretrained_D15}" if pretrained_D15 else ""
    
    cmd = (
        f"{config.python_cmd} train_nsf_sim_cache_sid_load_pretrain.py -e {exp_dir1} -sr {sr2} -f0 {int(if_f0_3)} -bs {batch_size12}"
        f" -g {gpus16 if gpus16 is not None else ''} -te {total_epoch11} -se {save_epoch10} {G_train} {D_train} -l {int(if_save_latest13)}"
        f" -c {int(if_cache_gpu17)} -sw {int(if_save_every_weights18)} -v {version19} -li {log_interval}"
    )

    print(cmd)

    global p
    p = Popen(cmd, shell=True, cwd=now_dir)
    global PID
    PID = p.pid

    p.wait()

    return "Training is done, check train.log", {"visible": False, "__type__": "update"}, {"visible": True, "__type__": "update"}

def train_index(exp_dir1, version19):
    exp_dir = os.path.join(now_dir, 'logs', exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)

    feature_dim = '256' if version19 == "v1" else '768'
    feature_dir = os.path.join(exp_dir, f"3_feature{feature_dim}")

    if not os.path.exists(feature_dir) or len(os.listdir(feature_dir)) == 0:
        return "请先进行特征提取!"

    npys = [np.load(os.path.join(feature_dir, name)) for name in sorted(os.listdir(feature_dir))]
            
    big_npy = np.concatenate(npys, 0)
    np.random.shuffle(big_npy)

    infos = []
    if big_npy.shape[0] > 2*10**5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * config.n_cpu, 
                                      compute_labels=False,init="random").fit(big_npy).cluster_centers_
        except Exception as e:
            infos.append(str(e))
            yield "\n".join(infos)

    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)

    index = faiss.index_factory(int(feature_dim), f"IVF{n_ivf},Flat")

    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    index.train(big_npy)

    index_file_base = f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    faiss.write_index(index, index_file_base)

    infos.append("adding")
    yield "\n".join(infos)

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i:i + batch_size_add])
    
    index_file_base = f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    faiss.write_index(index, index_file_base)

    infos.append(f"Successful Index Construction，added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index")
    yield "\n".join(infos)

#def setBoolean(status): #true to false and vice versa / not implemented yet, dont touch!!!!!!!
#    status = not status
#    return status

def change_info_(ckpt_path):
    train_log_path = os.path.join(os.path.dirname(ckpt_path), "train.log")
    
    if not os.path.exists(train_log_path):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

    try:
        with open(train_log_path, "r") as f:
            info_line = next(f).strip()
            info = eval(info_line.split("\t")[-1])
            
            sr, f0 = info.get("sample_rate"), info.get("if_f0")
            version = "v2" if info.get("version") == "v2" else "v1"

            return sr, str(f0), version

    except Exception as e:
        print(f"Exception occurred: {str(e)}, Traceback: {traceback.format_exc()}")
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

def export_onnx(model_path, exported_path):
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    vec_channels = 256 if checkpoint.get("version", "v1") == "v1" else 768
    
    test_inputs = {
        "phone": torch.rand(1, 200, vec_channels),
        "phone_lengths": torch.LongTensor([200]),
        "pitch": torch.randint(5, 255, (1, 200)),
        "pitchf": torch.rand(1, 200),
        "ds": torch.zeros(1).long(),
        "rnd": torch.rand(1, 192, 200)
    }
    
    checkpoint["config"][-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
    net_g = SynthesizerTrnMsNSFsidM(*checkpoint["config"], is_half=False, version=checkpoint.get("version", "v1"))
    
    net_g.load_state_dict(checkpoint["weight"], strict=False)
    net_g = net_g.to(device)

    dynamic_axes = {"phone": [1], "pitch": [1], "pitchf": [1], "rnd": [2]}

    torch.onnx.export(
        net_g,
        tuple(value.to(device) for value in test_inputs.values()),
        exported_path,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=list(test_inputs.keys()),
        output_names=["audio"],
    )
    return "Finished"


#region Mangio-RVC-Fork CLI App

import scipy.io.wavfile as wavfile

cli_current_page = "HOME"

def cli_split_command(com):
    exp = r'(?:(?<=\s)|^)"(.*?)"(?=\s|$)|(\S+)'
    split_array = re.findall(exp, com)
    split_array = [group[0] if group[0] else group[1] for group in split_array]
    return split_array

execute_generator_function = lambda genObject: all(x is not None for x in genObject)

def cli_infer(com):
    model_name, source_audio_path, output_file_name, feature_index_path, speaker_id, transposition, f0_method, crepe_hop_length, harvest_median_filter, resample, mix, feature_ratio, protection_amnt, _, f0_min, f0_max, do_formant = cli_split_command(com)[:17]

    speaker_id, crepe_hop_length, harvest_median_filter, resample = map(int, [speaker_id, crepe_hop_length, harvest_median_filter, resample])
    transposition, mix, feature_ratio, protection_amnt = map(float, [transposition, mix, feature_ratio, protection_amnt])

    if do_formant.lower() == 'false':
        Quefrency = 1.0
        Timbre = 1.0
    else:
        Quefrency, Timbre = map(float, cli_split_command(com)[17:19])

    rvc_globals.DoFormant = do_formant.lower() == 'true'
    rvc_globals.Quefrency = Quefrency
    rvc_globals.Timbre = Timbre

    output_message = 'Mangio-RVC-Fork Infer-CLI:'
    output_path = f'audio-outputs/{output_file_name}'
    
    print(f"{output_message} Starting the inference...")
    vc_data = get_vc(model_name, protection_amnt, protection_amnt)
    print(vc_data)

    print(f"{output_message} Performing inference...")
    conversion_data = vc_single(
        speaker_id,
        source_audio_path,
        source_audio_path,
        transposition,
        None, # f0 file support not implemented
        f0_method,
        feature_index_path,
        feature_index_path,
        feature_ratio,
        harvest_median_filter,
        resample,
        mix,
        protection_amnt,
        crepe_hop_length,
        f0_min=f0_min,
        note_min=None,
        f0_max=f0_max,
        note_max=None
    )

    if "Success." in conversion_data[0]:
        print(f"{output_message} Inference succeeded. Writing to {output_path}...")
        wavfile.write(output_path, conversion_data[1][0], conversion_data[1][1])
        print(f"{output_message} Finished! Saved output to {output_path}")
    else:
        print(f"{output_message} Inference failed. Here's the traceback: {conversion_data[0]}")
        
def cli_pre_process(com):
    print("Mangio-RVC-Fork Pre-process: Starting...")
    execute_generator_function(
        preprocess_dataset(
            *cli_split_command(com)[:3],
            int(cli_split_command(com)[3])
        )
    )
    print("Mangio-RVC-Fork Pre-process: Finished")

def cli_extract_feature(com):
    model_name, gpus, num_processes, has_pitch_guidance, f0_method, crepe_hop_length, version = cli_split_command(com)

    num_processes = int(num_processes)
    has_pitch_guidance = bool(int(has_pitch_guidance)) 
    crepe_hop_length = int(crepe_hop_length)

    print(
        f"Mangio-RVC-CLI: Extract Feature Has Pitch: {has_pitch_guidance}"
        f"Mangio-RVC-CLI: Extract Feature Version: {version}"
        "Mangio-RVC-Fork Feature Extraction: Starting..."
    )
    generator = extract_f0_feature(
        gpus, 
        num_processes, 
        f0_method, 
        has_pitch_guidance, 
        model_name, 
        version, 
        crepe_hop_length
    )
    execute_generator_function(generator)
    print("Mangio-RVC-Fork Feature Extraction: Finished")

def cli_train(com):
    com = cli_split_command(com)
    model_name = com[0]
    sample_rate = com[1]
    bool_flags = [bool(int(i)) for i in com[2:11]]
    version = com[11]

    pretrained_base = "pretrained/" if version == "v1" else "pretrained_v2/"
    
    g_pretrained_path = f"{pretrained_base}f0G{sample_rate}.pth"
    d_pretrained_path = f"{pretrained_base}f0D{sample_rate}.pth"

    print("Mangio-RVC-Fork Train-CLI: Training...")
    click_train(model_name, sample_rate, *bool_flags, g_pretrained_path, d_pretrained_path, version)

def cli_train_feature(com):
    output_message = 'Mangio-RVC-Fork Train Feature Index-CLI'
    print(f"{output_message}: Training... Please wait")
    execute_generator_function(train_index(*cli_split_command(com)))
    print(f"{output_message}: Done!")

def cli_extract_model(com):
    extract_small_model_process = extract_small_model(*cli_split_command(com))
    print(
        "Mangio-RVC-Fork Extract Small Model: Success!" 
        if extract_small_model_process == "Success." 
        else f"{extract_small_model_process}\nMangio-RVC-Fork Extract Small Model: Failed!"
    )

def preset_apply(preset, qfer, tmbr):
    if preset:
        try:
            with open(preset, 'r') as p:
                content = p.read().splitlines()  
            qfer, tmbr = content[0], content[1]
            formant_apply(qfer, tmbr)
        except IndexError:
            print("Error: File does not have enough lines to read 'qfer' and 'tmbr'")
        except FileNotFoundError:
            print("Error: File does not exist")
        except Exception as e: 
            print("An unexpected error occurred", e)

    return ({"value": qfer, "__type__": "update"}, {"value": tmbr, "__type__": "update"})

def print_page_details():
    page_description = {

        'HOME':
            "\n    go home            : Takes you back to home with a navigation list."
            "\n    go infer           : Takes you to inference command execution."
            "\n    go pre-process     : Takes you to training step.1) pre-process command execution."
            "\n    go extract-feature : Takes you to training step.2) extract-feature command execution."
            "\n    go train           : Takes you to training step.3) being or continue training command execution."
            "\n    go train-feature   : Takes you to the train feature index command execution."
            "\n    go extract-model   : Takes you to the extract small model command execution."

        , 'INFER': 
            "\n    arg 1) model name with .pth in ./weights: mi-test.pth"
            "\n    arg 2) source audio path: myFolder\\MySource.wav"
            "\n    arg 3) output file name to be placed in './audio-outputs': MyTest.wav"
            "\n    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index"
            "\n    arg 5) speaker id: 0"
            "\n    arg 6) transposition: 0"
            "\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe)"
            "\n    arg 8) crepe hop length: 160"
            "\n    arg 9) harvest median filter radius: 3 (0-7)"
            "\n    arg 10) post resample rate: 0"
            "\n    arg 11) mix volume envelope: 1"
            "\n    arg 12) feature index ratio: 0.78 (0-1)"
            "\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)"
            "\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)"
            "\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)"
            "\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false) \n"
            "\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2"

        , 'PRE-PROCESS':
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set"
            "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
            "\n    arg 4) Number of CPU threads to use: 8 \n"
            "\nExample: mi-test mydataset 40k 24"

        , 'EXTRACT-FEATURE':
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)"
            "\n    arg 3) Number of CPU threads to use: 8"
            "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
            "\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)"
            "\n    arg 6) Crepe hop length: 128"
            "\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n"
            "\nExample: mi-test 0 24 1 harvest 128 v2"

        , 'TRAIN':
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Sample rate: 40k (32k, 40k, 48k)"
            "\n    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
            "\n    arg 4) speaker id: 0"
            "\n    arg 5) Save epoch iteration: 50"
            "\n    arg 6) Total epochs: 10000"
            "\n    arg 7) Batch size: 8"
            "\n    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)"
            "\n    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)"
            "\n    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)"
            "\n    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)"
            "\n    arg 12) Model architecture version: v2 (use either v1 or v2)\n"
            "\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2"

        , 'TRAIN-FEATURE':
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n"
            "\nExample: mi-test v2"

        , 'EXTRACT-MODEL':
            "\n    arg 1) Model Path: logs/mi-test/G_168000.pth"
            "\n    arg 2) Model save name: MyModel"
            "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
            "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
            '\n    arg 5) Model information: "My Model"'
            "\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n"
            '\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2'

    }
    
    print(page_description.get(cli_current_page, 'Invalid page'))


def change_page(page):
    global cli_current_page
    cli_current_page = page
    return 0
def execute_command(com):
    command_to_page = {
        "go home": "HOME",
        "go infer": "INFER",
        "go pre-process": "PRE-PROCESS",
        "go extract-feature": "EXTRACT-FEATURE",
        "go train": "TRAIN",
        "go train-feature": "TRAIN-FEATURE",
        "go extract-model": "EXTRACT-MODEL",
    }
    
    page_to_function = {
        "INFER": cli_infer,
        "PRE-PROCESS": cli_pre_process,
        "EXTRACT-FEATURE": cli_extract_feature,
        "TRAIN": cli_train,
        "TRAIN-FEATURE": cli_train_feature,
        "EXTRACT-MODEL": cli_extract_model,
    }

    if com in command_to_page:
        return change_page(command_to_page[com])
    
    if com[:3] == "go ":
        print(f"page '{com[3:]}' does not exist!")
        return 0

    if cli_current_page in page_to_function:
        page_to_function[cli_current_page](com)

def cli_navigation_loop():
    while True:
        print(f"\nYou are currently in '{cli_current_page}':")
        print_page_details()
        print(f"{cli_current_page}: ", end="")
        try: execute_command(input())
        except Exception as e: print(f"An error occurred: {traceback.format_exc()}")

if(config.is_cli):
    print(
        "\n\nMangio-RVC-Fork v2 CLI App!\n"
        "Welcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.\n"
    )
    cli_navigation_loop()

#endregion

#region RVC WebUI App
'''
def get_presets():
    data = None
    with open('../inference-presets.json', 'r') as file:
        data = json.load(file)
    preset_names = []
    for preset in data['presets']:
        preset_names.append(preset['name'])
    
    return preset_names
'''

def switch_pitch_controls(f0method0):
    is_visible = f0method0 != 'rmvpe'

    if rvc_globals.NotesOrHertz:
        return (
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"}
        )
    else:
        return (
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"}
        )

def match_index(sid0: str) -> tuple:
    sid0strip = re.sub(r'\.pth|\.onnx$', '', sid0)
    sid0name = os.path.split(sid0strip)[-1]  # Extract only the name, not the directory

    # Check if the sid0strip has the specific ending format _eXXX_sXXX
    if re.match(r'.+_e\d+_s\d+$', sid0name):
        base_model_name = sid0name.rsplit('_', 2)[0]
    else:
        base_model_name = sid0name

    sid_directory = os.path.join(index_root, base_model_name)
    directories_to_search = [sid_directory] if os.path.exists(sid_directory) else []
    directories_to_search.append(index_root)

    matching_index_files = []

    for directory in directories_to_search:
        for filename in os.listdir(directory):
            if filename.endswith('.index') and 'trained' not in filename:
                # Condition to match the name
                name_match = any(name.lower() in filename.lower() for name in [sid0name, base_model_name])
                
                # If in the specific directory, it's automatically a match
                folder_match = directory == sid_directory

                if name_match or folder_match:
                    index_path = os.path.join(directory, filename)
                    if index_path in indexes_list:
                        matching_index_files.append((index_path, os.path.getsize(index_path), ' ' not in filename))

    if matching_index_files:
        # Sort by favoring files without spaces and by size (largest size first)
        matching_index_files.sort(key=lambda x: (-x[2], -x[1]))
        best_match_index_path = matching_index_files[0][0]
        return best_match_index_path, best_match_index_path

    return '', ''
def stoptraining(mim):
    if mim:
        try:
            with open('csvdb/stop.csv', 'w+') as file: file.write("True")
            os.kill(PID, SIGTERM)
        except Exception as e:
            print(f"Couldn't click due to {e}")
        return (
            {"visible": True , "__type__": "update"},
            {"visible": False, "__type__": "update"})
    return (
        {"visible": False, "__type__": "update"},
        {"visible": True , "__type__": "update"})

tab_faq = i18n("常见问题解答")
faq_file = "docs/faq.md" if tab_faq == "常见问题解答" else "docs/faq_en.md"
weights_dir = 'weights/'

def note_to_hz(note_name):
    SEMITONES = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    pitch_class, octave = note_name[:-1], int(note_name[-1])
    semitone = SEMITONES[pitch_class]
    note_number = 12 * (octave - 4) + semitone
    frequency = 440.0 * (2.0 ** (1.0/12)) ** note_number
    return frequency

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
    target_path = os.path.join('audios', os.path.basename(file_path))

    if os.path.exists(target_path):
        os.remove(target_path)
        print('Replacing old dropdown file...')

    shutil.move(file_path, target_path)
    return target_path
    
def change_choices2():
    return ""

def GradioSetup(UTheme=gr.themes.Soft()):

    default_weight = names[0] if names else '' # Set the first found weight as the preloaded model

    with gr.Blocks(theme='JohnSmith9982/small_and_pretty', title="Applio") as app:
        gr.HTML("<h1> 🍏 Applio (Mangio-RVC-Fork) </h1>")
        # gr.Markdown(
        #     value=i18n(
        #         "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>使用需遵守的协议-LICENSE.txt</b>."
        #     )
        #)
        with gr.Tabs():
            with gr.TabItem(i18n("模型推理")):
                with gr.Row():
                    sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names), value=default_weight)
                    refresh_button = gr.Button(i18n("刷新音色列表和索引路径"), variant="primary")
                    clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                    clean_button.click(fn=lambda: ({"value": "", "__type__": "update"}), inputs=[], outputs=[sid0])

                
                with gr.TabItem(i18n("单个")):
                    with gr.Row(): 
                        spk_item = gr.Slider(
                            minimum=0,
                            maximum=2333,
                            step=1,
                            label=i18n("请选择说话人id"),
                            value=0,
                            visible=False,
                            interactive=True,
                        )
                        #clean_button.click(fn=lambda: ({"value": "", "__type__": "update"}), inputs=[], outputs=[sid0])

                    with gr.Group(): # Defines whole single inference option section
                        with gr.Row():
                            with gr.Column(): # First column for audio-related inputs
                                dropbox = gr.File(label=i18n("将音频拖到此处，然后点击刷新按钮"))
                                record_button=gr.Audio(source="microphone", label=i18n("或录制音频"), type="filepath")
                                input_audio0 = gr.Textbox(
                                    label=i18n("Manual path to the audio file to be processed"),
                                    value=os.path.join(now_dir, "audios", "someguy.mp3"),
                                    visible=False
                                )
                                input_audio1 = gr.Dropdown(
                                    label=i18n("自动检测音频路径并从下拉菜单中选择："),
                                    choices=sorted(audio_paths),
                                    value='',
                                    interactive=True,
                                )
                                
                                input_audio1.select(fn=lambda:'',inputs=[],outputs=[input_audio0])
                                input_audio0.input(fn=lambda:'',inputs=[],outputs=[input_audio1])
                                
                                dropbox.upload(fn=save_to_wav2, inputs=[dropbox], outputs=[input_audio0])
                                dropbox.upload(fn=change_choices2, inputs=[], outputs=[input_audio1])
                                record_button.change(fn=save_to_wav, inputs=[record_button], outputs=[input_audio0])
                                record_button.change(fn=change_choices2, inputs=[], outputs=[input_audio1])

                            best_match_index_path1, _ = match_index(sid0.value) # Get initial index from default sid0 (first voice model in list)

                            with gr.Column(): # Second column for pitch shift and other options
                                file_index2 = gr.Dropdown(
                                    label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                                    choices=get_indexes(),
                                    value=best_match_index_path1,
                                    interactive=True,
                                    allow_custom_value=True,
                                )
                                index_rate1 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n("检索特征占比"),
                                    value=0.75,
                                    interactive=True,
                                )
                                refresh_button.click(
                                    fn=change_choices, inputs=[], outputs=[sid0, file_index2, input_audio1]
                                )
                                with gr.Column():
                                    vc_transform0 = gr.Number(
                                        label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                                    )
        
                    # Create a checkbox for advanced settings
                    advanced_settings_checkbox = gr.Checkbox(
                        value=False,
                        label=i18n("高级设置"),
                        interactive=True,
                    )
                    
                    # Advanced settings container        
                    with gr.Column(visible=False) as advanced_settings: # Initially hidden
                        with gr.Row(label = i18n("高级设置"), open = False):
                            with gr.Column():
                                f0method0 = gr.Radio(
                                    label=i18n(
                                        "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"
                                    ),
                                    choices=["pm", "harvest", "dio", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny", "rmvpe", "rmvpe+"], 
                                    value="rmvpe+",
                                    interactive=True,
                                )
                                crepe_hop_length = gr.Slider(
                                    minimum=1,
                                    maximum=512,
                                    step=1,
                                    label=i18n("crepe_hop_length"),
                                    value=120,
                                    interactive=True,
                                    visible=False,
                                )
                                filter_radius0 = gr.Slider(
                                    minimum=0,
                                    maximum=7,
                                    label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )    

                                minpitch_slider = gr.Slider(
                                    label       = i18n("音高最小值"),
                                    info        = i18n("指定推断的最小音高 [HZ]"),
                                    step        = 0.1,
                                    minimum     = 1,
                                    scale       = 0,
                                    value       = 50,
                                    maximum     = 16000,
                                    interactive = True,
                                    visible     = (not rvc_globals.NotesOrHertz) and (f0method0.value != 'rmvpe'),
                                )
                                minpitch_txtbox = gr.Textbox(
                                    label       = i18n("音高最小值"),
                                    info        = i18n("为推断指定最小音高 [音符][八度]"),
                                    placeholder = "C5",
                                    visible     = (rvc_globals.NotesOrHertz) and (f0method0.value != 'rmvpe'),
                                    interactive = True,
                                )

                                maxpitch_slider = gr.Slider(
                                    label       = i18n("音高最大值"),
                                    info        = i18n("指定推断的最大音高 [HZ]"),
                                    step        = 0.1,
                                    minimum     = 1,
                                    scale       = 0,
                                    value       = 1100,
                                    maximum     = 16000,
                                    interactive = True,
                                    visible     = (not rvc_globals.NotesOrHertz) and (f0method0.value != 'rmvpe'),
                                )
                                maxpitch_txtbox = gr.Textbox(
                                    label       = i18n("音高最大值"),
                                    info        = i18n("为推断指定最大音高 [音符][八度]"),
                                    placeholder = "C6",
                                    visible     = (rvc_globals.NotesOrHertz) and (f0method0.value != 'rmvpe'),
                                    interactive = True,
                                )

                            with gr.Column():
                                file_index1 = gr.Textbox(
                                    label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                                    value="",
                                    interactive=True,
                                )
                            
                                with gr.Accordion(label = i18n("自定义 f0 [根音] 文件"), open = False):
                                    f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"))

                            f0method0.change(
                                fn=lambda radio: (
                                    {
                                        "visible": radio in ['mangio-crepe', 'mangio-crepe-tiny'],
                                        "__type__": "update"
                                    }
                                ),
                                inputs=[f0method0],
                                outputs=[crepe_hop_length]
                            )

                            f0method0.change(
                                fn=switch_pitch_controls,
                                inputs=[f0method0],
                                outputs=[minpitch_slider, minpitch_txtbox,
                                         maxpitch_slider, maxpitch_txtbox]
                            )                            
                            
                            with gr.Column():
                                resample_sr0 = gr.Slider(
                                    minimum=0,
                                    maximum=48000,
                                    label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                    value=0,
                                    step=1,
                                    interactive=True,
                                )
                                rms_mix_rate0 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                                    value=0.25,
                                    interactive=True,
                                )
                                protect0 = gr.Slider(
                                    minimum=0,
                                    maximum=0.5,
                                    label=i18n(
                                        "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                    ),
                                    value=0.33,
                                    step=0.01,
                                    interactive=True,
                                )
                                formanting = gr.Checkbox(
                                    value=bool(DoFormant),
                                    label=i18n("共振声移动推理音频"),
                                    info=i18n("用于将男性转换为女性，反之亦然"),
                                    interactive=True,
                                    visible=True,
                                )
                                
                                formant_preset = gr.Dropdown(
                                    value='',
                                    choices=get_fshift_presets(),
                                    label=i18n("浏览共振峰预设"),
                                    info=i18n("预设位于 formantshiftcfg/ 文件夹中"),
                                    visible=bool(DoFormant),
                                )
                                
                                formant_refresh_button = gr.Button(
                                    value='\U0001f504',
                                    visible=bool(DoFormant),
                                    variant='primary',
                                )
                                
                                qfrency = gr.Slider(
                                        value=Quefrency,
                                        info=i18n("默认值为 1.0"),
                                        label=i18n("用于共振峰变换的 Quefrency"),
                                        minimum=0.0,
                                        maximum=16.0,
                                        step=0.1,
                                        visible=bool(DoFormant),
                                        interactive=True,
                                )
                                    
                                tmbre = gr.Slider(
                                    value=Timbre,
                                    info=i18n("默认值为 1.0"),
                                    label=i18n("用于共振峰变换的音色"),
                                    minimum=0.0,
                                    maximum=16.0,
                                    step=0.1,
                                    visible=bool(DoFormant),
                                    interactive=True,
                                )
                                frmntbut = gr.Button(i18n("应用"), variant="primary", visible=bool(DoFormant))

                            formant_preset.change(fn=preset_apply, inputs=[formant_preset, qfrency, tmbre], outputs=[qfrency, tmbre])
                            
                            formanting.change(fn=formant_enabled,inputs=[formanting,qfrency,tmbre],outputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button])
                            frmntbut.click(fn=formant_apply,inputs=[qfrency, tmbre], outputs=[qfrency, tmbre])
                            formant_refresh_button.click(fn=update_fshift_presets,inputs=[formant_preset, qfrency, tmbre],outputs=[formant_preset, qfrency, tmbre])

                    # Function to toggle advanced settings
                    def toggle_advanced_settings(checkbox):
                        return {"visible": checkbox, "__type__": "update"}

                    # Attach the change event
                    advanced_settings_checkbox.change(
                        fn=toggle_advanced_settings,
                        inputs=[advanced_settings_checkbox],
                        outputs=[advanced_settings]
                    )                           
                    
                    but0 = gr.Button(i18n("转换"), variant="primary").style(full_width=True)
                    
                    with gr.Row(): # Defines output info + output audio download after conversion
                        vc_output1 = gr.Textbox(label=i18n("输出信息"))
                        vc_output2 = gr.Audio(label=i18n("输出音频(右下角三个点,点了可以下载)"))

                    with gr.Group(): # I think this defines the big convert button
                        with gr.Row():
                            but0.click(
                                vc_single,
                                [
                                    spk_item,
                                    input_audio0,
                                    input_audio1,
                                    vc_transform0,
                                    f0_file,
                                    f0method0,
                                    file_index1,
                                    file_index2,
                                    index_rate1,
                                    filter_radius0,
                                    resample_sr0,
                                    rms_mix_rate0,
                                    protect0,
                                    crepe_hop_length,
                                    minpitch_slider, minpitch_txtbox,
                                    maxpitch_slider, maxpitch_txtbox,
                                ],
                                [vc_output1, vc_output2],
                            )
                           
                    
                with gr.TabItem(i18n("批处理")):
                    with gr.Group(): # Markdown explanation of batch inference
                        gr.Markdown(
                            value=i18n("批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. ")
                        )
                        with gr.Row():
                            with gr.Column():
                                vc_transform1 = gr.Number(
                                    label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                                )
                                opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt")
                            with gr.Column():
                                file_index4 = gr.Dropdown(
                                    label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                                    choices=get_indexes(),
                                    value=best_match_index_path1,
                                    interactive=True,
                                )
                                sid0.select(fn=match_index, inputs=[sid0], outputs=[file_index2, file_index4])

                                refresh_button.click(
                                    fn=lambda: change_choices()[1],
                                    inputs=[],
                                    outputs=file_index4,
                                )
                                index_rate2 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n("检索特征占比"),
                                    value=0.75,
                                    interactive=True,
                                )
                            with gr.Row():
                                dir_input = gr.Textbox(
                                    label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                                    value=os.path.join(now_dir, "audios"),
                                )
                                inputs = gr.File(
                                    file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                                )

                        with gr.Row():
                            with gr.Column():
                                # Create a checkbox for advanced batch settings
                                advanced_settings_batch_checkbox = gr.Checkbox(
                                    value=False,
                                    label=i18n("高级设置"),
                                    interactive=True,
                                )
                            
                                # Advanced batch settings container        
                                with gr.Row(visible=False) as advanced_settings_batch: # Initially hidden
                                    with gr.Row(label = i18n("高级设置[批量]"), open = False):
                                        with gr.Column():
                                            file_index3 = gr.Textbox(
                                                label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                                                value="",
                                                interactive=True,
                                            )

                                    f0method1 = gr.Radio(
                                        label=i18n(
                                            "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"
                                        ),
                                        choices=["pm", "harvest", "crepe", "rmvpe"],
                                        value="rmvpe",
                                        interactive=True,
                                    )
                                    filter_radius1 = gr.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                                        value=3,
                                        step=1,
                                        interactive=True,
                                    )
                                
                                    with gr.Row():
                                        format1 = gr.Radio(
                                            label=i18n("导出文件格式"),
                                            choices=["wav", "flac", "mp3", "m4a"],
                                            value="flac",
                                            interactive=True,
                                        )
                                        

                                    with gr.Column():
                                        resample_sr1 = gr.Slider(
                                            minimum=0,
                                            maximum=48000,
                                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                            value=0,
                                            step=1,
                                            interactive=True,
                                        )
                                        rms_mix_rate1 = gr.Slider(
                                            minimum=0,
                                            maximum=1,
                                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                                            value=1,
                                            interactive=True,
                                        )
                                        protect1 = gr.Slider(
                                            minimum=0,
                                            maximum=0.5,
                                            label=i18n(
                                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                            ),
                                            value=0.33,
                                            step=0.01,
                                            interactive=True,
                                        )
                                vc_output3 = gr.Textbox(label=i18n("输出信息")) 
                                but1 = gr.Button(i18n("转换"), variant="primary")
                                but1.click(
                                    vc_multi,
                                    [
                                        spk_item,
                                        dir_input,
                                        opt_input,
                                        inputs,
                                        vc_transform1,
                                        f0method1,
                                        file_index3,
                                        file_index4,
                                        index_rate2,
                                        filter_radius1,
                                        resample_sr1,
                                        rms_mix_rate1,
                                        protect1,
                                        format1,
                                        crepe_hop_length,
                                        minpitch_slider if (not rvc_globals.NotesOrHertz) else minpitch_txtbox,
                                        maxpitch_slider if (not rvc_globals.NotesOrHertz) else maxpitch_txtbox,
                                    ],
                                    [vc_output3],
                                )

                    sid0.change(
                        fn=get_vc,
                        inputs=[sid0, protect0, protect1],
                        outputs=[spk_item, protect0, protect1],
                    )

                    spk_item, protect0, protect1 = get_vc(sid0.value, protect0, protect1) # Set VC parameters for the preloaded model

                    # Function to toggle advanced settings
                    def toggle_advanced_settings_batch(checkbox):
                        return {"visible": checkbox, "__type__": "update"}

                    # Attach the change event
                    advanced_settings_batch_checkbox.change(
                        fn=toggle_advanced_settings_batch,
                        inputs=[advanced_settings_batch_checkbox],
                        outputs=[advanced_settings_batch]
                    )                           
                    
          # with gr.TabItem(i18n("伴奏人声分离&去混响&去回声")): # UVR section 
          #     with gr.Group():
          #         gr.Markdown(
          #             value=i18n(
          #                 "人声伴奏分离批量处理， 使用UVR5模型。 <br>"
          #                 "合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>"
          #                 "模型分为三类： <br>"
          #                 "1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>"
          #                 "2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> "
          #                 "3、去混响、去延迟模型（by FoxJoy）：<br>"
          #                 "  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>"
          #                 "&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>"
          #                 "去混响/去延迟，附：<br>"
          #                 "1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>"
          #                 "2、MDX-Net-Dereverb模型挺慢的；<br>"
          #                 "3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"
          #             )
          #         )
          #         with gr.Row():
          #             with gr.Column():
          #                 dir_wav_input = gr.Textbox(
          #                     label=i18n("输入待处理音频文件夹路径"),
          #                     value=os.path.join(now_dir, "audios")
          #                 )
          #                 wav_inputs = gr.File(
          #                     file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
          #                 )
          #             with gr.Column():
          #                 model_choose = gr.Dropdown(label=i18n("模型"), choices=uvr5_names)
          #                 agg = gr.Slider(
          #                     minimum=0,
          #                     maximum=20,
          #                     step=1,
          #                     label="人声提取激进程度",
          #                     value=10,
          #                     interactive=True,
          #                     visible=False,
          #                 )
          #                 opt_vocal_root = gr.Textbox(
          #                     label=i18n("指定输出主人声文件夹"), value="opt"
          #                 )
          #                 opt_ins_root = gr.Textbox(
          #                     label=i18n("指定输出非主人声文件夹"), value="opt"
          #                 )
          #                 format0 = gr.Radio(
          #                     label=i18n("导出文件格式"),
          #                     choices=["wav", "flac", "mp3", "m4a"],
          #                     value="flac",
          #                     interactive=True,
          #                 )
          #             but2 = gr.Button(i18n("转换"), variant="primary")
          #             vc_output4 = gr.Textbox(label=i18n("输出信息"))
          #             but2.click(
          #                 uvr,
          #                 [
          #                     model_choose,
          #                     dir_wav_input,
          #                     opt_vocal_root,
          #                     wav_inputs,
          #                     opt_ins_root,
          #                     agg,
          #                     format0,
          #                 ],
          #                 [vc_output4],
          #             )
            with gr.TabItem(i18n("训练")):
                gr.Markdown(
                    value=i18n(
                        "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
                    )
                )
                with gr.Row():
                    exp_dir1 = gr.Textbox(label=i18n("输入实验名"), value=i18n("宓模型"))
                    sr2 = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["40k", "48k", "32k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_3 = gr.Checkbox(
                        label=i18n("模型是否具有俯仰引导功能"),
                        value=True,
                        interactive=True,
                    )
                    version19 = gr.Radio(
                        label=i18n("版本"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                        visible=True,
                    )
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label=i18n("提取音高和处理数据使用的CPU进程数"),
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
                        )
                    )
                    with gr.Row():
                        trainset_dir4 = gr.Textbox(label=i18n("输入训练文件夹路径"), value=os.path.join(now_dir, datasets_root, datasets_name))
                        spk_id5 = gr.Slider(
                            minimum=0,
                            maximum=4,
                            step=1,
                            label=i18n("请指定说话人id"),
                            value=0,
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("处理数据"), variant="primary")
                        info1 = gr.Textbox(label=i18n("输出信息"), value="")
                        but1.click(
                            preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info1]
                        )
                with gr.Group():
                    gr.Markdown(value=i18n("step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"))
                    with gr.Row():
                        with gr.Column():
                            gpus6 = gr.Textbox(
                                label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                                value=gpus,
                                interactive=True,
                            )
                            gr.Textbox(label=i18n("显卡信息"), value=gpu_info)
                        with gr.Column():
                            f0method8 = gr.Radio(
                                label=i18n(
                                    "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢"
                                ),
                                choices=["pm", "harvest", "dio", "crepe", "mangio-crepe", "rmvpe"],
                                # [ MANGIO ]: Fork feature: Crepe on f0 extraction for training.
                                value="rmvpe",
                                interactive=True,
                            )
                            
                            extraction_crepe_hop_length = gr.Slider(
                                minimum=1,
                                maximum=512,
                                step=1,
                                label=i18n("crepe_hop_length"),
                                value=64,
                                interactive=True,
                                visible=False,
                            )
                            
                            f0method8.change(
                                fn=lambda radio: (
                                    {
                                        "visible": radio in ['mangio-crepe', 'mangio-crepe-tiny'],
                                        "__type__": "update"
                                    }
                                ),
                                inputs=[f0method8],
                                outputs=[extraction_crepe_hop_length]
                            )
                        but2 = gr.Button(i18n("特征提取"), variant="primary")
                        info2 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8, interactive=False)
                        but2.click(
                            extract_f0_feature,
                            [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, extraction_crepe_hop_length],
                            [info2],
                        )
                with gr.Group():
                    gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                    with gr.Row():
                        save_epoch10 = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label=i18n("保存频率save_every_epoch"),
                            value=10,
                            interactive=True,
                            visible=True,
                        )
                        total_epoch11 = gr.Slider(
                            minimum=1,
                            maximum=10000,
                            step=2,
                            label=i18n("总训练轮数total_epoch"),
                            value=750,
                            interactive=True,
                        )
                        batch_size12 = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label=i18n("每张显卡的batch_size"),
                            #value=default_batch_size,
                            value=20,
                            interactive=True,
                        )
                        if_save_latest13 = gr.Checkbox(
                            label=i18n("是否只保存最新的 .ckpt 文件以节省硬盘空间"),
                            value=True,
                            interactive=True,
                        )
                        if_cache_gpu17 = gr.Checkbox(
                            label=i18n("将所有训练集缓存到 GPU 内存中。缓存小型数据集（少于 10 分钟）可以加快训练速度，但缓存大型数据集会消耗大量 GPU 内存，可能无法显著提高速度"),
                            value=False,
                            interactive=True,
                        )
                        if_save_every_weights18 = gr.Checkbox(
                            label=i18n("在每个保存点将一个小的最终模型保存到 权重 文件夹中"),
                            value=True,
                            interactive=True,
                        )
                    with gr.Row():
                        pretrained_G14 = gr.Textbox(
                            lines=2,
                            label=i18n("加载预训练底模G路径"),
                            value="pretrained_v2/f0G40k.pth",
                            interactive=True,
                        )
                        pretrained_D15 = gr.Textbox(
                            lines=2,
                            label=i18n("加载预训练底模D路径"),
                            value="pretrained_v2/f0D40k.pth",
                            interactive=True,
                        )
                        sr2.change(
                            change_sr2,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15],
                        )
                        version19.change(
                            change_version19,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15, sr2],
                        )
                        if_f0_3.change(
                                fn=change_f0,
                                inputs=[if_f0_3, sr2, version19],
                                outputs=[f0method8, pretrained_G14, pretrained_D15],
                        )
                        if_f0_3.change(fn=lambda radio: (
                                    {
                                        "visible": radio in ['mangio-crepe', 'mangio-crepe-tiny'],
                                        "__type__": "update"
                                    }
                                ), inputs=[f0method8], outputs=[extraction_crepe_hop_length])
                        gpus16 = gr.Textbox(
                            label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                            value=gpus,
                            interactive=True,
                        )
                        butstop = gr.Button(i18n("停止培训"),
                                variant='primary',
                                visible=False,
                        )
                        but3 = gr.Button(i18n("训练模型"), variant="primary", visible=True)
                        but3.click(fn=stoptraining, inputs=[gr.Number(value=0, visible=False)], outputs=[but3, butstop])
                        butstop.click(fn=stoptraining, inputs=[gr.Number(value=1, visible=False)], outputs=[but3, butstop])
                        
                        with gr.Column(scale=0):
                            gr.Markdown(value="<br>")
                            gr.Markdown(value="### " + i18n("保存前构建索引。"))
                            but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                            gr.Markdown(value="### " + i18n("训练结束后保存您的模型。"))
                            save_action = gr.Dropdown(label=i18n("存储类型"), choices=[i18n("保存所有"),i18n("保存 D 和 G"),i18n("保存声音")], value=i18n("选择模型保存方法"), interactive=True)
                            but7 = gr.Button(i18n("保存模型"), variant="primary")
                        
                    
                      # but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                        info3 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10)
                        
                        if_save_every_weights18.change(
                            fn=lambda if_save_every_weights: (
                                {
                                    "visible": if_save_every_weights,
                                    "__type__": "update"
                                }
                            ),
                            inputs=[if_save_every_weights18],
                            outputs=[save_epoch10]
                        )
                        
                        but3.click(
                            click_train,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                spk_id5,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                            ],
                            [info3, butstop, but3],
                        )
                            
                        but4.click(train_index, [exp_dir1, version19], info3)
                        but7.click(easy_infer.save_model, [exp_dir1, save_action], info3)
                with gr.Group():
                    gr.Markdown(value=i18n(
                        '步骤4：单击模型的导出最低点后，在模型图上的导出最低点，新文件将位于logs/[yourmodelname]/lowestvals/folder中')
                    )
                    
                    with gr.Row():
                        with gr.Accordion(label=i18n("最低点导出")):
                        
                            lowestval_weight_dir = gr.Textbox(visible=False)
                            ds = gr.Textbox(visible=False)
                            weights_dir1 = gr.Textbox(visible=False, value=weights_dir)
                            
                                
                            with gr.Row():
                                amntlastmdls = gr.Slider(
                                    minimum=1,
                                    maximum=25,
                                    label=i18n('保存多少个最低点'),
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )
                                lpexport = gr.Button(
                                    value=i18n('导出模型的最低点'),
                                    variant='primary',
                                )
                                lw_mdls = gr.File(
                                    file_count="multiple",
                                    label=i18n("输出型号"),
                                    interactive=False,
                                ) #####
                                
                            with gr.Row():
                                infolpex = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10)
                                mdlbl = gr.Dataframe(label=i18n('所选模型的统计数据'), datatype='number', type='pandas')
                            
                            lpexport.click(
                                lambda model_name: os.path.join("logs", model_name, "lowestvals"),
                                inputs=[exp_dir1],
                                outputs=[lowestval_weight_dir]
                            )
                            
                            lpexport.click(fn=tensorlowest.main, inputs=[exp_dir1, save_epoch10, amntlastmdls], outputs=[ds])
                            
                            ds.change(
                                fn=tensorlowest.selectweights,
                                inputs=[exp_dir1, ds, weights_dir1, lowestval_weight_dir],
                                outputs=[infolpex, lw_mdls, mdlbl],
                            )
            with gr.TabItem(i18n("伴奏人声分离&去混响&去回声")): # UVR section 
                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "人声伴奏分离批量处理， 使用UVR5模型。 <br>"
                            "合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>"
                            "模型分为三类： <br>"
                            "1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>"
                            "2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> "
                            "3、去混响、去延迟模型（by FoxJoy）：<br>"
                            "  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>"
                            "&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>"
                            "去混响/去延迟，附：<br>"
                            "1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>"
                            "2、MDX-Net-Dereverb模型挺慢的；<br>"
                            "3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"
                        )
                    )
                    with gr.Row():
                        with gr.Column():
                            dir_wav_input = gr.Textbox(
                                label=i18n("输入待处理音频文件夹路径"),
                                value=os.path.join(now_dir, "audios")
                            )
                            wav_inputs = gr.File(
                                file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                            )
                        with gr.Column():
                            model_choose = gr.Dropdown(label=i18n("模型"), choices=uvr5_names)
                            agg = gr.Slider(
                                minimum=0,
                                maximum=20,
                                step=1,
                                label="人声提取激进程度",
                                value=10,
                                interactive=True,
                                visible=False,
                            )
                            opt_vocal_root = gr.Textbox(
                                label=i18n("指定输出主人声文件夹"), value="opt"
                            )
                            opt_ins_root = gr.Textbox(
                                label=i18n("指定输出非主人声文件夹"), value="opt"
                            )
                            format0 = gr.Radio(
                                label=i18n("导出文件格式"),
                                choices=["wav", "flac", "mp3", "m4a"],
                                value="flac",
                                interactive=True,
                            )
                        but2 = gr.Button(i18n("转换"), variant="primary")
                        vc_output4 = gr.Textbox(label=i18n("输出信息"))
                        but2.click(
                            uvr,
                            [
                                model_choose,
                                dir_wav_input,
                                opt_vocal_root,
                                wav_inputs,
                                opt_ins_root,
                                agg,
                                format0,
                            ],
                            [vc_output4],
                        )    
          # with gr.TabItem(i18n("ckpt处理")):
          #     with gr.Group():
          #         gr.Markdown(value=i18n("模型融合, 可用于测试音色融合"))
          #         with gr.Row():
          #             ckpt_a = gr.Textbox(label=i18n("A模型路径"), value="", interactive=True, placeholder="Path to your model A.")
          #             ckpt_b = gr.Textbox(label=i18n("B模型路径"), value="", interactive=True, placeholder="Path to your model B.")
          #             alpha_a = gr.Slider(
          #                 minimum=0,
          #                 maximum=1,
          #                 label=i18n("A模型权重"),
          #                 value=0.5,
          #                 interactive=True,
          #             )
          #         with gr.Row():
          #             sr_ = gr.Radio(
          #                 label=i18n("目标采样率"),
          #                 choices=["40k", "48k"],
          #                 value="40k",
          #                 interactive=True,
          #             )
          #             if_f0_ = gr.Checkbox(
          #                 label=i18n("模型是否具有俯仰引导功能"),
          #                 value=True,
          #                 interactive=True,
          #             )
          #             info__ = gr.Textbox(
          #                 label=i18n("要置入的模型信息"), value="", max_lines=8, interactive=True, placeholder="Model information to be placed."
          #             )
          #             name_to_save0 = gr.Textbox(
          #                 label=i18n("保存的模型名不带后缀"),
          #                 value="",
          #                 placeholder="Name for saving.",
          #                  max_lines=1,
          #                 interactive=True,
          #             )
          #             version_2 = gr.Radio(
          #                 label=i18n("模型版本型号"),
          #                 choices=["v1", "v2"],
          #                 value="v1",
          #                 interactive=True,
          #             )
          #         with gr.Row():
          #             but6 = gr.Button(i18n("融合"), variant="primary")
          #             info4 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
          #         but6.click(
          #             merge,
          #             [
          #                 ckpt_a,
          #                 ckpt_b,
          #                 alpha_a,
          #                 sr_,
          #                 if_f0_,
          #                 info__,
          #                 name_to_save0,
          #                 version_2,
          #             ],
          #             info4,
          #         )  # def merge(path1,path2,alpha1,sr,f0,info):
          #     with gr.Group():
          #         gr.Markdown(value=i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)"))
          #         with gr.Row(): ######
          #             ckpt_path0 = gr.Textbox(
          #                 label=i18n("模型路径"), placeholder="Path to your Model.", value="", interactive=True
          #             )
          #             info_ = gr.Textbox(
          #                 label=i18n("要改的模型信息"), value="", max_lines=8, interactive=True, placeholder="Model information to be changed."
          #             )
          #             name_to_save1 = gr.Textbox(
          #                 label=i18n("保存的文件名, 默认空为和源文件同名"),
          #                 placeholder="Either leave empty or put in the Name of the Model to be saved.",
          #                 value="",
          #                 max_lines=8,
          #                 interactive=True,
          #             )
          #         with gr.Row():
          #             but7 = gr.Button(i18n("修改"), variant="primary")
          #             info5 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
          #         but7.click(change_info, [ckpt_path0, info_, name_to_save1], info5)
          #     with gr.Group():
          #         gr.Markdown(value=i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)"))
          #         with gr.Row():
          #             ckpt_path1 = gr.Textbox(
          #                 label=i18n("模型路径"), value="", interactive=True, placeholder="Model path here."
          #             )
          #             but8 = gr.Button(i18n("查看"), variant="primary")
          #             info6 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
          #         but8.click(show_info, [ckpt_path1], info6)
          #     with gr.Group():
          #         gr.Markdown(
          #             value=i18n(
          #                 "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
          #             )
          #         )
          #         with gr.Row():
          #             ckpt_path2 = gr.Textbox(
          #                 lines=3,
          #                 label=i18n("模型路径"),
          #                 value=os.path.join(now_dir, "logs", "[YOUR_MODEL]", "G_23333.pth"),
          #                 interactive=True,
          #             )
          #             save_name = gr.Textbox(
          #                 label=i18n("保存名"), value="", interactive=True,
          #                 placeholder="Your filename here.",
          #             )
          #             sr__ = gr.Radio(
          #                 label=i18n("目标采样率"),
          #                 choices=["32k", "40k", "48k"],
          #                 value="40k",
          #                 interactive=True,
          #             )
          #             if_f0__ = gr.Checkbox(
          #                 label=i18n("模型是否具有俯仰引导功能"),
          #                 value=True,
          #                 interactive=True,
          #             )
          #             version_1 = gr.Radio(
          #                 label=i18n("模型版本型号"),
          #                 choices=["v1", "v2"],
          #                 value="v2",
          #                 interactive=True,
          #             )
          #             info___ = gr.Textbox(
          #                 label=i18n("要置入的模型信息"), value="", max_lines=8, interactive=True, placeholder="Model info here."
          #             )
          #             but9 = gr.Button(i18n("提取"), variant="primary")
          #             info7 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
          #             ckpt_path2.change(
          #                 change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
          #             )
          #         but9.click(
          #             extract_small_model,
          #             [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
          #             info7,
          #         )

          # with gr.TabItem(i18n("Onnx导出")):
          #     with gr.Row():
          #         ckpt_dir = gr.Textbox(label=i18n("RVC模型路径"), value="", interactive=True, placeholder="RVC model path.")
          #     with gr.Row():
          #         onnx_dir = gr.Textbox(
          #             label=i18n("Onnx输出路径"), value="", interactive=True, placeholder="Onnx model output path."
          #         )
          #     with gr.Row():
          #         infoOnnx = gr.Label(label="info")
          #     with gr.Row():
          #         butOnnx = gr.Button(i18n("导出Onnx模型"), variant="primary")
          #     butOnnx.click(export_onnx, [ckpt_dir, onnx_dir], infoOnnx)
            
            with gr.TabItem(i18n("资源")):
            
                easy_infer.download_model()
                easy_infer.download_backup()
                easy_infer.download_dataset(trainset_dir4)
            
            with gr.TabItem(i18n("设置")):
                with gr.Row():
                    gr.Markdown(value=
                                i18n("音调设置")
                                )
                    noteshertz = gr.Checkbox(
                        label       = i18n("是否使用音符名称而不是它们的赫兹值。例如，使用[C5，D6]代替[523.25，1174.66]赫兹。"),
                        value       = rvc_globals.NotesOrHertz,
                        interactive = True,
                    )
            
            noteshertz.change(fn=lambda nhertz: rvc_globals.__setattr__('NotesOrHertz', nhertz), inputs=[noteshertz], outputs=[])

            noteshertz.change(
                fn=switch_pitch_controls,
                inputs=[f0method0],
                outputs=[
                    minpitch_slider, minpitch_txtbox,
                    maxpitch_slider, maxpitch_txtbox,]
            )

            #with gr.TabItem(tab_faq):
                #try:
                    #with open(faq_file, "r", encoding="utf8") as f:
                        #info = f.read()
                    #gr.Markdown(value=info)
                #except:
                    #gr.Markdown(traceback.format_exc())
        return app

def GradioRun(app):
    share_gradio_link = config.iscolab or config.paperspace
    concurrency_count = 511
    max_size = 1022

    if (
        config.iscolab or config.paperspace
    ):  
        app.queue(concurrency_count=concurrency_count, max_size=max_size).launch(
        server_name="0.0.0.0",
        inbrowser=not config.noautoopen,
        server_port=config.listen_port,
        quiet=True,
        favicon_path="./icon.png",
        share=share_gradio_link,
        )
    else:
        app.queue(concurrency_count=concurrency_count, max_size=max_size).launch(
        server_name="0.0.0.0",
        inbrowser=not config.noautoopen,
        server_port=config.listen_port,
        quiet=True,
        favicon_path=".\icon.png",
        share=share_gradio_link,
        )

#endregion

if __name__ == "__main__":
    if os.name == 'nt': # Weird Windows async error when replacing a file.
        print("Any ConnectionResetErrors post-conversion are irrelevant and purely visual; they can be ignored\n")
    app = GradioSetup(UTheme=config.grtheme)
    GradioRun(app)