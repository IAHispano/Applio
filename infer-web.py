import os, sys

from tensorboard import program

now_dir = os.getcwd()
sys.path.append(now_dir)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
import shutil
import threading
from assets.configs.config import Config
import lib.globals.globals as rvc_globals

import lib.tools.model_fetcher as model_fetcher
import math as math
import ffmpeg as ffmpeg
import traceback
import warnings
from random import shuffle
from subprocess import Popen
from time import sleep
import json
import pathlib
import fairseq
import socket
import requests
import subprocess

logging.getLogger("faiss").setLevel(logging.WARNING)
import faiss
import gradio as gr
import numpy as np
import torch as torch
import regex as re
import soundfile as SF

SFWrite = SF.write
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans
import datetime


from glob import glob1
import signal
from signal import SIGTERM
from assets.i18n.i18n import I18nAuto
from lib.modules.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from lib.modules.uvr5.mdxnet import MDXNetDereverb
from lib.modules.uvr5.preprocess import AudioPre, AudioPreDeEcho
from lib.modules.vc.modules import VC
from lib.modules.vc.utils import *
import lib.globals.globals as rvc_globals
import nltk

nltk.download("punkt", quiet=True)

import tabs.resources as resources
import tabs.tts as tts
import tabs.merge as mergeaudios
import tabs.processing as processing

from lib.modules.infer.csvutil import CSVutil
import time
import csv
from shlex import quote as SQuote

logger = logging.getLogger(__name__)

RQuote = lambda val: SQuote(str(val))

tmp = os.path.join(now_dir, "temp")

# directories = ["logs", "datasets", "weights", "audio-others", "audio-outputs"]

shutil.rmtree(tmp, ignore_errors=True)

os.makedirs(tmp, exist_ok=True)

# Start the download server
if True == True:
    host = "localhost"
    port = 8000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)  # Timeout in seconds

    try:
        sock.connect((host, port))
        logger.warn(
            f"Something is listening on port {port}; check open connection and restart Applio."
        )
        logger.warn("Trying to start it anyway")
        sock.close()
        requests.post("http://localhost:8000/shutdown")
        time.sleep(3)
        script_path = os.path.join(now_dir, "lib", "tools", "server.py")
        try:
            subprocess.Popen(f"python {script_path}", shell=True)
        except Exception as e:
            logger.error(f"Failed to start the Flask server")
            logger.error(e)
    except Exception as e:
        sock.close()
        script_path = os.path.join(now_dir, "lib", "tools", "server.py")
        try:
            subprocess.Popen(f"python {script_path}", shell=True)
        except Exception as e:
            logger.error("Failed to start the Flask server")
            logger.error(e)

os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs/weights"), exist_ok=True)
os.environ["temp"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)

if not os.path.isdir("lib/csvdb/"):
    os.makedirs("lib/csvdb")
    frmnt, stp = open("lib/csvdb/formanting.csv", "w"), open("lib/csvdb/stop.csv", "w")
    frmnt.close()
    stp.close()

global DoFormant, Quefrency, Timbre

try:
    DoFormant, Quefrency, Timbre = CSVutil(
        "lib/csvdb/formanting.csv", "r", "formanting"
    )
    DoFormant = (
        lambda DoFormant: True
        if DoFormant.lower() == "true"
        else (False if DoFormant.lower() == "false" else DoFormant)
    )(DoFormant)
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil(
        "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre
    )

load_dotenv()
config = Config()
vc = VC(config)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

i18n = I18nAuto()
i18n.print()
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

isinterrupted = 0


if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_infos.append("%s\t%s" % (i, gpu_name))
        mem.append(
            int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4
            )
        )
if len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = (
        "Unfortunately, there is no compatible GPU available to support your training."
    )
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


import lib.modules.uvr5.mdx as mdx
from lib.modules.uvr5.mdxprocess import (
    get_model_list,
    get_demucs_model_list,
    id_to_ptm,
    prepare_mdx,
    run_mdx,
)

hubert_model = None
weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
datasets_root = "datasets"
fshift_root = "lib/modules/infer/formantshiftcfg"
audio_root = "assets\\audios"
audio_others_root = "assets/audios/audio-others"
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

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(weight_root)
    for file in files
    if file.endswith((".pth", ".onnx"))
]

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(index_root, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_root
]

audio_others_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_others_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_others_root
]

check_for_name = lambda: sorted(names)[0] if names else ""

datasets = []
for foldername in os.listdir(os.path.join(now_dir, datasets_root)):
    if os.path.isdir(os.path.join(now_dir, "datasets", foldername)):
        datasets.append(foldername)


def get_dataset():
    if len(datasets) > 0:
        return sorted(datasets)[0]
    else:
        return ""


def change_dataset(trainset_dir4):
    return gr.Textbox.update(value=trainset_dir4)


uvr5_names = [
    "HP2_all_vocals.pth",
    "HP3_all_vocals.pth",
    "HP5_only_main_vocal.pth",
    "VR-DeEchoAggressive.pth",
    "VR-DeEchoDeReverb.pth",
    "VR-DeEchoNormal.pth",
]

__s = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/"


def id_(mkey):
    if mkey in uvr5_names:
        model_name, ext = os.path.splitext(mkey)
        mpath = f"{now_dir}/assets/uvr5_weights/{mkey}"
        if not os.path.exists(f"{now_dir}/assets/uvr5_weights/{mkey}"):
            print("Downloading model...", end=" ")
            subprocess.run(["python", "-m", "wget", "-o", mpath, __s + mkey])
            print(f"saved to {mpath}")
            return model_name
        else:
            return model_name
    else:
        return None


def update_model_choices(select_value):
    model_ids = get_model_list()
    model_ids_list = list(model_ids)
    demucs_model_ids = get_demucs_model_list()
    demucs_model_ids_list = list(demucs_model_ids)
    if select_value == "VR":
        return {"choices": uvr5_names, "__type__": "update"}
    elif select_value == "MDX":
        return {"choices": model_ids_list, "__type__": "update"}
    elif select_value == "Demucs (Beta)":
        return {"choices": demucs_model_ids_list, "__type__": "update"}


def update_dataset_list(name):
    new_datasets = []
    for foldername in os.listdir(os.path.join(now_dir, datasets_root)):
        if os.path.isdir(os.path.join(now_dir, "datasets", foldername)):
            new_datasets.append(
                os.path.join(
                    now_dir,
                    "datasets",
                    foldername,
                )
            )
    return gr.Dropdown.update(choices=new_datasets)


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(index_root)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def get_fshift_presets():
    fshift_presets_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(fshift_root)
        for filename in filenames
        if filename.endswith(".txt")
    ]

    return fshift_presets_list if fshift_presets_list else ""


def uvr(
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    agg,
    format0,
    architecture,
):
    infos = []
    if architecture == "VR":
        try:
            inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            save_root_vocal = (
                save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )
            save_root_ins = (
                save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )

            model_name = id_(model_name)
            if model_name == None:
                return ""
            else:
                pass

            infos.append(
                i18n("Starting audio conversion... (This might take a moment)")
            )

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
                paths = [
                    os.path.join(inp_root, name)
                    for root, _, files in os.walk(inp_root, topdown=False)
                    for name in files
                    if name.endswith(tuple(sup_audioext)) and root == inp_root
                ]
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
                        os.path.join(os.environ["tmp"]),
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
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    yield "\n".join(infos)
                except:
                    try:
                        if done == 0:
                            pre_fun._path_audio_(
                                inp_path, save_root_ins, save_root_vocal, format0
                            )
                        infos.append("%s->Success" % (os.path.basename(inp_path)))
                        yield "\n".join(infos)
                    except:
                        infos.append(
                            "%s->%s"
                            % (os.path.basename(inp_path), traceback.format_exc())
                        )
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
            except:
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Executed torch.cuda.empty_cache()")
        yield "\n".join(infos)
    elif architecture == "MDX":
        try:
            infos.append(
                i18n("Starting audio conversion... (This might take a moment)")
            )
            yield "\n".join(infos)
            inp_root, save_root_vocal, save_root_ins = [
                x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
                for x in [inp_root, save_root_vocal, save_root_ins]
            ]

            if inp_root != "":
                paths = [
                    os.path.join(inp_root, name)
                    for root, _, files in os.walk(inp_root, topdown=False)
                    for name in files
                    if name.endswith(tuple(sup_audioext)) and root == inp_root
                ]
            else:
                paths = [path.name for path in paths]
            print(paths)
            invert = True
            denoise = True
            use_custom_parameter = True
            dim_f = 3072
            dim_t = 256
            n_fft = 7680
            use_custom_compensation = True
            compensation = 1.025
            suffix = "Vocals_custom"  # @param ["Vocals", "Drums", "Bass", "Other"]{allow-input: true}
            suffix_invert = "Instrumental_custom"  # @param ["Instrumental", "Drumless", "Bassless", "Instruments"]{allow-input: true}
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
                    infos.append(f"{os.path.basename(filename)}->Success")
                    yield "\n".join(infos)
        except:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)
        finally:
            try:
                del mdx_model
            except:
                traceback.print_exc()

            print("clean_empty_cache")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif architecture == "Demucs (Beta)":
        try:
            infos.append(
                i18n("Starting audio conversion... (This might take a moment)")
            )
            yield "\n".join(infos)
            inp_root, save_root_vocal, save_root_ins = [
                x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
                for x in [inp_root, save_root_vocal, save_root_ins]
            ]

            if inp_root != "":
                paths = [
                    os.path.join(inp_root, name)
                    for root, _, files in os.walk(inp_root, topdown=False)
                    for name in files
                    if name.endswith(tuple(sup_audioext)) and root == inp_root
                ]
            else:
                paths = [path.name for path in paths]

            # Loop through the audio files and separate sources
            for path in paths:
                input_audio_path = os.path.join(inp_root, path)
                filename_without_extension = os.path.splitext(
                    os.path.basename(input_audio_path)
                )[0]
                _output_dir = os.path.join(tmp, model_name, filename_without_extension)
                vocals = os.path.join(_output_dir, "vocals.wav")
                no_vocals = os.path.join(_output_dir, "no_vocals.wav")

                os.makedirs(tmp, exist_ok=True)

                if torch.cuda.is_available():
                    cpu_insted = ""
                else:
                    cpu_insted = "-d cpu"
                print(cpu_insted)

                # Use with os.system  to separate audio sources becuase at invoking from the command line it is faster than invoking from python
                os.system(
                    f"python -m .separate --two-stems=vocals -n {model_name} {cpu_insted} {input_audio_path} -o {tmp}"
                )

                # Move vocals and no_vocals to the output directory assets/audios for the vocal and assets/audios/audio-others for the instrumental
                shutil.move(vocals, save_root_vocal)
                shutil.move(no_vocals, save_root_ins)

                # And now rename the vocals and no vocals with the name of the input audio file and the suffix vocals or instrumental
                os.rename(
                    os.path.join(save_root_vocal, "vocals.wav"),
                    os.path.join(
                        save_root_vocal, f"{filename_without_extension}_vocals.wav"
                    ),
                )
                os.rename(
                    os.path.join(save_root_ins, "no_vocals.wav"),
                    os.path.join(
                        save_root_ins, f"{filename_without_extension}_instrumental.wav"
                    ),
                )

                # Remove the temporary directory
                os.rmdir(tmp, model_name)

                infos.append(f"{os.path.basename(input_audio_path)}->Success")
                yield "\n".join(infos)

        except:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)


def change_choices():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(weight_root)
        for file in files
        if file.endswith((".pth", ".onnx"))
    ]
    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(index_root, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]
    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext)) and root == audio_root
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
    )


def change_choices2():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(weight_root)
        for file in files
        if file.endswith((".pth", ".onnx"))
    ]
    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(index_root, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
    )


def clean():
    return {"value": "", "__type__": "update"}


def export_onnx():
    from lib.modules.onnx.export import export_onnx as eo

    eo()


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def formant_enabled(
    cbox, qfrency, tmbre, frmntapply, formantpreset, formant_refresh_button
):
    if cbox:
        DoFormant = True
        CSVutil(
            "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, qfrency, tmbre
        )

        # print(f"is checked? - {cbox}\ngot {DoFormant}")

        return (
            {"value": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
        )

    else:
        DoFormant = False
        CSVutil(
            "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, qfrency, tmbre
        )

        # print(f"is checked? - {cbox}\ngot {DoFormant}")
        return (
            {"value": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )


def formant_apply(qfrency, tmbre):
    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True
    CSVutil("lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, qfrency, tmbre)

    return (
        {"value": Quefrency, "__type__": "update"},
        {"value": Timbre, "__type__": "update"},
    )


def update_fshift_presets(preset, qfrency, tmbre):
    if preset:
        with open(preset, "r") as p:
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


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p, dataset_path):
    if re.search(r"[^0-9a-zA-Z !@#$%^&\(\)_+=\-`~\[\]\{\};',.]", exp_dir):
        raise gr.Error("Model name contains non-ASCII characters!")
    if not dataset_path.strip() == "":
        trainset_dir = dataset_path
    else:
        trainset_dir = os.path.join(now_dir, "datasets", trainset_dir)
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" lib/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, echl):
    if re.search(r"[^0-9a-zA-Z !@#$%^&\(\)_+=\-`~\[\]\{\};',.]", exp_dir):
        raise gr.Error("Model name contains non-ASCII characters!")
    gpus_rmvpe = gpus
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" lib/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s %s'
                % (config.python_cmd, now_dir, exp_dir, n_p, f0method, RQuote(echl))
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" lib/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' lib/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" lib/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warn(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warn(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0", sr2),
    )


global log_interval


def set_log_interval(exp_dir, batch_size12):
    log_interval = 1
    folder_path = os.path.join(exp_dir, "1_16k_wavs")

    if os.path.isdir(folder_path):
        wav_files_num = len(glob1(folder_path, "*.wav"))

        if wav_files_num > 0:
            log_interval = math.ceil(wav_files_num / batch_size12)
            if log_interval > 1:
                log_interval += 1

    return log_interval


global PID, PROCESS, TB


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
    if_retrain_collapse20,
    if_stop_on_fit21,
    smoothness22,
    collapse_threshold23,
):
    CSVutil("lib/csvdb/stop.csv", "w+", "formanting", False)
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    log_interval = set_log_interval(exp_dir, batch_size12)

    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    cmd = (
        '"%s" lib/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s %s %s'
        % (
            config.python_cmd,
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            ("-g %s" % gpus16) if gpus16 else "",
            total_epoch11,
            save_epoch10,
            "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
            "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
            1 if if_save_latest13 == True else 0,
            1 if if_cache_gpu17 == True else 0,
            1 if if_save_every_weights18 == True else 0,
            version19,
            ("-sof %s -sm %s" % (1 if if_stop_on_fit21 == True else 0, smoothness22))
            if if_stop_on_fit21
            else "",
            (
                "-rc %s -ct %s"
                % (1 if if_retrain_collapse20 == True else 0, collapse_threshold23)
            )
            if if_retrain_collapse20
            else "",
        )
    )
    logger.info(cmd)
    global p, PID
    p = Popen(cmd, shell=True, cwd=now_dir)
    PID = p.pid

    p.wait()
    batchSize = batch_size12
    colEpoch = 0
    while if_retrain_collapse20:
        if not os.path.exists(f"logs/{exp_dir1}/col"):
            break
        with open(f"logs/{exp_dir1}/col") as f:
            col = f.read().split(",")
            if colEpoch < int(col[1]):
                colEpoch = int(col[1])
                logger.info(f"Epoch to beat {col[1]}")
                if batchSize != batch_size12:
                    batchSize = batch_size12 + 1
            batchSize -= 1
        if batchSize < 1:
            break
        p = Popen(
            cmd.replace(f"-bs {batch_size12}", f"-bs {batchSize}"),
            shell=True,
            cwd=now_dir,
        )
        PID = p.pid
        p.wait()

    return (
        i18n("Training is done, check train.log"),
        {"visible": False, "__type__": "update"},
        {"visible": True, "__type__": "update"},
    )


def train_index(exp_dir1, version19):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    # exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "Please do the feature extraction first"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform the feature extraction first"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    # infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("Generating training file...")
    print("Generating training file...")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("Generating adding file...")
    print("Generating adding file...")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append("Files generated successfully!")
    print("Files generated successfully!")


def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = config.dml == False


import re as regex
import scipy.io.wavfile as wavfile

cli_current_page = "HOME"


def cli_split_command(com):
    exp = r'(?:(?<=\s)|^)"(.*?)"(?=\s|$)|(\S+)'
    split_array = regex.findall(exp, com)
    split_array = [group[0] if group[0] else group[1] for group in split_array]
    return split_array


def execute_generator_function(genObject):
    for _ in genObject:
        pass


def cli_infer(com):
    # get VC first
    com = cli_split_command(com)
    model_name = com[0]
    source_audio_path = com[1]
    format1_ = com[2]
    feature_index_path = com[3]
    f0_file = None  # Not Implemented Yet

    # Get parameters for inference
    speaker_id = int(com[4])
    transposition = float(com[5])
    f0_method = com[6]
    crepe_hop_length = int(com[7])
    harvest_median_filter = int(com[8])
    resample = int(com[9])
    mix = float(com[10])
    feature_ratio = float(com[11])
    protection_amnt = float(com[12])
    protect1 = 0.5

    if com[13] == "False" or com[13] == "false":
        DoFormant = False
        Quefrency = 0.0
        Timbre = 0.0
        CSVutil(
            "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre
        )

    else:
        DoFormant = True
        Quefrency = float(com[14])
        Timbre = float(com[15])
        CSVutil(
            "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre
        )
    split_audio = True if (com[16] == 1) else False
    f0_autotune = True if (com[17] == 1) else False
    minpitch_slider = com[18]
    minpitch_txtbox = minpitch_slider
    maxpitch_slider = com[19]
    maxpitch_txtbox = maxpitch_slider

    print("Applio-RVC-Fork Infer-CLI: Starting the inference...")
    vc_data = vc.get_vc(model_name, protection_amnt, protect1)
    print(vc_data)
    print("Applio-RVC-Fork Infer-CLI: Performing inference...")
    conversion_data = vc.vc_single(
        speaker_id,
        source_audio_path,
        transposition,
        f0_file,
        f0_method,
        feature_index_path,
        feature_index_path,
        feature_ratio,
        harvest_median_filter,
        resample,
        mix,
        protection_amnt,
        format1_,
        split_audio,
        crepe_hop_length,
        minpitch_slider,
        minpitch_txtbox,
        maxpitch_slider,
        maxpitch_txtbox,
        f0_autotune,
    )
    if "Success." in conversion_data[0]:
        print("Applio-RVC-Fork Infer-CLI: Inference succeeded.")
    else:
        print("Applio-RVC-Fork Infer-CLI: Inference failed. Here's the traceback: ")
        print(conversion_data[0])


def cli_pre_process(com):
    com = cli_split_command(com)
    model_name = com[0]
    trainset_directory = com[1]
    sample_rate = com[2]
    num_processes = int(com[3])

    print("Applio-RVC-Fork Pre-process: Starting...")
    generator = preprocess_dataset(
        trainset_directory, model_name, sample_rate, num_processes
    )
    execute_generator_function(generator)
    print("Applio-RVC-Fork Pre-process: Finished")


def cli_extract_feature(com):
    com = cli_split_command(com)
    model_name = com[0]
    gpus = com[1]
    num_processes = int(com[2])
    has_pitch_guidance = True if (int(com[3]) == 1) else False
    f0_method = com[4]
    crepe_hop_length = int(com[5])
    version = com[6]  # v1 or v2

    print("Applio-RVC-CLI: Extract Feature Has Pitch: " + str(has_pitch_guidance))
    print("Applio-RVC-CLI: Extract Feature Version: " + str(version))
    print("Applio-RVC-Fork Feature Extraction: Starting...")
    generator = extract_f0_feature(
        gpus,
        num_processes,
        f0_method,
        has_pitch_guidance,
        model_name,
        version,
        crepe_hop_length,
    )
    execute_generator_function(generator)
    print("Applio-RVC-Fork Feature Extraction: Finished")


def cli_train(com):
    com = cli_split_command(com)
    model_name = com[0]
    sample_rate = com[1]
    has_pitch_guidance = True if (int(com[2]) == 1) else False
    speaker_id = int(com[3])
    save_epoch_iteration = int(com[4])
    total_epoch = int(com[5])  # 10000
    batch_size = int(com[6])
    gpu_card_slot_numbers = com[7]
    if_save_latest = True if (int(com[8]) == 1) else False
    if_cache_gpu = True if (int(com[9]) == 1) else False
    if_save_every_weight = True if (int(com[10]) == 1) else False
    version = com[11]
    if_retrain_collapse20 = True if (int(com[12]) == 1) else False
    if_stop_on_fit21 = True if (int(com[13]) == 1) else False
    smoothness23 = float(com[14]) if com[14] != "" else 0.975
    collapse_threshold22 = int(com[15]) if com[15] != "" else 25

    pretrained_base = "pretrained/" if version == "v1" else "pretrained_v2/"

    g_pretrained_path = "%sf0G%s.pth" % (pretrained_base, sample_rate)
    d_pretrained_path = "%sf0D%s.pth" % (pretrained_base, sample_rate)

    print("Applio-RVC-Fork Train-CLI: Training...")
    click_train(
        model_name,
        sample_rate,
        has_pitch_guidance,
        speaker_id,
        save_epoch_iteration,
        total_epoch,
        batch_size,
        if_save_latest,
        g_pretrained_path,
        d_pretrained_path,
        gpu_card_slot_numbers,
        if_cache_gpu,
        if_save_every_weight,
        version,
        if_retrain_collapse20,
        if_stop_on_fit21,
        smoothness23,
        collapse_threshold22,
    )


def cli_train_feature(com):
    com = cli_split_command(com)
    model_name = com[0]
    version = com[1]
    print("Applio-RVC-Fork Train Feature Index-CLI: Training... Please wait")
    generator = train_index(model_name, version)
    execute_generator_function(generator)
    print("Applio-RVC-Fork Train Feature Index-CLI: Done!")


def cli_extract_model(com):
    com = cli_split_command(com)
    model_path = com[0]
    save_name = com[1]
    sample_rate = com[2]
    has_pitch_guidance = com[3]
    info = com[4]
    version = com[5]
    extract_small_model_process = extract_small_model(
        model_path, save_name, sample_rate, has_pitch_guidance, info, version
    )
    if extract_small_model_process == "Success.":
        print("Applio-RVC-Fork Extract Small Model: Success!")
    else:
        print(str(extract_small_model_process))
        print("Applio-RVC-Fork Extract Small Model: Failed!")


def preset_apply(preset, qfer, tmbr):
    if str(preset) != "":
        with open(str(preset), "r") as p:
            content = p.readlines()
            qfer, tmbr = content[0].split("\n")[0], content[1]
            formant_apply(qfer, tmbr)
    else:
        pass
    return (
        {"value": qfer, "__type__": "update"},
        {"value": tmbr, "__type__": "update"},
    )


def print_page_details():
    if cli_current_page == "HOME":
        print(
            "\n    go home            : Takes you back to home with a navigation list."
            "\n    go infer           : Takes you to inference command execution."
            "\n    go pre-process     : Takes you to training step.1) pre-process command execution."
            "\n    go extract-feature : Takes you to training step.2) extract-feature command execution."
            "\n    go train           : Takes you to training step.3) being or continue training command execution."
            "\n    go train-feature   : Takes you to the train feature index command execution."
            "\n    go extract-model   : Takes you to the extract small model command execution."
        )
    elif cli_current_page == "INFER":
        print(
            "\n    arg 1) model name with .pth in logs/weights: mi-test.pth"
            "\n    arg 2) source audio path: assets/audios/MySource.wav"
            "\n    arg 3) export format (wav,flac,mp3) : wav"
            "\n    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index"
            "\n    arg 5) speaker id: 0"
            "\n    arg 6) transposition: 0"
            "\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe, rmvpe+)"
            "\n    arg 8) crepe hop length: 160"
            "\n    arg 9) harvest median filter radius: 3 (0-7)"
            "\n    arg 10) post resample rate: 0"
            "\n    arg 11) mix volume envelope: 1"
            "\n    arg 12) feature index ratio: 0.75 (0-1)"
            "\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)"
            "\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)"
            "\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)"
            "\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false)"
            "\n    arg 17)* Audio split depending on silence: 0 (0 for no, 1 for yes)"
            "\n    arg 18)* Extra autotune: 0 (0 for no, 1 for yes)\n"
            "\n    Only for rmvpe+ algorithm (If it is another algorithm, set default.):"
            "\n    arg 19)* Min pitch [HZ] / [NOTE][OCTAVE]: 50 or C5"
            "\n    arg 20)* Max pitch [HZ] / [NOTE][OCTAVE]: 1000 or C6\n"
            "\nExample: mi-test.pth assets/audios/Sidney.wav wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 True 8.0 1.2 1 0 50 1000"
        )
    elif cli_current_page == "PRE-PROCESS":
        print(
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set"
            "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
            "\n    arg 4) Number of CPU threads to use: 8 \n"
            "\nExample: mi-test mydataset 40k 24"
        )
    elif cli_current_page == "EXTRACT-FEATURE":
        print(
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)"
            "\n    arg 3) Number of CPU threads to use: 8"
            "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
            "\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)"
            "\n    arg 6) Crepe hop length: 128"
            "\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n"
            "\nExample: mi-test 0 24 1 harvest 128 v2"
        )
    elif cli_current_page == "TRAIN":
        print(
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
            "\n    arg 12) Model architecture version: v2 (use either v1 or v2)"
            "\n    arg 13) Reload from checkpoint before a mode collapse and try training it again: 0 (0 for no, 1 for yes)"
            "\n    arg 14) Stop training early if no improvement detected. (Set Training Epochs to something high like 9999): 0 (0 for no, 1 for yes)\n"
            "\n    arg 15) Threshold %% for collapse: Default 25"
            "\n    arg 16) Improvement smoothness calculation: Default 0.975\n"
            "\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2 0 0 25 0.975"
        )
    elif cli_current_page == "TRAIN-FEATURE":
        print(
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n"
            "\nExample: mi-test v2"
        )
    elif cli_current_page == "EXTRACT-MODEL":
        print(
            "\n    arg 1) Model Path: logs/mi-test/G_168000.pth"
            "\n    arg 2) Model save name: MyModel"
            "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
            "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
            '\n    arg 5) Model information: "My Model"'
            "\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n"
            '\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2'
        )


def change_page(page):
    global cli_current_page
    cli_current_page = page
    return 0


def execute_command(com):
    if com == "go home":
        return change_page("HOME")
    elif com == "go infer":
        return change_page("INFER")
    elif com == "go pre-process":
        return change_page("PRE-PROCESS")
    elif com == "go extract-feature":
        return change_page("EXTRACT-FEATURE")
    elif com == "go train":
        return change_page("TRAIN")
    elif com == "go train-feature":
        return change_page("TRAIN-FEATURE")
    elif com == "go extract-model":
        return change_page("EXTRACT-MODEL")
    else:
        if com[:3] == "go ":
            print("page '%s' does not exist!" % com[3:])
            return 0

    if cli_current_page == "INFER":
        cli_infer(com)
    elif cli_current_page == "PRE-PROCESS":
        cli_pre_process(com)
    elif cli_current_page == "EXTRACT-FEATURE":
        cli_extract_feature(com)
    elif cli_current_page == "TRAIN":
        cli_train(com)
    elif cli_current_page == "TRAIN-FEATURE":
        cli_train_feature(com)
    elif cli_current_page == "EXTRACT-MODEL":
        cli_extract_model(com)


def cli_navigation_loop():
    while True:
        print("\nYou are currently in '%s':" % cli_current_page)
        print_page_details()
        command = input("%s: " % cli_current_page)
        try:
            execute_command(command)
        except:
            print(traceback.format_exc())


if config.is_cli:
    print("\n\nApplio-RVC-Fork CLI\n")
    print(
        "Welcome to the CLI version of RVC. Please read the documentation on README.MD to understand how to use this app.\n"
    )
    cli_navigation_loop()


def switch_pitch_controls(f0method0):
    is_visible = f0method0 != "rmvpe"

    if rvc_globals.NotesOrHertz:
        return (
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"},
        )
    else:
        return (
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": is_visible, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )


def match_index(sid0: str) -> tuple:
    sid0strip = re.sub(r"\.pth|\.onnx$", "", sid0)
    sid0name = os.path.split(sid0strip)[-1]  # Extract only the name, not the directory

    # Check if the sid0strip has the specific ending format _eXXX_sXXX
    if re.match(r".+_e\d+_s\d+$", sid0name):
        base_model_name = sid0name.rsplit("_", 2)[0]
    else:
        base_model_name = sid0name

    sid_directory = os.path.join(index_root, base_model_name)
    directories_to_search = [sid_directory] if os.path.exists(sid_directory) else []
    directories_to_search.append(index_root)

    matching_index_files = []

    for directory in directories_to_search:
        for filename in os.listdir(directory):
            if filename.endswith(".index") and "trained" not in filename:
                # Condition to match the name
                name_match = any(
                    name.lower() in filename.lower()
                    for name in [sid0name, base_model_name]
                )

                # If in the specific directory, it's automatically a match
                folder_match = directory == sid_directory

                if name_match or folder_match:
                    index_path = os.path.join(directory, filename)
                    if index_path in indexes_list:
                        matching_index_files.append(
                            (
                                index_path,
                                os.path.getsize(index_path),
                                " " not in filename,
                            )
                        )

    if matching_index_files:
        # Sort by favoring files without spaces and by size (largest size first)
        matching_index_files.sort(key=lambda x: (-x[2], -x[1]))
        best_match_index_path = matching_index_files[0][0]
        return best_match_index_path, best_match_index_path

    return "", ""


def stoptraining(mim):
    if int(mim) == 1:
        CSVutil("lib/csvdb/stop.csv", "w+", "stop", "True")
        # p.terminate()
        # p.kill()
        try:
            os.kill(PID, signal.SIGTERM)
        except Exception as e:
            print(f"Couldn't click due to {e}")
            pass
    else:
        pass

    return (
        {"visible": False, "__type__": "update"},
        {"visible": True, "__type__": "update"},
    )


weights_dir = "weights/"


def note_to_hz(note_name):
    SEMITONES = {
        "C": -9,
        "C#": -8,
        "D": -7,
        "D#": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        "G": -2,
        "G#": -1,
        "A": 0,
        "A#": 1,
        "B": 2,
    }
    pitch_class, octave = note_name[:-1], int(note_name[-1])
    semitone = SEMITONES[pitch_class]
    note_number = 12 * (octave - 4) + semitone
    frequency = 440.0 * (2.0 ** (1.0 / 12)) ** note_number
    return frequency


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join("assets", "audios", os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path


def save_to_wav2_edited(dropbox):
    if dropbox is None:
        pass
    else:
        file_path = dropbox.name
        target_path = os.path.join("assets", "audios", os.path.basename(file_path))

        if os.path.exists(target_path):
            os.remove(target_path)

        shutil.move(file_path, target_path)
    return


def save_to_wav2(dropbox):
    file_path = dropbox.name
    target_path = os.path.join("assets", "audios", os.path.basename(file_path))

    if os.path.exists(target_path):
        os.remove(target_path)

    shutil.move(file_path, target_path)
    return target_path


import lib.tools.loader_themes as loader_themes

my_applio = loader_themes.load_json()
if my_applio:
    pass
else:
    my_applio = "JohnSmith9982/small_and_pretty"


def GradioSetup():
    default_weight = ""

    with gr.Blocks(theme=my_applio, title="Applio-RVC-Fork") as app:
        gr.HTML("<h1> 🍏 Applio-RVC-Fork </h1>")
        gr.HTML(
            "<h3>Discover over 15,000 voice models with our Discord bot — <a href='https://bot.applio.org'>Invite it here!</a></h3>"
        )
        with gr.Tabs():
            with gr.TabItem(i18n("Model Inference")):
                with gr.Row():
                    sid0 = gr.Dropdown(
                        label=i18n("Inferencing voice:"),
                        choices=sorted(names),
                        value=default_weight,
                    )
                    best_match_index_path1, _ = match_index(sid0.value)
                    file_index2 = gr.Dropdown(
                        label=i18n(
                            "Auto-detect index path and select from the dropdown:"
                        ),
                        choices=get_indexes(),
                        value=best_match_index_path1,
                        interactive=True,
                        allow_custom_value=True,
                    )
                    with gr.Column():
                        refresh_button = gr.Button(i18n("Refresh"), variant="primary")
                        clean_button = gr.Button(
                            i18n("Unload voice to save GPU memory"), variant="primary"
                        )
                    clean_button.click(
                        fn=lambda: ({"value": "", "__type__": "update"}),
                        inputs=[],
                        outputs=[sid0],
                        api_name="infer_clean",
                    )

                with gr.TabItem(i18n("Single")):
                    with gr.Row():
                        spk_item = gr.Slider(
                            minimum=0,
                            maximum=2333,
                            step=1,
                            label=i18n("Select Speaker/Singer ID:"),
                            value=0,
                            visible=False,
                            interactive=True,
                        )
                    with gr.Row():
                        with gr.Column():  # First column for audio-related inputs
                            dropbox = gr.File(label=i18n("Drag your audio here:"))
                            record_button = gr.Audio(
                                source="microphone",
                                label=i18n("Or record an audio:"),
                                type="filepath",
                            )

                        with gr.Column():  # Second column for pitch shift and other options
                            with gr.Column():
                                input_audio1 = gr.Dropdown(
                                    label=i18n(
                                        "Auto detect audio path and select from the dropdown:"
                                    ),
                                    choices=sorted(audio_paths),
                                    value="",
                                    interactive=True,
                                )
                                vc_transform0 = gr.Number(
                                    label=i18n(
                                        "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):"
                                    ),
                                    value=0,
                                )

                            dropbox.upload(
                                fn=save_to_wav2,
                                inputs=[dropbox],
                                outputs=[input_audio1],
                            )
                            record_button.change(
                                fn=save_to_wav,
                                inputs=[record_button],
                                outputs=[input_audio1],
                            )
                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=[sid0, file_index2, input_audio1],
                                api_name="infer_refresh",
                            )
                    # Create a checkbox for advanced settings
                    advanced_settings_checkbox = gr.Checkbox(
                        value=False,
                        label=i18n("Advanced Settings"),
                        interactive=True,
                    )

                    # Advanced settings container
                    with gr.Column(
                        visible=False
                    ) as advanced_settings:  # Initially hidden
                        with gr.Row(label=i18n("Advanced Settings"), open=False):
                            with gr.Column():
                                f0method0 = gr.Radio(
                                    label=i18n(
                                        "Select the pitch extraction algorithm:"
                                    ),
                                    choices=[
                                        "pm",
                                        "harvest",
                                        "dio",
                                        "crepe",
                                        "crepe-tiny",
                                        "mangio-crepe",
                                        "mangio-crepe-tiny",
                                        "rmvpe",
                                        "rmvpe+",
                                    ]
                                    if config.dml == False
                                    else [
                                        "pm",
                                        "harvest",
                                        "dio",
                                        "rmvpe",
                                        "rmvpe+",
                                    ],
                                    value="rmvpe+",
                                    interactive=True,
                                )
                                format1_ = gr.Radio(
                                    label=i18n("Export file format:"),
                                    choices=["wav", "flac", "mp3", "m4a"],
                                    value="wav",
                                    interactive=True,
                                )

                                f0_autotune = gr.Checkbox(
                                    label=i18n("Enable autotune"),
                                    interactive=True,
                                    value=False,
                                )
                                split_audio = gr.Checkbox(
                                    label=i18n("Split Audio (Better Results)"),
                                    interactive=True,
                                )

                                crepe_hop_length = gr.Slider(
                                    minimum=1,
                                    maximum=512,
                                    step=1,
                                    label=i18n(
                                        "Mangio-Crepe Hop Length (Only applies to mangio-crepe): Hop length refers to the time it takes for the speaker to jump to a dramatic pitch. Lower hop lengths take more time to infer but are more pitch accurate."
                                    ),
                                    value=120,
                                    interactive=True,
                                    visible=False,
                                )

                                minpitch_slider = gr.Slider(
                                    label=i18n("Min pitch:"),
                                    info=i18n(
                                        "Specify minimal pitch for inference [HZ]"
                                    ),
                                    step=0.1,
                                    minimum=1,
                                    scale=0,
                                    value=50,
                                    maximum=16000,
                                    interactive=True,
                                    visible=(not rvc_globals.NotesOrHertz)
                                    and (f0method0.value != "rmvpe"),
                                )
                                minpitch_txtbox = gr.Textbox(
                                    label=i18n("Min pitch:"),
                                    info=i18n(
                                        "Specify minimal pitch for inference [NOTE][OCTAVE]"
                                    ),
                                    placeholder="C5",
                                    visible=(rvc_globals.NotesOrHertz)
                                    and (f0method0.value != "rmvpe"),
                                    interactive=True,
                                )

                                maxpitch_slider = gr.Slider(
                                    label=i18n("Max pitch:"),
                                    info=i18n("Specify max pitch for inference [HZ]"),
                                    step=0.1,
                                    minimum=1,
                                    scale=0,
                                    value=1100,
                                    maximum=16000,
                                    interactive=True,
                                    visible=(not rvc_globals.NotesOrHertz)
                                    and (f0method0.value != "rmvpe"),
                                )
                                maxpitch_txtbox = gr.Textbox(
                                    label=i18n("Max pitch:"),
                                    info=i18n(
                                        "Specify max pitch for inference [NOTE][OCTAVE]"
                                    ),
                                    placeholder="C6",
                                    visible=(rvc_globals.NotesOrHertz)
                                    and (f0method0.value != "rmvpe"),
                                    interactive=True,
                                )

                                file_index1 = gr.Textbox(
                                    label=i18n("Feature search database file path:"),
                                    value="",
                                    interactive=True,
                                )

                                f0_file = gr.File(
                                    label=i18n(
                                        "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:"
                                    )
                                )

                            f0method0.change(
                                fn=lambda radio: (
                                    {
                                        "visible": radio
                                        in ["mangio-crepe", "mangio-crepe-tiny"],
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[f0method0],
                                outputs=[crepe_hop_length],
                            )

                            f0method0.change(
                                fn=switch_pitch_controls,
                                inputs=[f0method0],
                                outputs=[
                                    minpitch_slider,
                                    minpitch_txtbox,
                                    maxpitch_slider,
                                    maxpitch_txtbox,
                                ],
                            )

                            with gr.Column():
                                resample_sr0 = gr.Slider(
                                    minimum=0,
                                    maximum=48000,
                                    label=i18n(
                                        "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:"
                                    ),
                                    value=0,
                                    step=1,
                                    interactive=True,
                                )
                                rms_mix_rate0 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n(
                                        "Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used:"
                                    ),
                                    value=0.25,
                                    interactive=True,
                                )
                                protect0 = gr.Slider(
                                    minimum=0,
                                    maximum=0.5,
                                    label=i18n(
                                        "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:"
                                    ),
                                    value=0.33,
                                    step=0.01,
                                    interactive=True,
                                )
                                filter_radius0 = gr.Slider(
                                    minimum=0,
                                    maximum=7,
                                    label=i18n(
                                        "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
                                    ),
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )
                                index_rate1 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n("Search feature ratio:"),
                                    value=0.75,
                                    interactive=True,
                                )
                                formanting = gr.Checkbox(
                                    value=bool(DoFormant),
                                    label=i18n("Formant shift inference audio"),
                                    info=i18n(
                                        "Used for male to female and vice-versa conversions"
                                    ),
                                    interactive=True,
                                    visible=True,
                                )

                                formant_preset = gr.Dropdown(
                                    value="",
                                    choices=get_fshift_presets(),
                                    label=i18n("Browse presets for formanting"),
                                    info=i18n(
                                        "Presets are located in formantshiftcfg/ folder"
                                    ),
                                    visible=bool(DoFormant),
                                )

                                formant_refresh_button = gr.Button(
                                    value="\U0001f504",
                                    visible=bool(DoFormant),
                                    variant="primary",
                                )

                                qfrency = gr.Slider(
                                    value=Quefrency,
                                    info=i18n("Default value is 1.0"),
                                    label=i18n("Quefrency for formant shifting"),
                                    minimum=0.0,
                                    maximum=16.0,
                                    step=0.1,
                                    visible=bool(DoFormant),
                                    interactive=True,
                                )

                                tmbre = gr.Slider(
                                    value=Timbre,
                                    info=i18n("Default value is 1.0"),
                                    label=i18n("Timbre for formant shifting"),
                                    minimum=0.0,
                                    maximum=16.0,
                                    step=0.1,
                                    visible=bool(DoFormant),
                                    interactive=True,
                                )
                                frmntbut = gr.Button(
                                    "Apply", variant="primary", visible=bool(DoFormant)
                                )

                            formant_preset.change(
                                fn=preset_apply,
                                inputs=[formant_preset, qfrency, tmbre],
                                outputs=[qfrency, tmbre],
                            )
                            formanting.change(
                                fn=formant_enabled,
                                inputs=[
                                    formanting,
                                    qfrency,
                                    tmbre,
                                    frmntbut,
                                    formant_preset,
                                    formant_refresh_button,
                                ],
                                outputs=[
                                    formanting,
                                    qfrency,
                                    tmbre,
                                    frmntbut,
                                    formant_preset,
                                    formant_refresh_button,
                                ],
                            )
                            frmntbut.click(
                                fn=formant_apply,
                                inputs=[qfrency, tmbre],
                                outputs=[qfrency, tmbre],
                            )
                            formant_refresh_button.click(
                                fn=update_fshift_presets,
                                inputs=[formant_preset, qfrency, tmbre],
                                outputs=[formant_preset, qfrency, tmbre],
                            )

                    # Function to toggle advanced settings
                    def toggle_advanced_settings(checkbox):
                        return {"visible": checkbox, "__type__": "update"}

                    # Attach the change event
                    advanced_settings_checkbox.change(
                        fn=toggle_advanced_settings,
                        inputs=[advanced_settings_checkbox],
                        outputs=[advanced_settings],
                    )

                    but0 = gr.Button(i18n("Convert"), variant="primary").style(
                        full_width=True
                    )

                    with gr.Row():  # Defines output info + output audio download after conversion
                        vc_output1 = gr.Textbox(label=i18n("Output information:"))
                        vc_output2 = gr.Audio(
                            label=i18n(
                                "Export audio (click on the three dots in the lower right corner to download)"
                            )
                        )

                    with gr.Group():  # I think this defines the big convert button
                        with gr.Row():
                            but0.click(
                                vc.vc_single,
                                [
                                    spk_item,
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
                                    format1_,
                                    split_audio,
                                    crepe_hop_length,
                                    minpitch_slider,
                                    minpitch_txtbox,
                                    maxpitch_slider,
                                    maxpitch_txtbox,
                                    f0_autotune,
                                ],
                                [vc_output1, vc_output2],
                                api_name="infer_convert",
                            )

                with gr.TabItem(i18n("Batch")):  # Dont Change
                    with gr.Row():
                        with gr.Column():
                            vc_transform1 = gr.Number(
                                label=i18n(
                                    "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):"
                                ),
                                value=0,
                            )
                            opt_input = gr.Textbox(
                                label=i18n("Specify output folder:"),
                                value="assets/audios/audio-outputs",
                            )
                        with gr.Column():
                            dir_input = gr.Textbox(
                                label=i18n(
                                    "Enter the path of the audio folder to be processed (copy it from the address bar of the file manager):"
                                ),
                                value=os.path.join(now_dir, "assets", "audios"),
                            )
                            sid0.select(
                                fn=match_index,
                                inputs=[sid0],
                                outputs=[file_index2],
                            )

                        with gr.Column():
                            inputs = gr.File(
                                file_count="multiple",
                                label=i18n(
                                    "You can also input audio files in batches. Choose one of the two options. Priority is given to reading from the folder."
                                ),
                            )
                    with gr.Row():
                        with gr.Column():
                            # Create a checkbox for advanced batch settings
                            advanced_settings_batch_checkbox = gr.Checkbox(
                                value=False,
                                label=i18n("Advanced Settings"),
                                interactive=True,
                            )

                            # Advanced batch settings container
                            with gr.Row(
                                visible=False
                            ) as advanced_settings_batch:  # Initially hidden
                                with gr.Row(
                                    label=i18n("Advanced Settings"), open=False
                                ):
                                    with gr.Column():
                                        file_index3 = gr.Textbox(
                                            label=i18n(
                                                "Feature search database file path:"
                                            ),
                                            value="",
                                            interactive=True,
                                        )
                                        f0method1 = gr.Radio(
                                            label=i18n(
                                                "Select the pitch extraction algorithm:"
                                            ),
                                            choices=[
                                                "pm",
                                                "harvest",
                                                "dio",
                                                "crepe",
                                                "crepe-tiny",
                                                "mangio-crepe",
                                                "mangio-crepe-tiny",
                                                "rmvpe",
                                            ]
                                            if config.dml == False
                                            else [
                                                "pm",
                                                "harvest",
                                                "dio",
                                                "rmvpe",
                                            ],
                                            value="rmvpe",
                                            interactive=True,
                                        )

                                        format1 = gr.Radio(
                                            label=i18n("Export file format:"),
                                            choices=["wav", "flac", "mp3", "m4a"],
                                            value="wav",
                                            interactive=True,
                                        )

                                with gr.Column():
                                    resample_sr1 = gr.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=i18n(
                                            "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:"
                                        ),
                                        value=0,
                                        step=1,
                                        interactive=True,
                                    )
                                    rms_mix_rate1 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n(
                                            "Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used:"
                                        ),
                                        value=1,
                                        interactive=True,
                                    )
                                    protect1 = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=i18n(
                                            "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:"
                                        ),
                                        value=0.33,
                                        step=0.01,
                                        interactive=True,
                                    )
                                    filter_radius1 = gr.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=i18n(
                                            "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
                                        ),
                                        value=3,
                                        step=1,
                                        interactive=True,
                                    )

                                    index_rate2 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("Search feature ratio:"),
                                        value=0.75,
                                        interactive=True,
                                    )
                                    f0_autotune = gr.Checkbox(
                                        label="Enable autotune",
                                        interactive=True,
                                        value=False,
                                    )
                                    hop_length = gr.Slider(
                                        minimum=1,
                                        maximum=512,
                                        step=1,
                                        label=i18n(
                                            "Hop Length (lower hop lengths take more time to infer but are more pitch accurate):"
                                        ),
                                        value=120,
                                        interactive=True,
                                        visible=False,
                                    )

                            but1 = gr.Button(i18n("Convert"), variant="primary")
                            vc_output3 = gr.Textbox(label=i18n("Output information:"))
                            but1.click(
                                vc.vc_multi,
                                [
                                    spk_item,
                                    dir_input,
                                    opt_input,
                                    inputs,
                                    vc_transform1,
                                    f0method1,
                                    file_index3,
                                    file_index2,
                                    index_rate2,
                                    filter_radius1,
                                    resample_sr1,
                                    rms_mix_rate1,
                                    protect1,
                                    format1,
                                    hop_length,
                                    minpitch_slider,
                                    minpitch_txtbox,
                                    maxpitch_slider,
                                    maxpitch_txtbox,
                                    f0_autotune,
                                ],
                                [vc_output3],
                                api_name="infer_convert_batch",
                            )

                    sid0.change(
                        fn=vc.get_vc,
                        inputs=[sid0, protect0, protect1],
                        outputs=[spk_item, protect0, protect1],
                        api_name="infer_change_voice",
                    )
                    if not sid0.value == "":
                        spk_item, protect0, protect1 = vc.get_vc(
                            sid0.value, protect0, protect1
                        )

                    # spk_item, protect0, protect1 = vc.get_vc(sid0.value, protect0, protect1)

                    # Function to toggle advanced settings
                    def toggle_advanced_settings_batch(checkbox):
                        return {"visible": checkbox, "__type__": "update"}

                    # Attach the change event
                    advanced_settings_batch_checkbox.change(
                        fn=toggle_advanced_settings_batch,
                        inputs=[advanced_settings_batch_checkbox],
                        outputs=[advanced_settings_batch],
                    )

            with gr.TabItem(i18n("Train")):
                with gr.Accordion(label=i18n("Step 1: Processing data")):
                    with gr.Row():
                        with gr.Column():
                            exp_dir1 = gr.Textbox(
                                label=i18n("Enter the model name:"),
                                value=i18n("Model_Name"),
                            )
                            if_f0_3 = gr.Checkbox(
                                label=i18n("Whether the model has pitch guidance."),
                                value=True,
                                interactive=True,
                            )
                        sr2 = gr.Radio(
                            label=i18n("Target sample rate:"),
                            choices=["40k", "48k", "32k"],
                            value="40k",
                            interactive=True,
                        )
                        version19 = gr.Radio(
                            label=i18n("Version:"),
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                            visible=True,
                        )

                        with gr.Column():
                            np7 = gr.Slider(
                                minimum=1,
                                maximum=config.n_cpu,
                                step=1,
                                label=i18n("Number of CPU processes:"),
                                value=config.n_cpu,
                                interactive=True,
                            )
                            spk_id5 = gr.Slider(
                                minimum=0,
                                maximum=4,
                                step=1,
                                label=i18n("Specify the model ID:"),
                                value=0,
                                interactive=True,
                            )

                    with gr.Row():
                        with gr.Column():
                            trainset_dir4 = gr.Dropdown(
                                choices=sorted(datasets),
                                label=i18n("Select your dataset:"),
                                value="",
                            )
                            trainset_dir4.change(
                                change_dataset, [trainset_dir4], [exp_dir1]
                            )
                            dataset_path = gr.Textbox(
                                label=i18n("Or add your dataset path:"),
                                interactive=True,
                            )
                            btn_update_dataset_list = gr.Button(
                                i18n("Update list"), variant="primary"
                            )

                        btn_update_dataset_list.click(
                            resources.update_dataset_list, [spk_id5], trainset_dir4
                        )
                        but1 = gr.Button(i18n("Process data"), variant="primary")
                        info1 = gr.Textbox(label=i18n("Output information:"), value="")
                        but1.click(
                            preprocess_dataset,
                            [trainset_dir4, exp_dir1, sr2, np7, dataset_path],
                            [info1],
                            api_name="train_preprocess",
                        )

                with gr.Accordion(label=i18n("Step 2: Extracting features")):
                    with gr.Row():
                        with gr.Column():
                            gpus6 = gr.Textbox(
                                label=i18n(
                                    "Provide the GPU index(es) separated by '-', like 0-1-2 for using GPUs 0, 1, and 2:"
                                ),
                                value=gpus,
                                interactive=True,
                            )
                            gpu_info9 = gr.Textbox(
                                label=i18n("GPU Information:"),
                                value=gpu_info,
                                visible=F0GPUVisible,
                            )
                        with gr.Column():
                            f0method8 = gr.Radio(
                                label=i18n("Select the pitch extraction algorithm:"),
                                choices=[
                                    "pm",
                                    "harvest",
                                    "dio",
                                    "crepe",
                                    "mangio-crepe",
                                    "rmvpe",
                                    "rmvpe_gpu",
                                ]
                                if config.dml == False
                                else [
                                    "pm",
                                    "harvest",
                                    "dio",
                                    "rmvpe",
                                    "rmvpe_gpu",
                                ],
                                value="rmvpe",
                                interactive=True,
                            )
                            hop_length = gr.Slider(
                                minimum=1,
                                maximum=512,
                                step=1,
                                label=i18n(
                                    "Hop Length (lower hop lengths take more time to infer but are more pitch accurate):"
                                ),
                                value=64,
                                interactive=True,
                            )

                    with gr.Row():
                        but2 = gr.Button(i18n("Feature extraction"), variant="primary")
                        info2 = gr.Textbox(
                            label=i18n("Output information:"),
                            value="",
                            max_lines=8,
                            interactive=False,
                        )

                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            exp_dir1,
                            version19,
                            hop_length,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )

                with gr.Row():
                    with gr.Accordion(label=i18n("Step 3: Model training started")):
                        with gr.Row():
                            total_epoch11 = gr.Slider(
                                minimum=1,
                                maximum=10000,
                                step=2,
                                label=i18n("Training epochs:"),
                                value=100,
                                interactive=True,
                            )
                            batch_size12 = gr.Slider(
                                minimum=1,
                                maximum=50,
                                step=1,
                                label=i18n("Batch size per GPU:"),
                                value=default_batch_size,
                                # value=20,
                                interactive=True,
                            )
                            save_epoch10 = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                label=i18n("Save frequency:"),
                                value=10,
                                interactive=True,
                                visible=True,
                            )
                            collapse_threshold22 = gr.Slider(
                                minimum=1,
                                maximum=50,
                                step=1,
                                label="Threshold % for collapse:",
                                value=25,
                                interactive=True,
                                visible=False,
                            )
                            smoothness23 = gr.Slider(
                                minimum=0,
                                maximum=0.99,
                                step=0.005,
                                label="Improvement smoothness calculation:",
                                value=0.975,
                                interactive=True,
                                visible=False,
                            )

                        with gr.Row():
                            if_save_latest13 = gr.Checkbox(
                                label=i18n(
                                    "Whether to save only the latest .ckpt file to save hard drive space"
                                ),
                                value=True,
                                interactive=True,
                            )
                            if_cache_gpu17 = gr.Checkbox(
                                label=i18n(
                                    "Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training"
                                ),
                                value=False,
                                interactive=True,
                            )
                            if_save_every_weights18 = gr.Checkbox(
                                label=i18n(
                                    "Save a small final model to the 'weights' folder at each save point"
                                ),
                                value=True,
                                interactive=True,
                            )
                            if_retrain_collapse20 = gr.Checkbox(
                                label=i18n(
                                    "Reload from checkpoint before a mode collapse and try training it again"
                                ),
                                value=False,
                                interactive=True,
                            )
                            if_stop_on_fit21 = gr.Checkbox(
                                label=i18n(
                                    "Stop training early if no improvement detected"
                                ),
                                value=False,
                                interactive=True,
                            )
                        with gr.Column():
                            with gr.Row():
                                pretrained_G14 = gr.Textbox(
                                    label=i18n("Load pre-trained base model G path:"),
                                    value="assets/pretrained_v2/f0G40k.pth",
                                    interactive=True,
                                )
                                pretrained_D15 = gr.Textbox(
                                    label=i18n("Load pre-trained base model D path:"),
                                    value="assets/pretrained_v2/f0D40k.pth",
                                    interactive=True,
                                )
                                with gr.Row():
                                    gpus16 = gr.Textbox(
                                        label=i18n(
                                            "Provide the GPU index(es) separated by '-', like 0-1-2 for using GPUs 0, 1, and 2:"
                                        ),
                                        value=gpus,
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
                        with gr.Row():
                            butstop = gr.Button(
                                i18n("Stop training"),
                                variant="primary",
                                visible=False,
                            )
                            but3 = gr.Button(
                                i18n("Train model"), variant="primary", visible=True
                            )
                            but3.click(
                                fn=stoptraining,
                                inputs=[gr.Number(value=0, visible=False)],
                                outputs=[but3, butstop],
                                api_name="train_stop",
                            )
                            butstop.click(
                                fn=stoptraining,
                                inputs=[gr.Number(value=1, visible=False)],
                                outputs=[but3, butstop],
                            )
                            info3 = gr.Textbox(
                                label=i18n("Output information:"),
                                value="",
                                lines=4,
                                max_lines=4,
                            )

                            with gr.Column():
                                save_action = gr.Dropdown(
                                    label=i18n("Save type"),
                                    choices=[
                                        i18n("Save all"),
                                        i18n("Save D and G"),
                                        i18n("Save voice"),
                                    ],
                                    value=i18n("Choose the method"),
                                    interactive=True,
                                )
                                but4 = gr.Button(
                                    i18n("Train feature index"), variant="primary"
                                )

                                but7 = gr.Button(i18n("Save model"), variant="primary")

                            # if_save_every_weights18.change(
                            #     fn=lambda if_save_every_weights: (
                            #         {
                            #             "visible": if_save_every_weights,
                            #             "__type__": "update",
                            #         }
                            #     ),
                            #     inputs=[if_save_every_weights18],
                            #     outputs=[save_epoch10],
                            # )
                            if_retrain_collapse20.change(
                                fn=lambda if_retrain_collapse20: (
                                    {
                                        "visible": if_retrain_collapse20,
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[if_retrain_collapse20],
                                outputs=[collapse_threshold22],
                            )
                            if_stop_on_fit21.change(
                                fn=lambda if_stop_on_fit21: (
                                    {
                                        "visible": if_stop_on_fit21,
                                        "__type__": "update",
                                    }
                                ),
                                inputs=[if_stop_on_fit21],
                                outputs=[smoothness23],
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
                                if_retrain_collapse20,
                                if_stop_on_fit21,
                                smoothness23,
                                collapse_threshold22,
                            ],
                            [info3, butstop, but3],
                            api_name="train_start",
                        )

                        but4.click(train_index, [exp_dir1, version19], info3)
                        but7.click(resources.save_model, [exp_dir1, save_action], info3)

            with gr.TabItem(i18n("UVR5")):  # UVR section
                with gr.Row():
                    with gr.Column():
                        model_select = gr.Radio(
                            label=i18n("Model Architecture:"),
                            choices=["VR", "MDX", "Demucs (Beta)"],
                            value="VR",
                            interactive=True,
                        )
                        dir_wav_input = gr.Textbox(
                            label=i18n(
                                "Enter the path of the audio folder to be processed:"
                            ),
                            value=os.path.join(now_dir, "assets", "audios"),
                        )
                        wav_inputs = gr.File(
                            file_count="multiple",
                            label=i18n(
                                "You can also input audio files in batches. Choose one of the two options. Priority is given to reading from the folder."
                            ),
                        )

                    with gr.Column():
                        model_choose = gr.Dropdown(
                            label=i18n("Model:"), choices=uvr5_names
                        )
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label="Vocal Extraction Aggressive",
                            value=10,
                            interactive=True,
                            visible=False,
                        )
                        opt_vocal_root = gr.Textbox(
                            label=i18n("Specify the output folder for vocals:"),
                            value="assets/audios",
                        )
                        opt_ins_root = gr.Textbox(
                            label=i18n("Specify the output folder for accompaniment:"),
                            value="assets/audios/audio-others",
                        )
                        format0 = gr.Radio(
                            label=i18n("Export file format:"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    model_select.change(
                        fn=update_model_choices,
                        inputs=model_select,
                        outputs=model_choose,
                    )
                    but2 = gr.Button(i18n("Convert"), variant="primary")
                    vc_output4 = gr.Textbox(label=i18n("Output information:"))
                    # wav_inputs.upload(fn=save_to_wav2_edited, inputs=[wav_inputs], outputs=[])
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
                            model_select,
                        ],
                        [vc_output4],
                        api_name="uvr_convert",
                    )
            with gr.TabItem(i18n("TTS")):
                with gr.Column():
                    text_test = gr.Textbox(
                        label=i18n("Text:"),
                        placeholder=i18n(
                            "Enter the text you want to convert to voice..."
                        ),
                        lines=6,
                    )

                with gr.Row():
                    with gr.Column():
                        tts_methods_voice = ["Edge-tts", "Google-tts"]
                        ttsmethod_test = gr.Dropdown(
                            tts_methods_voice,
                            value="Edge-tts",
                            label=i18n("TTS Method:"),
                            visible=True,
                        )
                        tts_test = gr.Dropdown(
                            tts.set_edge_voice,
                            label=i18n("TTS Model:"),
                            visible=True,
                        )
                        ttsmethod_test.change(
                            fn=tts.update_tts_methods_voice,
                            inputs=ttsmethod_test,
                            outputs=tts_test,
                        )

                    with gr.Column():
                        model_voice_path07 = gr.Dropdown(
                            label=i18n("RVC Model:"),
                            choices=sorted(names),
                            value=default_weight,
                        )
                        best_match_index_path1, _ = match_index(
                            model_voice_path07.value
                        )

                        file_index2_07 = gr.Dropdown(
                            label=i18n("Select the .index file:"),
                            choices=get_indexes(),
                            value=best_match_index_path1,
                            interactive=True,
                            allow_custom_value=True,
                        )
                with gr.Row():
                    refresh_button_ = gr.Button(i18n("Refresh"), variant="primary")
                    refresh_button_.click(
                        fn=change_choices2,
                        inputs=[],
                        outputs=[model_voice_path07, file_index2_07],
                    )
                with gr.Row():
                    original_ttsvoice = gr.Audio(label=i18n("Audio TTS:"))
                    ttsvoice = gr.Audio(label=i18n("Audio RVC:"))

                with gr.Row():
                    button_test = gr.Button(i18n("Convert"), variant="primary")

                button_test.click(
                    tts.use_tts,
                    inputs=[
                        text_test,
                        tts_test,
                        model_voice_path07,
                        file_index2_07,
                        # transpose_test,
                        vc_transform0,
                        f0method8,
                        index_rate1,
                        crepe_hop_length,
                        f0_autotune,
                        ttsmethod_test,
                    ],
                    outputs=[ttsvoice, original_ttsvoice],
                )

            with gr.TabItem(i18n("Resources")):
                resources.download_model()
                resources.download_backup()
                resources.download_dataset(trainset_dir4)
                resources.download_audio()
                resources.audio_downloader_separator()
            with gr.TabItem(i18n("Extra")):
                gr.Markdown(
                    value=i18n(
                        "This section contains some extra utilities that often may be in experimental phases"
                    )
                )
                with gr.TabItem(i18n("Merge Audios")):
                    mergeaudios.merge_audios()

                with gr.TabItem(i18n("Processing")):
                    processing.processing_()

            with gr.TabItem(i18n("Settings")):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(value=i18n("Pitch settings"))
                        noteshertz = gr.Checkbox(
                            label=i18n(
                                "Whether to use note names instead of their hertz value. E.G. [C5, D6] instead of [523.25, 1174.66]Hz"
                            ),
                            value=rvc_globals.NotesOrHertz,
                            interactive=True,
                        )
                        themes_select = gr.Dropdown(
                            loader_themes.get_list(),
                            value=loader_themes.read_json(),
                            label=i18n("Select Theme:"),
                            visible=True,
                        )
                        themes_select.change(
                            fn=loader_themes.select_theme,
                            inputs=themes_select,
                            outputs=[],
                        )

            noteshertz.change(
                fn=lambda nhertz: rvc_globals.__setattr__("NotesOrHertz", nhertz),
                inputs=[noteshertz],
                outputs=[],
            )

            noteshertz.change(
                fn=switch_pitch_controls,
                inputs=[f0method0],
                outputs=[
                    minpitch_slider,
                    minpitch_txtbox,
                    maxpitch_slider,
                    maxpitch_txtbox,
                ],
            )

        return app


def GradioRun(app):
    share_gradio_link = config.iscolab or config.paperspace
    concurrency_count = 511
    max_size = 1022

    if config.iscolab or config.paperspace:
        app.queue(concurrency_count=concurrency_count, max_size=max_size).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
            favicon_path="./assets/icon.png",
            share=share_gradio_link,
        )
    else:
        app.queue(concurrency_count=concurrency_count, max_size=max_size).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
            favicon_path="./assets/icon.png",
            share=share_gradio_link,
        )


if __name__ == "__main__":
    app = GradioSetup()
    GradioRun(app)
