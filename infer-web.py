import sys
from shutil import rmtree
import shutil
import json  # Mangio fork using json for preset saving
import datetime
import unicodedata
from glob import glob1
from signal import SIGTERM
import librosa
import os

now_dir = os.getcwd()
sys.path.append(now_dir)
import lib.globals.globals as rvc_globals
from LazyImport import lazyload
import mdx
from mdx_processing_script import get_model_list, id_to_ptm, prepare_mdx, run_mdx

math = lazyload("math")
import traceback
import warnings

tensorlowest = lazyload("tensorlowest")
import faiss

ffmpeg = lazyload("ffmpeg")
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize
from bark import generate_audio, SAMPLE_RATE

np = lazyload("numpy")
torch = lazyload("torch")
re = lazyload("regex")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
from random import shuffle
from subprocess import Popen
import easy_infer
import audioEffects

gr = lazyload("gradio")
SF = lazyload("soundfile")
SFWrite = SF.write
from config import Config
import fairseq
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

# Tabs


nonen = ""
RQuote = lambda val: SQuote(str(val))

tmp = os.path.join(now_dir, "TEMP")
runtime_dir = os.path.join(now_dir, "runtime/Lib/site-packages")
directories = ["logs", "audios", "datasets", "weights"]

rmtree(tmp, ignore_errors=True)
rmtree(os.path.join(runtime_dir, "infer_pack"), ignore_errors=True)
rmtree(os.path.join(runtime_dir, "uvr5_pack"), ignore_errors=True)
os.makedirs(os.path.join(now_dir, "audio-outputs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "audio-others"), exist_ok=True)
os.makedirs(tmp, exist_ok=True)
for folder in directories:
    os.makedirs(os.path.join(now_dir, folder), exist_ok=True)

os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)
try:
    file = open("csvdb/stop.csv", "x")
    file.close()
except FileExistsError:
    pass

global DoFormant, Quefrency, Timbre

DoFormant = rvc_globals.DoFormant
Quefrency = rvc_globals.Quefrency
Timbre = rvc_globals.Timbre

config = Config()
if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
i18n = I18nAuto()
i18n.print()
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

keywords = [
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
]

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i).upper()
        if any(keyword in gpu_name for keyword in keywords):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(torch.cuda.get_device_properties(i).total_memory / 1e9 + 0.4)
            )

gpu_info = (
    "\n".join(gpu_infos)
    if if_gpu_ok and gpu_infos
    else "Unfortunately, there is no compatible GPU available to support your training."
)
default_batch_size = min(mem) // 2 if if_gpu_ok and gpu_infos else 1
gpus = "-".join(i[0] for i in gpu_infos)

hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"], suffix=""
    )
    hubert_model = models[0].to(config.device)

    if config.is_half:
        hubert_model = hubert_model.half()

    hubert_model.eval()


datasets_root = "datasets"
weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
fshift_root = "formantshiftcfg"
audio_root = "audios"
audio_others_root = "audio-others"

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
    if name.endswith(tuple(sup_audioext))
    if name.endswith(tuple(sup_audioext)) and not name.endswith(".gitignore")
]

audio_others_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_others_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext))
]

uvr5_names = [
    name.replace(".pth", "")
    for name in os.listdir(weight_uvr5_root)
    if name.endswith(".pth") or "onnx" in name
]


check_for_name = lambda: sorted(names)[0] if names else ""

datasets = []
for foldername in os.listdir(os.path.join(now_dir, datasets_root)):
    if "." not in foldername:
        datasets.append(
            os.path.join(
                easy_infer.find_folder_parent(".", "pretrained"), "datasets", foldername
            )
        )


def get_dataset():
    if len(datasets) > 0:
        return sorted(datasets)[0]
    else:
        return ""


def update_model_choices(select_value):
    model_ids = get_model_list()
    model_ids_list = list(model_ids)
    if select_value == "VR":
        return {"choices": uvr5_names, "__type__": "update"}
    elif select_value == "MDX":
        return {"choices": model_ids_list, "__type__": "update"}


set_bark_voice = easy_infer.get_bark_voice()
set_edge_voice = easy_infer.get_edge_voice()


def update_tts_methods_voice(select_value):
    # ["Edge-tts", "RVG-tts", "Bark-tts"]
    if select_value == "Edge-tts":
        return {"choices": set_edge_voice, "value": "", "__type__": "update"}
    elif select_value == "Bark-tts":
        return {"choices": set_bark_voice, "value": "", "__type__": "update"}


def update_dataset_list(name):
    new_datasets = []
    for foldername in os.listdir(os.path.join(now_dir, datasets_root)):
        if "." not in foldername:
            new_datasets.append(
                os.path.join(
                    easy_infer.find_folder_parent(".", "pretrained"),
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


import soundfile as sf


def generate_output_path(output_folder, base_name, extension):
    # Generar un nombre único para el archivo de salida
    index = 1
    while True:
        output_path = os.path.join(output_folder, f"{base_name}_{index}.{extension}")
        if not os.path.exists(output_path):
            return output_path
        index += 1


def combine_and_save_audios(
    audio1_path, audio2_path, output_path, volume_factor_audio1, volume_factor_audio2
):
    audio1, sr1 = librosa.load(audio1_path, sr=None)
    audio2, sr2 = librosa.load(audio2_path, sr=None)

    # Alinear las tasas de muestreo
    if sr1 != sr2:
        if sr1 > sr2:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        else:
            audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=sr2)

    # Ajustar los audios para que tengan la misma longitud
    target_length = min(len(audio1), len(audio2))
    audio1 = librosa.util.fix_length(audio1, target_length)
    audio2 = librosa.util.fix_length(audio2, target_length)

    # Ajustar el volumen de los audios multiplicando por el factor de ganancia
    if volume_factor_audio1 != 1.0:
        audio1 *= volume_factor_audio1
    if volume_factor_audio2 != 1.0:
        audio2 *= volume_factor_audio2

    # Combinar los audios
    combined_audio = audio1 + audio2

    sf.write(output_path, combined_audio, sr1)


# Resto de tu código...


# Define función de conversión llamada por el botón
def audio_combined(
    audio1_path,
    audio2_path,
    volume_factor_audio1=1.0,
    volume_factor_audio2=1.0,
    reverb_enabled=False,
    compressor_enabled=False,
    noise_gate_enabled=False,
):
    output_folder = os.path.join(now_dir, "audio-outputs")
    os.makedirs(output_folder, exist_ok=True)

    # Generar nombres únicos para los archivos de salida
    base_name = "combined_audio"
    extension = "wav"
    output_path = generate_output_path(output_folder, base_name, extension)
    print(reverb_enabled)
    print(compressor_enabled)
    print(noise_gate_enabled)

    if reverb_enabled or compressor_enabled or noise_gate_enabled:
        # Procesa el primer audio con los efectos habilitados
        base_name = "effect_audio"
        output_path = generate_output_path(output_folder, base_name, extension)
        processed_audio_path = audioEffects.process_audio(
            audio2_path,
            output_path,
            reverb_enabled,
            compressor_enabled,
            noise_gate_enabled,
        )
        base_name = "combined_audio"
        output_path = generate_output_path(output_folder, base_name, extension)
        # Combina el audio procesado con el segundo audio usando audio_combined
        combine_and_save_audios(
            audio1_path,
            processed_audio_path,
            output_path,
            volume_factor_audio1,
            volume_factor_audio2,
        )

        return i18n("Conversion complete!"), output_path
    else:
        base_name = "combined_audio"
        output_path = generate_output_path(output_folder, base_name, extension)
        # No hay efectos habilitados, combina directamente los audios sin procesar
        combine_and_save_audios(
            audio1_path,
            audio2_path,
            output_path,
            volume_factor_audio1,
            volume_factor_audio2,
        )

        return i18n("Conversion complete!"), output_path


def vc_single(
    sid: str,
    input_audio_path0: str,
    input_audio_path1: str,
    f0_up_key: int,
    f0_file: str,
    f0_method: str,
    file_index: str,
    file_index2: str,
    index_rate: float,
    filter_radius: int,
    resample_sr: int,
    rms_mix_rate: float,
    protect: float,
    crepe_hop_length: int,
    f0_min: int,
    note_min: str,
    f0_max: int,
    note_max: str,
    f0_autotune: bool,
):
    global total_time
    total_time = 0
    start_time = time.time()
    global tgt_sr, net_g, vc, hubert_model, version
    rmvpe_onnx = True if f0_method == "rmvpe_onnx" else False
    if not input_audio_path0 and not input_audio_path1:
        return "You need to upload an audio", None

    if (not os.path.exists(input_audio_path0)) and (
        not os.path.exists(os.path.join(now_dir, input_audio_path0))
    ):
        return "Audio was not properly selected or doesn't exist", None

    input_audio_path1 = input_audio_path1 or input_audio_path0
    print(f"\nStarting inference for '{os.path.basename(input_audio_path1)}'")
    print("-------------------")

    f0_up_key = int(f0_up_key)

    if rvc_globals.NotesOrHertz and f0_method != "rmvpe":
        f0_min = note_to_hz(note_min) if note_min else 50
        f0_max = note_to_hz(note_max) if note_max else 1100
        print(
            f"Converted Min pitch: freq - {f0_min}\n"
            f"Converted Max pitch: freq - {f0_max}"
        )
    else:
        f0_min = f0_min or 50
        f0_max = f0_max or 1100
    try:
        input_audio_path1 = input_audio_path1 or input_audio_path0
        print(f"Attempting to load {input_audio_path1}....")
        audio = load_audio(
            input_audio_path1,
            16000,
            DoFormant=rvc_globals.DoFormant,
            Quefrency=rvc_globals.Quefrency,
            Timbre=rvc_globals.Timbre,
        )

        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        times = [0, 0, 0]
        if not hubert_model:
            print("Loading hubert for the first time...")
            load_hubert()

        try:
            if_f0 = cpt.get("f0", 1)
        except NameError:
            message = "Model was not properly selected"
            print(message)
            return message, None

        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )

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
                f0_autotune,
                rmvpe_onnx,
                f0_file=f0_file,
                f0_min=f0_min,
                f0_max=f0_max,
            )
        except AssertionError:
            message = "Mismatching index version detected (v1 with v2, or v2 with v1)."
            print(message)
            return message, None
        except NameError:
            message = (
                "RVC libraries are still loading. Please try again in a few seconds."
            )
            print(message)
            return message, None

        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr

        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )

        end_time = time.time()
        total_time = end_time - start_time

        output_folder = "audio-outputs"
        os.makedirs(output_folder, exist_ok=True)
        output_filename = "generated_audio_{}.wav"
        output_count = 1
        while True:
            current_output_path = os.path.join(
                output_folder, output_filename.format(output_count)
            )
            if not os.path.exists(current_output_path):
                break
            output_count += 1

        # Guardar el audio generado como archivo WAV
        wavfile.write(current_output_path, tgt_sr, audio_opt)

        print(f"Generated audio saved to: {current_output_path}")

        return (
            f"Success.\n {index_info}\nTime:\n npy:{times[0]}, f0:{times[1]}, infer:{times[2]}\nTotal Time: {total_time} seconds",
            (tgt_sr, audio_opt),
        )
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
    f0_autotune,
):
    if rvc_globals.NotesOrHertz and f0_method != "rmvpe":
        f0_min = note_to_hz(note_min) if note_min else 50
        f0_max = note_to_hz(note_max) if note_max else 1100
        print(
            f"Converted Min pitch: freq - {f0_min}\n"
            f"Converted Max pitch: freq - {f0_max}"
        )
    else:
        f0_min = f0_min or 50
        f0_max = f0_max or 1100

    try:
        dir_path, opt_root = [
            x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            for x in [dir_path, opt_root]
        ]
        os.makedirs(opt_root, exist_ok=True)

        paths = (
            [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            if dir_path
            else [path.name for path in paths]
        )
        infos = []

        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                None,
                f0_up_key,
                None,
                f0_method,
                file_index,
                file_index2,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                crepe_hop_length,
                f0_min,
                note_min,
                f0_max,
                note_max,
                f0_autotune,
            )

            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    output_path = f"{opt_root}/{base_name}.{format1}"
                    path, extension = output_path, format1
                    path, extension = (
                        output_path
                        if format1 in ["wav", "flac", "mp3", "ogg", "aac", "m4a"]
                        else f"{output_path}.wav",
                        format1,
                    )
                    SFWrite(path, audio_opt, tgt_sr)
                    # sys.stdout.write("\nFile Written Successfully with SFWrite") # Debugging print
                    if os.path.exists(path) and extension not in [
                        "wav",
                        "flac",
                        "mp3",
                        "ogg",
                        "aac",
                        "m4a",
                    ]:
                        sys.stdout.write(
                            f"Running command: ffmpeg -i {RQuote(path)} -vn {RQuote(path[:-4] + '.' + extension)} -q:a 2 -y"
                        )
                        os.system(
                            f"ffmpeg -i {RQuote(path)} -vn {RQuote(path[:-4] + '.' + extension)} -q:a 2 -y"
                        )
                        # print(f"\nFile Converted to {extension} using ffmpeg") # Debugging print
                except:
                    info += traceback.format_exc()
                    print(f"\nException encountered: {info}")  # Debugging print
            infos.append(f"{os.path.basename(path)}->{info}")
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


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
            inp_root, save_root_vocal, save_root_ins = [
                x.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
                for x in [inp_root, save_root_vocal, save_root_ins]
            ]
            usable_files = [
                os.path.join(inp_root, file)
                for file in os.listdir(inp_root)
                if file.endswith(tuple(sup_audioext))
            ]

            pre_fun = (
                MDXNetDereverb(15)
                if model_name == "onnx_dereverb_By_FoxJoy"
                else (_audio_pre_ if "DeEcho" not in model_name else _audio_pre_new)(
                    agg=int(agg),
                    model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                    device=config.device,
                    is_half=config.is_half,
                )
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
                    traceback.print_exc()

                if need_reformat:
                    tmp_path = (
                        f"{tmp}/{os.path.basename(RQuote(inp_path))}.reformatted.wav"
                    )
                    os.system(
                        f"ffmpeg -i {RQuote(inp_path)} -vn -acodec pcm_s16le -ac 2 -ar 44100 {RQuote(tmp_path)} -y"
                    )
                    inp_path = tmp_path

                try:
                    if not done:
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                    infos.append(f"{os.path.basename(inp_path)}->Success")
                    yield "\n".join(infos)
                except:
                    infos.append(
                        f"{os.path.basename(inp_path)}->{traceback.format_exc()}"
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

            print("clean_empty_cache")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            net_g = (
                (
                    SynthesizerTrnMs256NSFsid
                    if version == "v1"
                    else SynthesizerTrnMs768NSFsid
                )(*cpt["config"], is_half=config.is_half)
                if if_f0 == 1
                else (
                    SynthesizerTrnMs256NSFsid_nono
                    if version == "v1"
                    else SynthesizerTrnMs768NSFsid_nono
                )(*cpt["config"])
            )
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
    net_g = (
        (SynthesizerTrnMs256NSFsid if version == "v1" else SynthesizerTrnMs768NSFsid)(
            *cpt["config"], is_half=config.is_half
        )
        if cpt.get("f0", 1) == 1
        else (
            SynthesizerTrnMs256NSFsid_nono
            if version == "v1"
            else SynthesizerTrnMs768NSFsid_nono
        )(*cpt["config"])
    )
    del net_g.enc_q

    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    return (
        {"visible": False, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )


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
        os.path.join(audio_root, file)
        for file in os.listdir(os.path.join(now_dir, "audios"))
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


def change_choices3():
    audio_paths = [
        os.path.join(audio_root, file)
        for file in os.listdir(os.path.join(now_dir, "audios"))
    ]
    audio_others_paths = [
        os.path.join(audio_others_root, file)
        for file in os.listdir(os.path.join(now_dir, "audio-others"))
    ]

    return (
        {"choices": sorted(audio_others_paths), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
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

    return ({"value": DoFormant, "__type__": "update"},) + (visibility_update,) * 6


def formant_apply(qfrency, tmbre):
    global Quefrency, Timbre, DoFormant

    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True

    rvc_globals.DoFormant = True
    rvc_globals.Quefrency = qfrency
    rvc_globals.Timbre = tmbre

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


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]

    log_dir = os.path.join(now_dir, "logs", exp_dir)
    log_file = os.path.join(log_dir, "preprocess.log")

    os.makedirs(log_dir, exist_ok=True)

    with open(log_file, "w") as f:
        pass

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

    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()

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
    with open(log_file, "w") as f:
        pass

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
    choices_update = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )

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
    model_desc = {"G": "", "D": ""}

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
        {"visible": if_f0_3, "__type__": "update"},
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
    with open("csvdb/stop.csv", "w+") as file:
        file.write("False")
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

    names = set(
        name.split(".")[0]
        for directory in required_dirs
        for name in os.listdir(directory)
    )

    def generate_paths(name):
        paths = [gt_wavs_dir, feature_dir]
        if if_f0_3:
            paths.extend([f0_dir, f0nsf_dir])
        return "|".join(
            [
                path.replace("\\", "\\\\")
                + "/"
                + name
                + (
                    ".wav.npy"
                    if path in [f0_dir, f0nsf_dir]
                    else ".wav"
                    if path == gt_wavs_dir
                    else ".npy"
                )
                for path in paths
            ]
        )

    opt = [f"{generate_paths(name)}|{spk_id5}" for name in names]
    mute_dir = f"{now_dir}/logs/mute"

    for _ in range(2):
        mute_string = f"{mute_dir}/0_gt_wavs/mute{sr2}.wav|{mute_dir}/3_feature{feature_dim}/mute.npy"
        if if_f0_3:
            mute_string += (
                f"|{mute_dir}/2a_f0/mute.wav.npy|{mute_dir}/2b-f0nsf/mute.wav.npy"
            )
        opt.append(mute_string + f"|{spk_id5}")

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

    return (
        i18n("Training is done, check train.log"),
        {"visible": False, "__type__": "update"},
        {"visible": True, "__type__": "update"},
    )


def train_index(exp_dir1, version19):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)

    feature_dim = "256" if version19 == "v1" else "768"
    feature_dir = os.path.join(exp_dir, f"3_feature{feature_dim}")

    if not os.path.exists(feature_dir) or len(os.listdir(feature_dir)) == 0:
        return "请先进行特征提取!"

    npys = [
        np.load(os.path.join(feature_dir, name))
        for name in sorted(os.listdir(feature_dir))
    ]

    big_npy = np.concatenate(npys, 0)
    np.random.shuffle(big_npy)

    infos = []
    if big_npy.shape[0] > 2 * 10**5:
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
        index.add(big_npy[i : i + batch_size_add])

    index_file_base = f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    faiss.write_index(index, index_file_base)

    infos.append(
        f"Successful Index Construction，added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    )
    yield "\n".join(infos)


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
        "rnd": torch.rand(1, 192, 200),
    }

    checkpoint["config"][-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
    net_g = SynthesizerTrnMsNSFsidM(
        *checkpoint["config"], is_half=False, version=checkpoint.get("version", "v1")
    )

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


import scipy.io.wavfile as wavfile

cli_current_page = "HOME"


def cli_split_command(com):
    exp = r'(?:(?<=\s)|^)"(.*?)"(?=\s|$)|(\S+)'
    split_array = re.findall(exp, com)
    split_array = [group[0] if group[0] else group[1] for group in split_array]
    return split_array


execute_generator_function = lambda genObject: all(x is not None for x in genObject)


def cli_infer(com):
    (
        model_name,
        source_audio_path,
        output_file_name,
        feature_index_path,
        speaker_id,
        transposition,
        f0_method,
        crepe_hop_length,
        harvest_median_filter,
        resample,
        mix,
        feature_ratio,
        protection_amnt,
        _,
        f0_min,
        f0_max,
        do_formant,
    ) = cli_split_command(com)[:17]

    speaker_id, crepe_hop_length, harvest_median_filter, resample = map(
        int, [speaker_id, crepe_hop_length, harvest_median_filter, resample]
    )
    transposition, mix, feature_ratio, protection_amnt = map(
        float, [transposition, mix, feature_ratio, protection_amnt]
    )

    if do_formant.lower() == "false":
        Quefrency = 1.0
        Timbre = 1.0
    else:
        Quefrency, Timbre = map(float, cli_split_command(com)[17:19])

    rvc_globals.DoFormant = do_formant.lower() == "true"
    rvc_globals.Quefrency = Quefrency
    rvc_globals.Timbre = Timbre

    output_message = "Infer-CLI:"
    output_path = f"audio-others/{output_file_name}"

    print(f"{output_message} Starting the inference...")
    vc_data = get_vc(model_name, protection_amnt, protection_amnt)
    print(vc_data)

    print(f"{output_message} Performing inference...")
    conversion_data = vc_single(
        speaker_id,
        source_audio_path,
        source_audio_path,
        transposition,
        None,  # f0 file support not implemented
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
        note_max=None,
        f0_autotune=False,
    )

    if "Success." in conversion_data[0]:
        print(f"{output_message} Inference succeeded. Writing to {output_path}...")
        wavfile.write(output_path, conversion_data[1][0], conversion_data[1][1])
        print(f"{output_message} Finished! Saved output to {output_path}")
    else:
        print(
            f"{output_message} Inference failed. Here's the traceback: {conversion_data[0]}"
        )


def cli_pre_process(com):
    print("Pre-process: Starting...")
    execute_generator_function(
        preprocess_dataset(*cli_split_command(com)[:3], int(cli_split_command(com)[3]))
    )
    print("Pre-process: Finished")


def cli_extract_feature(com):
    (
        model_name,
        gpus,
        num_processes,
        has_pitch_guidance,
        f0_method,
        crepe_hop_length,
        version,
    ) = cli_split_command(com)

    num_processes = int(num_processes)
    has_pitch_guidance = bool(int(has_pitch_guidance))
    crepe_hop_length = int(crepe_hop_length)

    print(
        f"Extract Feature Has Pitch: {has_pitch_guidance}"
        f"Extract Feature Version: {version}"
        "Feature Extraction: Starting..."
    )
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
    print("Feature Extraction: Finished")


def cli_train(com):
    com = cli_split_command(com)
    model_name = com[0]
    sample_rate = com[1]
    bool_flags = [bool(int(i)) for i in com[2:11]]
    version = com[11]

    pretrained_base = "pretrained/" if version == "v1" else "pretrained_v2/"

    g_pretrained_path = f"{pretrained_base}f0G{sample_rate}.pth"
    d_pretrained_path = f"{pretrained_base}f0D{sample_rate}.pth"

    print("Train-CLI: Training...")
    click_train(
        model_name,
        sample_rate,
        *bool_flags,
        g_pretrained_path,
        d_pretrained_path,
        version,
    )


def cli_train_feature(com):
    output_message = "Train Feature Index-CLI"
    print(f"{output_message}: Training... Please wait")
    execute_generator_function(train_index(*cli_split_command(com)))
    print(f"{output_message}: Done!")


def cli_extract_model(com):
    extract_small_model_process = extract_small_model(*cli_split_command(com))
    print(
        "Extract Small Model: Success!"
        if extract_small_model_process == "Success."
        else f"{extract_small_model_process}\nExtract Small Model: Failed!"
    )


def preset_apply(preset, qfer, tmbr):
    if preset:
        try:
            with open(preset, "r") as p:
                content = p.read().splitlines()
            qfer, tmbr = content[0], content[1]
            formant_apply(qfer, tmbr)
        except IndexError:
            print("Error: File does not have enough lines to read 'qfer' and 'tmbr'")
        except FileNotFoundError:
            print("Error: File does not exist")
        except Exception as e:
            print("An unexpected error occurred", e)

    return (
        {"value": qfer, "__type__": "update"},
        {"value": tmbr, "__type__": "update"},
    )


def print_page_details():
    page_description = {
        "HOME": "\n    go home            : Takes you back to home with a navigation list."
        "\n    go infer           : Takes you to inference command execution."
        "\n    go pre-process     : Takes you to training step.1) pre-process command execution."
        "\n    go extract-feature : Takes you to training step.2) extract-feature command execution."
        "\n    go train           : Takes you to training step.3) being or continue training command execution."
        "\n    go train-feature   : Takes you to the train feature index command execution."
        "\n    go extract-model   : Takes you to the extract small model command execution.",
        "INFER": "\n    arg 1) model name with .pth in ./weights: mi-test.pth"
        "\n    arg 2) source audio path: myFolder\\MySource.wav"
        "\n    arg 3) output file name to be placed in './audio-others': MyTest.wav"
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
        "\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2",
        "PRE-PROCESS": "\n    arg 1) Model folder name in ./logs: mi-test"
        "\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set"
        "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
        "\n    arg 4) Number of CPU threads to use: 8 \n"
        "\nExample: mi-test mydataset 40k 24",
        "EXTRACT-FEATURE": "\n    arg 1) Model folder name in ./logs: mi-test"
        "\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)"
        "\n    arg 3) Number of CPU threads to use: 8"
        "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
        "\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)"
        "\n    arg 6) Crepe hop length: 128"
        "\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n"
        "\nExample: mi-test 0 24 1 harvest 128 v2",
        "TRAIN": "\n    arg 1) Model folder name in ./logs: mi-test"
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
        "\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2",
        "TRAIN-FEATURE": "\n    arg 1) Model folder name in ./logs: mi-test"
        "\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n"
        "\nExample: mi-test v2",
        "EXTRACT-MODEL": "\n    arg 1) Model Path: logs/mi-test/G_168000.pth"
        "\n    arg 2) Model save name: MyModel"
        "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
        "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
        '\n    arg 5) Model information: "My Model"'
        "\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n"
        '\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2',
    }

    print(page_description.get(cli_current_page, "Invalid page"))


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
        try:
            execute_command(input())
        except Exception as e:
            print(f"An error occurred: {traceback.format_exc()}")


if config.is_cli:
    print(
        "\n\nMangio-RVC-Fork v2 CLI App!\n"
        "Welcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.\n"
    )
    cli_navigation_loop()

"""
def get_presets():
    data = None
    with open('../inference-presets.json', 'r') as file:
        data = json.load(file)
    preset_names = []
    for preset in data['presets']:
        preset_names.append(preset['name'])
    
    return preset_names
"""


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
    if mim:
        try:
            with open("csvdb/stop.csv", "w+") as file:
                file.write("True")
            os.kill(PID, SIGTERM)
        except Exception as e:
            print(f"Couldn't click due to {e}")
        return (
            {"visible": True, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )
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
        new_path = "./audios/" + new_name
        shutil.move(path_to_file, new_path)
        return new_name


def save_to_wav2_edited(dropbox):
    if dropbox is None:
        pass
    else:
        file_path = dropbox.name
        target_path = os.path.join("audios", os.path.basename(file_path))

        if os.path.exists(target_path):
            os.remove(target_path)
            print("Replacing old dropdown file...")

        shutil.move(file_path, target_path)
    return


def save_to_wav2(dropbox):
    file_path = dropbox.name
    target_path = os.path.join("audios", os.path.basename(file_path))

    if os.path.exists(target_path):
        os.remove(target_path)
        print("Replacing old dropdown file...")

    shutil.move(file_path, target_path)
    return target_path


from gtts import gTTS
import edge_tts
import asyncio


def vc_single_tts(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
    f0_autotune,
    rmvpe_onnx,
):
    global tgt_sr, net_g, vc, hubert_model, version, cpt
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # reemplace for 2
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_autotune,
            rmvpe_onnx,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def custom_voice(
    _values,  # filter indices
    audio_files,  # all audio files
    model_voice_path="",
    transpose=0,
    f0method="pm",
    index_rate_=float(0.66),
    crepe_hop_length_=float(64),
    f0_autotune=False,
    rmvpe_onnx=False,
    file_index="",
    file_index2="",
):
    # hubert_model = None

    get_vc(
        sid=model_voice_path,  # model path
        to_return_protect0=0.33,
        to_return_protect1=0.33,
    )

    for _value_item in _values:
        filename = (
            "audio2/" + audio_files[_value_item]
            if _value_item != "converted_tts"
            else audio_files[0]
        )
        # filename = "audio2/"+audio_files[_value_item]
        try:
            print(audio_files[_value_item], model_voice_path)
        except:
            pass

        info_, (sample_, audio_output_) = vc_single_tts(
            sid=0,
            input_audio_path=filename,  # f"audio2/{filename}",
            f0_up_key=transpose,  # transpose for m to f and reverse 0 12
            f0_file=None,
            f0_method=f0method,
            file_index=file_index,  # dir pwd?
            file_index2=file_index2,
            # file_big_npy1,
            index_rate=index_rate_,
            filter_radius=int(3),
            resample_sr=int(0),
            rms_mix_rate=float(0.25),
            protect=float(0.33),
            crepe_hop_length=crepe_hop_length_,
            f0_autotune=f0_autotune,
            rmvpe_onnx=rmvpe_onnx,
        )

        sf.write(
            file=filename,  # f"audio2/{filename}",
            samplerate=sample_,
            data=audio_output_,
        )


def cast_to_device(tensor, device):
    try:
        return tensor.to(device)
    except Exception as e:
        print(e)
        return tensor


def __bark__(text, voice_preset):
    os.makedirs(os.path.join(now_dir, "tts"), exist_ok=True)
    from transformers import AutoProcessor, BarkModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if "cpu" in device else torch.float16
    bark_processor = AutoProcessor.from_pretrained(
        "suno/bark",
        cache_dir=os.path.join(now_dir, "tts", "suno/bark"),
        torch_dtype=dtype,
    )
    bark_model = BarkModel.from_pretrained(
        "suno/bark",
        cache_dir=os.path.join(now_dir, "tts", "suno/bark"),
        torch_dtype=dtype,
    ).to(device)
    # bark_model.enable_cpu_offload()
    inputs = bark_processor(text=[text], return_tensors="pt", voice_preset=voice_preset)
    tensor_dict = {
        k: cast_to_device(v, device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }
    speech_values = bark_model.generate(**tensor_dict, do_sample=True)
    sampling_rate = bark_model.generation_config.sample_rate
    speech = speech_values.cpu().numpy().squeeze()
    return speech, sampling_rate


def make_test(
    tts_text,
    tts_voice,
    model_path,
    index_path,
    transpose,
    f0_method,
    index_rate,
    crepe_hop_length,
    f0_autotune,
    tts_method,
):
    if tts_voice == None:
        return

    filename = os.path.join(now_dir, "audio-outputs", "converted_tts.wav")
    rmvpe_onnx = True if f0_method == "rmvpe_onnx" else False
    if "SET_LIMIT" == os.getenv("DEMO"):
        if len(tts_text) > 60:
            tts_text = tts_text[:60]
            print("DEMO; limit to 60 characters")

    language = tts_voice[:2]
    if tts_method == "Edge-tts":
        try:
            # nest_asyncio.apply() # gradio;not
            asyncio.run(
                edge_tts.Communicate(
                    tts_text, "-".join(tts_voice.split("-")[:-1])
                ).save(filename)
            )
        except:
            try:
                tts = gTTS(tts_text, lang=language)
                tts.save(filename)
                tts.save
                print(
                    f"No audio was received. Please change the tts voice for {tts_voice}. USING gTTS."
                )
            except:
                tts = gTTS("a", lang=language)
                tts.save(filename)
                print("Error: Audio will be replaced.")

        os.system("cp audio-outputs/converted_tts.wav audio-outputs/real_tts.wav")

        custom_voice(
            ["converted_tts"],  # filter indices
            ["audio-outputs/converted_tts.wav"],  # all audio files
            model_voice_path=model_path,
            transpose=transpose,
            f0method=f0_method,
            index_rate_=index_rate,
            crepe_hop_length_=crepe_hop_length,
            f0_autotune=f0_autotune,
            rmvpe_onnx=rmvpe_onnx,
            file_index="",
            file_index2=index_path,
        )
        return os.path.join(
            now_dir, "audio-outputs", "converted_tts.wav"
        ), os.path.join(now_dir, "audio-outputs", "real_tts.wav")
    elif tts_method == "Bark-tts":
        try:
            get_vc(
                sid=model_path,  # model path
                to_return_protect0=0.33,
                to_return_protect1=0.33,
            )
            script = tts_text.replace("\n", " ").strip()
            sentences = sent_tokenize(script)
            print(sentences)
            silence = np.zeros(int(0.25 * SAMPLE_RATE))
            pieces = []
            nombre_archivo = os.path.join(now_dir, "audio-outputs", "bark_out.wav")
            for sentence in sentences:
                audio_array, _ = __bark__(sentence, tts_voice.split("-")[0])
                pieces += [audio_array, silence.copy()]

            sf.write(
                file=nombre_archivo, samplerate=SAMPLE_RATE, data=np.concatenate(pieces)
            )
            info_, (sample_, audio_output_) = vc_single_tts(
                sid=0,
                input_audio_path=os.path.join(
                    now_dir, "audio-outputs", "bark_out.wav"
                ),  # f"audio2/{filename}",
                f0_up_key=transpose,  # transpose for m to f and reverse 0 12
                f0_file=None,
                f0_method=f0_method,
                file_index="",  # dir pwd?
                file_index2=index_path,
                # file_big_npy1,
                index_rate=index_rate,
                filter_radius=int(3),
                resample_sr=int(0),
                rms_mix_rate=float(0.25),
                protect=float(0.33),
                crepe_hop_length=crepe_hop_length,
                f0_autotune=f0_autotune,
                rmvpe_onnx=rmvpe_onnx,
            )
            wavfile.write(
                os.path.join(now_dir, "audio-outputs", "converted_bark.wav"),
                rate=sample_,
                data=audio_output_,
            )
            return (
                os.path.join(now_dir, "audio-outputs", "converted_bark.wav"),
                nombre_archivo,
            )

        except Exception as e:
            print(f"{e}")
            return None, None


def GradioSetup(UTheme=gr.themes.Soft()):
    default_weight = names[0] if names else ""

    with gr.Blocks(theme="JohnSmith9982/small_and_pretty", title="Applio") as app:
        gr.HTML("<h1> 🍏 Applio (Mangio-RVC-Fork) </h1>")
        with gr.Tabs():
            with gr.TabItem(i18n("Model Inference")):
                with gr.Row():
                    sid0 = gr.Dropdown(
                        label=i18n("Inferencing voice:"),
                        choices=sorted(names),
                        value=default_weight,
                    )
                    refresh_button = gr.Button(i18n("Refresh"), variant="primary")
                    clean_button = gr.Button(
                        i18n("Unload voice to save GPU memory"), variant="primary"
                    )
                    clean_button.click(
                        fn=lambda: ({"value": "", "__type__": "update"}),
                        inputs=[],
                        outputs=[sid0],
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

                    with gr.Group():
                        with gr.Row():
                            with gr.Column():  # First column for audio-related inputs
                                dropbox = gr.File(label=i18n("Drag your audio here:"))
                                record_button = gr.Audio(
                                    source="microphone",
                                    label=i18n("Or record an audio:"),
                                    type="filepath",
                                )
                                input_audio0 = gr.Textbox(
                                    label=i18n(
                                        "Manual path to the audio file to be processed"
                                    ),
                                    value=os.path.join(
                                        now_dir, "audios", "someguy.mp3"
                                    ),
                                    visible=False,
                                )
                                input_audio1 = gr.Dropdown(
                                    label=i18n(
                                        "Auto detect audio path and select from the dropdown:"
                                    ),
                                    choices=sorted(audio_paths),
                                    value="",
                                    interactive=True,
                                )

                                input_audio1.select(
                                    fn=lambda: "", inputs=[], outputs=[input_audio0]
                                )
                                input_audio0.input(
                                    fn=lambda: "", inputs=[], outputs=[input_audio1]
                                )

                                dropbox.upload(
                                    fn=save_to_wav2,
                                    inputs=[dropbox],
                                    outputs=[input_audio0],
                                )
                                dropbox.upload(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[input_audio1],
                                )
                                record_button.change(
                                    fn=save_to_wav,
                                    inputs=[record_button],
                                    outputs=[input_audio0],
                                )
                                record_button.change(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[input_audio1],
                                )

                            best_match_index_path1, _ = match_index(
                                sid0.value
                            )  # Get initial index from default sid0 (first voice model in list)

                            with gr.Column():  # Second column for pitch shift and other options
                                file_index2 = gr.Dropdown(
                                    label=i18n(
                                        "Auto-detect index path and select from the dropdown:"
                                    ),
                                    choices=get_indexes(),
                                    value=best_match_index_path1,
                                    interactive=True,
                                    allow_custom_value=True,
                                )
                                index_rate1 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n("Search feature ratio:"),
                                    value=0.75,
                                    interactive=True,
                                )
                                refresh_button.click(
                                    fn=change_choices,
                                    inputs=[],
                                    outputs=[sid0, file_index2, input_audio1],
                                )
                                with gr.Column():
                                    vc_transform0 = gr.Number(
                                        label=i18n(
                                            "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):"
                                        ),
                                        value=0,
                                    )

                    # Advanced settings container
                    with gr.Group():
                        with gr.Accordion(label=i18n("Advanced Settings"), open=False):
                            with gr.Row():
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
                                            "rmvpe_onnx",
                                            "rmvpe+",
                                        ],
                                        value="rmvpe+",
                                        interactive=True,
                                    )
                                    f0_autotune = gr.Checkbox(
                                        label="Enable autotune", interactive=True
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
                                        info=i18n(
                                            "Specify max pitch for inference [HZ]"
                                        ),
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
                                        label=i18n(
                                            "Feature search database file path:"
                                        ),
                                        value="",
                                        interactive=True,
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

                                    f0_file = gr.File(
                                        label=i18n(
                                            "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:"
                                        )
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
                                        i18n("Apply"),
                                        variant="primary",
                                        visible=bool(DoFormant),
                                    )
                                formant_preset.change(
                                    fn=preset_apply,
                                    inputs=[formant_preset, qfrency, tmbre],
                                    outputs=[qfrency, tmbre],
                                )
                                formanting.change(
                                    fn=formant_enabled,
                                    inputs=[formanting, qfrency, tmbre],
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

                    with gr.Row():  # Defines output info + output audio download after conversion
                        vc_output1 = gr.Textbox(label=i18n("Output information:"))
                        vc_output2 = gr.Audio(
                            label=i18n(
                                "Export audio (click on the three dots in the lower right corner to download)"
                            )
                        )
                    but0 = gr.Button(i18n("Convert"), variant="primary").style(
                        full_width=True
                    )
                    with gr.Group():  # I think this defines the big convert button
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
                                    minpitch_slider,
                                    minpitch_txtbox,
                                    maxpitch_slider,
                                    maxpitch_txtbox,
                                    f0_autotune,
                                ],
                                [vc_output1, vc_output2],
                            )

                with gr.TabItem(i18n("Batch")):  # Dont Change
                    with gr.Group():  # Markdown explanation of batch inference
                        with gr.Row():
                            with gr.Column():
                                vc_transform1 = gr.Number(
                                    label=i18n(
                                        "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):"
                                    ),
                                    value=0,
                                )
                                opt_input = gr.Textbox(
                                    label=i18n("Specify output folder:"), value="opt"
                                )
                                file_index4 = gr.Dropdown(
                                    label=i18n(
                                        "Auto-detect index path and select from the dropdown:"
                                    ),
                                    choices=get_indexes(),
                                    value=best_match_index_path1,
                                    interactive=True,
                                )
                                sid0.select(
                                    fn=match_index,
                                    inputs=[sid0],
                                    outputs=[file_index2, file_index4],
                                )

                                refresh_button.click(
                                    fn=lambda: change_choices()[1],
                                    inputs=[],
                                    outputs=file_index4,
                                )
                                index_rate2 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=i18n("Search feature ratio:"),
                                    value=0.75,
                                    interactive=True,
                                )
                            with gr.Column():
                                dir_input = gr.Textbox(
                                    label=i18n(
                                        "Enter the path of the audio folder to be processed (copy it from the address bar of the file manager):"
                                    ),
                                    value=os.path.join(now_dir, "audios"),
                                    lines=2,
                                )
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
                                                "rmvpe_onnx",
                                                "rmvpe+",
                                            ],
                                            value="rmvpe+",
                                            interactive=True,
                                        )
                                        f0_autotune = gr.Checkbox(
                                            label="Enable autotune", interactive=True
                                        )
                                        format1 = gr.Radio(
                                            label=i18n("Export file format"),
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
                            vc_output3 = gr.Textbox(label=i18n("Output information:"))
                            but1 = gr.Button(i18n("Convert"), variant="primary")
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
                                    minpitch_slider
                                    if (not rvc_globals.NotesOrHertz)
                                    else minpitch_txtbox,
                                    maxpitch_slider
                                    if (not rvc_globals.NotesOrHertz)
                                    else maxpitch_txtbox,
                                    f0_autotune,
                                ],
                                [vc_output3],
                            )

                    sid0.change(
                        fn=get_vc,
                        inputs=[sid0, protect0, protect1],
                        outputs=[spk_item, protect0, protect1],
                    )

                    spk_item, protect0, protect1 = get_vc(
                        sid0.value, protect0, protect1
                    )

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
                        exp_dir1 = gr.Textbox(
                            label=i18n("Enter the model name:"),
                            value=i18n("Model_Name"),
                        )
                        sr2 = gr.Radio(
                            label=i18n("Target sample rate:"),
                            choices=["40k", "48k", "32k"],
                            value="40k",
                            interactive=True,
                        )
                        if_f0_3 = gr.Checkbox(
                            label=i18n("Whether the model has pitch guidance."),
                            value=True,
                            interactive=True,
                        )
                        version19 = gr.Radio(
                            label=i18n("Version:"),
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                            visible=True,
                        )
                        np7 = gr.Slider(
                            minimum=0,
                            maximum=config.n_cpu,
                            step=1,
                            label=i18n("Number of CPU processes:"),
                            value=int(np.ceil(config.n_cpu / 1.5)),
                            interactive=True,
                        )
                with gr.Group():
                    with gr.Accordion(label=i18n("Step 2: Skipping pitch extraction")):
                        with gr.Row():
                            #  trainset_dir4 = gr.Textbox(
                            #      label=i18n("Enter the path of the training folder:"), value=os.path.join(now_dir, datasets_root)
                            #  )
                            with gr.Column():
                                trainset_dir4 = gr.Dropdown(
                                    choices=sorted(datasets),
                                    label=i18n("Select your dataset:"),
                                    value=get_dataset(),
                                )
                                btn_update_dataset_list = gr.Button(
                                    i18n("Update list"), variant="primary"
                                )
                            spk_id5 = gr.Slider(
                                minimum=0,
                                maximum=4,
                                step=1,
                                label=i18n("Specify the model ID:"),
                                value=0,
                                interactive=True,
                            )
                            btn_update_dataset_list.click(
                                easy_infer.update_dataset_list, [spk_id5], trainset_dir4
                            )
                            but1 = gr.Button(i18n("Process data"), variant="primary")
                            info1 = gr.Textbox(
                                label=i18n("Output information:"), value=""
                            )
                            but1.click(
                                preprocess_dataset,
                                [trainset_dir4, exp_dir1, sr2, np7],
                                [info1],
                            )
                with gr.Group():
                    with gr.Accordion(label=i18n("Step 3: Extracting features")):
                        with gr.Row():
                            with gr.Column():
                                gpus6 = gr.Textbox(
                                    label=i18n(
                                        "Provide the GPU index(es) separated by '-', like 0-1-2 for using GPUs 0, 1, and 2:"
                                    ),
                                    value=gpus,
                                    interactive=True,
                                )
                                gr.Textbox(
                                    label=i18n("GPU Information:"), value=gpu_info
                                )
                            with gr.Column():
                                f0method8 = gr.Radio(
                                    label=i18n(
                                        "Select the pitch extraction algorithm:"
                                    ),
                                    choices=[
                                        "pm",
                                        "harvest",
                                        "dio",
                                        "crepe",
                                        "mangio-crepe",
                                        "rmvpe",
                                    ],
                                    # [ MANGIO ]: Fork feature: Crepe on f0 extraction for training.
                                    value="rmvpe",
                                    interactive=True,
                                )

                                extraction_crepe_hop_length = gr.Slider(
                                    minimum=1,
                                    maximum=512,
                                    step=1,
                                    label=i18n(
                                        "Mangio-Crepe Hop Length (Only applies to mangio-crepe): Hop length refers to the time it takes for the speaker to jump to a dramatic pitch. Lower hop lengths take more time to infer but are more pitch accurate."
                                    ),
                                    value=64,
                                    interactive=True,
                                    visible=False,
                                )

                                f0method8.change(
                                    fn=lambda radio: (
                                        {
                                            "visible": radio
                                            in ["mangio-crepe", "mangio-crepe-tiny"],
                                            "__type__": "update",
                                        }
                                    ),
                                    inputs=[f0method8],
                                    outputs=[extraction_crepe_hop_length],
                                )
                            but2 = gr.Button(
                                i18n("Feature extraction"), variant="primary"
                            )
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
                                    extraction_crepe_hop_length,
                                ],
                                [info2],
                            )
                with gr.Group():
                    with gr.Row():
                        with gr.Accordion(label=i18n("Step 4: Model training started")):
                            with gr.Row():
                                save_epoch10 = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    label=i18n("Save frequency:"),
                                    value=10,
                                    interactive=True,
                                    visible=True,
                                )
                                total_epoch11 = gr.Slider(
                                    minimum=1,
                                    maximum=10000,
                                    step=2,
                                    label=i18n("Training epochs:"),
                                    value=750,
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

                            with gr.Row():
                                pretrained_G14 = gr.Textbox(
                                    lines=4,
                                    label=i18n("Load pre-trained base model G path:"),
                                    value="pretrained_v2/f0G40k.pth",
                                    interactive=True,
                                )
                                pretrained_D15 = gr.Textbox(
                                    lines=4,
                                    label=i18n("Load pre-trained base model D path:"),
                                    value="pretrained_v2/f0D40k.pth",
                                    interactive=True,
                                )
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
                                if_f0_3.change(
                                    fn=lambda radio: (
                                        {
                                            "visible": radio
                                            in ["mangio-crepe", "mangio-crepe-tiny"],
                                            "__type__": "update",
                                        }
                                    ),
                                    inputs=[f0method8],
                                    outputs=[extraction_crepe_hop_length],
                                )

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
                                )
                                butstop.click(
                                    fn=stoptraining,
                                    inputs=[gr.Number(value=1, visible=False)],
                                    outputs=[but3, butstop],
                                )

                                with gr.Column():
                                    info3 = gr.Textbox(
                                        label=i18n("Output information:"),
                                        value="",
                                        max_lines=4,
                                    )
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

                                    but7 = gr.Button(
                                        i18n("Save model"), variant="primary"
                                    )
                                    but4 = gr.Button(
                                        i18n("Train feature index"), variant="primary"
                                    )

                                if_save_every_weights18.change(
                                    fn=lambda if_save_every_weights: (
                                        {
                                            "visible": if_save_every_weights,
                                            "__type__": "update",
                                        }
                                    ),
                                    inputs=[if_save_every_weights18],
                                    outputs=[save_epoch10],
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
                            but7.click(
                                easy_infer.save_model, [exp_dir1, save_action], info3
                            )
                with gr.Group():
                    with gr.Row():
                        with gr.Accordion(
                            label=i18n(
                                "Step 5: Export lowest points on a graph of the model"
                            )
                        ):
                            lowestval_weight_dir = gr.Textbox(visible=False)
                            ds = gr.Textbox(visible=False)
                            weights_dir1 = gr.Textbox(visible=False, value=weights_dir)

                            with gr.Row():
                                amntlastmdls = gr.Slider(
                                    minimum=1,
                                    maximum=25,
                                    label=i18n("How many lowest points to save:"),
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )
                                lpexport = gr.Button(
                                    value=i18n("Export lowest points of a model"),
                                    variant="primary",
                                )
                                lw_mdls = gr.File(
                                    file_count="multiple",
                                    label=i18n("Output models:"),
                                    interactive=False,
                                )  #####

                            with gr.Row():
                                infolpex = gr.Textbox(
                                    label=i18n("Output information:"),
                                    value="",
                                    max_lines=10,
                                )
                                mdlbl = gr.Dataframe(
                                    label=i18n("Stats of selected models:"),
                                    datatype="number",
                                    type="pandas",
                                )

                            lpexport.click(
                                lambda model_name: os.path.join(
                                    "logs", model_name, "lowestvals"
                                ),
                                inputs=[exp_dir1],
                                outputs=[lowestval_weight_dir],
                            )

                            lpexport.click(
                                fn=tensorlowest.main,
                                inputs=[exp_dir1, save_epoch10, amntlastmdls],
                                outputs=[ds],
                            )

                            ds.change(
                                fn=tensorlowest.selectweights,
                                inputs=[
                                    exp_dir1,
                                    ds,
                                    weights_dir1,
                                    lowestval_weight_dir,
                                ],
                                outputs=[infolpex, lw_mdls, mdlbl],
                            )
            with gr.TabItem(i18n("UVR5")):  # UVR section
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            model_select = gr.Radio(
                                label=i18n("Model Architecture:"),
                                choices=["VR", "MDX"],
                                value="VR",
                                interactive=True,
                            )
                            dir_wav_input = gr.Textbox(
                                label=i18n(
                                    "Enter the path of the audio folder to be processed:"
                                ),
                                value=os.path.join(now_dir, "audios"),
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
                                value="opt",
                            )
                            opt_ins_root = gr.Textbox(
                                label=i18n(
                                    "Specify the output folder for accompaniment:"
                                ),
                                value="opt",
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
                        )
            with gr.TabItem(i18n("TTS")):
                with gr.Group():
                    with gr.Column():
                        text_test = gr.Textbox(
                            label=i18n("Text:"),
                            placeholder=i18n(
                                "Enter the text you want to convert to voice..."
                            ),
                            lines=6,
                        )

                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            tts_methods_voice = ["Edge-tts", "Bark-tts"]
                            ttsmethod_test = gr.Dropdown(
                                tts_methods_voice,
                                value="Edge-tts",
                                label=i18n("TTS Method:"),
                                visible=True,
                            )
                            tts_test = gr.Dropdown(
                                set_edge_voice, label=i18n("TTS Model:"), visible=True
                            )
                            ttsmethod_test.change(
                                fn=update_tts_methods_voice,
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
                            # transpose_test = gr.Number(label = i18n('Transpose (integer, number Fof semitones, raise by an octave: 12, lower by an octave: -12):'), value=0, visible=True, interactive= True)

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
                    make_test,
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
                easy_infer.download_model()
                easy_infer.download_backup()
                easy_infer.download_dataset(trainset_dir4)
                easy_infer.download_audio()
                easy_infer.youtube_separator()
            with gr.TabItem(i18n("Extra")):
                gr.Markdown(
                    value=i18n(
                        "This section contains some extra utilities that often may be in experimental phases"
                    )
                )
                with gr.TabItem(i18n("Merge Audios")):
                    with gr.Group():
                        gr.Markdown(
                            value="## "
                            + i18n("Merge your generated audios with the instrumental")
                        )
                        gr.Markdown(value="", scale="-0.5", visible=True)
                        gr.Markdown(value="", scale="-0.5", visible=True)
                        with gr.Row():
                            with gr.Column():
                                dropbox = gr.File(label=i18n("Drag your audio here:"))
                                gr.Markdown(value=i18n("### Instrumental settings:"))
                                input_audio1 = gr.Dropdown(
                                    label=i18n("Choose your instrumental:"),
                                    choices=sorted(audio_others_paths),
                                    value="",
                                    interactive=True,
                                )
                                input_audio1_scale = gr.Slider(
                                    minimum=0,
                                    maximum=10,
                                    label=i18n("Volume of the instrumental audio:"),
                                    value=1.00,
                                    interactive=True,
                                )
                                gr.Markdown(value=i18n("### Audio settings:"))
                                input_audio3 = gr.Dropdown(
                                    label=i18n("Select the generated audio"),
                                    choices=sorted(audio_paths),
                                    value="",
                                    interactive=True,
                                )
                                with gr.Row():
                                    input_audio3_scale = gr.Slider(
                                        minimum=0,
                                        maximum=10,
                                        label=i18n("Volume of the generated audio:"),
                                        value=1.00,
                                        interactive=True,
                                    )

                                gr.Markdown(value=i18n("### Add the effects:"))
                                reverb_ = gr.Checkbox(
                                    label=i18n("Reverb"),
                                    value=False,
                                    interactive=True,
                                )
                                compressor_ = gr.Checkbox(
                                    label=i18n("Compressor"),
                                    value=False,
                                    interactive=True,
                                )
                                noise_gate_ = gr.Checkbox(
                                    label=i18n("Noise Gate"),
                                    value=False,
                                    interactive=True,
                                )

                                butnone = gr.Button(
                                    i18n("Merge"), variant="primary"
                                ).style(full_width=True)

                                vc_output1 = gr.Textbox(
                                    label=i18n("Output information:")
                                )
                                vc_output2 = gr.Audio(
                                    label=i18n(
                                        "Export audio (click on the three dots in the lower right corner to download)"
                                    ),
                                    type="filepath",
                                )

                                dropbox.upload(
                                    fn=save_to_wav2,
                                    inputs=[dropbox],
                                    outputs=[input_audio1],
                                )
                                dropbox.upload(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[input_audio1],
                                )

                                refresh_button.click(
                                    fn=lambda: change_choices3(),
                                    inputs=[],
                                    outputs=[input_audio1, input_audio3],
                                )

                                butnone.click(
                                    fn=audio_combined,
                                    inputs=[
                                        input_audio1,
                                        input_audio3,
                                        input_audio1_scale,
                                        input_audio3_scale,
                                        reverb_,
                                        compressor_,
                                        noise_gate_,
                                    ],
                                    outputs=[vc_output1, vc_output2],
                                )

                with gr.TabItem(i18n("Processing")):
                    with gr.Group():
                        with gr.Accordion(
                            label=i18n(
                                "Model fusion, can be used to test timbre fusion"
                            )
                        ):
                            with gr.Row():
                                with gr.Column():
                                    name_to_save0 = gr.Textbox(
                                        label=i18n("Name:"),
                                        value="",
                                        max_lines=1,
                                        interactive=True,
                                        placeholder=i18n("Name for saving"),
                                    )
                                    alpha_a = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("Weight for Model A:"),
                                        value=0.5,
                                        interactive=True,
                                    )
                                    if_f0_ = gr.Checkbox(
                                        label=i18n(
                                            "Whether the model has pitch guidance."
                                        ),
                                        value=True,
                                        interactive=True,
                                    )
                                    version_2 = gr.Radio(
                                        label=i18n("Model architecture version:"),
                                        choices=["v1", "v2"],
                                        value="v2",
                                        interactive=True,
                                    )
                                    sr_ = gr.Radio(
                                        label=i18n("Target sample rate:"),
                                        choices=["40k", "48k"],
                                        value="40k",
                                        interactive=True,
                                    )

                                with gr.Column():
                                    ckpt_a = gr.Textbox(
                                        label=i18n("Path to Model A:"),
                                        value="",
                                        interactive=True,
                                        placeholder=i18n("Path to model"),
                                    )

                                    ckpt_b = gr.Textbox(
                                        label=i18n("Path to Model B:"),
                                        value="",
                                        interactive=True,
                                        placeholder=i18n("Path to model"),
                                    )

                                    info__ = gr.Textbox(
                                        label=i18n("Model information to be placed:"),
                                        value="",
                                        max_lines=8,
                                        interactive=True,
                                        placeholder=i18n(
                                            "Model information to be placed"
                                        ),
                                    )
                                    info4 = gr.Textbox(
                                        label=i18n("Output information:"),
                                        value="",
                                        max_lines=8,
                                    )

                            but6 = gr.Button(i18n("Fusion"), variant="primary")

                            but6.click(
                                merge,
                                [
                                    ckpt_a,
                                    ckpt_b,
                                    alpha_a,
                                    sr_,
                                    if_f0_,
                                    info__,
                                    name_to_save0,
                                    version_2,
                                ],
                                info4,
                            )  # def merge(path1,path2,alpha1,sr,f0,info):
                    with gr.Group():
                        with gr.Accordion(label=i18n("Modify model information")):
                            with gr.Row():  ######
                                with gr.Column():
                                    ckpt_path0 = gr.Textbox(
                                        label=i18n("Path to Model:"),
                                        value="",
                                        interactive=True,
                                        placeholder=i18n("Path to model"),
                                    )
                                    info_ = gr.Textbox(
                                        label=i18n("Model information to be modified:"),
                                        value="",
                                        max_lines=8,
                                        interactive=True,
                                        placeholder=i18n(
                                            "Model information to be placed"
                                        ),
                                    )

                                with gr.Column():
                                    name_to_save1 = gr.Textbox(
                                        label=i18n("Save file name:"),
                                        placeholder=i18n("Name for saving"),
                                        value="",
                                        max_lines=8,
                                        interactive=True,
                                    )

                                    info5 = gr.Textbox(
                                        label=i18n("Output information:"),
                                        value="",
                                        max_lines=8,
                                    )
                            but7 = gr.Button(i18n("Modify"), variant="primary")
                            but7.click(
                                change_info, [ckpt_path0, info_, name_to_save1], info5
                            )
                    with gr.Group():
                        with gr.Accordion(label=i18n("View model information")):
                            with gr.Row():
                                with gr.Column():
                                    ckpt_path1 = gr.Textbox(
                                        label=i18n("Path to Model:"),
                                        value="",
                                        interactive=True,
                                        placeholder=i18n("Path to model"),
                                    )

                                    info6 = gr.Textbox(
                                        label=i18n("Output information:"),
                                        value="",
                                        max_lines=8,
                                    )
                                    but8 = gr.Button(i18n("View"), variant="primary")
                            but8.click(show_info, [ckpt_path1], info6)
                    with gr.Group():
                        with gr.Accordion(label=i18n("Model extraction")):
                            with gr.Row():
                                with gr.Column():
                                    save_name = gr.Textbox(
                                        label=i18n("Name:"),
                                        value="",
                                        interactive=True,
                                        placeholder=i18n("Name for saving"),
                                    )
                                    if_f0__ = gr.Checkbox(
                                        label=i18n(
                                            "Whether the model has pitch guidance."
                                        ),
                                        value=True,
                                        interactive=True,
                                    )
                                    version_1 = gr.Radio(
                                        label=i18n("Model architecture version:"),
                                        choices=["v1", "v2"],
                                        value="v2",
                                        interactive=True,
                                    )
                                    sr__ = gr.Radio(
                                        label=i18n("Target sample rate:"),
                                        choices=["32k", "40k", "48k"],
                                        value="40k",
                                        interactive=True,
                                    )

                                with gr.Column():
                                    ckpt_path2 = gr.Textbox(
                                        label=i18n("Path to Model:"),
                                        placeholder=i18n("Path to model"),
                                        interactive=True,
                                    )
                                    info___ = gr.Textbox(
                                        label=i18n("Model information to be placed:"),
                                        value="",
                                        max_lines=8,
                                        interactive=True,
                                        placeholder=i18n(
                                            "Model information to be placed"
                                        ),
                                    )
                                    info7 = gr.Textbox(
                                        label=i18n("Output information:"),
                                        value="",
                                        max_lines=8,
                                    )

                            with gr.Row():
                                but9 = gr.Button(i18n("Extract"), variant="primary")
                                ckpt_path2.change(
                                    change_info_,
                                    [ckpt_path2],
                                    [sr__, if_f0__, version_1],
                                )
                            but9.click(
                                extract_small_model,
                                [
                                    ckpt_path2,
                                    save_name,
                                    sr__,
                                    if_f0__,
                                    info___,
                                    version_1,
                                ],
                                info7,
                            )

            with gr.TabItem(i18n("Settings")):
                with gr.Row():
                    gr.Markdown(value=i18n("Pitch settings"))
                    noteshertz = gr.Checkbox(
                        label=i18n(
                            "Whether to use note names instead of their hertz value. E.G. [C5, D6] instead of [523.25, 1174.66]Hz"
                        ),
                        value=rvc_globals.NotesOrHertz,
                        interactive=True,
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
            favicon_path="./images/icon.png",
            share=share_gradio_link,
        )
    else:
        app.queue(concurrency_count=concurrency_count, max_size=max_size).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
            favicon_path=".\images\icon.png",
            share=share_gradio_link,
        )


if __name__ == "__main__":
    if os.name == "nt":
        print(
            i18n(
                "Any ConnectionResetErrors post-conversion are irrelevant and purely visual; they can be ignored.\n"
            )
        )
    app = GradioSetup(UTheme=config.grtheme)
    GradioRun(app)
