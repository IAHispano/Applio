import os
import sys
import json
import argparse
import subprocess

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.config import Config

from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.lib.tools.pretrained_selector import pretrained_selector

from rvc.train.process.model_blender import model_blender
from rvc.train.process.model_information import model_information
from rvc.train.process.extract_small_model import extract_small_model

from rvc.infer.infer import VoiceConverter

infer_pipeline = VoiceConverter().infer_pipeline

from rvc.lib.tools.analyzer import analyze_audio

from rvc.lib.tools.launch_tensorboard import launch_tensorboard_pipeline

from rvc.lib.tools.model_download import model_download_pipeline

config = Config()
current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")

# Get TTS Voices -> https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list?trustedclienttoken=6A5AA1D4EAFF4E9FB37E23D68491D6F4
with open(os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r") as f:
    voices_data = json.load(f)


locales = list({voice["Locale"] for voice in voices_data})
python = sys.executable


# Infer
def run_infer_script(
    f0_up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0_method,
    input_path,
    output_path,
    pth_path,
    index_path,
    split_audio,
    f0_autotune,
    clean_audio,
    clean_strength,
    export_format,
    embedder_model,
    embedder_model_custom,
    upscale_audio,
    f0_file,
):
    f0_autotune = "True" if str(f0_autotune) == "True" else "False"
    clean_audio = "True" if str(clean_audio) == "True" else "False"
    upscale_audio = "True" if str(upscale_audio) == "True" else "False"
    infer_pipeline(
        f0_up_key,
        filter_radius,
        index_rate,
        rms_mix_rate,
        protect,
        hop_length,
        f0_method,
        input_path,
        output_path,
        pth_path,
        index_path,
        split_audio,
        f0_autotune,
        clean_audio,
        clean_strength,
        export_format,
        embedder_model,
        embedder_model_custom,
        upscale_audio,
        f0_file,
    )
    return f"File {input_path} inferred successfully.", output_path.replace(
        ".wav", f".{export_format.lower()}"
    )


# Batch infer
def run_batch_infer_script(
    f0_up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0_method,
    input_folder,
    output_folder,
    pth_path,
    index_path,
    split_audio,
    f0_autotune,
    clean_audio,
    clean_strength,
    export_format,
    embedder_model,
    embedder_model_custom,
    upscale_audio,
    f0_file,
):
    f0_autotune = "True" if str(f0_autotune) == "True" else "False"
    clean_audio = "True" if str(clean_audio) == "True" else "False"
    upscale_audio = "True" if str(upscale_audio) == "True" else "False"
    audio_files = [
        f for f in os.listdir(input_folder) if f.endswith((".mp3", ".wav", ".flac"))
    ]
    print(f"Detected {len(audio_files)} audio files for inference.")

    for audio_file in audio_files:
        if "_output" in audio_file:
            pass
        else:
            input_path = os.path.join(input_folder, audio_file)
            output_file_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(
                output_folder,
                f"{output_file_name}_output{os.path.splitext(audio_file)[1]}",
            )
            print(f"Inferring {input_path}...")

            infer_pipeline(
                f0_up_key,
                filter_radius,
                index_rate,
                rms_mix_rate,
                protect,
                hop_length,
                f0_method,
                input_path,
                output_path,
                pth_path,
                index_path,
                split_audio,
                f0_autotune,
                clean_audio,
                clean_strength,
                export_format,
                embedder_model,
                embedder_model_custom,
                upscale_audio,
                f0_file,
            )

    return f"Files from {input_folder} inferred successfully."


# TTS
def run_tts_script(
    tts_text,
    tts_voice,
    tts_rate,
    f0_up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0_method,
    output_tts_path,
    output_rvc_path,
    pth_path,
    index_path,
    split_audio,
    f0_autotune,
    clean_audio,
    clean_strength,
    export_format,
    embedder_model,
    embedder_model_custom,
    upscale_audio,
    f0_file,
):
    f0_autotune = "True" if str(f0_autotune) == "True" else "False"
    clean_audio = "True" if str(clean_audio) == "True" else "False"
    upscale_audio = "True" if str(upscale_audio) == "True" else "False"
    tts_script_path = os.path.join("rvc", "lib", "tools", "tts.py")

    if os.path.exists(output_tts_path):
        os.remove(output_tts_path)

    command_tts = [
        python,
        tts_script_path,
        tts_text,
        tts_voice,
        str(tts_rate),
        output_tts_path,
    ]
    subprocess.run(command_tts)

    infer_pipeline(
        f0_up_key,
        filter_radius,
        index_rate,
        rms_mix_rate,
        protect,
        hop_length,
        f0_method,
        output_tts_path,
        output_rvc_path,
        pth_path,
        index_path,
        split_audio,
        f0_autotune,
        clean_audio,
        clean_strength,
        export_format,
        embedder_model,
        embedder_model_custom,
        upscale_audio,
        f0_file,
    )

    return f"Text {tts_text} synthesized successfully.", output_rvc_path.replace(
        ".wav", f".{export_format.lower()}"
    )


# Preprocess
def run_preprocess_script(model_name, dataset_path, sampling_rate, cpu_cores):
    per = 3.0 if config.is_half else 3.7
    preprocess_script_path = os.path.join("rvc", "train", "preprocess", "preprocess.py")
    command = [
        python,
        preprocess_script_path,
        *map(
            str,
            [
                os.path.join(logs_path, model_name),
                dataset_path,
                sampling_rate,
                per,
                cpu_cores,
            ],
        ),
    ]

    os.makedirs(os.path.join(logs_path, model_name), exist_ok=True)
    subprocess.run(command)
    return f"Model {model_name} preprocessed successfully."


# Extract
def run_extract_script(
    model_name,
    rvc_version,
    f0_method,
    pitch_guidance,
    hop_length,
    cpu_cores,
    sampling_rate,
    embedder_model,
    embedder_model_custom,
):
    model_path = os.path.join(logs_path, model_name)
    extract_f0_script_path = os.path.join(
        "rvc", "train", "extract", "extract_f0_print.py"
    )
    extract_feature_script_path = os.path.join(
        "rvc", "train", "extract", "extract_feature_print.py"
    )

    command_1 = [
        python,
        extract_f0_script_path,
        *map(
            str,
            [
                model_path,
                f0_method,
                hop_length,
                cpu_cores,
            ],
        ),
    ]
    command_2 = [
        python,
        extract_feature_script_path,
        *map(
            str,
            [
                config.device,
                "1",
                "0",
                "0",
                model_path,
                rvc_version,
                "True",
                embedder_model,
                embedder_model_custom,
            ],
        ),
    ]
    subprocess.run(command_1)
    subprocess.run(command_2)

    f0 = 1 if str(pitch_guidance) == "True" else 0
    generate_config(rvc_version, sampling_rate, model_path)
    generate_filelist(f0, model_path, rvc_version, sampling_rate)
    return f"Model {model_name} extracted successfully."


# Train
def run_train_script(
    model_name,
    rvc_version,
    save_every_epoch,
    save_only_latest,
    save_every_weights,
    total_epoch,
    sampling_rate,
    batch_size,
    gpu,
    pitch_guidance,
    overtraining_detector,
    overtraining_threshold,
    pretrained,
    custom_pretrained,
    sync_graph,
    cache_data_in_gpu,
    g_pretrained_path=None,
    d_pretrained_path=None,
):
    f0 = 1 if str(pitch_guidance) == "True" else 0
    latest = 1 if str(save_only_latest) == "True" else 0
    save_every = 1 if str(save_every_weights) == "True" else 0
    detector = 1 if str(overtraining_detector) == "True" else 0
    sync = 1 if str(sync_graph) == "True" else 0
    cache_data = 1 if str(cache_data_in_gpu) == "True" else 0

    if str(pretrained) == "True":
        if str(custom_pretrained) == "False":
            pg, pd = pretrained_selector(f0)[rvc_version][sampling_rate]
        else:
            if g_pretrained_path is None or d_pretrained_path is None:
                raise ValueError(
                    "Please provide the path to the pretrained G and D models."
                )
            pg, pd = g_pretrained_path, d_pretrained_path
    else:
        pg, pd = "", ""

    train_script_path = os.path.join("rvc", "train", "train.py")
    command = [
        python,
        train_script_path,
        *map(
            str,
            [
                "-se",
                save_every_epoch,
                "-te",
                total_epoch,
                "-pg",
                pg,
                "-pd",
                pd,
                "-sr",
                sampling_rate,
                "-bs",
                batch_size,
                "-g",
                gpu,
                "-e",
                os.path.join(logs_path, model_name),
                "-v",
                rvc_version,
                "-l",
                latest,
                "-c",
                cache_data,
                "-sw",
                save_every,
                "-f0",
                f0,
                "-od",
                detector,
                "-ot",
                overtraining_threshold,
                "-sg",
                sync,
            ],
        ),
    ]

    subprocess.run(command)
    run_index_script(model_name, rvc_version)
    return f"Model {model_name} trained successfully."


# Index
def run_index_script(model_name, rvc_version):
    index_script_path = os.path.join("rvc", "train", "process", "extract_index.py")
    command = [
        python,
        index_script_path,
        os.path.join(logs_path, model_name),
        rvc_version,
    ]

    subprocess.run(command)
    return f"Index file for {model_name} generated successfully."


# Model extract
def run_model_extract_script(
    pth_path, model_name, sampling_rate, pitch_guidance, rvc_version, epoch, step
):
    f0 = 1 if str(pitch_guidance) == "True" else 0
    extract_small_model(
        pth_path, model_name, sampling_rate, f0, rvc_version, epoch, step
    )
    return f"Model {model_name} extracted successfully."


# Model information
def run_model_information_script(pth_path):
    print(model_information(pth_path))


# Model blender
def run_model_blender_script(model_name, pth_path_1, pth_path_2, ratio):
    message, model_blended = model_blender(model_name, pth_path_1, pth_path_2, ratio)
    return message, model_blended


# Tensorboard
def run_tensorboard_script():
    launch_tensorboard_pipeline()


# Download
def run_download_script(model_link):
    model_download_pipeline(model_link)
    return f"Model downloaded successfully."


# Prerequisites
def run_prerequisites_script(pretraineds_v1, pretraineds_v2, models, exe):
    prequisites_download_pipeline(pretraineds_v1, pretraineds_v2, models, exe)
    return "Prerequisites installed successfully."


# Audio analyzer
def run_audio_analyzer_script(input_path, save_plot_path="logs/audio_analysis.png"):
    audio_info, plot_path = analyze_audio(input_path, save_plot_path)
    print(
        f"Audio info of {input_path}: {audio_info}",
        f"Audio file {input_path} analyzed successfully. Plot saved at: {plot_path}",
    )
    return audio_info, plot_path


# API
def run_api_script(ip, port):
    command = [
        "env/Scripts/uvicorn.exe" if os.name == "nt" else "uvicorn",
        "api:app",
        "--host",
        ip,
        "--port",
        port,
    ]
    subprocess.run(command)


# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the main.py script with specific parameters."
    )
    subparsers = parser.add_subparsers(
        title="subcommands", dest="mode", help="Choose a mode"
    )

    # Parser for 'infer' mode
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--f0_up_key",
        type=str,
        help="Value for f0_up_key",
        choices=[str(i) for i in range(-24, 25)],
        default="0",
    )
    infer_parser.add_argument(
        "--filter_radius",
        type=str,
        help="Value for filter_radius",
        choices=[str(i) for i in range(11)],
        default="3",
    )
    infer_parser.add_argument(
        "--index_rate",
        type=str,
        help="Value for index_rate",
        choices=[str(i / 10) for i in range(11)],
        default="0.3",
    )
    infer_parser.add_argument(
        "--rms_mix_rate",
        type=str,
        help="Value for rms_mix_rate",
        choices=[str(i / 10) for i in range(11)],
        default="1",
    )
    infer_parser.add_argument(
        "--protect",
        type=str,
        help="Value for protect",
        choices=[str(i / 10) for i in range(6)],
        default="0.33",
    )
    infer_parser.add_argument(
        "--hop_length",
        type=str,
        help="Value for hop_length",
        choices=[str(i) for i in range(1, 513)],
        default="128",
    )
    infer_parser.add_argument(
        "--f0_method",
        type=str,
        help="Value for f0_method",
        choices=[
            "pm",
            "harvest",
            "dio",
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
            "hybrid[crepe+rmvpe]",
            "hybrid[crepe+fcpe]",
            "hybrid[rmvpe+fcpe]",
            "hybrid[crepe+rmvpe+fcpe]",
        ],
        default="rmvpe",
    )
    infer_parser.add_argument("--input_path", type=str, help="Input path")
    infer_parser.add_argument("--output_path", type=str, help="Output path")
    infer_parser.add_argument("--pth_path", type=str, help="Path to the .pth file")
    infer_parser.add_argument(
        "--index_path",
        type=str,
        help="Path to the .index file",
    )
    infer_parser.add_argument(
        "--split_audio",
        type=str,
        help="Enable split audio",
        choices=["True", "False"],
        default="False",
    )
    infer_parser.add_argument(
        "--f0_autotune",
        type=str,
        help="Enable autotune",
        choices=["True", "False"],
        default="False",
    )
    infer_parser.add_argument(
        "--clean_audio",
        type=str,
        help="Enable clean audio",
        choices=["True", "False"],
        default="False",
    )
    infer_parser.add_argument(
        "--clean_strength",
        type=str,
        help="Value for clean_strength",
        choices=[str(i / 10) for i in range(11)],
        default="0.7",
    )
    infer_parser.add_argument(
        "--export_format",
        type=str,
        help="Export format",
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    infer_parser.add_argument(
        "--embedder_model",
        type=str,
        help="Embedder model",
        choices=[
            "contentvec",
            "japanese-hubert-base",
            "chinese-hubert-large",
            "custom",
        ],
        default="contentvec",
    )
    infer_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help="Custom Embedder model",
        default=None,
    )
    infer_parser.add_argument(
        "--upscale_audio",
        type=str,
        help="Enable audio upscaling",
        choices=["True", "False"],
        default="False",
    )
    infer_parser.add_argument(
        "--f0_file",
        type=str,
        help="Path to the f0 file",
        default=None,
    )

    # Parser for 'batch_infer' mode
    batch_infer_parser = subparsers.add_parser(
        "batch_infer", help="Run batch inference"
    )
    batch_infer_parser.add_argument(
        "--f0_up_key",
        type=str,
        help="Value for f0_up_key",
        choices=[str(i) for i in range(-24, 25)],
        default="0",
    )
    batch_infer_parser.add_argument(
        "--filter_radius",
        type=str,
        help="Value for filter_radius",
        choices=[str(i) for i in range(11)],
        default="3",
    )
    batch_infer_parser.add_argument(
        "--index_rate",
        type=str,
        help="Value for index_rate",
        choices=[str(i / 10) for i in range(11)],
        default="0.3",
    )
    batch_infer_parser.add_argument(
        "--rms_mix_rate",
        type=str,
        help="Value for rms_mix_rate",
        choices=[str(i / 10) for i in range(11)],
        default="1",
    )
    batch_infer_parser.add_argument(
        "--protect",
        type=str,
        help="Value for protect",
        choices=[str(i / 10) for i in range(6)],
        default="0.33",
    )
    batch_infer_parser.add_argument(
        "--hop_length",
        type=str,
        help="Value for hop_length",
        choices=[str(i) for i in range(1, 513)],
        default="128",
    )
    batch_infer_parser.add_argument(
        "--f0_method",
        type=str,
        help="Value for f0_method",
        choices=[
            "pm",
            "harvest",
            "dio",
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
            "hybrid[crepe+rmvpe]",
            "hybrid[crepe+fcpe]",
            "hybrid[rmvpe+fcpe]",
            "hybrid[crepe+rmvpe+fcpe]",
        ],
        default="rmvpe",
    )
    batch_infer_parser.add_argument("--input_folder", type=str, help="Input folder")
    batch_infer_parser.add_argument("--output_folder", type=str, help="Output folder")
    batch_infer_parser.add_argument(
        "--pth_path", type=str, help="Path to the .pth file"
    )
    batch_infer_parser.add_argument(
        "--index_path",
        type=str,
        help="Path to the .index file",
    )
    batch_infer_parser.add_argument(
        "--split_audio",
        type=str,
        help="Enable split audio",
        choices=["True", "False"],
        default="False",
    )
    batch_infer_parser.add_argument(
        "--f0_autotune",
        type=str,
        help="Enable autotune",
        choices=["True", "False"],
        default="False",
    )
    batch_infer_parser.add_argument(
        "--clean_audio",
        type=str,
        help="Enable clean audio",
        choices=["True", "False"],
        default="False",
    )
    batch_infer_parser.add_argument(
        "--clean_strength",
        type=str,
        help="Value for clean_strength",
        choices=[str(i / 10) for i in range(11)],
        default="0.7",
    )
    batch_infer_parser.add_argument(
        "--export_format",
        type=str,
        help="Export format",
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    batch_infer_parser.add_argument(
        "--embedder_model",
        type=str,
        help="Embedder model",
        choices=[
            "contentvec",
            "japanese-hubert-base",
            "chinese-hubert-large",
            "custom",
        ],
        default="contentvec",
    )
    batch_infer_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help="Custom Embedder model",
        default=None,
    )
    batch_infer_parser.add_argument(
        "--upscale_audio",
        type=str,
        help="Enable audio upscaling",
        choices=["True", "False"],
        default="False",
    )
    batch_infer_parser.add_argument(
        "--f0_file",
        type=str,
        help="Path to the f0 file",
        default=None,
    )

    # Parser for 'tts' mode
    tts_parser = subparsers.add_parser("tts", help="Run TTS")
    tts_parser.add_argument(
        "--tts_text",
        type=str,
        help="Text to be synthesized",
    )
    tts_parser.add_argument(
        "--tts_voice",
        type=str,
        help="Voice to be used",
        choices=locales,
    )
    tts_parser.add_argument(
        "--tts_rate",
        type=str,
        help="Increase or decrease TTS speed",
        choices=[str(i) for i in range(-100, 100)],
        default="0",
    )
    tts_parser.add_argument(
        "--f0_up_key",
        type=str,
        help="Value for f0_up_key",
        choices=[str(i) for i in range(-24, 25)],
        default="0",
    )
    tts_parser.add_argument(
        "--filter_radius",
        type=str,
        help="Value for filter_radius",
        choices=[str(i) for i in range(11)],
        default="3",
    )
    tts_parser.add_argument(
        "--index_rate",
        type=str,
        help="Value for index_rate",
        choices=[str(i / 10) for i in range(11)],
        default="0.3",
    )
    tts_parser.add_argument(
        "--rms_mix_rate",
        type=str,
        help="Value for rms_mix_rate",
        choices=[str(i / 10) for i in range(11)],
        default="1",
    )
    tts_parser.add_argument(
        "--protect",
        type=str,
        help="Value for protect",
        choices=[str(i / 10) for i in range(6)],
        default="0.33",
    )
    tts_parser.add_argument(
        "--hop_length",
        type=str,
        help="Value for hop_length",
        choices=[str(i) for i in range(1, 513)],
        default="128",
    )
    tts_parser.add_argument(
        "--f0_method",
        type=str,
        help="Value for f0_method",
        choices=[
            "pm",
            "harvest",
            "dio",
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
            "hybrid[crepe+rmvpe]",
            "hybrid[crepe+fcpe]",
            "hybrid[rmvpe+fcpe]",
            "hybrid[crepe+rmvpe+fcpe]",
        ],
        default="rmvpe",
    )
    tts_parser.add_argument("--output_tts_path", type=str, help="Output tts path")
    tts_parser.add_argument("--output_rvc_path", type=str, help="Output rvc path")
    tts_parser.add_argument("--pth_path", type=str, help="Path to the .pth file")
    tts_parser.add_argument(
        "--index_path",
        type=str,
        help="Path to the .index file",
    )
    tts_parser.add_argument(
        "--split_audio",
        type=str,
        help="Enable split audio",
        choices=["True", "False"],
        default="False",
    )
    tts_parser.add_argument(
        "--f0_autotune",
        type=str,
        help="Enable autotune",
        choices=["True", "False"],
        default="False",
    )
    tts_parser.add_argument(
        "--clean_audio",
        type=str,
        help="Enable clean audio",
        choices=["True", "False"],
        default="False",
    )
    tts_parser.add_argument(
        "--clean_strength",
        type=str,
        help="Value for clean_strength",
        choices=[str(i / 10) for i in range(11)],
        default="0.7",
    )
    tts_parser.add_argument(
        "--export_format",
        type=str,
        help="Export format",
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    tts_parser.add_argument(
        "--embedder_model",
        type=str,
        help="Embedder model",
        choices=[
            "contentvec",
            "japanese-hubert-base",
            "chinese-hubert-large",
            "custom",
        ],
        default="contentvec",
    )
    tts_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help="Custom Embedder model",
        default=None,
    )
    tts_parser.add_argument(
        "--upscale_audio",
        type=str,
        help="Enable audio upscaling",
        choices=["True", "False"],
        default="False",
    )
    tts_parser.add_argument(
        "--f0_file",
        type=str,
        help="Path to the f0 file",
        default=None,
    )

    # Parser for 'preprocess' mode
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing")
    preprocess_parser.add_argument("--model_name", type=str, help="Name of the model")
    preprocess_parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
    )
    preprocess_parser.add_argument(
        "--sampling_rate",
        type=str,
        help="Sampling rate",
        choices=["32000", "40000", "48000"],
    )
    preprocess_parser.add_argument(
        "--cpu_cores",
        type=str,
        help="Number of CPU cores to use",
        choices=[str(i) for i in range(1, 64)],
        default=None,
    )

    # Parser for 'extract' mode
    extract_parser = subparsers.add_parser("extract", help="Run extract")
    extract_parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
    )
    extract_parser.add_argument(
        "--rvc_version",
        type=str,
        help="Version of the model",
        choices=["v1", "v2"],
        default="v2",
    )
    extract_parser.add_argument(
        "--f0_method",
        type=str,
        help="Value for f0_method",
        choices=[
            "pm",
            "harvest",
            "dio",
            "crepe",
            "crepe-tiny",
            "rmvpe",
        ],
        default="rmvpe",
    )
    extract_parser.add_argument(
        "--pitch_guidance",
        type=str,
        help="Pitch guidance",
        choices=["True", "False"],
        default="True",
    )
    extract_parser.add_argument(
        "--hop_length",
        type=str,
        help="Value for hop_length",
        choices=[str(i) for i in range(1, 513)],
        default="128",
    )
    extract_parser.add_argument(
        "--cpu_cores",
        type=str,
        help="Number of CPU cores to use",
        choices=[str(i) for i in range(1, 64)],
        default=None,
    )
    extract_parser.add_argument(
        "--sampling_rate",
        type=str,
        help="Sampling rate",
        choices=["32000", "40000", "48000"],
    )
    extract_parser.add_argument(
        "--embedder_model",
        type=str,
        help="Embedder model",
        choices=[
            "contentvec",
            "japanese-hubert-base",
            "chinese-hubert-large",
            "custom",
        ],
        default="contentvec",
    )
    extract_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help="Custom Embedder model",
        default=None,
    )

    # Parser for 'train' mode
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
    )
    train_parser.add_argument(
        "--rvc_version",
        type=str,
        help="Version of the model",
        choices=["v1", "v2"],
        default="v2",
    )
    train_parser.add_argument(
        "--save_every_epoch",
        type=str,
        help="Save every epoch",
        choices=[str(i) for i in range(1, 101)],
    )
    train_parser.add_argument(
        "--save_only_latest",
        type=str,
        help="Save weight only at last epoch",
        choices=["True", "False"],
        default="False",
    )
    train_parser.add_argument(
        "--save_every_weights",
        type=str,
        help="Save weight every epoch",
        choices=["True", "False"],
        default="True",
    )
    train_parser.add_argument(
        "--total_epoch",
        type=str,
        help="Total epoch",
        choices=[str(i) for i in range(1, 10001)],
        default="1000",
    )
    train_parser.add_argument(
        "--sampling_rate",
        type=str,
        help="Sampling rate",
        choices=["32000", "40000", "48000"],
    )
    train_parser.add_argument(
        "--batch_size",
        type=str,
        help="Batch size",
        choices=[str(i) for i in range(1, 51)],
        default="8",
    )
    train_parser.add_argument(
        "--gpu",
        type=str,
        help="GPU number",
        default="0",
    )
    train_parser.add_argument(
        "--pitch_guidance",
        type=str,
        help="Pitch guidance",
        choices=["True", "False"],
        default="True",
    )
    train_parser.add_argument(
        "--pretrained",
        type=str,
        help="Pretrained",
        choices=["True", "False"],
        default="True",
    )
    train_parser.add_argument(
        "--custom_pretrained",
        type=str,
        help="Custom pretrained",
        choices=["True", "False"],
        default="False",
    )
    train_parser.add_argument(
        "--g_pretrained_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the pretrained G file",
    )
    train_parser.add_argument(
        "--d_pretrained_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the pretrained D file",
    )
    train_parser.add_argument(
        "--overtraining_detector",
        type=str,
        help="Overtraining detector",
        choices=["True", "False"],
        default="False",
    )
    train_parser.add_argument(
        "--overtraining_threshold",
        type=str,
        help="Overtraining threshold",
        choices=[str(i) for i in range(1, 101)],
        default="50",
    )
    train_parser.add_argument(
        "--sync_graph",
        type=str,
        help="Sync graph",
        choices=["True", "False"],
        default="False",
    )
    train_parser.add_argument(
        "--cache_data_in_gpu",
        type=str,
        help="Cache data in GPU",
        choices=["True", "False"],
        default="False",
    )

    # Parser for 'index' mode
    index_parser = subparsers.add_parser("index", help="Generate index file")
    index_parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
    )
    index_parser.add_argument(
        "--rvc_version",
        type=str,
        help="Version of the model",
        choices=["v1", "v2"],
        default="v2",
    )

    # Parser for 'model_extract' mode
    model_extract_parser = subparsers.add_parser("model_extract", help="Extract model")
    model_extract_parser.add_argument(
        "--pth_path",
        type=str,
        help="Path to the .pth file",
    )
    model_extract_parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
    )
    model_extract_parser.add_argument(
        "--sampling_rate",
        type=str,
        help="Sampling rate",
        choices=["40000", "48000"],
    )
    model_extract_parser.add_argument(
        "--pitch_guidance",
        type=str,
        help="Pitch guidance",
        choices=["True", "False"],
    )
    model_extract_parser.add_argument(
        "--rvc_version",
        type=str,
        help="Version of the model",
        choices=["v1", "v2"],
        default="v2",
    )
    model_extract_parser.add_argument(
        "--epoch",
        type=str,
        help="Epochs of the model",
        choices=[str(i) for i in range(1, 10001)],
    )
    model_extract_parser.add_argument(
        "--step",
        type=str,
        help="Steps of the model",
    )

    # Parser for 'model_information' mode
    model_information_parser = subparsers.add_parser(
        "model_information", help="Print model information"
    )
    model_information_parser.add_argument(
        "--pth_path",
        type=str,
        help="Path to the .pth file",
    )

    # Parser for 'model_blender' mode
    model_blender_parser = subparsers.add_parser(
        "model_blender", help="Fuse two models"
    )
    model_blender_parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
    )
    model_blender_parser.add_argument(
        "--pth_path_1",
        type=str,
        help="Path to the first .pth file",
    )
    model_blender_parser.add_argument(
        "--pth_path_2",
        type=str,
        help="Path to the second .pth file",
    )
    model_blender_parser.add_argument(
        "--ratio",
        type=str,
        help="Value for blender ratio",
        choices=[str(i / 10) for i in range(11)],
        default="0.5",
    )

    # Parser for 'tensorboard' mode
    subparsers.add_parser("tensorboard", help="Run tensorboard")

    # Parser for 'download' mode
    download_parser = subparsers.add_parser("download", help="Download models")
    download_parser.add_argument(
        "--model_link",
        type=str,
        help="Link of the model",
    )

    # Parser for 'prerequisites' mode
    prerequisites_parser = subparsers.add_parser(
        "prerequisites", help="Install prerequisites"
    )
    prerequisites_parser.add_argument(
        "--pretraineds_v1",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Download pretrained models for v1",
    )
    prerequisites_parser.add_argument(
        "--pretraineds_v2",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Download pretrained models for v2",
    )
    prerequisites_parser.add_argument(
        "--models",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Donwload models",
    )
    prerequisites_parser.add_argument(
        "--exe",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Download executables",
    )

    # Parser for 'audio_analyzer' mode
    audio_analyzer = subparsers.add_parser("audio_analyzer", help="Run audio analyzer")
    audio_analyzer.add_argument(
        "--input_path",
        type=str,
        help="Path to the input audio file",
    )

    # Parser for 'api' mode
    api_parser = subparsers.add_parser("api", help="Run the API")
    api_parser.add_argument(
        "--host", type=str, help="Host address", default="127.0.0.1"
    )
    api_parser.add_argument("--port", type=str, help="Port", default="8000")

    return parser.parse_args()


def main():
    if len(sys.argv) == 1:
        print("Please run the script with '-h' for more information.")
        sys.exit(1)

    args = parse_arguments()

    try:
        if args.mode == "infer":
            run_infer_script(
                str(args.f0_up_key),
                str(args.filter_radius),
                str(args.index_rate),
                str(args.rms_mix_rate),
                str(args.protect),
                str(args.hop_length),
                str(args.f0_method),
                str(args.input_path),
                str(args.output_path),
                str(args.pth_path),
                str(args.index_path),
                str(args.split_audio),
                str(args.f0_autotune),
                str(args.clean_audio),
                str(args.clean_strength),
                str(args.export_format),
                str(args.embedder_model),
                str(args.embedder_model_custom),
                str(args.upscale_audio),
                str(args.f0_file),
            )
        elif args.mode == "batch_infer":
            run_batch_infer_script(
                str(args.f0_up_key),
                str(args.filter_radius),
                str(args.index_rate),
                str(args.rms_mix_rate),
                str(args.protect),
                str(args.hop_length),
                str(args.f0_method),
                str(args.input_folder),
                str(args.output_folder),
                str(args.pth_path),
                str(args.index_path),
                str(args.split_audio),
                str(args.f0_autotune),
                str(args.clean_audio),
                str(args.clean_strength),
                str(args.export_format),
                str(args.embedder_model),
                str(args.embedder_model_custom),
                str(args.upscale_audio),
                str(args.f0_file),
            )
        elif args.mode == "tts":
            run_tts_script(
                str(args.tts_text),
                str(args.tts_voice),
                str(args.tts_rate),
                str(args.f0_up_key),
                str(args.filter_radius),
                str(args.index_rate),
                str(args.rms_mix_rate),
                str(args.protect),
                str(args.hop_length),
                str(args.f0_method),
                str(args.output_tts_path),
                str(args.output_rvc_path),
                str(args.pth_path),
                str(args.index_path),
                str(args.split_audio),
                str(args.f0_autotune),
                str(args.clean_audio),
                str(args.clean_strength),
                str(args.export_format),
                str(args.embedder_model),
                str(args.embedder_model_custom),
                str(args.upscale_audio),
                str(args.f0_file),
            )
        elif args.mode == "preprocess":
            run_preprocess_script(
                str(args.model_name),
                str(args.dataset_path),
                str(args.sampling_rate),
                str(args.cpu_cores),
            )
        elif args.mode == "extract":
            run_extract_script(
                str(args.model_name),
                str(args.rvc_version),
                str(args.f0_method),
                str(args.pitch_guidance),
                str(args.hop_length),
                str(args.cpu_cores),
                str(args.sampling_rate),
                str(args.embedder_model),
                str(args.embedder_model_custom),
            )
        elif args.mode == "train":
            run_train_script(
                str(args.model_name),
                str(args.rvc_version),
                str(args.save_every_epoch),
                str(args.save_only_latest),
                str(args.save_every_weights),
                str(args.total_epoch),
                str(args.sampling_rate),
                str(args.batch_size),
                str(args.gpu),
                str(args.pitch_guidance),
                str(args.overtraining_detector),
                str(args.overtraining_threshold),
                str(args.pretrained),
                str(args.custom_pretrained),
                str(args.sync_graph),
                str(args.cache_data_in_gpu),
                str(args.g_pretrained_path),
                str(args.d_pretrained_path),
            )
        elif args.mode == "index":
            run_index_script(
                str(args.model_name),
                str(args.rvc_version),
            )
        elif args.mode == "model_extract":
            run_model_extract_script(
                str(args.pth_path),
                str(args.model_name),
                str(args.sampling_rate),
                str(args.pitch_guidance),
                str(args.rvc_version),
                str(args.epoch),
                str(args.step),
            )
        elif args.mode == "model_information":
            run_model_information_script(
                str(args.pth_path),
            )
        elif args.mode == "model_blender":
            run_model_blender_script(
                str(args.model_name),
                str(args.pth_path_1),
                str(args.pth_path_2),
                str(args.ratio),
            )
        elif args.mode == "tensorboard":
            run_tensorboard_script()
        elif args.mode == "download":
            run_download_script(
                str(args.model_link),
            )
        elif args.mode == "prerequisites":
            run_prerequisites_script(
                str(args.pretraineds_v1),
                str(args.pretraineds_v2),
                str(args.models),
                str(args.exe),
            )
        elif args.mode == "audio_analyzer":
            run_audio_analyzer_script(
                str(args.input_path),
            )
        elif args.mode == "api":
            run_api_script(
                str(args.host),
                str(args.port),
            )
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
