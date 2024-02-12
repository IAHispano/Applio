import os
import sys
import argparse
import subprocess

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.config import Config
from rvc.lib.tools.validators import (
    validate_sampling_rate,
    validate_f0up_key,
    validate_f0method,
    validate_true_false,
    validate_tts_voices,
)

from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.lib.tools.pretrained_selector import pretrained_selector

from rvc.lib.process.model_fusion import model_fusion
from rvc.lib.process.model_information import model_information

config = Config()
current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")
subprocess.run(
    ["python", os.path.join("rvc", "lib", "tools", "prerequisites_download.py")]
)


# Infer
def run_infer_script(
    f0up_key,
    filter_radius,
    index_rate,
    hop_length,
    f0method,
    input_path,
    output_path,
    pth_file,
    index_path,
    split_audio,
    f0_autotune,
):
    infer_script_path = os.path.join("rvc", "infer", "infer.py")
    command = [
        "python",
        infer_script_path,
        str(f0up_key),
        str(filter_radius),
        str(index_rate),
        str(hop_length),
        f0method,
        input_path,
        output_path,
        pth_file,
        index_path,
        str(split_audio),
        str(f0_autotune),
    ]
    subprocess.run(command)
    return f"File {input_path} inferred successfully.", output_path


# Batch infer
def run_batch_infer_script(
    f0up_key,
    filter_radius,
    index_rate,
    hop_length,
    f0method,
    input_folder,
    output_folder,
    pth_file,
    index_path,
    split_audio,
    f0_autotune,
):
    infer_script_path = os.path.join("rvc", "infer", "infer.py")

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

        command = [
            "python",
            infer_script_path,
            str(f0up_key),
            str(filter_radius),
            str(index_rate),
            str(hop_length),
            f0method,
            input_path,
            output_path,
            pth_file,
            index_path,
            str(split_audio),
            str(f0_autotune),
        ]
        subprocess.run(command)

    return f"Files from {input_folder} inferred successfully."


# TTS
def run_tts_script(
    tts_text,
    tts_voice,
    f0up_key,
    filter_radius,
    index_rate,
    hop_length,
    f0method,
    output_tts_path,
    output_rvc_path,
    pth_file,
    index_path,
    split_audio,
    f0_autotune,
):
    tts_script_path = os.path.join("rvc", "lib", "tools", "tts.py")
    infer_script_path = os.path.join("rvc", "infer", "infer.py")

    if os.path.exists(output_tts_path):
        os.remove(output_tts_path)

    command_tts = [
        "python",
        tts_script_path,
        tts_text,
        tts_voice,
        output_tts_path,
    ]

    command_infer = [
        "python",
        infer_script_path,
        str(f0up_key),
        str(filter_radius),
        str(index_rate),
        str(hop_length),
        f0method,
        output_tts_path,
        output_rvc_path,
        pth_file,
        index_path,
        str(split_audio),
        str(f0_autotune),
    ]
    subprocess.run(command_tts)
    subprocess.run(command_infer)
    return f"Text {tts_text} synthesized successfully.", output_rvc_path


# Preprocess
def run_preprocess_script(model_name, dataset_path, sampling_rate):
    per = 3.0 if config.is_half else 3.7
    preprocess_script_path = os.path.join("rvc", "train", "preprocess", "preprocess.py")
    command = [
        "python",
        preprocess_script_path,
        os.path.join(logs_path, str(model_name)),
        dataset_path,
        str(sampling_rate),
        str(per),
    ]

    os.makedirs(os.path.join(logs_path, str(model_name)), exist_ok=True)
    subprocess.run(command)
    return f"Model {model_name} preprocessed successfully."


# Extract
def run_extract_script(model_name, rvc_version, f0method, hop_length, sampling_rate):
    model_path = os.path.join(logs_path, str(model_name))
    extract_f0_script_path = os.path.join(
        "rvc", "train", "extract", "extract_f0_print.py"
    )
    extract_feature_script_path = os.path.join(
        "rvc", "train", "extract", "extract_feature_print.py"
    )

    command_1 = [
        "python",
        extract_f0_script_path,
        model_path,
        f0method,
        str(hop_length),
    ]
    command_2 = [
        "python",
        extract_feature_script_path,
        config.device,
        "1",
        "0",
        "0",
        model_path,
        rvc_version,
        "True",
    ]
    subprocess.run(command_1)
    subprocess.run(command_2)

    generate_config(rvc_version, sampling_rate, model_path)
    generate_filelist(f0method, model_path, rvc_version, sampling_rate)
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
    pretrained,
    custom_pretrained,
    g_pretrained_path=None,
    d_pretrained_path=None,
):
    f0 = 1 if str(pitch_guidance) == "True" else 0
    latest = 1 if str(save_only_latest) == "True" else 0
    save_every = 1 if str(save_every_weights) == "True" else 0

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
        "python",
        str(train_script_path),
        "-se",
        str(save_every_epoch),
        "-te",
        str(total_epoch),
        "-pg",
        str(pg),
        "-pd",
        str(pd),
        "-sr",
        str(sampling_rate),
        "-bs",
        str(batch_size),
        "-g",
        str(gpu),
        "-e",
        os.path.join(logs_path, str(model_name)),
        "-v",
        str(rvc_version),
        "-l",
        str(latest),
        "-c",
        "0",
        "-sw",
        str(save_every),
        "-f0",
        str(f0),
    ]

    subprocess.run(command)
    run_index_script(model_name, rvc_version)
    return f"Model {model_name} trained successfully."


# Index
def run_index_script(model_name, rvc_version):
    index_script_path = os.path.join("rvc", "train", "index_generator.py")
    command = [
        "python",
        index_script_path,
        os.path.join(logs_path, str(model_name)),
        rvc_version,
    ]

    subprocess.run(command)
    return f"Index file for {model_name} generated successfully."


# Model information
def run_model_information_script(pth_path):
    print(model_information(pth_path))


# Model fusion
def run_model_fusion_script(model_name, pth_path_1, pth_path_2):
    model_fusion(model_name, pth_path_1, pth_path_2)


# Tensorboard
def run_tensorboard_script():
    tensorboard_script_path = os.path.join(
        "rvc", "lib", "tools", "launch_tensorboard.py"
    )
    command = [
        "python",
        tensorboard_script_path,
    ]
    subprocess.run(command)


# Download
def run_download_script(model_link):
    download_script_path = os.path.join("rvc", "lib", "tools", "model_download.py")
    command = [
        "python",
        download_script_path,
        model_link,
    ]
    subprocess.run(command)
    return f"Model downloaded successfully."


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
        "f0up_key",
        type=validate_f0up_key,
        help="Value for f0up_key (-24 to +24)",
    )
    infer_parser.add_argument(
        "filter_radius",
        type=str,
        help="Value for filter_radius (0 to 10)",
    )
    infer_parser.add_argument(
        "index_rate",
        type=str,
        help="Value for index_rate (0.0 to 1)",
    )
    infer_parser.add_argument(
        "hop_length",
        type=str,
        help="Value for hop_length (1 to 512)",
    )
    infer_parser.add_argument(
        "f0method",
        type=validate_f0method,
        help="Value for f0method (pm, dio, crepe, crepe-tiny, harvest, rmvpe)",
    )
    infer_parser.add_argument(
        "input_path", type=str, help="Input path (enclose in double quotes)"
    )
    infer_parser.add_argument(
        "output_path", type=str, help="Output path (enclose in double quotes)"
    )
    infer_parser.add_argument(
        "pth_file", type=str, help="Path to the .pth file (enclose in double quotes)"
    )
    infer_parser.add_argument(
        "index_path",
        type=str,
        help="Path to the .index file (enclose in double quotes)",
    )
    infer_parser.add_argument(
        "split_audio",
        type=str,
        help="Enable split audio ( better results )",
    )
    infer_parser.add_argument(
        "f0_autotune",
        type=str,
        help="Enable autotune",
    )

    # Parser for 'batch_infer' mode
    batch_infer_parser = subparsers.add_parser(
        "batch_infer", help="Run batch inference"
    )
    batch_infer_parser.add_argument(
        "f0up_key",
        type=validate_f0up_key,
        help="Value for f0up_key (-24 to +24)",
    )
    batch_infer_parser.add_argument(
        "filter_radius",
        type=str,
        help="Value for filter_radius (0 to 10)",
    )
    batch_infer_parser.add_argument(
        "index_rate",
        type=str,
        help="Value for index_rate (0.0 to 1)",
    )
    batch_infer_parser.add_argument(
        "hop_length",
        type=str,
        help="Value for hop_length (1 to 512)",
    )
    batch_infer_parser.add_argument(
        "f0method",
        type=validate_f0method,
        help="Value for f0method (pm, dio, crepe, crepe-tiny, harvest, rmvpe)",
    )
    batch_infer_parser.add_argument(
        "input_folder", type=str, help="Input folder (enclose in double quotes)"
    )
    batch_infer_parser.add_argument(
        "output_folder", type=str, help="Output folder (enclose in double quotes)"
    )
    batch_infer_parser.add_argument(
        "pth_file", type=str, help="Path to the .pth file (enclose in double quotes)"
    )
    batch_infer_parser.add_argument(
        "index_path",
        type=str,
        help="Path to the .index file (enclose in double quotes)",
    )
    batch_infer_parser.add_argument(
        "split_audio",
        type=str,
        help="Enable split audio ( better results )",
    )
    batch_infer_parser.add_argument(
        "f0_autotune",
        type=str,
        help="Enable autotune",
    )

    # Parser for 'tts' mode
    tts_parser = subparsers.add_parser("tts", help="Run TTS")
    tts_parser.add_argument(
        "tts_text",
        type=str,
        help="Text to be synthesized (enclose in double quotes)",
    )
    tts_parser.add_argument(
        "tts_voice",
        type=validate_tts_voices,
        help="Voice to be used (enclose in double quotes)",
    )
    tts_parser.add_argument(
        "f0up_key",
        type=validate_f0up_key,
        help="Value for f0up_key (-24 to +24)",
    )
    tts_parser.add_argument(
        "filter_radius",
        type=str,
        help="Value for filter_radius (0 to 10)",
    )
    tts_parser.add_argument(
        "index_rate",
        type=str,
        help="Value for index_rate (0.0 to 1)",
    )
    tts_parser.add_argument(
        "hop_length",
        type=str,
        help="Value for hop_length (1 to 512)",
    )
    tts_parser.add_argument(
        "f0method",
        type=validate_f0method,
        help="Value for f0method (pm, dio, crepe, crepe-tiny, harvest, rmvpe)",
    )
    tts_parser.add_argument(
        "output_tts_path", type=str, help="Output tts path (enclose in double quotes)"
    )
    tts_parser.add_argument(
        "output_rvc_path", type=str, help="Output rvc path (enclose in double quotes)"
    )
    tts_parser.add_argument(
        "pth_file", type=str, help="Path to the .pth file (enclose in double quotes)"
    )
    tts_parser.add_argument(
        "index_path",
        type=str,
        help="Path to the .index file (enclose in double quotes)",
    )
    tts_parser.add_argument(
        "split_audio",
        type=str,
        help="Enable split audio ( better results )",
    )
    tts_parser.add_argument(
        "f0_autotune",
        type=str,
        help="Enable autotune",
    )

    # Parser for 'preprocess' mode
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing")
    preprocess_parser.add_argument(
        "model_name", type=str, help="Name of the model (enclose in double quotes)"
    )
    preprocess_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset (enclose in double quotes)",
    )
    preprocess_parser.add_argument(
        "sampling_rate",
        type=validate_sampling_rate,
        help="Sampling rate (32000, 40000 or 48000)",
    )

    # Parser for 'extract' mode
    extract_parser = subparsers.add_parser("extract", help="Run extract")
    extract_parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model (enclose in double quotes)",
    )
    extract_parser.add_argument(
        "rvc_version",
        type=str,
        help="Version of the model (v1 or v2)",
    )
    extract_parser.add_argument(
        "f0method",
        type=validate_f0method,
        help="Value for f0method (pm, dio, crepe, crepe-tiny, mangio-crepe, mangio-crepe-tiny, harvest, rmvpe)",
    )
    extract_parser.add_argument(
        "hop_length",
        type=str,
        help="Value for hop_length (1 to 512)",
    )
    extract_parser.add_argument(
        "sampling_rate",
        type=validate_sampling_rate,
        help="Sampling rate (32000, 40000 or 48000)",
    )

    # Parser for 'train' mode
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model (enclose in double quotes)",
    )
    train_parser.add_argument(
        "rvc_version",
        type=str,
        help="Version of the model (v1 or v2)",
    )
    train_parser.add_argument(
        "save_every_epoch",
        type=str,
        help="Save every epoch",
    )
    train_parser.add_argument(
        "save_only_latest",
        type=str,
        help="Save weight only at last epoch",
    )
    train_parser.add_argument(
        "save_every_weights",
        type=str,
        help="Save weight every epoch",
    )
    train_parser.add_argument(
        "total_epoch",
        type=str,
        help="Total epoch",
    )
    train_parser.add_argument(
        "sampling_rate",
        type=validate_sampling_rate,
        help="Sampling rate (32000, 40000, or 48000)",
    )
    train_parser.add_argument(
        "batch_size",
        type=str,
        help="Batch size",
    )
    train_parser.add_argument(
        "gpu",
        type=str,
        help="GPU number (0 to 10 separated by -)",
    )
    train_parser.add_argument(
        "pitch_guidance",
        type=validate_true_false,
        help="Pitch guidance (True or False)",
    )
    train_parser.add_argument(
        "pretrained",
        type=validate_true_false,
        help="Pretrained (True or False)",
    )
    train_parser.add_argument(
        "custom_pretrained",
        type=validate_true_false,
        help="Custom pretrained (True or False)",
    )
    train_parser.add_argument(
        "g_pretrained_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the pretrained G file (enclose in double quotes)",
    )
    train_parser.add_argument(
        "d_pretrained_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the pretrained D file (enclose in double quotes)",
    )

    # Parser for 'index' mode
    index_parser = subparsers.add_parser("index", help="Generate index file")
    index_parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model (enclose in double quotes)",
    )
    index_parser.add_argument(
        "rvc_version",
        type=str,
        help="Version of the model (v1 or v2)",
    )

    # Parser for 'model_information' mode
    model_information_parser = subparsers.add_parser(
        "model_information", help="Print model information"
    )
    model_information_parser.add_argument(
        "pth_path",
        type=str,
        help="Path to the .pth file (enclose in double quotes)",
    )

    # Parser for 'model_fusion' mode
    model_fusion_parser = subparsers.add_parser("model_fusion", help="Fuse two models")
    model_fusion_parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model (enclose in double quotes)",
    )
    model_fusion_parser.add_argument(
        "pth_path_1",
        type=str,
        help="Path to the first .pth file (enclose in double quotes)",
    )
    model_fusion_parser.add_argument(
        "pth_path_2",
        type=str,
        help="Path to the second .pth file (enclose in double quotes)",
    )

    # Parser for 'tensorboard' mode
    subparsers.add_parser("tensorboard", help="Run tensorboard")

    # Parser for 'download' mode
    download_parser = subparsers.add_parser("download", help="Download models")
    download_parser.add_argument(
        "model_link",
        type=str,
        help="Link of the model (enclose in double quotes)",
    )

    return parser.parse_args()


def main():
    if len(sys.argv) == 1:
        print("Please run the script with '-h' for more information.")
        sys.exit(1)

    args = parse_arguments()

    try:
        if args.mode == "infer":
            run_infer_script(
                args.f0up_key,
                args.filter_radius,
                args.index_rate,
                args.hop_length,
                args.f0method,
                args.input_path,
                args.output_path,
                args.pth_file,
                args.index_path,
                args.split_audio,
                args.f0_autotune,

            )
        elif args.mode == "batch_infer":
            run_batch_infer_script(
                args.f0up_key,
                args.filter_radius,
                args.index_rate,
                args.hop_length,
                args.f0method,
                args.input_folder,
                args.output_folder,
                args.pth_file,
                args.index_path,
                args.split_audio,
                args.f0_autotune,
            )
        elif args.mode == "tts":
            run_tts_script(
                args.tts_text,
                args.tts_voice,
                args.f0up_key,
                args.filter_radius,
                args.index_rate,
                args.hop_length,
                args.f0method,
                args.output_tts_path,
                args.output_rvc_path,
                args.pth_file,
                args.index_path,
                args.split_audio,
                args.f0_autotune,
            )
        elif args.mode == "preprocess":
            run_preprocess_script(
                args.model_name,
                args.dataset_path,
                str(args.sampling_rate),
            )

        elif args.mode == "extract":
            run_extract_script(
                args.model_name,
                args.rvc_version,
                args.f0method,
                args.hop_length,
                args.sampling_rate,
            )
        elif args.mode == "train":
            run_train_script(
                args.model_name,
                args.rvc_version,
                args.save_every_epoch,
                args.save_only_latest,
                args.save_every_weights,
                args.total_epoch,
                args.sampling_rate,
                args.batch_size,
                args.gpu,
                args.pitch_guidance,
                args.pretrained,
                args.custom_pretrained,
                args.g_pretrained_path,
                args.d_pretrained_path,
            )
        elif args.mode == "index":
            run_index_script(
                args.model_name,
                args.rvc_version,
            )
        elif args.mode == "model_information":
            run_model_information_script(
                args.pth_path,
            )
        elif args.mode == "model_fusion":
            run_model_fusion_script(
                args.model_name,
                args.pth_path_1,
                args.pth_path_2,
            )
        elif args.mode == "tensorboard":
            run_tensorboard_script()
        elif args.mode == "download":
            run_download_script(
                args.model_link,
            )
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
