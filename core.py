import os
import sys
import json
import argparse
import subprocess
from functools import lru_cache
from distutils.util import strtobool

now_dir = os.getcwd()
sys.path.append(now_dir)

current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")

from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
from rvc.train.process.model_blender import model_blender
from rvc.train.process.model_information import model_information
from rvc.lib.tools.analyzer import analyze_audio
from rvc.lib.tools.launch_tensorboard import launch_tensorboard_pipeline
from rvc.lib.tools.model_download import model_download_pipeline

python = sys.executable


# Get TTS Voices -> https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list?trustedclienttoken=6A5AA1D4EAFF4E9FB37E23D68491D6F4
@lru_cache(maxsize=1)  # Cache only one result since the file is static
def load_voices_data():
    with open(
        os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r", encoding="utf-8"
    ) as file:
        return json.load(file)


voices_data = load_voices_data()
locales = list({voice["ShortName"] for voice in voices_data})


@lru_cache(maxsize=None)
def import_voice_converter():
    from rvc.infer.infer import VoiceConverter

    return VoiceConverter()


@lru_cache(maxsize=1)
def get_config():
    from rvc.configs.config import Config

    return Config()


# Infer
def run_infer_script(
    pitch: int,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_method: str,
    input_path: str,
    output_path: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    proposed_pitch: bool,
    proposed_pitch_threshold: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    embedder_model: str,
    embedder_model_custom: str = None,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    post_process: bool = False,
    reverb: bool = False,
    pitch_shift: bool = False,
    limiter: bool = False,
    gain: bool = False,
    distortion: bool = False,
    chorus: bool = False,
    bitcrush: bool = False,
    clipping: bool = False,
    compressor: bool = False,
    delay: bool = False,
    reverb_room_size: float = 0.5,
    reverb_damping: float = 0.5,
    reverb_wet_gain: float = 0.5,
    reverb_dry_gain: float = 0.5,
    reverb_width: float = 0.5,
    reverb_freeze_mode: float = 0.5,
    pitch_shift_semitones: float = 0.0,
    limiter_threshold: float = -6,
    limiter_release_time: float = 0.01,
    gain_db: float = 0.0,
    distortion_gain: float = 25,
    chorus_rate: float = 1.0,
    chorus_depth: float = 0.25,
    chorus_center_delay: float = 7,
    chorus_feedback: float = 0.0,
    chorus_mix: float = 0.5,
    bitcrush_bit_depth: int = 8,
    clipping_threshold: float = -6,
    compressor_threshold: float = 0,
    compressor_ratio: float = 1,
    compressor_attack: float = 1.0,
    compressor_release: float = 100,
    delay_seconds: float = 0.5,
    delay_feedback: float = 0.0,
    delay_mix: float = 0.5,
    sid: int = 0,
):
    kwargs = {
        "audio_input_path": input_path,
        "audio_output_path": output_path,
        "model_path": pth_path,
        "index_path": index_path,
        "volume_envelope": volume_envelope,
        "pitch": pitch,
        "index_rate": index_rate,
        "protect": protect,
        "f0_method": f0_method,
        "pth_path": pth_path,
        "index_path": index_path,
        "split_audio": split_audio,
        "f0_autotune": f0_autotune,
        "f0_autotune_strength": f0_autotune_strength,
        "proposed_pitch": proposed_pitch,
        "proposed_pitch_threshold": proposed_pitch_threshold,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "export_format": export_format,
        "embedder_model": embedder_model,
        "embedder_model_custom": embedder_model_custom,
        "post_process": post_process,
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "reverb": reverb,
        "pitch_shift": pitch_shift,
        "limiter": limiter,
        "gain": gain,
        "distortion": distortion,
        "chorus": chorus,
        "bitcrush": bitcrush,
        "clipping": clipping,
        "compressor": compressor,
        "delay": delay,
        "reverb_room_size": reverb_room_size,
        "reverb_damping": reverb_damping,
        "reverb_wet_level": reverb_wet_gain,
        "reverb_dry_level": reverb_dry_gain,
        "reverb_width": reverb_width,
        "reverb_freeze_mode": reverb_freeze_mode,
        "pitch_shift_semitones": pitch_shift_semitones,
        "limiter_threshold": limiter_threshold,
        "limiter_release": limiter_release_time,
        "gain_db": gain_db,
        "distortion_gain": distortion_gain,
        "chorus_rate": chorus_rate,
        "chorus_depth": chorus_depth,
        "chorus_delay": chorus_center_delay,
        "chorus_feedback": chorus_feedback,
        "chorus_mix": chorus_mix,
        "bitcrush_bit_depth": bitcrush_bit_depth,
        "clipping_threshold": clipping_threshold,
        "compressor_threshold": compressor_threshold,
        "compressor_ratio": compressor_ratio,
        "compressor_attack": compressor_attack,
        "compressor_release": compressor_release,
        "delay_seconds": delay_seconds,
        "delay_feedback": delay_feedback,
        "delay_mix": delay_mix,
        "sid": sid,
    }
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio(
        **kwargs,
    )
    return f"File {input_path} inferred successfully.", output_path.replace(
        ".wav", f".{export_format.lower()}"
    )


# Batch infer
def run_batch_infer_script(
    pitch: int,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_method: str,
    input_folder: str,
    output_folder: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    proposed_pitch: bool,
    proposed_pitch_threshold: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    embedder_model: str,
    embedder_model_custom: str = None,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    post_process: bool = False,
    reverb: bool = False,
    pitch_shift: bool = False,
    limiter: bool = False,
    gain: bool = False,
    distortion: bool = False,
    chorus: bool = False,
    bitcrush: bool = False,
    clipping: bool = False,
    compressor: bool = False,
    delay: bool = False,
    reverb_room_size: float = 0.5,
    reverb_damping: float = 0.5,
    reverb_wet_gain: float = 0.5,
    reverb_dry_gain: float = 0.5,
    reverb_width: float = 0.5,
    reverb_freeze_mode: float = 0.5,
    pitch_shift_semitones: float = 0.0,
    limiter_threshold: float = -6,
    limiter_release_time: float = 0.01,
    gain_db: float = 0.0,
    distortion_gain: float = 25,
    chorus_rate: float = 1.0,
    chorus_depth: float = 0.25,
    chorus_center_delay: float = 7,
    chorus_feedback: float = 0.0,
    chorus_mix: float = 0.5,
    bitcrush_bit_depth: int = 8,
    clipping_threshold: float = -6,
    compressor_threshold: float = 0,
    compressor_ratio: float = 1,
    compressor_attack: float = 1.0,
    compressor_release: float = 100,
    delay_seconds: float = 0.5,
    delay_feedback: float = 0.0,
    delay_mix: float = 0.5,
    sid: int = 0,
):
    kwargs = {
        "audio_input_paths": input_folder,
        "audio_output_path": output_folder,
        "model_path": pth_path,
        "index_path": index_path,
        "pitch": pitch,
        "index_rate": index_rate,
        "volume_envelope": volume_envelope,
        "protect": protect,
        "f0_method": f0_method,
        "pth_path": pth_path,
        "index_path": index_path,
        "split_audio": split_audio,
        "f0_autotune": f0_autotune,
        "f0_autotune_strength": f0_autotune_strength,
        "proposed_pitch": proposed_pitch,
        "proposed_pitch_threshold": proposed_pitch_threshold,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "export_format": export_format,
        "embedder_model": embedder_model,
        "embedder_model_custom": embedder_model_custom,
        "post_process": post_process,
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "reverb": reverb,
        "pitch_shift": pitch_shift,
        "limiter": limiter,
        "gain": gain,
        "distortion": distortion,
        "chorus": chorus,
        "bitcrush": bitcrush,
        "clipping": clipping,
        "compressor": compressor,
        "delay": delay,
        "reverb_room_size": reverb_room_size,
        "reverb_damping": reverb_damping,
        "reverb_wet_level": reverb_wet_gain,
        "reverb_dry_level": reverb_dry_gain,
        "reverb_width": reverb_width,
        "reverb_freeze_mode": reverb_freeze_mode,
        "pitch_shift_semitones": pitch_shift_semitones,
        "limiter_threshold": limiter_threshold,
        "limiter_release": limiter_release_time,
        "gain_db": gain_db,
        "distortion_gain": distortion_gain,
        "chorus_rate": chorus_rate,
        "chorus_depth": chorus_depth,
        "chorus_delay": chorus_center_delay,
        "chorus_feedback": chorus_feedback,
        "chorus_mix": chorus_mix,
        "bitcrush_bit_depth": bitcrush_bit_depth,
        "clipping_threshold": clipping_threshold,
        "compressor_threshold": compressor_threshold,
        "compressor_ratio": compressor_ratio,
        "compressor_attack": compressor_attack,
        "compressor_release": compressor_release,
        "delay_seconds": delay_seconds,
        "delay_feedback": delay_feedback,
        "delay_mix": delay_mix,
        "sid": sid,
    }
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio_batch(
        **kwargs,
    )

    return f"Files from {input_folder} inferred successfully."


# TTS
def run_tts_script(
    tts_file: str,
    tts_text: str,
    tts_voice: str,
    tts_rate: int,
    pitch: int,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_method: str,
    output_tts_path: str,
    output_rvc_path: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    proposed_pitch: bool,
    proposed_pitch_threshold: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    embedder_model: str,
    embedder_model_custom: str = None,
    sid: int = 0,
):

    tts_script_path = os.path.join("rvc", "lib", "tools", "tts.py")

    if os.path.exists(output_tts_path) and os.path.abspath(output_tts_path).startswith(
        os.path.abspath("assets")
    ):
        os.remove(output_tts_path)

    command_tts = [
        *map(
            str,
            [
                python,
                tts_script_path,
                tts_file,
                tts_text,
                tts_voice,
                tts_rate,
                output_tts_path,
            ],
        ),
    ]
    subprocess.run(command_tts)
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio(
        pitch=pitch,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        f0_method=f0_method,
        audio_input_path=output_tts_path,
        audio_output_path=output_rvc_path,
        model_path=pth_path,
        index_path=index_path,
        split_audio=split_audio,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        proposed_pitch=proposed_pitch,
        proposed_pitch_threshold=proposed_pitch_threshold,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        export_format=export_format,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        sid=sid,
        formant_shifting=None,
        formant_qfrency=None,
        formant_timbre=None,
        post_process=None,
        reverb=None,
        pitch_shift=None,
        limiter=None,
        gain=None,
        distortion=None,
        chorus=None,
        bitcrush=None,
        clipping=None,
        compressor=None,
        delay=None,
        sliders=None,
    )

    return f"Text {tts_text} synthesized successfully.", output_rvc_path.replace(
        ".wav", f".{export_format.lower()}"
    )


# Preprocess
def run_preprocess_script(
    model_name: str,
    dataset_path: str,
    sample_rate: int,
    cpu_cores: int,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    clean_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str = "none",
):
    preprocess_script_path = os.path.join("rvc", "train", "preprocess", "preprocess.py")
    command = [
        python,
        preprocess_script_path,
        *map(
            str,
            [
                os.path.join(logs_path, model_name),
                dataset_path,
                sample_rate,
                cpu_cores,
                cut_preprocess,
                process_effects,
                noise_reduction,
                clean_strength,
                chunk_len,
                overlap_len,
                normalization_mode,
            ],
        ),
    ]
    subprocess.run(command)
    return f"Model {model_name} preprocessed successfully."


# Extract
def run_extract_script(
    model_name: str,
    f0_method: str,
    cpu_cores: int,
    gpu: int,
    sample_rate: int,
    embedder_model: str,
    embedder_model_custom: str = None,
    include_mutes: int = 2,
):

    model_path = os.path.join(logs_path, model_name)
    extract = os.path.join("rvc", "train", "extract", "extract.py")

    command_1 = [
        python,
        extract,
        *map(
            str,
            [
                model_path,
                f0_method,
                cpu_cores,
                gpu,
                sample_rate,
                embedder_model,
                embedder_model_custom,
                include_mutes,
            ],
        ),
    ]

    subprocess.run(command_1)

    return f"Model {model_name} extracted successfully."


# Train
def run_train_script(
    model_name: str,
    save_every_epoch: int,
    save_only_latest: bool,
    save_every_weights: bool,
    total_epoch: int,
    sample_rate: int,
    batch_size: int,
    gpu: int,
    overtraining_detector: bool,
    overtraining_threshold: int,
    pretrained: bool,
    cleanup: bool,
    index_algorithm: str = "Auto",
    cache_data_in_gpu: bool = False,
    custom_pretrained: bool = False,
    g_pretrained_path: str = None,
    d_pretrained_path: str = None,
    vocoder: str = "HiFi-GAN",
    architecture: str = "RVC",
    checkpointing: bool = False,
):

    if pretrained == True:
        from rvc.lib.tools.pretrained_selector import pretrained_selector

        if custom_pretrained == False:
            pg, pd = pretrained_selector(str(vocoder), int(sample_rate))
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
                model_name,
                save_every_epoch,
                total_epoch,
                pg,
                pd,
                gpu,
                batch_size,
                sample_rate,
                save_only_latest,
                save_every_weights,
                cache_data_in_gpu,
                overtraining_detector,
                overtraining_threshold,
                cleanup,
                vocoder,
                architecture,
                checkpointing,
            ],
        ),
    ]
    subprocess.run(command)
    run_index_script(model_name, index_algorithm)
    return f"Model {model_name} trained successfully."


# Index
def run_index_script(model_name: str, index_algorithm: str):
    index_script_path = os.path.join("rvc", "train", "process", "extract_index.py")
    command = [
        python,
        index_script_path,
        os.path.join(logs_path, model_name),
        index_algorithm,
    ]

    subprocess.run(command)
    return f"Index file for {model_name} generated successfully."


# Model information
def run_model_information_script(pth_path: str):
    print(model_information(pth_path))
    return model_information(pth_path)


# Model blender
def run_model_blender_script(
    model_name: str, pth_path_1: str, pth_path_2: str, ratio: float
):
    message, model_blended = model_blender(model_name, pth_path_1, pth_path_2, ratio)
    return message, model_blended


# Tensorboard
def run_tensorboard_script():
    launch_tensorboard_pipeline()


# Download
def run_download_script(model_link: str):
    model_download_pipeline(model_link)
    return f"Model downloaded successfully."


# Prerequisites
def run_prerequisites_script(
    pretraineds_hifigan: bool,
    models: bool,
    exe: bool,
):
    prequisites_download_pipeline(
        pretraineds_hifigan,
        models,
        exe,
    )
    return "Prerequisites installed successfully."


# Audio analyzer
def run_audio_analyzer_script(
    input_path: str, save_plot_path: str = "logs/audio_analysis.png"
):
    audio_info, plot_path = analyze_audio(input_path, save_plot_path)
    print(
        f"Audio info of {input_path}: {audio_info}",
        f"Audio file {input_path} analyzed successfully. Plot saved at: {plot_path}",
    )
    return audio_info, plot_path


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
    pitch_description = (
        "Set the pitch of the audio. Higher values result in a higher pitch."
    )
    infer_parser.add_argument(
        "--pitch",
        type=int,
        help=pitch_description,
        choices=range(-24, 25),
        default=0,
    )
    index_rate_description = "Control the influence of the index file on the output. Higher values mean stronger influence. Lower values can help reduce artifacts but may result in less accurate voice cloning."
    infer_parser.add_argument(
        "--index_rate",
        type=float,
        help=index_rate_description,
        choices=[i / 100.0 for i in range(0, 101)],
        default=0.3,
    )
    volume_envelope_description = "Control the blending of the output's volume envelope. A value of 1 means the output envelope is fully used."
    infer_parser.add_argument(
        "--volume_envelope",
        type=float,
        help=volume_envelope_description,
        choices=[i / 100.0 for i in range(0, 101)],
        default=1,
    )
    protect_description = "Protect consonants and breathing sounds from artifacts. A value of 0.5 offers the strongest protection, while lower values may reduce the protection level but potentially mitigate the indexing effect."
    infer_parser.add_argument(
        "--protect",
        type=float,
        help=protect_description,
        choices=[i / 1000.0 for i in range(0, 501)],
        default=0.33,
    )
    f0_method_description = "Choose the pitch extraction algorithm for the conversion. 'rmvpe' is the default and generally recommended."
    infer_parser.add_argument(
        "--f0_method",
        type=str,
        help=f0_method_description,
        choices=[
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
    infer_parser.add_argument(
        "--input_path",
        type=str,
        help="Full path to the input audio file.",
        required=True,
    )
    infer_parser.add_argument(
        "--output_path",
        type=str,
        help="Full path to the output audio file.",
        required=True,
    )
    pth_path_description = "Full path to the RVC model file (.pth)."
    infer_parser.add_argument(
        "--pth_path", type=str, help=pth_path_description, required=True
    )
    index_path_description = "Full path to the index file (.index)."
    infer_parser.add_argument(
        "--index_path", type=str, help=index_path_description, required=True
    )
    split_audio_description = "Split the audio into smaller segments before inference. This can improve the quality of the output for longer audio files."
    infer_parser.add_argument(
        "--split_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=split_audio_description,
        default=False,
    )
    f0_autotune_description = "Apply a light autotune to the inferred audio. Particularly useful for singing voice conversions."
    infer_parser.add_argument(
        "--f0_autotune",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=f0_autotune_description,
        default=False,
    )
    f0_autotune_strength_description = "Set the autotune strength - the more you increase it the more it will snap to the chromatic grid."
    infer_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        help=f0_autotune_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=1.0,
    )
    proposed_pitch_description = "Proposed Pitch"
    infer_parser.add_argument(
        "--proposed_pitch",
        type=bool,
        help=proposed_pitch_description,
        choices=[True, False],
        default=False,
    )
    proposed_pitch_threshold_description = "Proposed Pitch Threshold"
    infer_parser.add_argument(
        "--proposed_pitch_threshold",
        type=float,
        help=proposed_pitch_threshold_description,
        choices=[i for i in range(50, 1200)],
        default=155.0,
    )
    clean_audio_description = "Clean the output audio using noise reduction algorithms. Recommended for speech conversions."
    infer_parser.add_argument(
        "--clean_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=clean_audio_description,
        default=False,
    )
    clean_strength_description = "Adjust the intensity of the audio cleaning process. Higher values result in stronger cleaning, but may lead to a more compressed sound."
    infer_parser.add_argument(
        "--clean_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=0.7,
    )
    export_format_description = "Select the desired output audio format."
    infer_parser.add_argument(
        "--export_format",
        type=str,
        help=export_format_description,
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    embedder_model_description = (
        "Choose the model used for generating speaker embeddings."
    )
    infer_parser.add_argument(
        "--embedder_model",
        type=str,
        help=embedder_model_description,
        choices=[
            "contentvec",
            "spin",
            "chinese-hubert-base",
            "japanese-hubert-base",
            "korean-hubert-base",
            "custom",
        ],
        default="contentvec",
    )
    embedder_model_custom_description = "Specify the path to a custom model for speaker embedding. Only applicable if 'embedder_model' is set to 'custom'."
    infer_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help=embedder_model_custom_description,
        default=None,
    )
    formant_shifting_description = "Apply formant shifting to the input audio. This can help adjust the timbre of the voice."
    infer_parser.add_argument(
        "--formant_shifting",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=formant_shifting_description,
        default=False,
        required=False,
    )
    formant_qfrency_description = "Control the frequency of the formant shifting effect. Higher values result in a more pronounced effect."
    infer_parser.add_argument(
        "--formant_qfrency",
        type=float,
        help=formant_qfrency_description,
        default=1.0,
        required=False,
    )
    formant_timbre_description = "Control the timbre of the formant shifting effect. Higher values result in a more pronounced effect."
    infer_parser.add_argument(
        "--formant_timbre",
        type=float,
        help=formant_timbre_description,
        default=1.0,
        required=False,
    )
    sid_description = "Speaker ID for multi-speaker models."
    infer_parser.add_argument(
        "--sid",
        type=int,
        help=sid_description,
        default=0,
        required=False,
    )
    post_process_description = "Apply post-processing effects to the output audio."
    infer_parser.add_argument(
        "--post_process",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=post_process_description,
        default=False,
        required=False,
    )
    reverb_description = "Apply reverb effect to the output audio."
    infer_parser.add_argument(
        "--reverb",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=reverb_description,
        default=False,
        required=False,
    )

    pitch_shift_description = "Apply pitch shifting effect to the output audio."
    infer_parser.add_argument(
        "--pitch_shift",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=pitch_shift_description,
        default=False,
        required=False,
    )

    limiter_description = "Apply limiter effect to the output audio."
    infer_parser.add_argument(
        "--limiter",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=limiter_description,
        default=False,
        required=False,
    )

    gain_description = "Apply gain effect to the output audio."
    infer_parser.add_argument(
        "--gain",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=gain_description,
        default=False,
        required=False,
    )

    distortion_description = "Apply distortion effect to the output audio."
    infer_parser.add_argument(
        "--distortion",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=distortion_description,
        default=False,
        required=False,
    )

    chorus_description = "Apply chorus effect to the output audio."
    infer_parser.add_argument(
        "--chorus",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=chorus_description,
        default=False,
        required=False,
    )

    bitcrush_description = "Apply bitcrush effect to the output audio."
    infer_parser.add_argument(
        "--bitcrush",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=bitcrush_description,
        default=False,
        required=False,
    )

    clipping_description = "Apply clipping effect to the output audio."
    infer_parser.add_argument(
        "--clipping",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=clipping_description,
        default=False,
        required=False,
    )

    compressor_description = "Apply compressor effect to the output audio."
    infer_parser.add_argument(
        "--compressor",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=compressor_description,
        default=False,
        required=False,
    )

    delay_description = "Apply delay effect to the output audio."
    infer_parser.add_argument(
        "--delay",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=delay_description,
        default=False,
        required=False,
    )

    reverb_room_size_description = "Control the room size of the reverb effect. Higher values result in a larger room size."
    infer_parser.add_argument(
        "--reverb_room_size",
        type=float,
        help=reverb_room_size_description,
        default=0.5,
        required=False,
    )

    reverb_damping_description = "Control the damping of the reverb effect. Higher values result in a more damped sound."
    infer_parser.add_argument(
        "--reverb_damping",
        type=float,
        help=reverb_damping_description,
        default=0.5,
        required=False,
    )

    reverb_wet_gain_description = "Control the wet gain of the reverb effect. Higher values result in a stronger reverb effect."
    infer_parser.add_argument(
        "--reverb_wet_gain",
        type=float,
        help=reverb_wet_gain_description,
        default=0.5,
        required=False,
    )

    reverb_dry_gain_description = "Control the dry gain of the reverb effect. Higher values result in a stronger dry signal."
    infer_parser.add_argument(
        "--reverb_dry_gain",
        type=float,
        help=reverb_dry_gain_description,
        default=0.5,
        required=False,
    )

    reverb_width_description = "Control the stereo width of the reverb effect. Higher values result in a wider stereo image."
    infer_parser.add_argument(
        "--reverb_width",
        type=float,
        help=reverb_width_description,
        default=0.5,
        required=False,
    )

    reverb_freeze_mode_description = "Control the freeze mode of the reverb effect. Higher values result in a stronger freeze effect."
    infer_parser.add_argument(
        "--reverb_freeze_mode",
        type=float,
        help=reverb_freeze_mode_description,
        default=0.5,
        required=False,
    )

    pitch_shift_semitones_description = "Control the pitch shift in semitones. Positive values increase the pitch, while negative values decrease it."
    infer_parser.add_argument(
        "--pitch_shift_semitones",
        type=float,
        help=pitch_shift_semitones_description,
        default=0.0,
        required=False,
    )

    limiter_threshold_description = "Control the threshold of the limiter effect. Higher values result in a stronger limiting effect."
    infer_parser.add_argument(
        "--limiter_threshold",
        type=float,
        help=limiter_threshold_description,
        default=-6,
        required=False,
    )

    limiter_release_time_description = "Control the release time of the limiter effect. Higher values result in a longer release time."
    infer_parser.add_argument(
        "--limiter_release_time",
        type=float,
        help=limiter_release_time_description,
        default=0.01,
        required=False,
    )

    gain_db_description = "Control the gain in decibels. Positive values increase the gain, while negative values decrease it."
    infer_parser.add_argument(
        "--gain_db",
        type=float,
        help=gain_db_description,
        default=0.0,
        required=False,
    )

    distortion_gain_description = "Control the gain of the distortion effect. Higher values result in a stronger distortion effect."
    infer_parser.add_argument(
        "--distortion_gain",
        type=float,
        help=distortion_gain_description,
        default=25,
        required=False,
    )

    chorus_rate_description = "Control the rate of the chorus effect. Higher values result in a faster chorus effect."
    infer_parser.add_argument(
        "--chorus_rate",
        type=float,
        help=chorus_rate_description,
        default=1.0,
        required=False,
    )

    chorus_depth_description = "Control the depth of the chorus effect. Higher values result in a stronger chorus effect."
    infer_parser.add_argument(
        "--chorus_depth",
        type=float,
        help=chorus_depth_description,
        default=0.25,
        required=False,
    )

    chorus_center_delay_description = "Control the center delay of the chorus effect. Higher values result in a longer center delay."
    infer_parser.add_argument(
        "--chorus_center_delay",
        type=float,
        help=chorus_center_delay_description,
        default=7,
        required=False,
    )

    chorus_feedback_description = "Control the feedback of the chorus effect. Higher values result in a stronger feedback effect."
    infer_parser.add_argument(
        "--chorus_feedback",
        type=float,
        help=chorus_feedback_description,
        default=0.0,
        required=False,
    )

    chorus_mix_description = "Control the mix of the chorus effect. Higher values result in a stronger chorus effect."
    infer_parser.add_argument(
        "--chorus_mix",
        type=float,
        help=chorus_mix_description,
        default=0.5,
        required=False,
    )

    bitcrush_bit_depth_description = "Control the bit depth of the bitcrush effect. Higher values result in a stronger bitcrush effect."
    infer_parser.add_argument(
        "--bitcrush_bit_depth",
        type=int,
        help=bitcrush_bit_depth_description,
        default=8,
        required=False,
    )

    clipping_threshold_description = "Control the threshold of the clipping effect. Higher values result in a stronger clipping effect."
    infer_parser.add_argument(
        "--clipping_threshold",
        type=float,
        help=clipping_threshold_description,
        default=-6,
        required=False,
    )

    compressor_threshold_description = "Control the threshold of the compressor effect. Higher values result in a stronger compressor effect."
    infer_parser.add_argument(
        "--compressor_threshold",
        type=float,
        help=compressor_threshold_description,
        default=0,
        required=False,
    )

    compressor_ratio_description = "Control the ratio of the compressor effect. Higher values result in a stronger compressor effect."
    infer_parser.add_argument(
        "--compressor_ratio",
        type=float,
        help=compressor_ratio_description,
        default=1,
        required=False,
    )

    compressor_attack_description = "Control the attack of the compressor effect. Higher values result in a stronger compressor effect."
    infer_parser.add_argument(
        "--compressor_attack",
        type=float,
        help=compressor_attack_description,
        default=1.0,
        required=False,
    )

    compressor_release_description = "Control the release of the compressor effect. Higher values result in a stronger compressor effect."
    infer_parser.add_argument(
        "--compressor_release",
        type=float,
        help=compressor_release_description,
        default=100,
        required=False,
    )

    delay_seconds_description = "Control the delay time in seconds. Higher values result in a longer delay time."
    infer_parser.add_argument(
        "--delay_seconds",
        type=float,
        help=delay_seconds_description,
        default=0.5,
        required=False,
    )
    delay_feedback_description = "Control the feedback of the delay effect. Higher values result in a stronger feedback effect."
    infer_parser.add_argument(
        "--delay_feedback",
        type=float,
        help=delay_feedback_description,
        default=0.0,
        required=False,
    )
    delay_mix_description = "Control the mix of the delay effect. Higher values result in a stronger delay effect."
    infer_parser.add_argument(
        "--delay_mix",
        type=float,
        help=delay_mix_description,
        default=0.5,
        required=False,
    )

    # Parser for 'batch_infer' mode
    batch_infer_parser = subparsers.add_parser(
        "batch_infer",
        help="Run batch inference",
    )
    batch_infer_parser.add_argument(
        "--pitch",
        type=int,
        help=pitch_description,
        choices=range(-24, 25),
        default=0,
    )
    batch_infer_parser.add_argument(
        "--index_rate",
        type=float,
        help=index_rate_description,
        choices=[i / 100.0 for i in range(0, 101)],
        default=0.3,
    )
    batch_infer_parser.add_argument(
        "--volume_envelope",
        type=float,
        help=volume_envelope_description,
        choices=[i / 100.0 for i in range(0, 101)],
        default=1,
    )
    batch_infer_parser.add_argument(
        "--protect",
        type=float,
        help=protect_description,
        choices=[i / 1000.0 for i in range(0, 501)],
        default=0.33,
    )
    batch_infer_parser.add_argument(
        "--f0_method",
        type=str,
        help=f0_method_description,
        choices=[
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
    batch_infer_parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing input audio files.",
        required=True,
    )
    batch_infer_parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder for saving output audio files.",
        required=True,
    )
    batch_infer_parser.add_argument(
        "--pth_path", type=str, help=pth_path_description, required=True
    )
    batch_infer_parser.add_argument(
        "--index_path", type=str, help=index_path_description, required=True
    )
    batch_infer_parser.add_argument(
        "--split_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=split_audio_description,
        default=False,
    )
    batch_infer_parser.add_argument(
        "--f0_autotune",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=f0_autotune_description,
        default=False,
    )
    batch_infer_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=1.0,
    )
    proposed_pitch_description = "Proposed Pitch adjustment"
    batch_infer_parser.add_argument(
        "--proposed_pitch",
        type=bool,
        help=proposed_pitch_description,
        choices=[True, False],
        default=False,
    )
    proposed_pitch_threshold_description = "Proposed Pitch adjustment value"
    batch_infer_parser.add_argument(
        "--proposed_pitch_threshold",
        type=float,
        help=proposed_pitch_threshold_description,
        choices=[i for i in range(50, 1200)],
        default=155.0,
    )
    batch_infer_parser.add_argument(
        "--clean_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=clean_audio_description,
        default=False,
    )
    batch_infer_parser.add_argument(
        "--clean_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=0.7,
    )
    batch_infer_parser.add_argument(
        "--export_format",
        type=str,
        help=export_format_description,
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    batch_infer_parser.add_argument(
        "--embedder_model",
        type=str,
        help=embedder_model_description,
        choices=[
            "contentvec",
            "spin",
            "chinese-hubert-base",
            "japanese-hubert-base",
            "korean-hubert-base",
            "custom",
        ],
        default="contentvec",
    )
    batch_infer_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help=embedder_model_custom_description,
        default=None,
    )
    batch_infer_parser.add_argument(
        "--formant_shifting",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=formant_shifting_description,
        default=False,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--formant_qfrency",
        type=float,
        help=formant_qfrency_description,
        default=1.0,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--formant_timbre",
        type=float,
        help=formant_timbre_description,
        default=1.0,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--sid",
        type=int,
        help=sid_description,
        default=0,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--post_process",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=post_process_description,
        default=False,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--reverb",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=reverb_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--pitch_shift",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=pitch_shift_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--limiter",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=limiter_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--gain",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=gain_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--distortion",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=distortion_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--chorus",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=chorus_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--bitcrush",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=bitcrush_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--clipping",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=clipping_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--compressor",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=compressor_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--delay",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=delay_description,
        default=False,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--reverb_room_size",
        type=float,
        help=reverb_room_size_description,
        default=0.5,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--reverb_damping",
        type=float,
        help=reverb_damping_description,
        default=0.5,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--reverb_wet_gain",
        type=float,
        help=reverb_wet_gain_description,
        default=0.5,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--reverb_dry_gain",
        type=float,
        help=reverb_dry_gain_description,
        default=0.5,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--reverb_width",
        type=float,
        help=reverb_width_description,
        default=0.5,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--reverb_freeze_mode",
        type=float,
        help=reverb_freeze_mode_description,
        default=0.5,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--pitch_shift_semitones",
        type=float,
        help=pitch_shift_semitones_description,
        default=0.0,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--limiter_threshold",
        type=float,
        help=limiter_threshold_description,
        default=-6,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--limiter_release_time",
        type=float,
        help=limiter_release_time_description,
        default=0.01,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--gain_db",
        type=float,
        help=gain_db_description,
        default=0.0,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--distortion_gain",
        type=float,
        help=distortion_gain_description,
        default=25,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--chorus_rate",
        type=float,
        help=chorus_rate_description,
        default=1.0,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--chorus_depth",
        type=float,
        help=chorus_depth_description,
        default=0.25,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--chorus_center_delay",
        type=float,
        help=chorus_center_delay_description,
        default=7,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--chorus_feedback",
        type=float,
        help=chorus_feedback_description,
        default=0.0,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--chorus_mix",
        type=float,
        help=chorus_mix_description,
        default=0.5,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--bitcrush_bit_depth",
        type=int,
        help=bitcrush_bit_depth_description,
        default=8,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--clipping_threshold",
        type=float,
        help=clipping_threshold_description,
        default=-6,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--compressor_threshold",
        type=float,
        help=compressor_threshold_description,
        default=0,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--compressor_ratio",
        type=float,
        help=compressor_ratio_description,
        default=1,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--compressor_attack",
        type=float,
        help=compressor_attack_description,
        default=1.0,
        required=False,
    )

    batch_infer_parser.add_argument(
        "--compressor_release",
        type=float,
        help=compressor_release_description,
        default=100,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--delay_seconds",
        type=float,
        help=delay_seconds_description,
        default=0.5,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--delay_feedback",
        type=float,
        help=delay_feedback_description,
        default=0.0,
        required=False,
    )
    batch_infer_parser.add_argument(
        "--delay_mix",
        type=float,
        help=delay_mix_description,
        default=0.5,
        required=False,
    )

    # Parser for 'tts' mode
    tts_parser = subparsers.add_parser("tts", help="Run TTS inference")
    tts_parser.add_argument(
        "--tts_file", type=str, help="File with a text to be synthesized", required=True
    )
    tts_parser.add_argument(
        "--tts_text", type=str, help="Text to be synthesized", required=True
    )
    tts_parser.add_argument(
        "--tts_voice",
        type=str,
        help="Voice to be used for TTS synthesis.",
        choices=locales,
        required=True,
    )
    tts_parser.add_argument(
        "--tts_rate",
        type=int,
        help="Control the speaking rate of the TTS. Values range from -100 (slower) to 100 (faster).",
        choices=range(-100, 101),
        default=0,
    )
    tts_parser.add_argument(
        "--pitch",
        type=int,
        help=pitch_description,
        choices=range(-24, 25),
        default=0,
    )
    tts_parser.add_argument(
        "--index_rate",
        type=float,
        help=index_rate_description,
        choices=[(i / 10) for i in range(11)],
        default=0.3,
    )
    tts_parser.add_argument(
        "--volume_envelope",
        type=float,
        help=volume_envelope_description,
        choices=[(i / 10) for i in range(11)],
        default=1,
    )
    tts_parser.add_argument(
        "--protect",
        type=float,
        help=protect_description,
        choices=[(i / 10) for i in range(6)],
        default=0.33,
    )
    tts_parser.add_argument(
        "--f0_method",
        type=str,
        help=f0_method_description,
        choices=[
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
    tts_parser.add_argument(
        "--output_tts_path",
        type=str,
        help="Full path to save the synthesized TTS audio.",
        required=True,
    )
    tts_parser.add_argument(
        "--output_rvc_path",
        type=str,
        help="Full path to save the voice-converted audio using the synthesized TTS.",
        required=True,
    )
    tts_parser.add_argument(
        "--pth_path", type=str, help=pth_path_description, required=True
    )
    tts_parser.add_argument(
        "--index_path", type=str, help=index_path_description, required=True
    )
    tts_parser.add_argument(
        "--split_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=split_audio_description,
        default=False,
    )
    tts_parser.add_argument(
        "--f0_autotune",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=f0_autotune_description,
        default=False,
    )
    tts_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=1.0,
    )
    proposed_pitch_description = "Proposed Pitch adjustment"
    tts_parser.add_argument(
        "--proposed_pitch",
        type=bool,
        help=proposed_pitch_description,
        choices=[True, False],
        default=False,
    )
    proposed_pitch_threshold_description = "Proposed Pitch adjustment value"
    tts_parser.add_argument(
        "--proposed_pitch_threshold",
        type=float,
        help=proposed_pitch_threshold_description,
        choices=[i for i in range(100, 500)],
        default=155.0,
    )
    tts_parser.add_argument(
        "--clean_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=clean_audio_description,
        default=False,
    )
    tts_parser.add_argument(
        "--clean_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=0.7,
    )
    tts_parser.add_argument(
        "--export_format",
        type=str,
        help=export_format_description,
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    tts_parser.add_argument(
        "--embedder_model",
        type=str,
        help=embedder_model_description,
        choices=[
            "contentvec",
            "spin",
            "chinese-hubert-base",
            "japanese-hubert-base",
            "korean-hubert-base",
            "custom",
        ],
        default="contentvec",
    )
    tts_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help=embedder_model_custom_description,
        default=None,
    )

    # Parser for 'preprocess' mode
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess a dataset for training."
    )
    preprocess_parser.add_argument(
        "--model_name", type=str, help="Name of the model to be trained.", required=True
    )
    preprocess_parser.add_argument(
        "--dataset_path", type=str, help="Path to the dataset directory.", required=True
    )
    preprocess_parser.add_argument(
        "--sample_rate",
        type=int,
        help="Target sampling rate for the audio data.",
        choices=[32000, 40000, 48000],
        required=True,
    )
    preprocess_parser.add_argument(
        "--cpu_cores",
        type=int,
        help="Number of CPU cores to use for preprocessing.",
        choices=range(1, 65),
    )
    preprocess_parser.add_argument(
        "--cut_preprocess",
        type=str,
        choices=["Skip", "Simple", "Automatic"],
        help="Cut the dataset into smaller segments for faster preprocessing.",
        default="Automatic",
        required=True,
    )
    preprocess_parser.add_argument(
        "--process_effects",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Disable all filters during preprocessing.",
        default=False,
        required=False,
    )
    preprocess_parser.add_argument(
        "--noise_reduction",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enable noise reduction during preprocessing.",
        default=False,
        required=False,
    )
    preprocess_parser.add_argument(
        "--noise_reduction_strength",
        type=float,
        help="Strength of the noise reduction filter.",
        choices=[(i / 10) for i in range(11)],
        default=0.7,
        required=False,
    )
    preprocess_parser.add_argument(
        "--chunk_len",
        type=float,
        help="Chunk length.",
        choices=[i * 0.5 for i in range(1, 11)],
        default=3.0,
        required=False,
    )
    preprocess_parser.add_argument(
        "--overlap_len",
        type=float,
        help="Overlap length.",
        choices=[0.0, 0.1, 0.2, 0.3, 0.4],
        default=0.3,
        required=False,
    )
    preprocess_parser.add_argument(
        "--norm_mode",
        type=str,
        help="Normalization mode.",
        choices=["none", "pre", "post"],
        default="none",
        required=False,
    )

    # Parser for 'extract' mode
    extract_parser = subparsers.add_parser(
        "extract", help="Extract features from a dataset."
    )
    extract_parser.add_argument(
        "--model_name", type=str, help="Name of the model.", required=True
    )
    extract_parser.add_argument(
        "--f0_method",
        type=str,
        help="Pitch extraction method to use.",
        choices=[
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
        ],
        default="rmvpe",
    )
    extract_parser.add_argument(
        "--cpu_cores",
        type=int,
        help="Number of CPU cores to use for feature extraction (optional).",
        choices=range(1, 65),
        default=None,
    )
    extract_parser.add_argument(
        "--gpu",
        type=str,
        help="GPU device to use for feature extraction (optional).",
        default="-",
    )
    extract_parser.add_argument(
        "--sample_rate",
        type=int,
        help="Target sampling rate for the audio data.",
        choices=[32000, 40000, 44100, 48000],
        required=True,
    )
    extract_parser.add_argument(
        "--embedder_model",
        type=str,
        help=embedder_model_description,
        choices=[
            "contentvec",
            "spin",
            "chinese-hubert-base",
            "japanese-hubert-base",
            "korean-hubert-base",
            "custom",
        ],
        default="contentvec",
    )
    extract_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help=embedder_model_custom_description,
        default=None,
    )
    extract_parser.add_argument(
        "--include_mutes",
        type=int,
        help="Number of silent files to include.",
        choices=range(0, 11),
        default=2,
        required=True,
    )

    # Parser for 'train' mode
    train_parser = subparsers.add_parser("train", help="Train an RVC model.")
    train_parser.add_argument(
        "--model_name", type=str, help="Name of the model to be trained.", required=True
    )
    train_parser.add_argument(
        "--vocoder",
        type=str,
        help="Vocoder name",
        choices=["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
        default="HiFi-GAN",
    )
    train_parser.add_argument(
        "--architecture",
        type=str,
        help="Chose the architecture to be used",
        choices=["RVC", "Applio"],
        default="RVC",
        required=True,
    )
    train_parser.add_argument(
        "--checkpointing",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enables memory-efficient training.",
        default=False,
        required=False,
    )
    train_parser.add_argument(
        "--save_every_epoch",
        type=int,
        help="Save the model every specified number of epochs.",
        choices=range(1, 101),
        required=True,
    )
    train_parser.add_argument(
        "--save_only_latest",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Save only the latest model checkpoint.",
        default=False,
    )
    train_parser.add_argument(
        "--save_every_weights",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Save model weights every epoch.",
        default=True,
    )
    train_parser.add_argument(
        "--total_epoch",
        type=int,
        help="Total number of epochs to train for.",
        choices=range(1, 10001),
        default=1000,
    )
    train_parser.add_argument(
        "--sample_rate",
        type=int,
        help="Sampling rate of the training data.",
        choices=[32000, 40000, 48000],
        required=True,
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training.",
        choices=range(1, 51),
        default=8,
    )
    train_parser.add_argument(
        "--gpu",
        type=str,
        help="GPU device to use for training (e.g., '0').",
        default="0",
    )
    train_parser.add_argument(
        "--pretrained",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Use a pretrained model for initialization.",
        default=True,
    )
    train_parser.add_argument(
        "--custom_pretrained",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Use a custom pretrained model.",
        default=False,
    )
    train_parser.add_argument(
        "--g_pretrained_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the pretrained generator model file.",
    )
    train_parser.add_argument(
        "--d_pretrained_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the pretrained discriminator model file.",
    )
    train_parser.add_argument(
        "--overtraining_detector",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enable overtraining detection.",
        default=False,
    )
    train_parser.add_argument(
        "--overtraining_threshold",
        type=int,
        help="Threshold for overtraining detection.",
        choices=range(1, 101),
        default=50,
    )
    train_parser.add_argument(
        "--cleanup",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Cleanup previous training attempt.",
        default=False,
    )
    train_parser.add_argument(
        "--cache_data_in_gpu",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Cache training data in GPU memory.",
        default=False,
    )
    train_parser.add_argument(
        "--index_algorithm",
        type=str,
        choices=["Auto", "Faiss", "KMeans"],
        help="Choose the method for generating the index file.",
        default="Auto",
        required=False,
    )

    # Parser for 'index' mode
    index_parser = subparsers.add_parser(
        "index", help="Generate an index file for an RVC model."
    )
    index_parser.add_argument(
        "--model_name", type=str, help="Name of the model.", required=True
    )
    index_parser.add_argument(
        "--index_algorithm",
        type=str,
        choices=["Auto", "Faiss", "KMeans"],
        help="Choose the method for generating the index file.",
        default="Auto",
        required=False,
    )

    # Parser for 'model_information' mode
    model_information_parser = subparsers.add_parser(
        "model_information", help="Display information about a trained model."
    )
    model_information_parser.add_argument(
        "--pth_path", type=str, help="Path to the .pth model file.", required=True
    )

    # Parser for 'model_blender' mode
    model_blender_parser = subparsers.add_parser(
        "model_blender", help="Fuse two RVC models together."
    )
    model_blender_parser.add_argument(
        "--model_name", type=str, help="Name of the new fused model.", required=True
    )
    model_blender_parser.add_argument(
        "--pth_path_1",
        type=str,
        help="Path to the first .pth model file.",
        required=True,
    )
    model_blender_parser.add_argument(
        "--pth_path_2",
        type=str,
        help="Path to the second .pth model file.",
        required=True,
    )
    model_blender_parser.add_argument(
        "--ratio",
        type=float,
        help="Ratio for blending the two models (0.0 to 1.0).",
        choices=[(i / 10) for i in range(11)],
        default=0.5,
    )

    # Parser for 'tensorboard' mode
    subparsers.add_parser(
        "tensorboard", help="Launch TensorBoard for monitoring training progress."
    )

    # Parser for 'download' mode
    download_parser = subparsers.add_parser(
        "download", help="Download a model from a provided link."
    )
    download_parser.add_argument(
        "--model_link", type=str, help="Direct link to the model file.", required=True
    )

    # Parser for 'prerequisites' mode
    prerequisites_parser = subparsers.add_parser(
        "prerequisites", help="Install prerequisites for RVC."
    )
    prerequisites_parser.add_argument(
        "--pretraineds_hifigan",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        default=True,
        help="Download pretrained models for RVC v2.",
    )
    prerequisites_parser.add_argument(
        "--models",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        default=True,
        help="Download additional models.",
    )
    prerequisites_parser.add_argument(
        "--exe",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        default=True,
        help="Download required executables.",
    )

    # Parser for 'audio_analyzer' mode
    audio_analyzer = subparsers.add_parser(
        "audio_analyzer", help="Analyze an audio file."
    )
    audio_analyzer.add_argument(
        "--input_path", type=str, help="Path to the input audio file.", required=True
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
                pitch=args.pitch,
                index_rate=args.index_rate,
                volume_envelope=args.volume_envelope,
                protect=args.protect,
                f0_method=args.f0_method,
                input_path=args.input_path,
                output_path=args.output_path,
                pth_path=args.pth_path,
                index_path=args.index_path,
                split_audio=args.split_audio,
                f0_autotune=args.f0_autotune,
                f0_autotune_strength=args.f0_autotune_strength,
                proposed_pitch=args.proposed_pitch,
                proposed_pitch_threshold=args.proposed_pitch_threshold,
                clean_audio=args.clean_audio,
                clean_strength=args.clean_strength,
                export_format=args.export_format,
                embedder_model=args.embedder_model,
                embedder_model_custom=args.embedder_model_custom,
                formant_shifting=args.formant_shifting,
                formant_qfrency=args.formant_qfrency,
                formant_timbre=args.formant_timbre,
                sid=args.sid,
                post_process=args.post_process,
                reverb=args.reverb,
                pitch_shift=args.pitch_shift,
                limiter=args.limiter,
                gain=args.gain,
                distortion=args.distortion,
                chorus=args.chorus,
                bitcrush=args.bitcrush,
                clipping=args.clipping,
                compressor=args.compressor,
                delay=args.delay,
                reverb_room_size=args.reverb_room_size,
                reverb_damping=args.reverb_damping,
                reverb_wet_gain=args.reverb_wet_gain,
                reverb_dry_gain=args.reverb_dry_gain,
                reverb_width=args.reverb_width,
                reverb_freeze_mode=args.reverb_freeze_mode,
                pitch_shift_semitones=args.pitch_shift_semitones,
                limiter_threshold=args.limiter_threshold,
                limiter_release_time=args.limiter_release_time,
                gain_db=args.gain_db,
                distortion_gain=args.distortion_gain,
                chorus_rate=args.chorus_rate,
                chorus_depth=args.chorus_depth,
                chorus_center_delay=args.chorus_center_delay,
                chorus_feedback=args.chorus_feedback,
                chorus_mix=args.chorus_mix,
                bitcrush_bit_depth=args.bitcrush_bit_depth,
                clipping_threshold=args.clipping_threshold,
                compressor_threshold=args.compressor_threshold,
                compressor_ratio=args.compressor_ratio,
                compressor_attack=args.compressor_attack,
                compressor_release=args.compressor_release,
                delay_seconds=args.delay_seconds,
                delay_feedback=args.delay_feedback,
                delay_mix=args.delay_mix,
            )
        elif args.mode == "batch_infer":
            run_batch_infer_script(
                pitch=args.pitch,
                index_rate=args.index_rate,
                volume_envelope=args.volume_envelope,
                protect=args.protect,
                f0_method=args.f0_method,
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                pth_path=args.pth_path,
                index_path=args.index_path,
                split_audio=args.split_audio,
                f0_autotune=args.f0_autotune,
                f0_autotune_strength=args.f0_autotune_strength,
                proposed_pitch=args.proposed_pitch,
                proposed_pitch_threshold=args.proposed_pitch_threshold,
                clean_audio=args.clean_audio,
                clean_strength=args.clean_strength,
                export_format=args.export_format,
                embedder_model=args.embedder_model,
                embedder_model_custom=args.embedder_model_custom,
                formant_shifting=args.formant_shifting,
                formant_qfrency=args.formant_qfrency,
                formant_timbre=args.formant_timbre,
                sid=args.sid,
                post_process=args.post_process,
                reverb=args.reverb,
                pitch_shift=args.pitch_shift,
                limiter=args.limiter,
                gain=args.gain,
                distortion=args.distortion,
                chorus=args.chorus,
                bitcrush=args.bitcrush,
                clipping=args.clipping,
                compressor=args.compressor,
                delay=args.delay,
                reverb_room_size=args.reverb_room_size,
                reverb_damping=args.reverb_damping,
                reverb_wet_gain=args.reverb_wet_gain,
                reverb_dry_gain=args.reverb_dry_gain,
                reverb_width=args.reverb_width,
                reverb_freeze_mode=args.reverb_freeze_mode,
                pitch_shift_semitones=args.pitch_shift_semitones,
                limiter_threshold=args.limiter_threshold,
                limiter_release_time=args.limiter_release_time,
                gain_db=args.gain_db,
                distortion_gain=args.distortion_gain,
                chorus_rate=args.chorus_rate,
                chorus_depth=args.chorus_depth,
                chorus_center_delay=args.chorus_center_delay,
                chorus_feedback=args.chorus_feedback,
                chorus_mix=args.chorus_mix,
                bitcrush_bit_depth=args.bitcrush_bit_depth,
                clipping_threshold=args.clipping_threshold,
                compressor_threshold=args.compressor_threshold,
                compressor_ratio=args.compressor_ratio,
                compressor_attack=args.compressor_attack,
                compressor_release=args.compressor_release,
                delay_seconds=args.delay_seconds,
                delay_feedback=args.delay_feedback,
                delay_mix=args.delay_mix,
            )
        elif args.mode == "tts":
            run_tts_script(
                tts_file=args.tts_file,
                tts_text=args.tts_text,
                tts_voice=args.tts_voice,
                tts_rate=args.tts_rate,
                pitch=args.pitch,
                index_rate=args.index_rate,
                volume_envelope=args.volume_envelope,
                protect=args.protect,
                f0_method=args.f0_method,
                output_tts_path=args.output_tts_path,
                output_rvc_path=args.output_rvc_path,
                pth_path=args.pth_path,
                index_path=args.index_path,
                split_audio=args.split_audio,
                f0_autotune=args.f0_autotune,
                f0_autotune_strength=args.f0_autotune_strength,
                proposed_pitch=args.proposed_pitch,
                proposed_pitch_threshold=args.proposed_pitch_threshold,
                clean_audio=args.clean_audio,
                clean_strength=args.clean_strength,
                export_format=args.export_format,
                embedder_model=args.embedder_model,
                embedder_model_custom=args.embedder_model_custom,
            )
        elif args.mode == "preprocess":
            run_preprocess_script(
                model_name=args.model_name,
                dataset_path=args.dataset_path,
                sample_rate=args.sample_rate,
                cpu_cores=args.cpu_cores,
                cut_preprocess=args.cut_preprocess,
                process_effects=args.process_effects,
                noise_reduction=args.noise_reduction,
                clean_strength=args.noise_reduction_strength,
                chunk_len=args.chunk_len,
                overlap_len=args.overlap_len,
                normalization_mode=args.normalization_mode,
            )
        elif args.mode == "extract":
            run_extract_script(
                model_name=args.model_name,
                f0_method=args.f0_method,
                cpu_cores=args.cpu_cores,
                gpu=args.gpu,
                sample_rate=args.sample_rate,
                embedder_model=args.embedder_model,
                embedder_model_custom=args.embedder_model_custom,
                include_mutes=args.include_mutes,
            )
        elif args.mode == "train":
            run_train_script(
                model_name=args.model_name,
                save_every_epoch=args.save_every_epoch,
                save_only_latest=args.save_only_latest,
                save_every_weights=args.save_every_weights,
                total_epoch=args.total_epoch,
                sample_rate=args.sample_rate,
                batch_size=args.batch_size,
                gpu=args.gpu,
                overtraining_detector=args.overtraining_detector,
                overtraining_threshold=args.overtraining_threshold,
                pretrained=args.pretrained,
                custom_pretrained=args.custom_pretrained,
                cleanup=args.cleanup,
                index_algorithm=args.index_algorithm,
                cache_data_in_gpu=args.cache_data_in_gpu,
                g_pretrained_path=args.g_pretrained_path,
                d_pretrained_path=args.d_pretrained_path,
                vocoder=args.vocoder,
                architecture=args.architecture,
                checkpointing=args.checkpointing,
            )
        elif args.mode == "index":
            run_index_script(
                model_name=args.model_name,
                index_algorithm=args.index_algorithm,
            )
        elif args.mode == "model_information":
            run_model_information_script(
                pth_path=args.pth_path,
            )
        elif args.mode == "model_blender":
            run_model_blender_script(
                model_name=args.model_name,
                pth_path_1=args.pth_path_1,
                pth_path_2=args.pth_path_2,
                ratio=args.ratio,
            )
        elif args.mode == "tensorboard":
            run_tensorboard_script()
        elif args.mode == "download":
            run_download_script(
                model_link=args.model_link,
            )
        elif args.mode == "prerequisites":
            run_prerequisites_script(
                pretraineds_hifigan=args.pretraineds_hifigan,
                models=args.models,
                exe=args.exe,
            )
        elif args.mode == "audio_analyzer":
            run_audio_analyzer_script(
                input_path=args.input_path,
            )
    except Exception as error:
        print(f"An error occurred during execution: {error}")

        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
