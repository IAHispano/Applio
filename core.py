import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from functools import lru_cache

import click

from rvc.configs.v3 import DEFAULT_BATCH_SIZE

now_dir = os.getcwd()
sys.path.append(now_dir)

current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")

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
    infer_pipeline.convert_audio(**kwargs)
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
    infer_pipeline.convert_audio_batch(**kwargs)
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
    result = subprocess.run(command_tts, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
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
        post_process=False,
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
    result = subprocess.run(command)
    if result.returncode != 0:
        return f"Preprocessing failed for model {model_name}. Please check the console logs for more details."

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

    result = subprocess.run(command_1)
    if result.returncode != 0:
        return f"Feature extraction failed for model {model_name}. Please check the console logs for more details."

    return f"Model {model_name} extracted successfully."


def shutdown_after_training():
    os_name = sys.platform
    shutdown_time = None

    # Windows
    if os_name == "win32":
        delay_seconds = 300
        shutdown_time = datetime.now() + timedelta(seconds=delay_seconds)
        os.system(f"shutdown /s /t {delay_seconds}")

    # MacOS
    elif os_name == "darwin":
        shutdown_time = datetime.now()
        os.system("osascript -e 'tell app \"System Events\" to shut down'")

    # Linux
    elif os_name.startswith("linux"):
        delay_minutes = 5
        shutdown_time = datetime.now() + timedelta(minutes=delay_minutes)
        os.system(f"shutdown -h +{delay_minutes}")

    # Unknown
    else:
        print("Unsupported OS")
        return os_name, None

    return os_name, shutdown_time


def append_data_shutdown_log(
    model_name, total_epoch, batch_size, sample_rate, gpu, shutdown_time, os_name
):
    log_file = "training_shutdown_log.txt"

    log_entry = (
        f"[{datetime.now()}] "
        f"Model: {model_name} | "
        f"Epochs: {total_epoch} | "
        f"Batch: {batch_size} | "
        f"SR: {sample_rate} | "
        f"GPU: {gpu} | "
        f"OS: {os_name} | "
        f"Shutdown at: {shutdown_time}\n"
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)


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
    pretrained: bool,
    cleanup: bool,
    index_algorithm: str = "Auto",
    cache_data_in_gpu: bool = False,
    custom_pretrained: bool = False,
    g_pretrained_path: str = None,
    d_pretrained_path: str = None,
    vocoder: str = "HiFi-GAN",
    checkpointing: bool = False,
    shutdown_check: bool = False,
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
                cleanup,
                vocoder,
                checkpointing,
            ],
        ),
    ]
    result = subprocess.run(command)
    if result.returncode != 0:
        return f"Training failed for model {model_name}. Please check the console logs for more details."

    run_index_script(model_name, index_algorithm)

    if shutdown_check:
        os_name, shutdown_datetime = shutdown_after_training()

        append_data_shutdown_log(
            model_name=model_name,
            total_epoch=total_epoch,
            batch_size=batch_size,
            sample_rate=sample_rate,
            gpu=gpu,
            shutdown_time=shutdown_datetime,
            os_name=os_name,
        )

        print(
            f"Model {model_name} trained successfully. Shutdown scheduled at {shutdown_datetime}"
        )
        return f"Model {model_name} trained successfully. Shutdown scheduled at {shutdown_datetime}"

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

    result = subprocess.run(command)
    if result.returncode != 0:
        return f"Index generation failed for model {model_name}. Make sure you have enough GPU available to generate the Index file. Please check the console logs for more details."

    return f"Index file for {model_name} generated successfully."


# Model information
def run_model_information_script(pth_path: str):
    from rvc.configs.architectures import inspect_model

    metadata = inspect_model(pth_path)
    if metadata["backend"] == "applio-v3":
        result = json.dumps(metadata, indent=2)
    else:
        from rvc.train.process.model_information import model_information as model_info

        result = model_info(pth_path)
    print(result)
    return result


# Model blender
def run_model_blender_script(
    model_name: str, pth_path_1: str, pth_path_2: str, ratio: float
):
    from rvc.train.process.model_blender import model_blender as blender

    message, model_blended = blender(model_name, pth_path_1, pth_path_2, ratio)
    return message, model_blended


# Tensorboard
def run_tensorboard_script():
    from rvc.lib.tools.launch_tensorboard import launch_tensorboard_pipeline

    launch_tensorboard_pipeline()


# Download
def run_download_script(model_link: str):
    from rvc.lib.tools.model_download import model_download_pipeline

    result = model_download_pipeline(model_link)
    if result == "Error" or result is None:
        return "An error occurred downloading the model. Please check the console logs for more details."
    return "Model downloaded successfully."


# Prerequisites
def run_prerequisites_script(
    pretraineds_hifigan: bool,
    models: bool,
    exe: bool,
):
    from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline

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
    from rvc.lib.tools.analyzer import analyze_audio

    audio_info, plot_path = analyze_audio(input_path, save_plot_path)
    print(
        f"Audio info of {input_path}: {audio_info}",
        f"Audio file {input_path} analyzed successfully. Plot saved at: {plot_path}",
    )
    return audio_info, plot_path


def v3_project(model_name):
    """Keep training artifacts inside the existing logs/<model> structure."""
    from pathlib import Path

    if (
        not model_name
        or Path(model_name).name != model_name
        or model_name in {".", ".."}
    ):
        raise ValueError("Model name must be one directory name")
    return Path(current_script_directory) / "logs" / model_name


def run_v3_prepare_script(
    model_name,
    dataset_path,
    encoder_path="rvc/models/embedders/contentvec",
    pitch_extractor="swift",
    pitch_path=None,
    profile="bounded",
    device="auto",
    validation_fraction=0.1,
    segment_seconds=4,
    seed=1234,
    speakers=(),
    recordings_per_speaker=0,
    progress=None,
):
    from rvc.train.extract.v3 import FeatureExtractor
    from rvc.train.v3.data import atomic_json, prepare
    from rvc.train.v3.trainer import resolve_device

    project = v3_project(model_name)
    extractor = FeatureExtractor(
        encoder_path,
        device=resolve_device(device),
        pitch=pitch_extractor,
        pitch_path=pitch_path or None,
        profile=profile,
    )
    result = prepare(
        dataset_path,
        project / "data",
        extractor,
        validation_fraction,
        segment_seconds,
        seed,
        speaker_names=speakers,
        recordings_per_speaker=int(recordings_per_speaker),
        progress=progress,
    )
    atomic_json(
        project / "preparation.json",
        dict(
            model_name=model_name,
            dataset_path=dataset_path,
            encoder_path=encoder_path,
            pitch_extractor=pitch_extractor,
            pitch_path=pitch_path,
            profile=profile,
            device=device,
            validation_fraction=validation_fraction,
            segment_seconds=segment_seconds,
            seed=seed,
            speakers=list(speakers),
            recordings_per_speaker=int(recordings_per_speaker),
        ),
    )
    return str(result)


def run_v3_train_script(
    model_name,
    stage="predictor",
    manifest=None,
    output_dir=None,
    base_model=None,
    config=None,
    **kwargs,
):
    from pathlib import Path

    from rvc.configs.v3 import AcousticConfig, VocoderConfig
    from rvc.train.v3.trainer import train

    project = v3_project(model_name)
    kind = "vocoder" if stage == "vocoder" else "acoustic"
    if config:
        constructor = VocoderConfig if kind == "vocoder" else AcousticConfig
        config = constructor(**json.loads(Path(config).read_text(encoding="utf-8")))
    yield from train(
        manifest or project / "data/manifest.json",
        output_dir or project / "checkpoints" / stage,
        kind=kind,
        phase="predictor" if kind == "vocoder" else stage,
        pretrained=base_model or None,
        config=config,
        **kwargs,
    )


def run_v3_infer_script(
    input_path,
    output_path,
    pth_path,
    vocoder_path,
    encoder_path="rvc/models/embedders/contentvec",
    pitch_path=None,
    sid=0,
    pitch=0,
    refinement_steps=4,
    seed=0,
    device="auto",
    ordinary_flow=False,
    batch=False,
):
    from pathlib import Path

    from rvc.infer.v3 import Converter

    if not vocoder_path:
        raise ValueError("Applio v3 requires a separate --vocoder-path")
    converter = Converter(
        pth_path, vocoder_path, encoder_path, pitch_path or None, device
    )
    options = dict(
        speaker=int(sid),
        semitones=float(pitch),
        steps=int(refinement_steps),
        seed=int(seed),
        ordinary=ordinary_flow,
    )
    if not batch:
        return converter.convert_file(input_path, output_path, **options)
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if not source.is_dir() or destination.is_relative_to(source):
        raise ValueError(
            "Batch input must be a directory and output must be outside its tree"
        )
    files = sorted(
        p
        for p in source.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif"}
    )
    if not files:
        raise ValueError("No supported batch inputs")
    return [
        converter.convert_file(
            file, destination / file.relative_to(source).with_suffix(".wav"), **options
        )
        for file in files
    ]


def run_v3_export_script(checkpoint, output_path):
    from rvc.train.process.v3_checkpoints import export_checkpoint

    return str(export_checkpoint(checkpoint, output_path))


def run_v3_evaluate_script(manifest, pth_path, vocoder_path, output_dir, **kwargs):
    from rvc.train.process.v3_evaluation import evaluate

    return evaluate(manifest, pth_path, vocoder_path, output_dir, **kwargs)


def _architecture_options(func):
    return click.option(
        "--architecture",
        type=click.Choice(["classic", "v3"]),
        default="classic",
        show_default=True,
    )(func)


def _v3_prepare_options(func):
    for option in reversed(
        [
            click.option(
                "--encoder-path",
                default="rvc/models/embedders/contentvec",
                type=click.Path(file_okay=False),
            ),
            click.option(
                "--pitch-extractor",
                type=click.Choice(["swift", "rmvpe"]),
                default="swift",
            ),
            click.option("--pitch-path", type=click.Path(exists=True)),
            click.option(
                "--profile",
                type=click.Choice(["bounded", "offline"]),
                default="bounded",
            ),
            click.option("--device", default="auto"),
            click.option(
                "--validation-fraction",
                type=click.FloatRange(0.001, 0.999),
                default=0.1,
            ),
            click.option(
                "--segment-seconds", type=click.FloatRange(min=0.05), default=4.0
            ),
            click.option("--seed", type=int, default=1234),
            click.option("--speaker", "speakers", multiple=True),
            click.option(
                "--recordings-per-speaker", type=click.IntRange(min=0), default=0
            ),
        ]
    ):
        func = option(func)
    return func


def _v3_inference_options(func):
    for option in reversed(
        [
            click.option(
                "--architecture",
                type=click.Choice(["auto", "classic", "v3"]),
                default="auto",
                show_default=True,
            ),
            click.option("--vocoder-path", type=click.Path(exists=True)),
            click.option("--encoder-path", default="rvc/models/embedders/contentvec"),
            click.option("--pitch-path", type=click.Path(exists=True)),
            click.option(
                "--refinement-steps",
                type=click.Choice(["0", "1", "2", "4", "8", "16", "32"]),
                default="4",
            ),
            click.option("--seed", type=int, default=0),
            click.option("--device", default="auto"),
            click.option("--ordinary-flow", is_flag=True),
        ]
    ):
        func = option(func)
    return func


def _dispatch_inference(kwargs, batch=False):
    from rvc.configs.architectures import resolve_architecture

    try:
        architecture = resolve_architecture(
            kwargs.pop("architecture"), kwargs["pth_path"]
        )
        names = (
            "vocoder_path",
            "encoder_path",
            "pitch_path",
            "refinement_steps",
            "seed",
            "device",
            "ordinary_flow",
        )
        options = {name: kwargs.pop(name) for name in names}
        if architecture == "v3":
            supported = set(names) | {
                "architecture",
                "input_path",
                "output_path",
                "input_folder",
                "output_folder",
                "pth_path",
                "sid",
                "pitch",
                "index_path",
            }
            _reject_explicit_options(
                set(click.get_current_context().params) - supported, "v3 inference"
            )
            if kwargs.get("index_path"):
                raise ValueError(
                    "Retrieval indexes belong to classic models; leave --index-path empty for v3"
                )
            return run_v3_infer_script(
                kwargs["input_folder" if batch else "input_path"],
                kwargs["output_folder" if batch else "output_path"],
                kwargs["pth_path"],
                sid=kwargs.get("sid", 0),
                pitch=kwargs["pitch"],
                batch=batch,
                **options,
            )
        _reject_explicit_options(names, "classic inference")
        kwargs["index_path"] = kwargs.get("index_path") or ""
        return (
            run_batch_infer_script(**kwargs) if batch else run_infer_script(**kwargs)[0]
        )
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error


def _reject_explicit_options(names, operation):
    context = click.get_current_context()
    unsupported = sorted(
        "--" + name.replace("_", "-")
        for name in names
        if context.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
    )
    if unsupported:
        raise click.ClickException(
            f"Options unavailable for {operation}: {', '.join(unsupported)}"
        )


def _get_version():
    config_path = os.path.join(
        current_script_directory, "assets", "config_template.json"
    )
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("version", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


VERSION = _get_version()


def _infer_opts(func):
    """Core inference options shared by infer, batch_infer and tts."""
    opts = [
        click.option(
            "--pitch",
            type=click.IntRange(-24, 24),
            default=0,
            help="Set the pitch of the audio. Higher values result in a higher pitch.",
        ),
        click.option(
            "--index-rate",
            type=click.FloatRange(0, 1),
            default=0.3,
            help="Control the influence of the index file on the output.",
        ),
        click.option(
            "--volume-envelope",
            type=click.FloatRange(0, 1),
            default=1.0,
            help="Control the blending of the output's volume envelope.",
        ),
        click.option(
            "--protect",
            type=click.FloatRange(0, 0.5),
            default=0.33,
            help="Protect consonants and breathing sounds from artifacts.",
        ),
        click.option(
            "--f0-method",
            type=click.Choice(
                [
                    "crepe",
                    "crepe-tiny",
                    "rmvpe",
                    "fcpe",
                    "hybrid[crepe+rmvpe]",
                    "hybrid[crepe+fcpe]",
                    "hybrid[rmvpe+fcpe]",
                    "hybrid[crepe+rmvpe+fcpe]",
                ]
            ),
            default="rmvpe",
            help="Choose the pitch extraction algorithm.",
        ),
        click.option(
            "--split-audio",
            is_flag=True,
            default=False,
            help="Split audio into smaller segments before inference.",
        ),
        click.option(
            "--f0-autotune",
            is_flag=True,
            default=False,
            help="Apply a light autotune to the inferred audio.",
        ),
        click.option(
            "--f0-autotune-strength",
            type=click.FloatRange(0, 1),
            default=1.0,
            help="Autotune strength (higher = more chromatic snap).",
        ),
        click.option(
            "--proposed-pitch",
            is_flag=True,
            default=False,
            help="Enable proposed pitch adjustment.",
        ),
        click.option(
            "--proposed-pitch-threshold",
            type=click.FloatRange(50, 1199),
            default=155.0,
            help="Proposed pitch threshold value.",
        ),
        click.option(
            "--clean-audio",
            is_flag=True,
            default=False,
            help="Clean output audio using noise reduction.",
        ),
        click.option(
            "--clean-strength",
            type=click.FloatRange(0, 1),
            default=0.7,
            help="Intensity of the audio cleaning process.",
        ),
        click.option(
            "--export-format",
            type=click.Choice(["WAV", "MP3", "FLAC", "OGG", "M4A"]),
            default="WAV",
            help="Output audio format.",
        ),
        click.option(
            "--embedder-model",
            type=click.Choice(
                [
                    "contentvec",
                    "spin",
                    "spin-v2",
                    "chinese-hubert-base",
                    "japanese-hubert-base",
                    "korean-hubert-base",
                    "custom",
                ]
            ),
            default="contentvec",
            help="Model used for generating speaker embeddings.",
        ),
        click.option(
            "--embedder-model-custom",
            type=str,
            default=None,
            help="Path to a custom embedding model (only when --embedder-model is 'custom').",
        ),
        click.option(
            "--sid", type=int, default=0, help="Speaker ID for multi-speaker models."
        ),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


def _post_process_opts(func):
    """Post-processing options shared by infer and batch_infer."""
    opts = [
        click.option(
            "--formant-shifting",
            is_flag=True,
            default=False,
            help="Apply formant shifting to the input audio.",
        ),
        click.option(
            "--formant-qfrency",
            type=float,
            default=1.0,
            help="Formant shift frequency.",
        ),
        click.option(
            "--formant-timbre", type=float, default=1.0, help="Formant shift timbre."
        ),
        click.option(
            "--post-process",
            is_flag=True,
            default=False,
            help="Apply post-processing effects.",
        ),
        click.option(
            "--reverb", is_flag=True, default=False, help="Apply reverb effect."
        ),
        click.option("--reverb-room-size", type=float, default=0.5),
        click.option("--reverb-damping", type=float, default=0.5),
        click.option("--reverb-wet-gain", type=float, default=0.5),
        click.option("--reverb-dry-gain", type=float, default=0.5),
        click.option("--reverb-width", type=float, default=0.5),
        click.option("--reverb-freeze-mode", type=float, default=0.5),
        click.option(
            "--pitch-shift",
            is_flag=True,
            default=False,
            help="Apply pitch shift effect.",
        ),
        click.option("--pitch-shift-semitones", type=float, default=0.0),
        click.option(
            "--limiter", is_flag=True, default=False, help="Apply limiter effect."
        ),
        click.option("--limiter-threshold", type=float, default=-6),
        click.option("--limiter-release-time", type=float, default=0.01),
        click.option("--gain", is_flag=True, default=False, help="Apply gain effect."),
        click.option("--gain-db", type=float, default=0.0),
        click.option(
            "--distortion", is_flag=True, default=False, help="Apply distortion effect."
        ),
        click.option("--distortion-gain", type=float, default=25),
        click.option(
            "--chorus", is_flag=True, default=False, help="Apply chorus effect."
        ),
        click.option("--chorus-rate", type=float, default=1.0),
        click.option("--chorus-depth", type=float, default=0.25),
        click.option("--chorus-center-delay", type=float, default=7),
        click.option("--chorus-feedback", type=float, default=0.0),
        click.option("--chorus-mix", type=float, default=0.5),
        click.option(
            "--bitcrush", is_flag=True, default=False, help="Apply bitcrush effect."
        ),
        click.option("--bitcrush-bit-depth", type=int, default=8),
        click.option(
            "--clipping", is_flag=True, default=False, help="Apply clipping effect."
        ),
        click.option("--clipping-threshold", type=float, default=-6),
        click.option(
            "--compressor", is_flag=True, default=False, help="Apply compressor effect."
        ),
        click.option("--compressor-threshold", type=float, default=0),
        click.option("--compressor-ratio", type=float, default=1),
        click.option("--compressor-attack", type=float, default=1.0),
        click.option("--compressor-release", type=float, default=100),
        click.option(
            "--delay", is_flag=True, default=False, help="Apply delay effect."
        ),
        click.option("--delay-seconds", type=float, default=0.5),
        click.option("--delay-feedback", type=float, default=0.0),
        click.option("--delay-mix", type=float, default=0.5),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


@click.group(invoke_without_command=True)
@click.version_option(
    version=VERSION, prog_name="Applio", message="%(prog)s v%(version)s"
)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@cli.command()
@click.option("--input-path", required=True, help="Full path to the input audio file.")
@click.option(
    "--output-path", required=True, help="Full path to the output audio file."
)
@click.option(
    "--pth-path",
    "--model-path",
    required=True,
    help="Full path to the RVC model file (.pth).",
)
@click.option("--index-path", default="", help="Full path to the index file (.index).")
@_v3_inference_options
@_infer_opts
@_post_process_opts
def infer(**kwargs):
    """Run voice conversion on a single audio file."""
    click.echo(_dispatch_inference(kwargs))


@cli.command()
@click.option(
    "--input-folder", required=True, help="Folder containing input audio files."
)
@click.option(
    "--output-folder", required=True, help="Folder for saving output audio files."
)
@click.option(
    "--pth-path",
    "--model-path",
    required=True,
    help="Full path to the RVC model file (.pth).",
)
@click.option("--index-path", default="", help="Full path to the index file (.index).")
@_v3_inference_options
@_infer_opts
@_post_process_opts
def batch_infer(**kwargs):
    """Run voice conversion on multiple audio files in a folder."""
    click.echo(_dispatch_inference(kwargs, batch=True))


@cli.command()
@click.option("--tts-file", required=True, help="File with text to be synthesized.")
@click.option("--tts-text", required=True, help="Text to be synthesized.")
@click.option(
    "--tts-voice",
    required=True,
    type=click.Choice(locales, case_sensitive=False),
    help="Voice to use for TTS synthesis.",
)
@click.option(
    "--tts-rate",
    type=click.IntRange(-100, 100),
    default=0,
    help="Speaking rate (-100 slower .. 100 faster).",
)
@click.option(
    "--output-tts-path", required=True, help="Path to save the synthesized TTS audio."
)
@click.option(
    "--output-rvc-path", required=True, help="Path to save the voice-converted audio."
)
@click.option(
    "--pth-path", required=True, help="Full path to the RVC model file (.pth)."
)
@click.option(
    "--index-path", required=True, help="Full path to the index file (.index)."
)
@_infer_opts
def tts(**kwargs):
    """Synthesise speech with TTS and apply voice conversion."""
    result = run_tts_script(**kwargs)
    click.echo(result[0])


@cli.command()
@click.option("--model-name", required=True, help="Name of the model to train.")
@click.option("--dataset-path", required=True, help="Path to the dataset directory.")
@click.option(
    "--sample-rate",
    default="40000",
    type=click.Choice(["32000", "40000", "44100", "48000"]),
    help="Target sampling rate.",
)
@click.option(
    "--cpu-cores",
    type=click.IntRange(1, 64),
    default=None,
    help="Number of CPU cores to use.",
)
@click.option(
    "--cut-preprocess",
    type=click.Choice(["Skip", "Simple", "Automatic"]),
    default="Automatic",
    help="Dataset cutting method.",
)
@click.option(
    "--process-effects",
    is_flag=True,
    default=False,
    help="Disable filters during preprocessing.",
)
@click.option(
    "--noise-reduction",
    is_flag=True,
    default=False,
    help="Enable noise reduction during preprocessing.",
)
@click.option(
    "--noise-reduction-strength",
    type=click.FloatRange(0, 1),
    default=0.7,
    help="Strength of the noise reduction filter.",
)
@click.option(
    "--chunk-len",
    type=click.Choice([str(i * 0.5) for i in range(1, 11)]),
    default="3.0",
    help="Chunk length in seconds.",
)
@click.option(
    "--overlap-len",
    type=click.Choice(["0.0", "0.1", "0.2", "0.3", "0.4"]),
    default="0.3",
    help="Overlap length.",
)
@click.option(
    "--normalization-mode",
    type=click.Choice(["none", "pre", "post"]),
    default="none",
    help="Normalization mode.",
)
@_architecture_options
@_v3_prepare_options
def preprocess(**kwargs):
    """Preprocess a dataset for training."""
    architecture = kwargs.pop("architecture")
    names = (
        "encoder_path",
        "pitch_extractor",
        "pitch_path",
        "profile",
        "device",
        "validation_fraction",
        "segment_seconds",
        "seed",
        "speakers",
        "recordings_per_speaker",
    )
    options = {name: kwargs.pop(name) for name in names}
    if architecture == "v3":
        _reject_explicit_options(
            set(kwargs) - {"model_name", "dataset_path", "sample_rate"},
            "v3 preparation",
        )
        if (
            click.get_current_context().get_parameter_source("sample_rate")
            == click.core.ParameterSource.COMMANDLINE
            and kwargs["sample_rate"] != "44100"
        ):
            raise click.ClickException("Applio v3 uses the fixed 44100 Hz mel contract")
        try:
            click.echo(
                run_v3_prepare_script(
                    kwargs["model_name"],
                    kwargs["dataset_path"],
                    progress=lambda update: click.echo(json.dumps(update)),
                    **options,
                )
            )
        except (ValueError, OSError) as error:
            raise click.ClickException(str(error)) from error
        return
    _reject_explicit_options(names, "classic preparation")
    if kwargs["sample_rate"] == "44100":
        raise click.ClickException(
            "Classic preprocessing supports 32000, 40000 or 48000 Hz"
        )
    kwargs["sample_rate"] = int(kwargs["sample_rate"])
    kwargs["noise_reduction_strength"] = float(kwargs["noise_reduction_strength"])
    kwargs["chunk_len"] = float(kwargs["chunk_len"])
    kwargs["overlap_len"] = float(kwargs["overlap_len"])
    kwargs["cpu_cores"] = kwargs.get("cpu_cores") or 1
    result = run_preprocess_script(
        model_name=kwargs["model_name"],
        dataset_path=kwargs["dataset_path"],
        sample_rate=kwargs["sample_rate"],
        cpu_cores=kwargs["cpu_cores"],
        cut_preprocess=kwargs["cut_preprocess"],
        process_effects=kwargs["process_effects"],
        noise_reduction=kwargs["noise_reduction"],
        clean_strength=kwargs["noise_reduction_strength"],
        chunk_len=kwargs["chunk_len"],
        overlap_len=kwargs["overlap_len"],
        normalization_mode=kwargs["normalization_mode"],
    )
    click.echo(result)


@cli.command()
@click.option("--model-name", required=True, help="Name of the model.")
@click.option(
    "--f0-method",
    type=click.Choice(["crepe", "crepe-tiny", "rmvpe", "fcpe"]),
    default="rmvpe",
    help="Pitch extraction method.",
)
@click.option(
    "--cpu-cores", type=click.IntRange(1, 64), default=None, help="Number of CPU cores."
)
@click.option("--gpu", type=str, default="-", help="GPU device to use (e.g. '0').")
@click.option(
    "--sample-rate",
    default="40000",
    type=click.Choice(["32000", "40000", "44100", "48000"]),
    help="Target sampling rate.",
)
@click.option(
    "--embedder-model",
    type=click.Choice(
        [
            "contentvec",
            "spin",
            "spin-v2",
            "chinese-hubert-base",
            "japanese-hubert-base",
            "korean-hubert-base",
            "custom",
        ]
    ),
    default="contentvec",
    help="Model used for generating speaker embeddings.",
)
@click.option(
    "--embedder-model-custom",
    type=str,
    default=None,
    help="Path to custom embedding model.",
)
@click.option(
    "--include-mutes",
    type=click.IntRange(0, 10),
    default=2,
    help="Number of silent files to include.",
)
@_architecture_options
def extract(**kwargs):
    """Extract features from a preprocessed dataset."""
    if kwargs.pop("architecture") == "v3":
        recipe = v3_project(kwargs["model_name"]) / "preparation.json"
        if not recipe.exists():
            raise click.ClickException(
                "Run preprocess --architecture v3 first; it prepares audio and features together"
            )
        click.echo(
            run_v3_prepare_script(**json.loads(recipe.read_text(encoding="utf-8")))
        )
        return
    kwargs["sample_rate"] = int(kwargs["sample_rate"])
    kwargs["cpu_cores"] = kwargs.get("cpu_cores") or 1
    result = run_extract_script(
        model_name=kwargs["model_name"],
        f0_method=kwargs["f0_method"],
        cpu_cores=kwargs["cpu_cores"],
        gpu=kwargs["gpu"],
        sample_rate=kwargs["sample_rate"],
        embedder_model=kwargs["embedder_model"],
        embedder_model_custom=kwargs["embedder_model_custom"],
        include_mutes=kwargs["include_mutes"],
    )
    click.echo(result)


@cli.command()
@click.option("--model-name", required=True, help="Name of the model to train.")
@click.option(
    "--vocoder",
    type=click.Choice(["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]),
    default="HiFi-GAN",
    help="Vocoder to use.",
)
@click.option(
    "--checkpointing",
    is_flag=True,
    default=False,
    help="Enable memory-efficient checkpointing.",
)
@click.option(
    "--save-every-epoch",
    default=10,
    type=click.IntRange(1, 100),
    help="Save checkpoint every N epochs.",
)
@click.option(
    "--save-only-latest",
    is_flag=True,
    default=False,
    help="Keep only the latest checkpoint.",
)
@click.option(
    "--save-every-weights",
    is_flag=True,
    default=True,
    help="Save model weights every epoch.",
)
@click.option(
    "--total-epoch",
    type=click.IntRange(1, 10000),
    default=1000,
    help="Total number of training epochs.",
)
@click.option(
    "--sample-rate",
    default="40000",
    type=click.Choice(["32000", "40000", "44100", "48000"]),
    help="Training sampling rate.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(1, 64),
    default=8,
    help="Training batch size; increase when memory permits. V3 defaults to 2.",
)
@click.option("--gpu", type=str, default="0", help="GPU device to use.")
@click.option(
    "--pretrained/--no-pretrained",
    default=True,
    help="Use pretrained model for initialisation.",
)
@click.option(
    "--custom-pretrained",
    is_flag=True,
    default=False,
    help="Use custom pretrained model paths.",
)
@click.option(
    "--g-pretrained-path", type=str, default=None, help="Path to pretrained generator."
)
@click.option(
    "--d-pretrained-path",
    type=str,
    default=None,
    help="Path to pretrained discriminator.",
)
@click.option(
    "--cleanup", is_flag=True, default=False, help="Clean up previous training attempt."
)
@click.option(
    "--cache-data-in-gpu",
    is_flag=True,
    default=False,
    help="Cache training data in GPU memory.",
)
@click.option(
    "--index-algorithm",
    type=click.Choice(["Auto", "Faiss", "KMeans"]),
    default="Auto",
    help="Index file generation algorithm.",
)
@_architecture_options
@click.option(
    "--stage",
    type=click.Choice(["predictor", "flow", "shortcut", "adapt", "vocoder"]),
    default="predictor",
)
@click.option("--manifest", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
@click.option("--base-model", type=click.Path(exists=True))
@click.option("--resume", type=click.Path(exists=True))
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Research-only override of the default V3 architecture for scratch training.",
)
@click.option("--steps", type=click.IntRange(min=1), default=10000)
@click.option("--crop-frames", type=click.IntRange(min=1), default=128)
@click.option(
    "--learning-rate", type=click.FloatRange(min=0, min_open=True), default=2e-4
)
@click.option(
    "--precision", type=click.Choice(["auto", "bf16", "fp16", "fp32"]), default="auto"
)
@click.option("--device", default="auto")
@click.option("--seed", type=int, default=1234)
@click.option("--checkpoint-every", type=click.IntRange(min=1), default=100)
@click.option("--adapter-rank", type=click.IntRange(min=1), default=8)
@click.option("--adaptation", type=click.Choice(["lora", "full"]), default="lora")
@click.option("--accumulation-steps", type=click.IntRange(min=1), default=1)
def train(**kwargs):
    """Train classic RVC or a selected Applio v3 stage."""
    architecture = kwargs.pop("architecture")
    names = (
        "stage",
        "manifest",
        "output_dir",
        "base_model",
        "resume",
        "config",
        "steps",
        "crop_frames",
        "learning_rate",
        "precision",
        "device",
        "seed",
        "checkpoint_every",
        "adapter_rank",
        "adaptation",
        "accumulation_steps",
    )
    options = {name: kwargs.pop(name) for name in names}
    if architecture == "v3":
        _reject_explicit_options(
            set(kwargs) - {"model_name", "sample_rate", "batch_size"}, "v3 training"
        )
        if (
            click.get_current_context().get_parameter_source("sample_rate")
            == click.core.ParameterSource.COMMANDLINE
            and kwargs["sample_rate"] != "44100"
        ):
            raise click.ClickException("Applio v3 uses 44100 Hz")
        if (
            click.get_current_context().get_parameter_source("batch_size")
            == click.core.ParameterSource.DEFAULT
        ):
            kwargs["batch_size"] = DEFAULT_BATCH_SIZE
        try:
            for update in run_v3_train_script(
                kwargs["model_name"], batch_size=kwargs["batch_size"], **options
            ):
                click.echo(json.dumps(update), color=False)
        except (ValueError, OSError) as error:
            raise click.ClickException(str(error)) from error
        return
    _reject_explicit_options(names, "classic training")
    if kwargs["sample_rate"] == "44100":
        raise click.ClickException("Classic training supports 32000, 40000 or 48000 Hz")
    result = run_train_script(
        model_name=kwargs["model_name"],
        save_every_epoch=kwargs["save_every_epoch"],
        save_only_latest=kwargs["save_only_latest"],
        save_every_weights=kwargs["save_every_weights"],
        total_epoch=kwargs["total_epoch"],
        sample_rate=int(kwargs["sample_rate"]),
        batch_size=kwargs["batch_size"],
        gpu=kwargs["gpu"],
        pretrained=kwargs["pretrained"],
        cleanup=kwargs["cleanup"],
        index_algorithm=kwargs["index_algorithm"],
        cache_data_in_gpu=kwargs["cache_data_in_gpu"],
        custom_pretrained=kwargs["custom_pretrained"],
        g_pretrained_path=kwargs.get("g_pretrained_path"),
        d_pretrained_path=kwargs.get("d_pretrained_path"),
        vocoder=kwargs["vocoder"],
        checkpointing=kwargs["checkpointing"],
    )
    click.echo(result)


@cli.command()
@click.option("--model-name", required=True, help="Name of the model.")
@click.option(
    "--index-algorithm",
    type=click.Choice(["Auto", "Faiss", "KMeans"]),
    default="Auto",
    help="Index file generation algorithm.",
)
def index(**kwargs):
    """Generate an index file for an RVC model."""
    result = run_index_script(kwargs["model_name"], kwargs["index_algorithm"])
    click.echo(result)


@cli.command()
@click.option("--pth-path", required=True, help="Path to the .pth model file.")
def model_information(**kwargs):
    """Display information about a trained model."""
    run_model_information_script(kwargs["pth_path"])


@cli.command()
@click.option("--model-name", required=True, help="Name of the new fused model.")
@click.option("--pth-path-1", required=True, help="Path to the first .pth model.")
@click.option("--pth-path-2", required=True, help="Path to the second .pth model.")
@click.option(
    "--ratio",
    type=click.Choice([str(i / 10) for i in range(11)]),
    default="0.5",
    help="Blending ratio (0.0 - 1.0).",
)
def model_blender(**kwargs):
    """Fuse two RVC models together."""
    kwargs["ratio"] = float(kwargs["ratio"])
    msg, path = run_model_blender_script(
        kwargs["model_name"],
        kwargs["pth_path_1"],
        kwargs["pth_path_2"],
        kwargs["ratio"],
    )
    click.echo(f"{msg} {path}")


@cli.command()
def tensorboard():
    """Launch TensorBoard for monitoring training progress."""
    run_tensorboard_script()


@cli.command()
@click.option("--model-link", required=True, help="Direct link to the model file.")
def download(**kwargs):
    """Download a model from a provided link."""
    result = run_download_script(kwargs["model_link"])
    click.echo(result)


@cli.command()
@click.option(
    "--pretraineds-hifigan/--no-pretraineds-hifigan",
    default=True,
    help="Download pretrained HiFi-GAN models.",
)
@click.option("--models/--no-models", default=True, help="Download additional models.")
@click.option("--exe/--no-exe", default=True, help="Download required executables.")
def prerequisites(**kwargs):
    """Install prerequisites for RVC."""
    result = run_prerequisites_script(
        kwargs["pretraineds_hifigan"], kwargs["models"], kwargs["exe"]
    )
    click.echo(result)


@cli.command()
@click.option("--input-path", required=True, help="Path to the input audio file.")
def audio_analyzer(**kwargs):
    """Analyze an audio file and display information."""
    run_audio_analyzer_script(kwargs["input_path"])


@cli.command("export-model")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--output-path", required=True, type=click.Path())
def export_model(checkpoint, output_path):
    """Export v3 EMA weights, merging adaptation adapters for inference."""
    click.echo(run_v3_export_script(checkpoint, output_path))


@cli.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--pth-path", "--model-path", required=True, type=click.Path(exists=True))
@click.option("--vocoder-path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option(
    "--budget",
    "budgets",
    multiple=True,
    type=click.Choice(["0", "1", "2", "4", "8", "16", "32"]),
    default=["0", "2", "4", "8", "16"],
)
@click.option("--device", default="auto")
@click.option("--seed", type=int, default=1234)
@click.option("--limit", type=click.IntRange(min=0), default=0)
def evaluate(budgets, **kwargs):
    """Render recording-disjoint validation audio, budget curves and vocoder ceiling."""
    result = run_v3_evaluate_script(budgets=[int(b) for b in budgets], **kwargs)
    click.echo(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


@cli.command("download-encoder")
@click.option(
    "--output-dir", default="rvc/models/embedders/contentvec", type=click.Path()
)
@click.option("--repo", default="IAHispano/Applio")
@click.option("--revision", default="70ed563897504c756ec94067c12c902c4fd42025")
@click.option("--prefix", default="Resources/embedders/contentvec")
def download_encoder(output_dir, repo, revision, prefix):
    """Download pinned content encoder weights without remote Python code."""
    import re
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise click.ClickException("Pin --revision to a full 40-character commit SHA")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "pytorch_model.bin"):
        source = hf_hub_download(repo, f"{prefix}/{name}", revision=revision)
        shutil.copy2(source, destination / name)
    click.echo(str(destination))


@cli.command("verify-architecture")
@click.option(
    "--encoder-path",
    default="rvc/models/embedders/contentvec",
    type=click.Path(exists=True),
)
@click.option("--output-dir", default="logs/v3-verification")
@click.option("--device", default="auto")
@click.option("--small", is_flag=True)
def verify_architecture(encoder_path, output_dir, device, small):
    """Measure graph, streaming and GPU memory on a synthetic engineering fixture."""
    from rvc.lib.tools.v3_verify import verify

    click.echo(
        json.dumps(
            verify(encoder_path, output_dir, device, full_size=not small), indent=2
        )
    )


def _v3_runtime_options(func):
    for option in reversed(
        [
            click.option(
                "--pth-path",
                "--model-path",
                required=True,
                type=click.Path(exists=True),
            ),
            click.option("--vocoder-path", required=True, type=click.Path(exists=True)),
            click.option(
                "--encoder-path",
                default="rvc/models/embedders/contentvec",
                type=click.Path(exists=True),
            ),
            click.option("--pitch-path", type=click.Path(exists=True)),
            click.option("--device", default="auto"),
        ]
    ):
        func = option(func)
    return func


@cli.command("serve-v3")
@_v3_runtime_options
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=click.IntRange(1, 65535), default=7862)
def serve_v3(pth_path, vocoder_path, encoder_path, pitch_path, device, host, port):
    """Serve the V3 WebSocket API; the microphone UI lives in Applio's Realtime tab."""
    import uvicorn

    from rvc.infer.v3 import Converter
    from rvc.realtime.v3_transport import create_app

    converter = Converter(pth_path, vocoder_path, encoder_path, pitch_path, device)
    uvicorn.run(create_app(converter), host=host, port=port)


@cli.command("realtime-v3")
@_v3_runtime_options
@click.option("--input-device", type=int)
@click.option("--output-device", type=int)
@click.option("--sample-rate", type=click.IntRange(min=8000), default=48000)
@click.option("--block-size", type=click.IntRange(min=1), default=1024)
@click.option("--sid", type=click.IntRange(min=0), default=0)
@click.option("--pitch", type=click.FloatRange(-24, 24), default=0)
@click.option(
    "--refinement-steps",
    type=click.Choice(["0", "1", "2", "4", "8", "16", "32"]),
    default="4",
)
@click.option("--seed", type=click.IntRange(min=0), default=0)
def realtime_v3(
    pth_path,
    vocoder_path,
    encoder_path,
    pitch_path,
    device,
    sid,
    pitch,
    refinement_steps,
    seed,
    **kwargs,
):
    """Run v3 conversion on explicitly selected native audio devices."""
    from rvc.infer.v3 import Converter
    from rvc.realtime.v3_transport import NativeSession

    converter = Converter(pth_path, vocoder_path, encoder_path, pitch_path, device)
    NativeSession(
        converter,
        speaker=sid,
        semitones=pitch,
        steps=int(refinement_steps),
        seed=seed,
        **kwargs,
    ).run()


@cli.command("test-vctk")
@click.option("--model-name", required=True)
@click.option("--predictor-steps", type=click.IntRange(min=2), default=1000)
@click.option("--vocoder-steps", type=click.IntRange(min=2), default=2000)
@click.option("--flow-steps", type=click.IntRange(min=1), default=500)
@click.option("--shortcut-steps", type=click.IntRange(min=1), default=500)
@click.option("--limit", type=click.IntRange(min=1), default=8)
@click.option("--device", default="cuda")
def test_vctk(**kwargs):
    """Run staged learning and held-out audio tests on an already prepared VCTK subset."""
    from rvc.lib.tools.v3_vctk import run_experiment

    run_experiment(**kwargs)


def main():
    cli()


if __name__ == "__main__":
    main()
