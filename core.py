import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from functools import lru_cache

import click

from rvc.configs.neural import DEFAULT_BATCH_SIZE

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


def acoustic_project(model_name):
    """Keep training artifacts inside the existing logs/<model> structure."""
    from pathlib import Path

    if (
        not model_name
        or Path(model_name).name != model_name
        or model_name in {".", ".."}
    ):
        raise ValueError("Model name must be one directory name")
    return Path(current_script_directory) / "logs" / model_name


def run_acoustic_preprocess_script(
    model_name,
    dataset_path,
    validation_fraction=0.1,
    segment_seconds=4,
    seed=1234,
    speakers=(),
    recordings_per_speaker=0,
    progress=None,
    base_model=None,
    vocoder_path=None,
):
    """Preprocess V3 audio on CPU; feature models are loaded only by Extract."""
    from rvc.train.acoustic.data import atomic_json, preprocess_audio

    project = acoustic_project(model_name)
    mel = None
    if base_model or vocoder_path:
        from rvc.configs.architectures import inspect_model
        from rvc.configs.neural import MelConfig, require_contract

        metadata = inspect_model(base_model or vocoder_path)
        expected_kind = "acoustic" if base_model else "vocoder"
        if metadata.get("kind") != expected_kind:
            raise ValueError(f"Preprocessing requires a {expected_kind} package")
        mel = MelConfig(**metadata["mel"])
        if vocoder_path and base_model:
            wave = inspect_model(vocoder_path)
            if wave.get("kind") != "vocoder":
                raise ValueError("Choose a universal vocoder package")
            require_contract(metadata["mel"], wave["mel"], "preprocessing voice/vocoder mel")
    result = preprocess_audio(
        dataset_path,
        project / "data",
        validation_fraction,
        segment_seconds,
        seed,
        speaker_names=speakers,
        recordings_per_speaker=int(recordings_per_speaker),
        mel_config=mel,
        progress=progress,
    )
    atomic_json(
        project / "preparation.json",
        dict(
            model_name=model_name,
            dataset_path=str(dataset_path),
            validation_fraction=validation_fraction,
            segment_seconds=segment_seconds,
            seed=seed,
            speakers=list(speakers),
            recordings_per_speaker=int(recordings_per_speaker),
            base_model=str(base_model) if base_model else None,
            vocoder_path=str(vocoder_path) if vocoder_path else None,
        ),
    )
    return str(result)


def run_acoustic_extract_script(
    model_name,
    encoder_path="rvc/models/embedders/contentvec",
    pitch_extractor="swift",
    pitch_path=None,
    profile="bounded",
    device="auto",
    progress=None,
    base_model=None,
):
    """Extract cached content, pitch, energy and mel from preprocessed V3 audio."""
    from rvc.train.extract.features import FeatureExtractor
    from rvc.train.acoustic.data import atomic_json, extract_preprocessed
    from rvc.train.acoustic.trainer import resolve_device
    from rvc.configs.neural import MelConfig, require_contract

    project = acoustic_project(model_name)
    audio = project / "data/audio_manifest.json"
    if not audio.exists():
        raise ValueError(
            "Run Preprocess Dataset for this V3 model before Extract Features"
        )
    prepared = json.loads(audio.read_text(encoding="utf-8"))
    feature_options = dict(mel=MelConfig(**prepared["mel"]), pitch=pitch_extractor, profile=profile)
    if base_model:
        from rvc.configs.architectures import inspect_model
        from dataclasses import asdict

        metadata = inspect_model(base_model)
        if metadata.get("kind") != "acoustic":
            raise ValueError(
                "Choose a V3 pretrained voice model before extracting features"
            )
        features = metadata["features"]
        require_contract(metadata["mel"], prepared["mel"], "pretrained preprocessing mel")
        feature_options = dict(
            mel=MelConfig(**metadata["mel"]),
            layer=features["layer"],
            pitch=features["pitch_method"],
            threshold=features["pitch_threshold"],
            profile=features["profile"],
            context_seconds=features["context_seconds"],
            lookahead_seconds=features["lookahead_seconds"],
            packet_seconds=features["packet_seconds"],
        )
    extractor = FeatureExtractor(
        encoder_path,
        device=resolve_device(device),
        pitch_path=pitch_path or None,
        **feature_options,
    )
    if base_model:
        require_contract(
            asdict(extractor.config), features, "pretrained feature extraction"
        )
    result = extract_preprocessed(audio, extractor, progress=progress)
    atomic_json(
        project / "extraction.json",
        dict(
            model_name=model_name,
            encoder_path=str(encoder_path),
            pitch_extractor=extractor.config.pitch_method,
            pitch_path=str(pitch_path) if pitch_path else None,
            profile=extractor.config.profile,
            device=device,
            base_model=str(base_model) if base_model else None,
        ),
    )
    return str(result)


def run_acoustic_prepare_script(
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
    from rvc.train.extract.features import FeatureExtractor
    from rvc.train.acoustic.data import atomic_json, prepare
    from rvc.train.acoustic.trainer import resolve_device

    project = acoustic_project(model_name)
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


def run_acoustic_train_script(
    model_name,
    stage="predictor",
    manifest=None,
    output_dir=None,
    base_model=None,
    config=None,
    **kwargs,
):
    from pathlib import Path

    from rvc.configs.neural import AcousticConfig, VocoderConfig
    from rvc.train.acoustic.trainer import train

    project = acoustic_project(model_name)
    kind = "vocoder" if stage == "vocoder" else "acoustic"
    if config:
        constructor = VocoderConfig if kind == "vocoder" else AcousticConfig
        config = constructor(**json.loads(Path(config).read_text(encoding="utf-8")))
    dataset = Path(manifest) if manifest else project / "data/manifest.json"
    if not dataset.exists():
        raise ValueError("Run Preprocess Dataset and Extract Features before training")
    yield from train(
        dataset,
        output_dir or project / "checkpoints" / stage,
        kind=kind,
        phase="predictor" if kind == "vocoder" else stage,
        pretrained=base_model or None,
        config=config,
        **kwargs,
    )


def run_acoustic_train_all_script(
    model_name,
    refine=False,
    steps=10000,
    manifest=None,
    output_dir=None,
    base_model=None,
    resume=None,
    config=None,
    **kwargs,
):
    """Train and export a usable voice in one action with the shared recipe.

    Predictor and vocoder are required. Flow and shortcut are optional and run
    in dependency order. Unlike individual-stage training's additional budget,
    steps here is the target per part: restarting the pipeline resumes last.pt
    and skips completed parts. Stop never starts the next part or exports an
    unfinished pipeline. Stage checkpoints remain available for advanced use.
    """
    import os
    from pathlib import Path

    import psutil

    from rvc.train.process.checkpoints import load_payload

    if base_model or resume or config:
        raise ValueError(
            "Use an individual-stage mode for custom base weights, resume paths or architecture overrides. Complete model resumes its own saved checkpoints automatically."
        )
    if int(steps) < 1:
        raise ValueError("Training duration must be positive")
    if kwargs.get("waveform_weight") or kwargs.get("waveform_adversarial_weight"):
        raise ValueError(
            "Use an individual predictor/adaptation stage for waveform supervision"
        )
    project = acoustic_project(model_name)
    dataset = Path(manifest) if manifest else project / "data/manifest.json"
    if not dataset.exists():
        raise ValueError("Run Preprocess Dataset and Extract Features before training")
    dataset_id = json.loads(dataset.read_text(encoding="utf-8"))["dataset_id"]
    campaign_path = project / "campaign_status.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(campaign["pid"])
            live = (
                process.is_running() and process.create_time() <= campaign["updated_at"]
            )
        except (psutil.Error, KeyError):
            live = False
        if live and process.pid != os.getpid():
            raise ValueError(
                "This model already has a background training job. Follow it in Live training progress or use a different model name."
            )
    root = Path(output_dir) if output_dir else project / "checkpoints"
    stages = ["predictor", "vocoder"] + (["flow", "shortcut"] if refine else [])
    kwargs.setdefault("checkpoint_every", 1000)
    stop = kwargs.get("stop_requested")
    acoustic_stage = "predictor"
    for part, stage in enumerate(stages, 1):
        if stop and stop():
            return
        checkpoint = root / stage / "last.pt"
        initial = 0
        if checkpoint.exists():
            payload = load_payload(checkpoint)
            if payload.get("dataset_id") != dataset_id or payload.get("phase") != stage:
                raise ValueError(
                    f"Saved {stage} progress belongs to another dataset or stage. Use a new model name."
                )
            initial = int(payload["step"])
            del payload
        base = None
        if stage in {"flow", "shortcut"}:
            base = root / acoustic_stage / "best.pt"
        if initial >= int(steps):
            yield {
                "status": "skipped",
                "phase": stage,
                "step": initial,
                "part": part,
                "parts": len(stages),
            }
        else:
            stage_options = dict(kwargs)
            if stage != "predictor":
                stage_options.pop("mel_detail_weight", None)
                stage_options.pop("mel_adversarial_weight", None)
                stage_options.pop("pitch_guidance", None)
            for update in run_acoustic_train_script(
                model_name,
                stage,
                manifest=str(dataset),
                output_dir=str(root / stage),
                base_model=str(base) if base and not checkpoint.exists() else None,
                resume=str(checkpoint) if checkpoint.exists() else None,
                steps=int(steps) - initial,
                **stage_options,
            ):
                yield dict(update, phase=stage, part=part, parts=len(stages))
                if update.get("stopped"):
                    return
        if stage != "vocoder":
            acoustic_stage = stage
    if stop and stop():
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return
    acoustic = project / f"{model_name}_acoustic.pth"
    vocoder = project / f"{model_name}_vocoder.pth"
    run_acoustic_export_script(str(root / acoustic_stage / "best.pt"), str(acoustic))
    run_acoustic_export_script(str(root / "vocoder" / "best.pt"), str(vocoder))
    yield {"status": "exported", "acoustic": str(acoustic), "vocoder": str(vocoder)}


def run_acoustic_finetune_script(
    model_name, base_model, vocoder_path, steps=10000, **kwargs
):
    """Adapt only the voice model, retaining the frozen shared vocoder.

    Resume uses this project's adapter checkpoint; export merges the adapters
    into a standalone acoustic package. The selected vocoder is referenced,
    never copied, optimized or overwritten by voice fine-tuning.
    """
    from pathlib import Path
    from rvc.train.process.checkpoints import load_payload
    from rvc.configs.neural import require_contract
    from rvc.train.acoustic.data import atomic_json

    if not base_model or not vocoder_path:
        raise ValueError(
            "Choose a pretrained voice model and universal vocoder in Model Settings first"
        )
    if int(steps) < 1:
        raise ValueError("Training duration must be positive")
    base, vocoder = load_payload(base_model), load_payload(vocoder_path)
    if base["kind"] != "acoustic" or vocoder["kind"] != "vocoder":
        raise ValueError("Choose a V3 voice model and its universal vocoder")
    if base.get("adapters"):
        raise ValueError(
            "Select an exported voice model with merged adapters as the pretrained base"
        )
    require_contract(base["mel"], vocoder["mel"], "pretrained voice/vocoder")
    project = acoustic_project(model_name)
    destination = project / f"{model_name}_acoustic.pth"
    if (
        Path(base_model).resolve() == destination.resolve()
        or Path(vocoder_path).resolve() == destination.resolve()
    ):
        raise ValueError(
            "Use a different model name to preserve the pretrained weights"
        )
    manifest = project / "data/manifest.json"
    campaign = project / "campaign_status.json"
    if campaign.exists():
        import psutil

        state = json.loads(campaign.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(state["pid"])
            running = (
                process.is_running() and process.create_time() <= state["updated_at"]
            )
        except (psutil.Error, KeyError):
            running = False
        if running:
            raise ValueError(
                "This project has an active background training job. Use a new model name for fine-tuning"
            )
    if not manifest.exists():
        raise ValueError(
            "Run Preprocess Dataset and Extract Features before fine-tuning"
        )
    from rvc.train.acoustic.data import AcousticDataset

    dataset = AcousticDataset(manifest)
    require_contract(
        base["features"],
        dataset.manifest["contract"]["features"],
        "pretrained features; extract again using the selected pretrained model",
    )
    require_contract(base["mel"], dataset.manifest["contract"]["mel"], "pretrained mel")
    checkpoint = project / "checkpoints/adapt/last.pt"
    from rvc.train.extract.features import file_hash

    recipe = dict(
        base_model=str(Path(base_model).resolve()),
        vocoder=str(Path(vocoder_path).resolve()),
        dataset_id=dataset.manifest["dataset_id"],
        base_hash=file_hash(Path(base_model)),
        vocoder_hash=file_hash(Path(vocoder_path)),
    )
    settings = project / "finetuning.json"
    if checkpoint.exists():
        previous = (
            json.loads(settings.read_text(encoding="utf-8"))
            if settings.exists()
            else {}
        )
        # The vocoder is independent of adaptation. Another compatible vocoder
        # may be selected without invalidating the acoustic optimizer state.
        if any(previous.get(key) != recipe[key] for key in ("base_hash", "dataset_id")):
            raise ValueError(
                "Saved fine-tuning uses another pretrained model or dataset. Use a new model name"
            )
        initial = int(load_payload(checkpoint)["step"])
    else:
        initial = 0
    # Do not overwrite a selected pretrained checkpoint through a reused name.
    if Path(base_model).resolve().is_relative_to(
        (project / "checkpoints").resolve()
    ) or Path(vocoder_path).resolve().is_relative_to(
        (project / "checkpoints/adapt").resolve()
    ):
        raise ValueError("Use a new model name for the voice you want to fine-tune")
    atomic_json(settings, recipe)
    del base, vocoder, dataset
    kwargs["adaptation"] = "lora"
    if initial < int(steps):
        for update in run_acoustic_train_script(
            model_name,
            "adapt",
            base_model=None if checkpoint.exists() else base_model,
            resume=str(checkpoint) if checkpoint.exists() else None,
            steps=int(steps) - initial,
            **kwargs,
        ):
            yield update
            if update.get("stopped"):
                return
    if kwargs.get("stop_requested") and kwargs["stop_requested"]():
        return
    run_acoustic_export_script(
        str(project / "checkpoints/adapt/best.pt"), str(destination)
    )
    yield dict(status="exported", acoustic=str(destination), vocoder=str(vocoder_path))


def run_acoustic_infer_script(
    input_path,
    output_path,
    pth_path,
    vocoder_path,
    encoder_path="rvc/models/embedders/contentvec",
    pitch_path=None,
    sid=0,
    pitch=0,
    refinement_steps=0,
    seed=0,
    device="auto",
    ordinary_flow=False,
    batch=False,
):
    from rvc.infer.acoustic import Converter

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
    return converter.convert_directory(input_path, output_path, **options)


def run_acoustic_export_script(checkpoint, output_path):
    from rvc.train.process.checkpoints import export_checkpoint

    return str(export_checkpoint(checkpoint, output_path))


def run_acoustic_evaluate_script(
    manifest, pth_path, vocoder_path, output_dir, **kwargs
):
    from rvc.train.process.evaluation import evaluate

    return evaluate(manifest, pth_path, vocoder_path, output_dir, **kwargs)


def _architecture_options(func):
    return click.option(
        "--architecture",
        type=click.Choice(["classic", "v3"]),
        default="classic",
        show_default=True,
    )(func)


def _acoustic_extract_options(func):
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
        ]
    ):
        func = option(func)
    return func


def _acoustic_prepare_options(func):
    for option in reversed(
        [
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


def _acoustic_inference_options(func):
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
                default="0",
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
            return run_acoustic_infer_script(
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
@_acoustic_inference_options
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
@_acoustic_inference_options
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
@_acoustic_prepare_options
@click.option("--base-model", type=click.Path(exists=True), help="Prepare audio with the pretrained acoustic model's mel contract for fine-tuning.")
@click.option("--vocoder-path", type=click.Path(exists=True), help="Prepare scratch acoustic targets with a frozen vocoder's exact mel contract.")
@click.option(
    "--json-logs",
    is_flag=True,
    help="Emit V3 machine-readable progress instead of terminal bars.",
)
def preprocess(**kwargs):
    """Preprocess a dataset for training."""
    architecture = kwargs.pop("architecture")
    json_logs = kwargs.pop("json_logs")
    names = (
        "validation_fraction",
        "segment_seconds",
        "seed",
        "speakers",
        "recordings_per_speaker",
        "base_model",
        "vocoder_path",
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
            from rvc.train.process.console import ConsoleProgress

            with ConsoleProgress(kwargs["model_name"]) as console:
                result = run_acoustic_preprocess_script(
                    kwargs["model_name"],
                    kwargs["dataset_path"],
                    progress=(lambda update: click.echo(json.dumps(update)))
                    if json_logs
                    else console.preparation,
                    **options,
                )
            click.echo(result)
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
@_acoustic_extract_options
@click.option("--base-model", type=click.Path(exists=True), help="Use the pretrained acoustic model's exact feature contract for fine-tuning.")
@click.option(
    "--json-logs",
    is_flag=True,
    help="Emit V3 machine-readable progress instead of terminal bars.",
)
def extract(**kwargs):
    """Extract features from a preprocessed dataset."""
    architecture = kwargs.pop("architecture")
    json_logs = kwargs.pop("json_logs")
    names = ("encoder_path", "pitch_extractor", "pitch_path", "profile", "device", "base_model")
    options = {name: kwargs.pop(name) for name in names}
    if architecture == "v3":
        _reject_explicit_options(
            set(kwargs) - {"model_name", "sample_rate"}, "v3 extraction"
        )
        if (
            click.get_current_context().get_parameter_source("sample_rate")
            == click.core.ParameterSource.COMMANDLINE
            and kwargs["sample_rate"] != "44100"
        ):
            raise click.ClickException("Applio v3 uses 44100 Hz")
        try:
            from rvc.train.process.console import ConsoleProgress

            with ConsoleProgress(kwargs["model_name"]) as console:
                if not json_logs:
                    click.echo("Loading content encoder and pitch model...")
                result = run_acoustic_extract_script(
                    kwargs["model_name"],
                    progress=(lambda update: click.echo(json.dumps(update)))
                    if json_logs
                    else console.preparation,
                    **options,
                )
            click.echo(result)
        except (ValueError, OSError) as error:
            raise click.ClickException(str(error)) from error
        return
    _reject_explicit_options(names, "classic extraction")
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
    flag_value=True,
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
    type=click.Choice(
        ["all", "all_refiners", "predictor", "flow", "shortcut", "adapt", "vocoder"]
    ),
    default="all",
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
@click.option("--checkpoint-every", type=click.IntRange(min=1), default=1000)
@click.option("--adapter-rank", type=click.IntRange(min=1), default=8)
@click.option("--adaptation", type=click.Choice(["lora", "full"]), default="lora")
@click.option("--accumulation-steps", type=click.IntRange(min=1), default=1)
@click.option(
    "--mel-detail-weight",
    type=click.FloatRange(min=0, max=1),
    default=0.0,
    help="Experimental acoustic spectral-detail objective; zero preserves legacy training.",
)
@click.option("--spectral-vocoder", type=click.Path(exists=True))
@click.option(
    "--waveform-adversarial-weight",
    type=click.FloatRange(min=0, max=1),
    default=0.0,
    help="Research-only critics of audio rendered through the frozen spectral vocoder; individual predictor/adaptation stages only.",
)
@click.option(
    "--pitch-guidance/--no-pitch-guidance",
    default=None,
    help="Research-only frame-local harmonic pitch conditioning; recorded in the acoustic model configuration.",
)
@click.option(
    "--mel-adversarial-weight",
    type=click.FloatRange(min=0, max=1),
    default=0.0,
    help="Research-only mel texture critics for predictor/adaptation training; zero preserves existing training.",
)
@click.option(
    "--waveform-weight",
    type=click.FloatRange(min=0, max=1),
    default=0.0,
    help="Experimental frozen-vocoder spectral supervision for individual predictor/adaptation stages.",
)
@click.option(
    "--json-logs",
    is_flag=True,
    help="Emit V3 machine-readable progress instead of terminal bars.",
)
def train(**kwargs):
    """Train classic RVC or a selected Applio v3 stage."""
    architecture = kwargs.pop("architecture")
    json_logs = kwargs.pop("json_logs")
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
        "mel_detail_weight",
        "spectral_vocoder",
        "waveform_weight",
        "mel_adversarial_weight",
        "pitch_guidance",
        "waveform_adversarial_weight",
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
            pipeline = options["stage"] in {"all", "all_refiners"}
            if pipeline and (
                options["waveform_weight"] or options["waveform_adversarial_weight"]
            ):
                raise ValueError(
                    "Use an individual predictor/adaptation stage for waveform supervision"
                )
            trainer = (
                run_acoustic_train_all_script if pipeline else run_acoustic_train_script
            )
            if pipeline:
                options["refine"] = options.pop("stage") == "all_refiners"
            from rvc.train.process.console import ConsoleProgress

            with ConsoleProgress(kwargs["model_name"]) as console:
                for update in trainer(
                    kwargs["model_name"], batch_size=kwargs["batch_size"], **options
                ):
                    if json_logs:
                        click.echo(json.dumps(update), color=False)
                    else:
                        console.training(
                            update, options["steps"], additional=not pipeline
                        )
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


@cli.command("train-corpus")
@click.option("--model-name", required=True)
@click.option(
    "--dataset-path", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "--vocoder-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--architecture-from",
    type=click.Path(exists=True, dir_okay=False),
    help="Reuse architecture/frontend semantics only; initialize new acoustic weights from scratch.",
)
@click.option(
    "--encoder-path",
    default="rvc/models/embedders/contentvec",
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--batch-size", default=DEFAULT_BATCH_SIZE, type=click.IntRange(min=1))
@click.option("--accumulation-steps", default=1, type=click.IntRange(min=1))
@click.option("--predictor-steps", default=60000, type=click.IntRange(min=1))
@click.option("--flow-steps", default=None, type=click.IntRange(min=0), help="Residual flow updates; defaults to 15000 for residual models and zero for joint shallow flow.")
@click.option("--shortcut-steps", default=None, type=click.IntRange(min=0), help="Residual shortcut updates; defaults to 15000 for residual models and zero for joint shallow flow.")
@click.option("--crop-frames", default=None, type=click.IntRange(min=1), help="Training context; defaults to 128 residual frames or 256 joint shallow-flow frames.")
@click.option("--device", default="auto")
@click.option(
    "--precision", default="auto", type=click.Choice(["auto", "bf16", "fp16", "fp32"])
)
@click.option("--checkpoint-every", default=1000, type=click.IntRange(min=1))
@click.option("--compact-cache", is_flag=True, help="Store derived audio as scaled PCM24 FLAC and content as FP16; originals remain unchanged.")
@click.option("--compact-cache-audit", type=click.Path(exists=True, dir_okay=False), help="Passed contract-matched corpus storage qualification for a measured disk estimate.")
@click.option("--sampling-mode", default="segments", type=click.Choice(["segments", "domain-speaker"]), help="Segment-uniform sampling or duration-tempered domains with balanced voices.")
@click.option("--validation-limit", default=0, type=click.IntRange(min=0), help="Fixed voice-covered checkpoint validation panel; zero validates every reserved segment.")
@click.option("--validation-manifest", type=click.Path(exists=True, dir_okay=False), help="Preserve prior held-out recordings and their corpus split groups in the new validation split.")
@click.option("--seed", default=1234, type=int)
@click.option(
    "--check-only",
    is_flag=True,
    help="Check inputs, disk space and resume compatibility without training.",
)
def train_corpus(**kwargs):
    """Preprocess, extract and train every corpus speaker with a frozen vocoder."""
    from rvc.train.acoustic.campaign import run_corpus

    try:
        run_corpus(**kwargs)
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error


@cli.command("prepare-pitch-views")
@click.option("--manifest", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-manifest", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--per-speaker", default=16, type=click.IntRange(min=4), show_default=True
)
@click.option("--seed", default=5678, type=int, show_default=True)
def prepare_pitch_views(manifest, output_manifest, per_speaker, seed):
    """Prepare optional V3 training pitch counterexamples; keep natural validation."""
    from rvc.train.acoustic.augmentation import prepare_pitch_views as prepare

    try:
        result = prepare(
            manifest,
            output_manifest,
            per_speaker,
            seed,
            progress=lambda item: (
                click.echo(f"Speaker {item['speaker']}: {item['views']} views prepared")
                if item["views"] % 16 == 0
                else None
            ),
        )
        click.echo(result)
    except (ValueError, ImportError, OSError) as error:
        raise click.ClickException(str(error)) from error


@cli.command("export-model")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--output-path", required=True, type=click.Path())
def export_model(checkpoint, output_path):
    """Export v3 EMA weights, merging adaptation adapters for inference."""
    click.echo(run_acoustic_export_script(checkpoint, output_path))


@cli.command("import-vocoder")
@click.option("--output-path", required=True, type=click.Path())
@click.option("--checkpoint", type=click.Path(exists=True), default=None)
@click.option("--config", "configuration", type=click.Path(exists=True), default=None)
@click.option("--backend", type=click.Choice(["bigvgan-v2", "nsf-hifigan"]), default="bigvgan-v2")
def import_vocoder(output_path, checkpoint, configuration, backend):
    """Package compatible frozen vocoder weights for file inference/fine-tuning.

    Without local checkpoint/config paths, download the pinned official
    44.1-kHz BigVGAN model. NSF-HiFiGAN requires a local converted OpenVPI export
    with its own mel contract and weight license. Both backends are file-only.
    """
    from rvc.train.process.pretrained import import_bigvgan, import_nsf_hifigan

    try:
        if backend == "nsf-hifigan":
            if not checkpoint or configuration:
                raise ValueError("NSF-HiFiGAN requires --checkpoint and uses its serialized configuration")
            click.echo(import_nsf_hifigan(output_path, checkpoint))
        else:
            click.echo(import_bigvgan(output_path, checkpoint, configuration))
    except (ValueError, OSError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error


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
@click.option("--oracle-start", is_flag=True, help="Research diagnostic: shallow flow initialized with reference mel unavailable during conversion.")
def evaluate(budgets, **kwargs):
    """Render recording-disjoint validation audio, budget curves and vocoder ceiling."""
    result = run_acoustic_evaluate_script(budgets=[int(b) for b in budgets], **kwargs)
    click.echo(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


@cli.command("prepare-evaluation")
@click.option("--audio-manifest", required=True, type=click.Path(exists=True))
@click.option("--output-path", required=True, type=click.Path())
@click.option("--segments-per-speaker", type=click.IntRange(min=1), default=2)
def prepare_evaluation(audio_manifest, output_path, segments_per_speaker):
    """Save a fixed held-out listening and conversion protocol without loading models."""
    from rvc.train.process.reports import prepare_plan

    try:
        plan = prepare_plan(audio_manifest, output_path, segments_per_speaker)
        click.echo(
            f"Prepared {len(plan['held_out'])} segments across {plan['speakers']} speakers. Use evaluate --limit {plan['evaluator_limit']} --seed {plan['seed']} after training."
        )
    except (ValueError, OSError, KeyError) as error:
        raise click.ClickException(str(error)) from error


@cli.command("summarize-evaluation")
@click.option("--report-path", required=True, type=click.Path(exists=True))
@click.option("--output-path", required=True, type=click.Path())
def summarize_evaluation(report_path, output_path):
    """Compare refinement budgets on identical cases with speaker-level uncertainty."""
    from rvc.train.process.reports import summarize_report

    try:
        result = summarize_report(report_path, output_path)
        click.echo(
            f"Saved paired summary. Lowest waveform mel error: budget {result['lowest_waveform_mel_error_budget']}. Listening and identity checks are still required."
        )
    except (ValueError, OSError, KeyError) as error:
        raise click.ClickException(str(error)) from error


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
    from rvc.lib.tools.verify_architecture import verify

    click.echo(
        json.dumps(
            verify(encoder_path, output_dir, device, full_size=not small), indent=2
        )
    )


def _acoustic_runtime_options(func):
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
@_acoustic_runtime_options
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=click.IntRange(1, 65535), default=7862)
def serve_acoustic(
    pth_path, vocoder_path, encoder_path, pitch_path, device, host, port
):
    """Serve the V3 WebSocket API; the microphone UI lives in Applio's Realtime tab."""
    import uvicorn

    from rvc.infer.acoustic import Converter
    from rvc.realtime.transport import create_app

    converter = Converter(pth_path, vocoder_path, encoder_path, pitch_path, device)
    uvicorn.run(create_app(converter), host=host, port=port)


@cli.command("realtime-v3")
@_acoustic_runtime_options
@click.option("--input-device", type=int)
@click.option("--output-device", type=int)
@click.option("--sample-rate", type=click.IntRange(min=8000), default=48000)
@click.option("--block-size", type=click.IntRange(min=1), default=1024)
@click.option("--sid", type=click.IntRange(min=0), default=0)
@click.option("--pitch", type=click.FloatRange(-24, 24), default=0)
@click.option(
    "--refinement-steps",
    type=click.Choice(["0", "1", "2", "4", "8", "16", "32"]),
    default="0",
)
@click.option("--seed", type=click.IntRange(min=0), default=0)
def realtime_acoustic(
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
    from rvc.infer.acoustic import Converter
    from rvc.realtime.transport import NativeSession

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
    from rvc.lib.tools.corpus_experiment import run_experiment

    run_experiment(**kwargs)


def main():
    cli()


if __name__ == "__main__":
    main()
