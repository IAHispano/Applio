import gradio as gr
import sounddevice as sd
import os
import sys
import time
import json
import regex as re
import shutil
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.callbacks import AudioCallbacks
from rvc.realtime.audio import list_audio_device
from rvc.realtime.core import AUDIO_SAMPLE_RATE

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

model_root = os.path.join(now_dir, "logs")
custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)


def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/").lower()


MODEL_FOLDER = re.compile(r"^(?:model.{0,4}|mdl(?:s)?|weight.{0,4}|zip(?:s)?)$")
INDEX_FOLDER = re.compile(r"^(?:ind.{0,4}|idx(?:s)?)$")


def is_mdl_alias(name: str) -> bool:
    return bool(MODEL_FOLDER.match(name))


def is_idx_alias(name: str) -> bool:
    return bool(INDEX_FOLDER.match(name))


def alias_score(path: str, want_model: bool) -> int:
    parts = normalize_path(os.path.dirname(path)).split("/")
    has_mdl = any(is_mdl_alias(p) for p in parts)
    has_idx = any(is_idx_alias(p) for p in parts)
    if want_model:
        return 2 if has_mdl else (1 if has_idx else 0)
    else:
        return 2 if has_idx else (1 if has_mdl else 0)


def get_files(type="model"):
    assert type in ("model", "index"), "Invalid type for get_files (models or index)"
    is_model = type == "model"
    exts = (".pth", ".onnx") if is_model else (".index",)
    exclude_prefixes = ("G_", "D_") if is_model else ()
    exclude_substr = None if is_model else "trained"

    best = {}
    order = 0

    for root, _, files in os.walk(model_root_relative, followlinks=True):
        for file in files:
            if not file.endswith(exts):
                continue
            if any(file.startswith(p) for p in exclude_prefixes):
                continue
            if exclude_substr and exclude_substr in file:
                continue

            full = os.path.join(root, file)
            real = os.path.realpath(full)
            score = alias_score(full, is_model)

            prev = best.get(real)
            if (
                prev is None
            ):  # Prefer higher score; if equal score, use first encountered
                best[real] = (score, order, full)
            else:
                prev_score, prev_order, _ = prev
                if score > prev_score:
                    best[real] = (score, prev_order, full)
            order += 1

    return [t[2] for t in sorted(best.values(), key=lambda x: x[1])]


def folders_same(
    a: str, b: str
) -> bool:  # Used to "pair" index and model folders based on path names
    """
    True if:
      1) The two normalized paths are totally identical..OR
      2) One lives under a MODEL_FOLDER and the other lives
         under an INDEX_FOLDER, at the same relative subpath
         i.e.  logs/models/miku  and  logs/index/miku  =  "SAME FOLDER"
    """
    a = normalize_path(a)
    b = normalize_path(b)
    if a == b:
        return True

    def split_after_alias(p):
        parts = p.split("/")
        for i, part in enumerate(parts):
            if is_mdl_alias(part) or is_idx_alias(part):
                base = part
                rel = "/".join(parts[i + 1 :])
                return base, rel
        return None, None

    base_a, rel_a = split_after_alias(a)
    base_b, rel_b = split_after_alias(b)

    if rel_a is None or rel_b is None:
        return False

    if rel_a == rel_b and (
        (is_mdl_alias(base_a) and is_idx_alias(base_b))
        or (is_idx_alias(base_a) and is_mdl_alias(base_b))
    ):
        return True
    return False


def match_index(model_file_value):
    if not model_file_value:
        return ""

    # Derive the information about the model's name and path for index matching
    model_folder = normalize_path(os.path.dirname(model_file_value))
    model_name = os.path.basename(model_file_value)
    base_name = os.path.splitext(model_name)[0]
    common = re.sub(r"[_\-\.\+](?:e|s|v|V)\d.*$", "", base_name)
    prefix_match = re.match(r"^(.*?)[_\-\.\+]", base_name)
    prefix = prefix_match.group(1) if prefix_match else None

    same_count = 0
    last_same = None
    same_substr = None
    same_prefixed = None
    external_exact = None
    external_substr = None
    external_pref = None

    for idx in get_files("index"):
        idx_folder = os.path.dirname(idx)
        idx_folder_n = normalize_path(idx_folder)
        idx_name = os.path.basename(idx)
        idx_base = os.path.splitext(idx_name)[0]

        in_same = folders_same(model_folder, idx_folder_n)
        if in_same:
            same_count += 1
            last_same = idx

            # 1) EXACT match to loaded model name and folders_same = True
            if idx_base == base_name:
                return idx

            # 2) Substring match to model name and folders_same
            if common in idx_base and same_substr is None:
                same_substr = idx

            # 3) Prefix match to model name and folders_same
            if prefix and idx_base.startswith(prefix) and same_prefixed is None:
                same_prefixed = idx

        # If it's NOT in a paired folder (folders_same = False) we look elseware:
        else:
            # 4) EXACT match to model name in external directory
            if idx_base == base_name and external_exact is None:
                external_exact = idx

            # 5) Substring match to model name in ED
            if common in idx_base and external_substr is None:
                external_substr = idx

            # 6) Prefix match to model name in ED
            if prefix and idx_base.startswith(prefix) and external_pref is None:
                external_pref = idx

    # Fallback: If there is exactly one index file in the same (or paired) folder,
    # we should assume that's the intended index file even if the name doesnt match
    if same_count == 1:
        return last_same

    # Then by remaining priority queue:
    if same_substr:
        return same_substr
    if same_prefixed:
        return same_prefixed
    if external_exact:
        return external_exact
    if external_substr:
        return external_substr
    if external_pref:
        return external_pref

    return ""


def extract_model_and_epoch(path):
    base_name = os.path.basename(path)
    match = re.match(r"(.+?)_(\d+)e_", base_name)
    if match:
        model, epoch = match.groups()
        return model, int(epoch)
    return "", 0


def get_speakers_id(model):
    if model:
        try:
            model_data = torch.load(
                os.path.join(now_dir, model), map_location="cpu", weights_only=True
            )
            speakers_id = model_data.get("speakers_id")
            if speakers_id:
                return list(range(speakers_id))
            else:
                return [0]
        except Exception as e:
            return [0]
    else:
        return [0]


def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)

    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))
    if config_file:
        shutil.copy(
            config_file, os.path.join(target_folder, os.path.basename(config_file))
        )

    return f"Files moved to folder {target_folder}"


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders


names = get_files("model")
default_weight = names[0] if names else None

client_mode = "--client" in sys.argv

PASS_THROUGH = False
interactive_true = gr.update(interactive=True)
interactive_false = gr.update(interactive=False)
running, callbacks, audio_manager = False, None, None
callbacks_kwargs = {}

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")


def save_realtime_settings(
    value, key
):
    """Save realtime settings to config.json"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        if "realtime" not in config:
            config["realtime"] = {}

        # Only save non-None values, preserve existing values for None inputs
        if value is not None:
            config["realtime"][key] = value or ""

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving realtime settings: {e}")


def load_realtime_settings():
    """Load realtime settings from config.json"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                realtime_config = config.get("realtime", {})
                return {
                    "input_device": realtime_config.get("input_device", ""),
                    "output_device": realtime_config.get("output_device", ""),
                    "monitor_device": realtime_config.get("monitor_device", ""),
                    "client_input_device": realtime_config.get("client_input_device", ""),
                    "client_output_device": realtime_config.get("client_output_device", ""),
                    "client_monitor_device": realtime_config.get("client_monitor_device", ""),
                    "model_file": realtime_config.get("model_file", ""),
                    "index_file": realtime_config.get("index_file", ""),
                }
    except Exception as e:
        print(f"Error loading realtime settings: {e}")

    return {
        "input_device": "",
        "output_device": "",
        "monitor_device": "",
        "client_input_device": "",
        "client_output_device": "",
        "client_monitor_device": "",
        "model_file": "",
        "index_file": "",
    }


def get_safe_dropdown_value(saved_value, choices, fallback_value=None):
    """Safely get a dropdown value, ensuring it exists in choices"""
    if saved_value and saved_value in choices:
        return saved_value
    elif fallback_value and fallback_value in choices:
        return fallback_value
    elif choices:
        return choices[0]
    else:
        return None


def get_safe_index_value(saved_value, choices, fallback_value=None):
    """Safely get an index file value, handling file path matching"""
    # Handle empty string, None, or whitespace-only values
    if not saved_value or (isinstance(saved_value, str) and not saved_value.strip()):
        if fallback_value and fallback_value in choices:
            return fallback_value
        elif choices:
            return choices[0]
        else:
            return None

    # Check exact match first
    if saved_value in choices:
        return saved_value

    # Check if saved value is a filename that matches any choice
    saved_filename = os.path.basename(saved_value)
    for choice in choices:
        if os.path.basename(choice) == saved_filename:
            return choice

    # Fallback to default or first choice
    if fallback_value and fallback_value in choices:
        return fallback_value
    elif choices:
        return choices[0]
    else:
        return None


def start_realtime(
    input_audio_device: str,
    input_audio_gain: int,
    input_asio_channels: int,
    output_audio_device: str,
    output_audio_gain: int,
    output_asio_channels: int,
    monitor_output_device: str,
    monitor_audio_gain: int,
    monitor_asio_channels: int,
    use_monitor_device: bool,
    exclusive_mode: bool,
    vad_enabled: bool,
    chunk_size: float,
    cross_fade_overlap_size: float,
    extra_convert_size: float,
    silent_threshold: int,
    pitch: int,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_method: str,
    pth_path: str,
    index_path: str,
    sid: int,
    f0_autotune: bool,
    f0_autotune_strength: float,
    proposed_pitch: bool,
    proposed_pitch_threshold: float,
    embedder_model: str,
    embedder_model_custom: str = None,
    clean_audio: bool = False,
    clean_strength: float = 0.5,
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
):
    global running, callbacks, audio_manager, callbacks_kwargs
    running = True

    if not input_audio_device or not output_audio_device:
        yield (
            "Please select valid input/output devices!",
            interactive_true,
            interactive_false,
        )
        return
    if use_monitor_device and not monitor_output_device:
        yield (
            "Please select a valid monitor device!",
            interactive_true,
            interactive_false,
        )
        return
    if not pth_path:
        yield (
            "Model path not provided. Aborting conversion.",
            interactive_true,
            interactive_false,
        )
        return

    yield "Starting Realtime...", interactive_false, interactive_true

    read_chunk_size = int(chunk_size * AUDIO_SAMPLE_RATE / 1000 / 128)

    sid = int(sid) if sid is not None else 0

    input_audio_gain /= 100.0
    output_audio_gain /= 100.0
    monitor_audio_gain /= 100.0

    try:
        input_devices, output_devices = get_audio_devices_formatted()
        input_device_id = input_devices[input_audio_device]
        output_device_id = output_devices[output_audio_device]
        output_monitor_id = (
            output_devices[monitor_output_device] if use_monitor_device else None
        )
    except (ValueError, IndexError):
        yield "Incorrectly formatted audio device. Stopping.", interactive_true, interactive_false
        return
    
    callbacks_kwargs = {
        "pass_through": PASS_THROUGH,
        "read_chunk_size": read_chunk_size,
        "cross_fade_overlap_size": cross_fade_overlap_size,
        "extra_convert_size": extra_convert_size,
        "model_path": pth_path,
        "index_path": str(index_path),
        "f0_method": f0_method,
        "embedder_model": embedder_model,
        "embedder_model_custom": embedder_model_custom,
        "silent_threshold": silent_threshold,
        "f0_up_key": pitch,
        "index_rate": index_rate,
        "protect": protect,
        "volume_envelope": volume_envelope,
        "f0_autotune": f0_autotune,
        "f0_autotune_strength": f0_autotune_strength,
        "proposed_pitch": proposed_pitch,
        "proposed_pitch_threshold": proposed_pitch_threshold,
        "input_audio_gain": input_audio_gain,
        "output_audio_gain": output_audio_gain,
        "monitor_audio_gain": monitor_audio_gain,
        "monitor": use_monitor_device,
        "vad_enabled": vad_enabled,
        "vad_sensitivity": 3,
        "vad_frame_ms": 30,
        "sid": sid,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "post_process": post_process,
        "kwargs": {
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
        }
    }

    callbacks = AudioCallbacks(**callbacks_kwargs)

    audio_manager = callbacks.audio
    audio_manager.start(
        input_device_id=input_device_id,
        output_device_id=output_device_id,
        output_monitor_id=output_monitor_id,
        exclusive_mode=exclusive_mode,
        asio_input_channel=input_asio_channels,
        asio_output_channel=output_asio_channels,
        asio_output_monitor_channel=monitor_asio_channels,
        read_chunk_size=read_chunk_size,
    )

    yield "Realtime is ready!", interactive_false, interactive_true

    while running and callbacks is not None and audio_manager is not None:
        time.sleep(0.1)
        if hasattr(audio_manager, "latency"):
            yield f"Latency: {audio_manager.latency:.2f} ms", interactive_false, interactive_true

    return gr.update(), gr.update(), gr.update()


def change_callbacks_config():
    global callbacks

    if running and audio_manager is not None and callbacks is not None:
        # print(callbacks_kwargs)

        # It will need to create a new stream to work.
        # callbacks.vc.block_frame = callbacks_kwargs.get("read_chunk_size", 192) * 128
        # callbacks.vc.crossfade_frame = int(callbacks_kwargs.get("cross_fade_overlap_size", 0.1) * AUDIO_SAMPLE_RATE)
        # callbacks.vc.extra_frame = int(callbacks_kwargs.get("extra_convert_size", 0.5) * AUDIO_SAMPLE_RATE)
        # callbacks.vc.vc_model.input_sensitivity = 10 ** (callbacks_kwargs.get("silent_threshold", -90) / 20)

        # callbacks.vc.vc_model.realloc(
        #     callbacks.vc.block_frame,
        #     callbacks.vc.extra_frame,
        #     callbacks.vc.crossfade_frame,
        #     callbacks.vc.sola_search_frame,
        # )
        # callbacks.vc.generate_strength()
        
        vad_enabled = callbacks_kwargs.get("vad_enabled", True)
        if vad_enabled is False:
            callbacks.vc.vc_model.vad = None
        elif vad_enabled and callbacks.vc.vc_model.vad is None:
            from rvc.realtime.utils.vad import VADProcessor

            callbacks.vc.vc_model.vad = VADProcessor(
                sensitivity_mode=3,
                sample_rate=callbacks.vc.vc_model.sample_rate,
                frame_duration_ms=30,  
            )

        # The VAD parameters have been assigned by default.
        # if callbacks.vc.vc_model.vad is not None:
        #     callbacks.vc.vc_model.vad.vad.set_mode(vad_sensitivity)
        #     callbacks.vc.vc_model.vad.frame_length = int(callbacks.vc.vc_model.sample_rate * (vad_frame_ms / 1000.0))

        clean_audio = callbacks_kwargs.get("clean_audio", False)
        clean_strength = callbacks_kwargs.get("clean_strength", 0.5)

        if clean_audio is False:
            callbacks.vc.vc_model.reduced_noise = None
        elif clean_audio and callbacks.vc.vc_model.reduced_noise is None:
            from noisereduce.torchgate import TorchGate

            callbacks.vc.vc_model.reduced_noise = (
                TorchGate(
                    callbacks.vc.vc_model.pipeline.tgt_sr,
                    prop_decrease=clean_strength,
                ).to(callbacks.vc.vc_model.device)
            )

        if callbacks.vc.vc_model.reduced_noise is not None:
            callbacks.vc.vc_model.reduced_noise.prop_decrease = clean_strength

        post_process = callbacks_kwargs.get("post_process", False)
        kwargs = callbacks_kwargs.get("kwargs", {})

        if post_process is False:
            callbacks.vc.vc_model.board = None
        elif post_process and callbacks.vc.vc_model.board is None:
            new_board = callbacks.vc.vc_model.setup_pedalboard(**kwargs)
            callbacks.vc.vc_model.board = new_board

        if callbacks.vc.vc_model.board is not None and callbacks.vc.vc_model.kwargs != kwargs:
            # Post-process requires creating a new pendalboard.
            new_board = callbacks.vc.vc_model.setup_pedalboard(**kwargs)
            callbacks.vc.vc_model.board = new_board      

        callbacks.audio.f0_up_key = callbacks_kwargs.get("f0_up_key", 0)
        callbacks.audio.index_rate = callbacks_kwargs.get("index_rate", 0.75)
        callbacks.audio.protect = callbacks_kwargs.get("protect", 0.5)
        callbacks.audio.volume_envelope = callbacks_kwargs.get("volume_envelope", 1)

        callbacks.audio.f0_autotune = callbacks_kwargs.get("f0_autotune", False)
        callbacks.audio.f0_autotune_strength = callbacks_kwargs.get("f0_autotune_strength", 1.0)
        callbacks.audio.proposed_pitch = callbacks_kwargs.get("proposed_pitch", False)
        callbacks.audio.proposed_pitch_threshold = callbacks_kwargs.get("proposed_pitch_threshold", 155.0)

        callbacks.audio.input_audio_gain = callbacks_kwargs.get("input_audio_gain", 1.0)
        callbacks.audio.output_audio_gain = callbacks_kwargs.get("output_audio_gain", 1.0)
        callbacks.audio.monitor_audio_gain = callbacks_kwargs.get("monitor_audio_gain", 1.0)


def change_config(value, key, if_kwargs=False):
    global callbacks_kwargs

    if running and audio_manager is not None and callbacks is not None:
        if if_kwargs:
            callbacks_kwargs["kwargs"][key] = value
        else:
            callbacks_kwargs[key] = value

        change_callbacks_config()


def stop_realtime():
    global running, callbacks, audio_manager
    if running and audio_manager is not None and callbacks is not None:
        audio_manager.stop()
        running = False
        if hasattr(audio_manager, "latency"):
            del audio_manager.latency
        audio_manager = callbacks = None
        time.sleep(0.1)

        return "Stopped", interactive_true, interactive_false,
    else:
        return "Realtime pipeline not found!", interactive_true, interactive_false


def get_audio_devices_formatted():
    try:
        input_devices, output_devices = list_audio_device()

        def priority(name: str) -> int:
            n = name.lower()
            if "virtual" in n:
                return 0
            if "vb" in n:
                return 1
            return 2

        output_sorted = sorted(output_devices, key=lambda d: priority(d.name))
        input_sorted = sorted(
            input_devices, key=lambda d: priority(d.name), reverse=True
        )

        input_device_list = {
            f"{input_sorted.index(d)+1}: {d.name} ({d.host_api})": d.index
            for d in input_sorted
        }
        output_device_list = {
            f"{output_sorted.index(d)+1}: {d.name} ({d.host_api})": d.index
            for d in output_sorted
        }

        return input_device_list, output_device_list
    except Exception:
        return [], []


def update_dropdowns_from_json(data):
    if not data:
        return [
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
        ]

    inputs = list(data.get("inputs", {}).keys())
    outputs = list(data.get("outputs", {}).keys())

    return [
        gr.update(choices=inputs, value=inputs[0] if len(inputs) > 0 else None),
        gr.update(choices=outputs, value=outputs[0] if len(outputs) > 0 else None),
        gr.update(choices=outputs, value=outputs[0] if len(outputs) > 0 else None),
    ]


def update_button_from_json(data):
    if not data:
        return [gr.update(interactive=True), gr.update(interactive=False)]

    return [
        gr.update(interactive=data.get("start_button", True)),
        gr.update(interactive=data.get("stop_button", False)),
    ]


def realtime_tab():
    input_devices, output_devices = [], []
    saved_settings = load_realtime_settings()

    if not client_mode:
        input_devices, output_devices = get_audio_devices_formatted()
        input_devices, output_devices = list(input_devices.keys()), list(
            output_devices.keys()
        )
    else:
        input_devices = [saved_settings["client_input_device"]] if saved_settings["client_input_device"] else []
        output_devices = [saved_settings["client_output_device"]] if saved_settings["client_output_device"] else []

    # Load saved settings

    with gr.Blocks() as ui:
        with gr.Row():
            start_button = gr.Button(i18n("Start"), variant="primary")
            stop_button = gr.Button(i18n("Stop"), interactive=False)
        latency_info = gr.Label(
            label=i18n("Status"),
            value=i18n("Realtime not started."),
            elem_id="realtime-status-info",
        )
        terms_checkbox = gr.Checkbox(
            label=i18n("I agree to the terms of use"),
            info=i18n(
                "Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your realtime."
            ),
            value=False,
            interactive=True,
        )

        with gr.Tabs():
            with gr.TabItem(i18n("Audio Settings")):
                with gr.Row():
                    refresh_devices_button = gr.Button(i18n("Refresh Audio Devices"))
                with gr.Row():
                    with gr.Accordion(i18n("Input Device"), open=True):
                        with gr.Column():
                            input_audio_device = gr.Dropdown(
                                label=i18n("Input Device"),
                                info=i18n(
                                    "Select the microphone or audio interface you will be speaking into."
                                ),
                                choices=input_devices,
                                value=get_safe_dropdown_value(
                                    saved_settings["client_input_device" if client_mode else "input_device"], input_devices
                                ),
                                interactive=True,
                            )
                            input_audio_gain = gr.Slider(
                                minimum=0,
                                maximum=200,
                                value=100,
                                label=i18n("Input Gain (%)"),
                                info=i18n(
                                    "Adjusts the input volume before processing. Prevents clipping or boosts a quiet mic."
                                ),
                                interactive=True,
                            )
                            input_asio_channels = gr.Slider(
                                minimum=-1,
                                maximum=16,
                                value=-1,
                                step=1,
                                label=i18n("Input ASIO Channel"),
                                info=i18n(
                                    "For ASIO drivers, selects a specific input channel. Leave at -1 for default."
                                ),
                                interactive=True,
                                visible=not client_mode,
                            )
                    with gr.Accordion("Output Device", open=True):
                        with gr.Column():
                            output_audio_device = gr.Dropdown(
                                label=i18n("Output Device"),
                                info=i18n(
                                    "Select the device where the final converted voice will be sent (e.g., a virtual cable)."
                                ),
                                choices=output_devices,
                                value=get_safe_dropdown_value(
                                    saved_settings["client_output_device" if client_mode else "output_device"], output_devices
                                ),
                                interactive=True,
                            )
                            output_audio_gain = gr.Slider(
                                minimum=0,
                                maximum=200,
                                value=100,
                                label=i18n("Output Gain (%)"),
                                info=i18n(
                                    "Adjusts the final volume of the converted voice after processing."
                                ),
                                interactive=True,
                            )
                            output_asio_channels = gr.Slider(
                                minimum=-1,
                                maximum=16,
                                value=-1,
                                step=1,
                                label=i18n("Output ASIO Channel"),
                                info=i18n(
                                    "For ASIO drivers, selects a specific output channel. Leave at -1 for default."
                                ),
                                interactive=True,
                                visible=not client_mode,
                            )
                with gr.Accordion(i18n("Monitor Device (Optional)"), open=False):
                    with gr.Column():
                        use_monitor_device = gr.Checkbox(
                            label=i18n("Use Monitor Device"),
                            value=False,
                            interactive=True,
                        )
                        monitor_output_device = gr.Dropdown(
                            label=i18n("Monitor Device"),
                            info=i18n(
                                "Select the device for monitoring your voice (e.g., your headphones)."
                            ),
                            choices=output_devices,
                            value=get_safe_dropdown_value(
                                saved_settings["client_monitor_device" if client_mode else "monitor_device"], output_devices
                            ),
                            interactive=True,
                        )
                        monitor_audio_gain = gr.Slider(
                            minimum=0,
                            maximum=200,
                            value=100,
                            label=i18n("Monitor Gain (%)"),
                            info=i18n(
                                "Adjusts the volume of the monitor feed, independent of the main output."
                            ),
                            interactive=True,
                        )
                        monitor_asio_channels = gr.Slider(
                            minimum=-1,
                            maximum=16,
                            value=-1,
                            step=1,
                            label=i18n("Monitor ASIO Channel"),
                            info=i18n(
                                "For ASIO drivers, selects a specific monitor output channel. Leave at -1 for default."
                            ),
                            interactive=True,
                            visible=not client_mode,
                        )
                with gr.Row():
                    exclusive_mode = gr.Checkbox(
                        label=i18n("Exclusive Mode"),
                        info=i18n(
                            "For WASAPI (Windows), gives the app exclusive control for potentially lower latency."
                            if not client_mode else
                            "Gives the app exclusive control for potentially lower latency."
                        ),
                        value=False,
                        interactive=True,
                        visible=client_mode # This is quite bad because it's prone to errors when used locally.
                    )
                    vad_enabled = gr.Checkbox(
                        label=i18n("Enable VAD"),
                        info=i18n(
                            "Enables Voice Activity Detection to only process audio when you are speaking, saving CPU."
                        ),
                        value=True,
                        interactive=True,
                    )

            with gr.TabItem(i18n("Model Settings")):
                with gr.Row():
                    model_choices = (
                        sorted(names, key=extract_model_and_epoch) if names else []
                    )
                    model_file = gr.Dropdown(
                        label=i18n("Voice Model"),
                        choices=model_choices,
                        interactive=True,
                        value=get_safe_dropdown_value(
                            saved_settings["model_file"], model_choices, default_weight
                        ),
                        allow_custom_value=True,
                    )
                    index_choices = get_files("index")
                    index_file = gr.Dropdown(
                        label=i18n("Index File"),
                        choices=index_choices,
                        value=get_safe_index_value(
                            saved_settings["index_file"],
                            index_choices,
                            match_index(default_weight) if default_weight else None,
                        ),
                        interactive=True,
                        allow_custom_value=True,
                    )

                with gr.Row():
                    unload_button = gr.Button(i18n("Unload Voice"))
                    refresh_button = gr.Button(i18n("Refresh"))
                with gr.Column():
                    autotune = gr.Checkbox(
                        label=i18n("Autotune"),
                        info=i18n(
                            "Apply a soft autotune to your inferences, recommended for singing conversions."
                        ),
                        visible=True,
                        value=False,
                        interactive=True,
                    )
                    autotune_strength = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Autotune Strength"),
                        info=i18n(
                            "Set the autotune strength - the more you increase it the more it will snap to the chromatic grid."
                        ),
                        visible=False,
                        value=1,
                        interactive=True,
                    )
                    proposed_pitch = gr.Checkbox(
                        label=i18n("Proposed Pitch"),
                        info=i18n(
                            "Adjust the input audio pitch to match the voice model range."
                        ),
                        visible=True,
                        value=False,
                        interactive=True,
                    )
                    proposed_pitch_threshold = gr.Slider(
                        minimum=50.0,
                        maximum=1200.0,
                        label=i18n("Proposed Pitch Threshold"),
                        info=i18n(
                            "Male voice models typically use 155.0 and female voice models typically use 255.0."
                        ),
                        visible=False,
                        value=155.0,
                        interactive=True,
                    )
                    clean_audio = gr.Checkbox(
                        label=i18n("Clean Audio"),
                        info=i18n(
                            "Clean your audio output using noise detection algorithms, recommended for speaking audios."
                        ),
                        visible=True,
                        value=False,
                        interactive=True,
                    )
                    clean_strength = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Clean Strength"),
                        info=i18n(
                            "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                        ),
                        visible=False,
                        value=0.5,
                        interactive=True,
                    )
                    post_process = gr.Checkbox(
                        label=i18n("Post-Process"),
                        info=i18n(
                            "Post-process the audio to apply effects to the output."
                        ),
                        value=False,
                        interactive=True,
                    )
                    reverb = gr.Checkbox(
                        label=i18n("Reverb"),
                        info=i18n("Apply reverb to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    reverb_room_size = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Reverb Room Size"),
                        info=i18n("Set the room size of the reverb."),
                        value=0.5,
                        interactive=True,
                        visible=False,
                    )
                    reverb_damping = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Reverb Damping"),
                        info=i18n("Set the damping of the reverb."),
                        value=0.5,
                        interactive=True,
                        visible=False,
                    )
                    reverb_wet_gain = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Reverb Wet Gain"),
                        info=i18n("Set the wet gain of the reverb."),
                        value=0.33,
                        interactive=True,
                        visible=False,
                    )
                    reverb_dry_gain = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Reverb Dry Gain"),
                        info=i18n("Set the dry gain of the reverb."),
                        value=0.4,
                        interactive=True,
                        visible=False,
                    )
                    reverb_width = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Reverb Width"),
                        info=i18n("Set the width of the reverb."),
                        value=1.0,
                        interactive=True,
                        visible=False,
                    )
                    reverb_freeze_mode = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Reverb Freeze Mode"),
                        info=i18n("Set the freeze mode of the reverb."),
                        value=0.0,
                        interactive=True,
                        visible=False,
                    )
                    pitch_shift = gr.Checkbox(
                        label=i18n("Pitch Shift"),
                        info=i18n("Apply pitch shift to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    pitch_shift_semitones = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        label=i18n("Pitch Shift Semitones"),
                        info=i18n("Set the pitch shift semitones."),
                        value=0,
                        interactive=True,
                        visible=False,
                    )
                    limiter = gr.Checkbox(
                        label=i18n("Limiter"),
                        info=i18n("Apply limiter to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    limiter_threshold = gr.Slider(
                        minimum=-60,
                        maximum=0,
                        label=i18n("Limiter Threshold dB"),
                        info=i18n("Set the limiter threshold dB."),
                        value=-6,
                        interactive=True,
                        visible=False,
                    )
                    limiter_release_time = gr.Slider(
                        minimum=0.01,
                        maximum=1,
                        label=i18n("Limiter Release Time"),
                        info=i18n("Set the limiter release time."),
                        value=0.05,
                        interactive=True,
                        visible=False,
                    )
                    gain = gr.Checkbox(
                        label=i18n("Gain"),
                        info=i18n("Apply gain to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    gain_db = gr.Slider(
                        minimum=-60,
                        maximum=60,
                        label=i18n("Gain dB"),
                        info=i18n("Set the gain dB."),
                        value=0,
                        interactive=True,
                        visible=False,
                    )
                    distortion = gr.Checkbox(
                        label=i18n("Distortion"),
                        info=i18n("Apply distortion to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    distortion_gain = gr.Slider(
                        minimum=-60,
                        maximum=60,
                        label=i18n("Distortion Gain"),
                        info=i18n("Set the distortion gain."),
                        value=25,
                        interactive=True,
                        visible=False,
                    )
                    chorus = gr.Checkbox(
                        label=i18n("Chorus"),
                        info=i18n("Apply chorus to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    chorus_rate = gr.Slider(
                        minimum=0,
                        maximum=100,
                        label=i18n("Chorus Rate Hz"),
                        info=i18n("Set the chorus rate Hz."),
                        value=1.0,
                        interactive=True,
                        visible=False,
                    )
                    chorus_depth = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Chorus Depth"),
                        info=i18n("Set the chorus depth."),
                        value=0.25,
                        interactive=True,
                        visible=False,
                    )
                    chorus_center_delay = gr.Slider(
                        minimum=7,
                        maximum=8,
                        label=i18n("Chorus Center Delay ms"),
                        info=i18n("Set the chorus center delay ms."),
                        value=7,
                        interactive=True,
                        visible=False,
                    )
                    chorus_feedback = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Chorus Feedback"),
                        info=i18n("Set the chorus feedback."),
                        value=0.0,
                        interactive=True,
                        visible=False,
                    )
                    chorus_mix = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Chorus Mix"),
                        info=i18n("Set the chorus mix."),
                        value=0.5,
                        interactive=True,
                        visible=False,
                    )
                    bitcrush = gr.Checkbox(
                        label=i18n("Bitcrush"),
                        info=i18n("Apply bitcrush to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    bitcrush_bit_depth = gr.Slider(
                        minimum=1,
                        maximum=32,
                        label=i18n("Bitcrush Bit Depth"),
                        info=i18n("Set the bitcrush bit depth."),
                        value=8,
                        interactive=True,
                        visible=False,
                    )
                    clipping = gr.Checkbox(
                        label=i18n("Clipping"),
                        info=i18n("Apply clipping to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    clipping_threshold = gr.Slider(
                        minimum=-60,
                        maximum=0,
                        label=i18n("Clipping Threshold"),
                        info=i18n("Set the clipping threshold."),
                        value=-6,
                        interactive=True,
                        visible=False,
                    )
                    compressor = gr.Checkbox(
                        label=i18n("Compressor"),
                        info=i18n("Apply compressor to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    compressor_threshold = gr.Slider(
                        minimum=-60,
                        maximum=0,
                        label=i18n("Compressor Threshold dB"),
                        info=i18n("Set the compressor threshold dB."),
                        value=0,
                        interactive=True,
                        visible=False,
                    )
                    compressor_ratio = gr.Slider(
                        minimum=1,
                        maximum=20,
                        label=i18n("Compressor Ratio"),
                        info=i18n("Set the compressor ratio."),
                        value=1,
                        interactive=True,
                        visible=False,
                    )
                    compressor_attack = gr.Slider(
                        minimum=0.0,
                        maximum=100,
                        label=i18n("Compressor Attack ms"),
                        info=i18n("Set the compressor attack ms."),
                        value=1.0,
                        interactive=True,
                        visible=False,
                    )
                    compressor_release = gr.Slider(
                        minimum=0.01,
                        maximum=100,
                        label=i18n("Compressor Release ms"),
                        info=i18n("Set the compressor release ms."),
                        value=100,
                        interactive=True,
                        visible=False,
                    )
                    delay = gr.Checkbox(
                        label=i18n("Delay"),
                        info=i18n("Apply delay to the audio."),
                        value=False,
                        interactive=True,
                        visible=False,
                    )
                    delay_seconds = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        label=i18n("Delay Seconds"),
                        info=i18n("Set the delay seconds."),
                        value=0.5,
                        interactive=True,
                        visible=False,
                    )
                    delay_feedback = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        label=i18n("Delay Feedback"),
                        info=i18n("Set the delay feedback."),
                        value=0.0,
                        interactive=True,
                        visible=False,
                    )
                    delay_mix = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        label=i18n("Delay Mix"),
                        info=i18n("Set the delay mix."),
                        value=0.5,
                        interactive=True,
                        visible=False,
                    )
                    sid = gr.Dropdown(
                        label=i18n("Speaker ID"),
                        choices=(
                            get_speakers_id(default_weight) if default_weight else [0]
                        ),
                        value=0,
                        interactive=True,
                    )
                    pitch = gr.Slider(
                        minimum=-24,
                        maximum=24,
                        step=1,
                        label=i18n("Pitch"),
                        info=i18n(
                            "Set the pitch of the audio, the higher the value, the higher the pitch."
                        ),
                        value=0,
                        interactive=True,
                    )
                    index_rate = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Search Feature Ratio"),
                        info=i18n(
                            "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                        ),
                        value=0, # The index is not always necessary, so disabling it can help improve latency.
                        interactive=True,
                    )
                    volume_envelope = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=1,
                        label=i18n("Volume Envelope"),
                        info=i18n(
                            "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                        ),
                        interactive=True,
                    )
                    protect = gr.Slider(
                        minimum=0,
                        maximum=0.5,
                        value=0.5,
                        label=i18n("Protect Voiceless Consonants"),
                        info=i18n(
                            "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                        ),
                        interactive=True,
                    )
                    f0_method = gr.Radio(
                        choices=["rmvpe", "fcpe"],
                        value="fcpe",
                        label=i18n("Pitch extraction algorithm"),
                        info=i18n(
                            "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                        ),
                        interactive=True,
                    )
                    embedder_model = gr.Radio(
                        choices=[
                            "contentvec",
                            "spin",
                            "spin-v2",
                            "chinese-hubert-base",
                            "japanese-hubert-base",
                            "korean-hubert-base",
                            "custom",
                        ],
                        value="contentvec",
                        label=i18n("Embedder Model"),
                        info=i18n("Model used for learning speaker embedding."),
                        interactive=True,
                    )
                    with gr.Column(visible=False) as embedder_custom:
                        with gr.Accordion(i18n("Custom Embedder"), open=True):
                            with gr.Row():
                                embedder_model_custom = gr.Dropdown(
                                    label=i18n("Select Custom Embedder"),
                                    choices=refresh_embedders_folders(),
                                    interactive=True,
                                    allow_custom_value=True,
                                )
                                refresh_embedders_button = gr.Button(
                                    i18n("Refresh embedders")
                                )
                            folder_name_input = gr.Textbox(
                                label=i18n("Folder Name"), interactive=True
                            )
                            with gr.Row():
                                bin_file_upload = gr.File(
                                    label=i18n("Upload .bin"),
                                    type="filepath",
                                    interactive=True,
                                )
                                config_file_upload = gr.File(
                                    label=i18n("Upload .json"),
                                    type="filepath",
                                    interactive=True,
                                )
                            move_files_button = gr.Button(
                                i18n("Move files to custom embedder folder")
                            )

            with gr.TabItem(i18n("Performance Settings")):
                chunk_size = gr.Slider(
                    minimum=2.7,
                    maximum=2730.7,
                    value=768,
                    step=1,
                    label=i18n("Chunk Size (ms)"),
                    info=i18n(
                        "Audio buffer size in milliseconds. Lower values may reduce latency but increase CPU load."
                    ),
                    interactive=True,
                )
                cross_fade_overlap_size = gr.Slider(
                    minimum=0.05,
                    maximum=0.2,
                    value=0.05,
                    step=0.01,
                    label=i18n("Crossfade Overlap Size (s)"),
                    info=i18n(
                        "Duration of the fade between audio chunks to prevent clicks. Higher values create smoother transitions but may increase latency."
                    ),
                    interactive=True,
                )
                extra_convert_size = gr.Slider(
                    minimum=0.1,
                    maximum=5,
                    value=0.5,
                    step=0.1,
                    label=i18n("Extra Conversion Size (s)"),
                    info=i18n(
                        "Amount of extra audio processed to provide context to the model. Improves conversion quality at the cost of higher CPU usage."
                    ),
                    interactive=True,
                )
                silent_threshold = gr.Slider(
                    minimum=-90,
                    maximum=-60,
                    value=-90,
                    step=1,
                    label=i18n("Silence Threshold (dB)"),
                    info=i18n(
                        "Volume level below which audio is treated as silence and not processed. Helps to save CPU resources and reduce background noise."
                    ),
                    interactive=True,
                )

        json_audio_hidden = gr.JSON(visible=False)
        json_button_hidden = gr.JSON(visible=False)

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                yield message, interactive_true, interactive_false
                return
            yield from start_realtime(*args)

        def update_on_model_change(model_path):
            new_index = match_index(model_path)
            new_sids = get_speakers_id(model_path)

            # Get updated index choices
            new_index_choices = get_files("index")
            # Use the matched index as fallback, but handle empty strings
            fallback_index = new_index if new_index and new_index.strip() else None
            safe_index_value = get_safe_index_value(
                "", new_index_choices, fallback_index
            )

            return gr.update(
                choices=new_index_choices, value=safe_index_value
            ), gr.update(choices=new_sids, value=0 if new_sids else None)

        def refresh_devices():
            sd._terminate()
            sd._initialize()

            input_choices, output_choices = get_audio_devices_formatted()
            input_choices, output_choices = list(input_choices.keys()), list(
                output_choices.keys()
            )
            return (
                gr.update(choices=input_choices),
                gr.update(choices=output_choices),
                gr.update(choices=output_choices),
            )

        def toggle_visible(checkbox):
            return {"visible": checkbox, "__type__": "update"}

        def toggle_visible_embedder_custom(embedder_model):
            if embedder_model == "custom":
                return {"visible": True, "__type__": "update"}
            return {"visible": False, "__type__": "update"}

        if client_mode:
            refresh_devices_button.click(
                fn=None,
                js="getAudioDevices",
                outputs=[json_audio_hidden],
            )

            json_audio_hidden.change(
                fn=update_dropdowns_from_json,
                inputs=[json_audio_hidden],
                outputs=[input_audio_device, output_audio_device, monitor_output_device],
            )

            json_button_hidden.change(
                fn=update_button_from_json,
                inputs=[json_button_hidden],
                outputs=[start_button, stop_button],
            )
        else:
            refresh_devices_button.click(
                fn=refresh_devices,
                outputs=[input_audio_device, output_audio_device, monitor_output_device],
            )

        autotune.change(
            fn=toggle_visible,
            inputs=[autotune],
            outputs=[autotune_strength],
        )

        proposed_pitch.change(
            fn=toggle_visible,
            inputs=[proposed_pitch],
            outputs=[proposed_pitch_threshold],
        )

        clean_audio.change(
            fn=toggle_visible,
            inputs=[clean_audio],
            outputs=[clean_strength],
        )

        embedder_model.change(
            fn=toggle_visible_embedder_custom,
            inputs=[embedder_model],
            outputs=[embedder_custom],
        )

        move_files_button.click(
            fn=create_folder_and_move_files,
            inputs=[folder_name_input, bin_file_upload, config_file_upload],
            outputs=[],
        )
        refresh_embedders_button.click(
            fn=lambda: gr.update(choices=refresh_embedders_folders()),
            inputs=[],
            outputs=[embedder_model_custom],
        )

        def update_visibility(checkbox, count):
            return [gr.update(visible=checkbox) for _ in range(count)]

        def post_process_visible(checkbox):
            return update_visibility(checkbox, 10)

        def reverb_visible(checkbox):
            return update_visibility(checkbox, 6)

        def limiter_visible(checkbox):
            return update_visibility(checkbox, 2)

        def chorus_visible(checkbox):
            return update_visibility(checkbox, 6)

        def bitcrush_visible(checkbox):
            return update_visibility(checkbox, 1)

        def compress_visible(checkbox):
            return update_visibility(checkbox, 4)

        def delay_visible(checkbox):
            return update_visibility(checkbox, 3)

        post_process.change(
            fn=post_process_visible,
            inputs=[post_process],
            outputs=[
                reverb,
                pitch_shift,
                limiter,
                gain,
                distortion,
                chorus,
                bitcrush,
                clipping,
                compressor,
                delay,
            ],
        )

        reverb.change(
            fn=reverb_visible,
            inputs=[reverb],
            outputs=[
                reverb_room_size,
                reverb_damping,
                reverb_wet_gain,
                reverb_dry_gain,
                reverb_width,
                reverb_freeze_mode,
            ],
        )
        pitch_shift.change(
            fn=toggle_visible,
            inputs=[pitch_shift],
            outputs=[pitch_shift_semitones],
        )
        limiter.change(
            fn=limiter_visible,
            inputs=[limiter],
            outputs=[limiter_threshold, limiter_release_time],
        )
        gain.change(
            fn=toggle_visible,
            inputs=[gain],
            outputs=[gain_db],
        )
        distortion.change(
            fn=toggle_visible,
            inputs=[distortion],
            outputs=[distortion_gain],
        )
        chorus.change(
            fn=chorus_visible,
            inputs=[chorus],
            outputs=[
                chorus_rate,
                chorus_depth,
                chorus_center_delay,
                chorus_feedback,
                chorus_mix,
            ],
        )
        bitcrush.change(
            fn=bitcrush_visible,
            inputs=[bitcrush],
            outputs=[bitcrush_bit_depth],
        )
        clipping.change(
            fn=toggle_visible,
            inputs=[clipping],
            outputs=[clipping_threshold],
        )
        compressor.change(
            fn=compress_visible,
            inputs=[compressor],
            outputs=[
                compressor_threshold,
                compressor_ratio,
                compressor_attack,
                compressor_release,
            ],
        )
        delay.change(
            fn=delay_visible,
            inputs=[delay],
            outputs=[delay_seconds, delay_feedback, delay_mix],
        )

        if client_mode:
            start_button.click(
                fn=None,
                js="StreamAudioRealtime",
                inputs=[
                    terms_checkbox,
                    input_audio_device,
                    input_audio_gain,
                    output_audio_device,
                    output_audio_gain,
                    monitor_output_device,
                    monitor_audio_gain,
                    use_monitor_device,
                    vad_enabled,
                    chunk_size,
                    cross_fade_overlap_size,
                    extra_convert_size,
                    silent_threshold,
                    pitch,
                    index_rate,
                    volume_envelope,
                    protect,
                    f0_method,
                    model_file,
                    index_file,
                    sid,
                    autotune,
                    autotune_strength,
                    proposed_pitch,
                    proposed_pitch_threshold,
                    embedder_model,
                    embedder_model_custom,
                    exclusive_mode,
                    clean_audio,
                    clean_strength,
                    post_process,
                    reverb,
                    pitch_shift,
                    limiter,
                    gain,
                    distortion,
                    chorus,
                    bitcrush,
                    clipping,
                    compressor,
                    delay,
                    reverb_room_size,
                    reverb_damping,
                    reverb_wet_gain,
                    reverb_dry_gain,
                    reverb_width,
                    reverb_freeze_mode,
                    pitch_shift_semitones,
                    limiter_threshold,
                    limiter_release_time,
                    gain_db,
                    distortion_gain,
                    chorus_rate,
                    chorus_depth,
                    chorus_center_delay,
                    chorus_feedback,
                    chorus_mix,
                    bitcrush_bit_depth,
                    clipping_threshold,
                    compressor_threshold,
                    compressor_ratio,
                    compressor_attack,
                    compressor_release,
                    delay_seconds,
                    delay_feedback,
                    delay_mix,
                ],
                outputs=[json_button_hidden],
            )

            stop_button.click(fn=None, js="StopAudioStream", outputs=[json_button_hidden])
        else:
            start_button.click(
                fn=enforce_terms,
                inputs=[
                    terms_checkbox,
                    input_audio_device,
                    input_audio_gain,
                    input_asio_channels,
                    output_audio_device,
                    output_audio_gain,
                    output_asio_channels,
                    monitor_output_device,
                    monitor_audio_gain,
                    monitor_asio_channels,
                    use_monitor_device,
                    exclusive_mode,
                    vad_enabled,
                    chunk_size,
                    cross_fade_overlap_size,
                    extra_convert_size,
                    silent_threshold,
                    pitch,
                    index_rate,
                    volume_envelope,
                    protect,
                    f0_method,
                    model_file,
                    index_file,
                    sid,
                    autotune,
                    autotune_strength,
                    proposed_pitch,
                    proposed_pitch_threshold,
                    embedder_model,
                    embedder_model_custom,
                    clean_audio,
                    clean_strength,
                    post_process,
                    reverb,
                    pitch_shift,
                    limiter,
                    gain,
                    distortion,
                    chorus,
                    bitcrush,
                    clipping,
                    compressor,
                    delay,
                    reverb_room_size,
                    reverb_damping,
                    reverb_wet_gain,
                    reverb_dry_gain,
                    reverb_width,
                    reverb_freeze_mode,
                    pitch_shift_semitones,
                    limiter_threshold,
                    limiter_release_time,
                    gain_db,
                    distortion_gain,
                    chorus_rate,
                    chorus_depth,
                    chorus_center_delay,
                    chorus_feedback,
                    chorus_mix,
                    bitcrush_bit_depth,
                    clipping_threshold,
                    compressor_threshold,
                    compressor_ratio,
                    compressor_attack,
                    compressor_release,
                    delay_seconds,
                    delay_feedback,
                    delay_mix,
                ],
                outputs=[latency_info, start_button, stop_button],
            )

            stop_button.click(
                fn=stop_realtime, outputs=[latency_info, start_button, stop_button]
            ) # .then(
            #     fn=lambda: (
            #         yield gr.update(value="Stopped"),
            #         interactive_true,
            #         interactive_false,
            #     ),
            #     inputs=None,
            #     outputs=[latency_info, start_button, stop_button],
            # )

        unload_button.click(
            fn=lambda: (
                {"value": "", "__type__": "update"},
                {"value": "", "__type__": "update"},
            ),
            inputs=[],
            outputs=[model_file, index_file],
        )
        model_file.select(
            fn=update_on_model_change, inputs=[model_file], outputs=[index_file, sid]
        )

        # Add event handlers to save settings
        input_audio_device.change(
            fn=lambda key: save_realtime_settings(key, "client_input_device" if client_mode else "input_device"), inputs=[input_audio_device], outputs=[]
        )

        output_audio_device.change(
            fn=lambda key: save_realtime_settings(key, "client_output_device" if client_mode else "output_device"), inputs=[output_audio_device], outputs=[]
        )

        monitor_output_device.change(
            fn=lambda key: save_realtime_settings(key, "client_monitor_device" if client_mode else "monitor_device"), inputs=[monitor_output_device], outputs=[]
        )

        model_file.change(fn=lambda key: save_realtime_settings(key, "model_file"), inputs=[model_file], outputs=[])

        index_file.change(fn=lambda key: save_realtime_settings(key, "index_file"), inputs=[index_file], outputs=[])

        if client_mode:
            def refresh_all():
                new_names = get_files("model")
                new_indexes = get_files("index")
                return (
                    gr.update(choices=sorted(new_names, key=extract_model_and_epoch)),
                    gr.update(choices=new_indexes),
                )

            refresh_button.click(
                fn=refresh_all,
                outputs=[
                    model_file,
                    index_file,
                ],
            )
        else:
            def refresh_all():
                new_names = get_files("model")
                new_indexes = get_files("index")
                input_choices, output_choices = get_audio_devices_formatted()
                input_choices, output_choices = list(input_choices.keys()), list(
                    output_choices.keys()
                )
                return (
                    gr.update(choices=sorted(new_names, key=extract_model_and_epoch)),
                    gr.update(choices=new_indexes),
                    gr.update(choices=input_choices),
                    gr.update(choices=output_choices),
                    gr.update(choices=output_choices),
                )

            refresh_button.click(
                fn=refresh_all,
                outputs=[
                    model_file,
                    index_file,
                    input_audio_device,
                    output_audio_device,
                    monitor_output_device,
                ],
            )

        input_audio_gain.change(js="(value) => window.ChangeConfig(value, 'input_audio_gain')", fn=lambda value: change_config(value / 100.0, "input_audio_gain") if value else None, inputs=[input_audio_gain], outputs=[])
        output_audio_gain.change(js="(value) => window.ChangeConfig(value, 'output_audio_gain')", fn=lambda value: change_config(value / 100.0, "output_audio_gain") if value else None, inputs=[output_audio_gain], outputs=[])
        monitor_audio_gain.change(js="(value) => window.ChangeConfig(value, 'monitor_audio_gain')", fn=lambda value: change_config(value / 100.0, "monitor_audio_gain") if value else None, inputs=[monitor_audio_gain], outputs=[])
        vad_enabled.change(js="(value) => window.ChangeConfig(value, 'vad_enabled')", fn=lambda value: change_config(value, "vad_enabled") if value else None, inputs=[vad_enabled], outputs=[])

        # chunk_size.change(fn=lambda value: change_config(int(value * AUDIO_SAMPLE_RATE / 1000 / 128) if value else None, "read_chunk_size"), inputs=[chunk_size], outputs=[])
        # cross_fade_overlap_size.change(fn=lambda value: change_config(value, "cross_fade_overlap_size"), inputs=[cross_fade_overlap_size], outputs=[])
        # extra_convert_size.change(fn=lambda value: change_config(value, "extra_convert_size"), inputs=[extra_convert_size], outputs=[])
        # silent_threshold.change(fn=lambda value: change_config(value, "silent_threshold"), inputs=[silent_threshold], outputs=[])

        pitch.change(js="(value) => window.ChangeConfig(value, 'f0_up_key')" if client_mode else None, fn=lambda value: change_config(value, "f0_up_key") if value else None, inputs=[pitch], outputs=[])
        index_rate.change(js="(value) => window.ChangeConfig(value, 'index_rate')" if client_mode else None, fn=lambda value: change_config(value, "index_rate") if value else None, inputs=[index_rate], outputs=[])
        volume_envelope.change(js="(value) => window.ChangeConfig(value, 'volume_envelope')" if client_mode else None, fn=lambda value: change_config(value, "volume_envelope") if value else None, inputs=[volume_envelope], outputs=[])
        protect.change(js="(value) => window.ChangeConfig(value, 'protect')" if client_mode else None if client_mode else None, fn=lambda value: change_config(value, "protect") if value else None, inputs=[protect], outputs=[])

        autotune.change(js="(value) => window.ChangeConfig(value, 'autotune')" if client_mode else None, fn=lambda value: change_config(value, "autotune") if value else None, inputs=[autotune], outputs=[])
        autotune_strength.change(js="(value) => window.ChangeConfig(value, 'autotune_strength')" if client_mode else None, fn=lambda value: change_config(value, "autotune_strength") if value else None, inputs=[autotune_strength], outputs=[])
        proposed_pitch.change(js="(value) => window.ChangeConfig(value, 'proposed_pitch')" if client_mode else None, fn=lambda value: change_config(value, "proposed_pitch") if value else None, inputs=[proposed_pitch], outputs=[])
        proposed_pitch_threshold.change(js="(value) => window.ChangeConfig(value, 'proposed_pitch_threshold')" if client_mode else None, fn=lambda value: change_config(value, "proposed_pitch_threshold") if value else None, inputs=[proposed_pitch_threshold], outputs=[])
        clean_audio.change(js="(value) => window.ChangeConfig(value, 'clean_audio')" if client_mode else None, fn=lambda value: change_config(value, "clean_audio") if value else None, inputs=[clean_audio], outputs=[])
        clean_strength.change(js="(value) => window.ChangeConfig(value, 'clean_strength')" if client_mode else None, fn=lambda value: change_config(value, "clean_strength") if value else None, inputs=[clean_strength], outputs=[])

        post_process.change(js="(value) => window.ChangeConfig(value, 'post_process')" if client_mode else None, fn=lambda value: change_config(value, "post_process") if value else None, inputs=[post_process], outputs=[])
        reverb.change(js="(value) => window.ChangeConfig(value, 'reverb', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "reverb", if_kwargs=True) if value else None, inputs=[reverb], outputs=[])
        pitch_shift.change(js="(value) => window.ChangeConfig(value, 'pitch_shift', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "pitch_shift", if_kwargs=True) if value else None, inputs=[pitch_shift], outputs=[])
        limiter.change(js="(value) => window.ChangeConfig(value, 'limiter', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "limiter", if_kwargs=True) if value else None, inputs=[limiter], outputs=[])
        gain.change(js="(value) => window.ChangeConfig(value, 'gain', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "gain", if_kwargs=True) if value else None, inputs=[gain], outputs=[])
        distortion.change(js="(value) => window.ChangeConfig(value, 'distortion', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "distortion", if_kwargs=True) if value else None, inputs=[distortion], outputs=[])
        chorus.change(js="(value) => window.ChangeConfig(value, 'chorus', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "chorus", if_kwargs=True) if value else None, inputs=[chorus], outputs=[])
        bitcrush.change(js="(value) => window.ChangeConfig(value, 'bitcrush', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "bitcrush", if_kwargs=True) if value else None, inputs=[bitcrush], outputs=[])
        clipping.change(js="(value) => window.ChangeConfig(value, 'clipping', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "clipping", if_kwargs=True) if value else None, inputs=[clipping], outputs=[])
        compressor.change(js="(value) => window.ChangeConfig(value, 'compressor', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "compressor", if_kwargs=True) if value else None, inputs=[compressor], outputs=[])
        delay.change(js="(value) => window.ChangeConfig(value, 'delay', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "delay", if_kwargs=True) if value else None, inputs=[delay], outputs=[])
    
        reverb_room_size.change(js="(value) => window.ChangeConfig(value, 'reverb_room_size', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "reverb_room_size", if_kwargs=True) if value else None, inputs=[reverb_room_size], outputs=[])
        reverb_damping.change(js="(value) => window.ChangeConfig(value, 'reverb_damping', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "reverb_damping", if_kwargs=True) if value else None, inputs=[reverb_damping], outputs=[])
        reverb_wet_gain.change(js="(value) => window.ChangeConfig(value, 'reverb_wet_gain', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "reverb_wet_gain", if_kwargs=True) if value else None, inputs=[reverb_wet_gain], outputs=[])
        reverb_dry_gain.change(js="(value) => window.ChangeConfig(value, 'reverb_dry_gain', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "reverb_dry_gain", if_kwargs=True) if value else None, inputs=[reverb_dry_gain], outputs=[])
        reverb_width.change(js="(value) => window.ChangeConfig(value, 'reverb_width', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "reverb_width", if_kwargs=True) if value else None, inputs=[reverb_width], outputs=[])
        reverb_freeze_mode.change(js="(value) => window.ChangeConfig(value, 'reverb_freeze_mode', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "reverb_freeze_mode", if_kwargs=True) if value else None, inputs=[reverb_freeze_mode], outputs=[])
    
        pitch_shift_semitones.change(js="(value) => window.ChangeConfig(value, 'pitch_shift_semitones', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "pitch_shift_semitones", if_kwargs=True) if value else None, inputs=[pitch_shift_semitones], outputs=[])
        limiter_threshold.change(js="(value) => window.ChangeConfig(value, 'limiter_threshold', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "limiter_threshold", if_kwargs=True) if value else None, inputs=[limiter_threshold], outputs=[])
        limiter_release_time.change(js="(value) => window.ChangeConfig(value, 'limiter_release_time', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "limiter_release_time", if_kwargs=True) if value else None, inputs=[limiter_release_time], outputs=[])
        gain_db.change(js="(value) => window.ChangeConfig(value, 'gain_db', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "gain_db", if_kwargs=True) if value else None, inputs=[gain_db], outputs=[])
        distortion_gain.change(js="(value) => window.ChangeConfig(value, 'distortion_gain', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "distortion_gain", if_kwargs=True) if value else None, inputs=[distortion_gain], outputs=[])

        chorus_rate.change(js="(value) => window.ChangeConfig(value, 'chorus_rate', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "chorus_rate", if_kwargs=True) if value else None, inputs=[chorus_rate], outputs=[])
        chorus_depth.change(js="(value) => window.ChangeConfig(value, 'chorus_depth', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "chorus_depth", if_kwargs=True) if value else None, inputs=[chorus_depth], outputs=[])
        chorus_center_delay.change(js="(value) => window.ChangeConfig(value, 'chorus_center_delay', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "chorus_center_delay", if_kwargs=True) if value else None, inputs=[chorus_center_delay], outputs=[])
        chorus_feedback.change(js="(value) => window.ChangeConfig(value, 'chorus_feedback', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "chorus_feedback", if_kwargs=True) if value else None, inputs=[chorus_feedback], outputs=[])
        chorus_mix.change(js="(value) => window.ChangeConfig(value, 'chorus_mix', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "chorus_mix", if_kwargs=True) if value else None, inputs=[chorus_mix], outputs=[])

        bitcrush_bit_depth.change(js="(value) => window.ChangeConfig(value, 'bitcrush_bit_depth', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "bitcrush_bit_depth", if_kwargs=True) if value else None, inputs=[bitcrush_bit_depth], outputs=[])
        clipping_threshold.change(js="(value) => window.ChangeConfig(value, 'clipping_threshold', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "clipping_threshold", if_kwargs=True) if value else None, inputs=[clipping_threshold], outputs=[])

        compressor_threshold.change(js="(value) => window.ChangeConfig(value, 'compressor_threshold', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "compressor_threshold", if_kwargs=True) if value else None, inputs=[compressor_threshold], outputs=[])
        compressor_ratio.change(js="(value) => window.ChangeConfig(value, 'compressor_ratio', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "compressor_ratio", if_kwargs=True) if value else None, inputs=[compressor_ratio], outputs=[])
        compressor_attack.change(js="(value) => window.ChangeConfig(value, 'compressor_attack', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "compressor_attack", if_kwargs=True) if value else None, inputs=[compressor_attack], outputs=[])
        compressor_release.change(js="(value) => window.ChangeConfig(value, 'compressor_release', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "compressor_release", if_kwargs=True) if value else None, inputs=[compressor_release], outputs=[])

        delay_seconds.change(js="(value) => window.ChangeConfig(value, 'delay_seconds', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "delay_seconds", if_kwargs=True) if value else None, inputs=[delay_seconds], outputs=[])
        delay_feedback.change(js="(value) => window.ChangeConfig(value, 'delay_feedback', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "delay_feedback", if_kwargs=True) if value else None, inputs=[delay_feedback], outputs=[])
        delay_mix.change(js="(value) => window.ChangeConfig(value, 'delay_mix', if_kwargs=true)" if client_mode else None, fn=lambda value: change_config(value, "delay_mix", if_kwargs=True) if value else None, inputs=[delay_mix], outputs=[])