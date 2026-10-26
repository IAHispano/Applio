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
from rvc.realtime.audio import list_audio_device, resolve_sample_rate
from rvc.realtime.core import AUDIO_SAMPLE_RATE

from assets.i18n.i18n import I18nAuto
from assets.app_shell import step_card

i18n = I18nAuto()


def progress_str(remaining, total):
    pct = max(0, min(100, int((total - remaining) / total * 100))) if total > 0 else 0
    filled = "â–ˆ" * (pct // 10)
    empty = "â–‘" * (10 - pct // 10)
    return f"[{filled}{empty}] {pct}%"


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
    exts = (".pth",) if is_model else (".index",)
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


def display_name(path):
    """Dropdown label for a path: the file or folder name, without extension."""
    return os.path.splitext(os.path.basename(path))[0]


def path_choices(paths):
    """
    Convert a list of paths into (label, value) dropdown choices so only the
    file name is displayed and announced instead of the full relative path.
    The order of `paths` is preserved, and duplicate names are disambiguated
    by appending the parent folder name.
    """
    names = [display_name(path) for path in paths]
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    choices = []
    for name, path in zip(names, paths):
        if counts[name] > 1:
            name = f"{name} ({os.path.basename(os.path.dirname(path))})"
        choices.append((name, path))
    return choices


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
        except Exception:
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
interactive_true = gr.update(interactive=True, visible=True)
interactive_false = gr.update(interactive=False, visible=False)
interactive_visible = gr.update(interactive=False, visible=True)
running, callbacks, audio_manager = False, None, None
callbacks_kwargs = {}

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")


def save_realtime_settings(
    input_device=None,
    output_device=None,
    monitor_device=None,
    model_file=None,
    index_file=None,
    asio_enabled=None,
    audio_sample_rate=None,
    **kwargs,
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
        if input_device is not None:
            config["realtime"][
                "client_input_device" if client_mode else "input_device"
            ] = (input_device or "")
        if output_device is not None:
            config["realtime"][
                "client_output_device" if client_mode else "output_device"
            ] = (output_device or "")
        if monitor_device is not None:
            config["realtime"][
                "client_monitor_device" if client_mode else "monitor_device"
            ] = (monitor_device or "")
        if model_file is not None:
            config["realtime"]["model_file"] = model_file or ""
        if index_file is not None:
            config["realtime"]["index_file"] = index_file or ""
        if asio_enabled is not None:
            config["realtime"]["asio_enabled"] = asio_enabled
        if audio_sample_rate is not None:
            config["realtime"]["audio_sample_rate"] = audio_sample_rate

        for key, val in kwargs.items():
            if val is not None:
                config["realtime"][key] = val

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving realtime settings: {e}")


def load_realtime_settings():
    """Load realtime settings from config.json"""
    defaults = {
        "input_device": "",
        "output_device": "",
        "monitor_device": "",
        "client_input_device": "",
        "client_output_device": "",
        "client_monitor_device": "",
        "use_monitor_device": False,
        "model_file": "",
        "index_file": "",
        "asio_enabled": False,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "input_audio_gain": 100,
        "output_audio_gain": 100,
        "monitor_audio_gain": 100,
        "chunk_size": 250.0,
        "extra_convert_size": 2.5,
        "cross_fade_overlap_size": 0.05,
        "silent_threshold": -60,
        "vad_enabled": True,
        "pitch": 0,
        "index_rate": 0.0,
        "volume_envelope": 1.0,
        "protect": 0.33,
        "terms_accepted": False,
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                realtime_config = config.get("realtime", {})
                for k in defaults.keys():
                    if k in realtime_config:
                        defaults[k] = realtime_config[k]
                return defaults
    except Exception as e:
        print(f"Error loading realtime settings: {e}")

    return defaults


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


def _cleanup_realtime():
    global running, callbacks, audio_manager
    running = False
    if audio_manager is not None:
        try:
            audio_manager.stop()
        except Exception:
            pass
    if callbacks is not None:
        if hasattr(callbacks, "vc"):
            try:
                callbacks.vc.stop()
            except Exception:
                pass
    audio_manager = None
    callbacks = None


def start_realtime(
    input_audio_device: str,
    input_audio_gain: int,
    input_asio_channels: int,
    output_audio_device: str,
    output_audio_gain: int,
    output_asio_channels: int,
    asio_output_stereo: bool,
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
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    record_audio: bool = False,
    record_audio_path: str = None,
    export_format: str = "WAV",
):
    global running, callbacks, audio_manager, callbacks_kwargs

    if not input_audio_device or not output_audio_device:
        _cleanup_realtime()
        yield (
            i18n("Please select valid input/output devices!"),
            interactive_true,
            interactive_false,
        )
        return
    if use_monitor_device and not monitor_output_device:
        _cleanup_realtime()
        yield (
            i18n("Please select a valid monitor device!"),
            interactive_true,
            interactive_false,
        )
        return
    if not pth_path:
        _cleanup_realtime()
        yield (
            i18n("Model path not provided. Aborting conversion."),
            interactive_true,
            interactive_false,
        )
        return

    running = True

    print(f"Starting Realtime...")
    yield i18n("Starting Realtime..."), interactive_false, interactive_visible

    sid = int(sid) if sid is not None else 0

    save_realtime_settings(
        input_device=input_audio_device,
        output_device=output_audio_device,
        monitor_device=monitor_output_device,
        model_file=pth_path,
        index_file=index_path,
        use_monitor_device=use_monitor_device,
        input_audio_gain=int(input_audio_gain),
        output_audio_gain=int(output_audio_gain),
        monitor_audio_gain=int(monitor_audio_gain),
        vad_enabled=vad_enabled,
        chunk_size=chunk_size,
        cross_fade_overlap_size=cross_fade_overlap_size,
        extra_convert_size=extra_convert_size,
        silent_threshold=silent_threshold,
        pitch=pitch,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
    )

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
        _cleanup_realtime()
        print(f"Error: incorrectly formatted audio device.")
        yield (
            i18n("Incorrectly formatted audio device. Stopping."),
            interactive_true,
            interactive_false,
        )
        return

    # Load ASIO and sample rate settings from config.
    _rt_cfg = load_realtime_settings()
    asio_enabled = _rt_cfg["asio_enabled"]
    audio_sample_rate = _rt_cfg["audio_sample_rate"]
    audio_sample_rate = resolve_sample_rate(
        input_device_id, asio_enabled, audio_sample_rate
    )
    block_frame = int(chunk_size * audio_sample_rate / 1000)

    callbacks_kwargs = {
        "pass_through": PASS_THROUGH,
        "block_frame": block_frame,
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
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "record_audio": record_audio,
        "record_audio_path": record_audio_path,
        "export_format": export_format,
        "audio_sample_rate": audio_sample_rate,
    }

    callbacks = AudioCallbacks(**callbacks_kwargs)

    audio_manager = callbacks.audio
    try:
        audio_manager.start(
            input_device_id=input_device_id,
            output_device_id=output_device_id,
            output_monitor_id=output_monitor_id,
            exclusive_mode=exclusive_mode,
            asio_input_channel=input_asio_channels if asio_enabled else -1,
            asio_output_channel=output_asio_channels if asio_enabled else -1,
            asio_output_monitor_channel=monitor_asio_channels if asio_enabled else -1,
            block_frame=block_frame,
            audio_sample_rate=audio_sample_rate,
            asio_output_stereo=asio_output_stereo,
        )
    except Exception as error:
        _cleanup_realtime()
        print(f"Realtime error: {error}")
        yield i18n("Error: {}").format(error), interactive_true, interactive_false
        return

    print(f"Loading model...")
    yield "Loading model...", interactive_false, interactive_visible

    # Wait for the worker process to finish loading the model
    load_start = time.time()
    last_report = 0
    while running and callbacks is not None:
        time.sleep(0.1)
        if (
            hasattr(callbacks.vc, "_process")
            and callbacks.vc._process is not None
            and not callbacks.vc._process.is_alive()
        ):
            print(f"Worker process died during model loading.")
            _cleanup_realtime()
            yield (
                "Worker process crashed during model loading.",
                interactive_true,
                interactive_false,
            )
            return
        if getattr(callbacks.vc, "ready", False):
            break
        if time.time() - load_start > 300:
            print(f"Model loading timed out.")
            _cleanup_realtime()
            yield (
                "Model loading timed out.",
                interactive_true,
                interactive_false,
            )
            return
        elapsed = int(time.time() - load_start)
        if elapsed > last_report:
            last_report = elapsed
            print(f"Loading model... ({elapsed}s)")
            yield (
                f"Loading model... ({elapsed}s)",
                interactive_false,
                interactive_visible,
            )

    if not running or callbacks is None:
        _cleanup_realtime()
        return (
            i18n("Realtime stopped."),
            interactive_true,
            interactive_false,
        )

    print(f"Realtime is starting!")
    yield i18n("Realtime is starting!"), interactive_false, interactive_visible

    warmup_total = getattr(callbacks.vc.vc_model, "warmup_blocks", 0)

    while running and callbacks is not None and audio_manager is not None:
        time.sleep(0.1)
        # Snapshot global callbacks and audio_manager into local variables to prevent
        # race conditions where stop_realtime() concurrently clears them to None.
        cb = callbacks
        am = audio_manager
        if not running or cb is None or am is None:
            break
        if (
            hasattr(cb, "vc")
            and hasattr(cb.vc, "_process")
            and cb.vc._process is not None
            and not cb.vc._process.is_alive()
        ):
            print(f"Worker process died during realtime processing.")
            _cleanup_realtime()
            yield (
                "Worker process crashed.",
                interactive_true,
                interactive_false,
            )
            return
        if hasattr(am, "latency") and hasattr(am, "volume"):
            warmup_remaining = (
                getattr(cb.vc.vc_model, "warmup_blocks", 0)
                if hasattr(cb, "vc") and hasattr(cb.vc, "vc_model")
                else 0
            )

            if warmup_remaining > 0:
                bar = progress_str(warmup_remaining, warmup_total)
                yield (
                    i18n("Warming up... ({} blocks) {}").format(warmup_remaining, bar),
                    interactive_false,
                    interactive_true,
                )
            else:
                yield (
                    i18n("Latency: {:.2f} ms | Volume: {:.2f} dB").format(
                        am.latency, am.volume
                    ),
                    interactive_false,
                    interactive_true,
                )

    _cleanup_realtime()
    yield (
        i18n("Realtime stopped."),
        interactive_true,
        interactive_false,
    )


def change_callbacks_config():
    global callbacks

    if running and audio_manager is not None and callbacks is not None:
        # print(callbacks_kwargs)

        # It will need to create a new stream to work.
        # callbacks.vc.block_frame = callbacks_kwargs.get("block_frame", 192)
        crossfade_frame = int(
            callbacks_kwargs.get("cross_fade_overlap_size", 0.1) * AUDIO_SAMPLE_RATE
        )
        extra_frame = int(
            callbacks_kwargs.get("extra_convert_size", 0.5) * AUDIO_SAMPLE_RATE
        )

        cfg = {}

        if (
            callbacks.vc.crossfade_frame != crossfade_frame
            or callbacks.vc.extra_frame != extra_frame
        ):
            cfg["crossfade_frame"] = crossfade_frame
            cfg["extra_frame"] = extra_frame
            callbacks.vc.crossfade_frame = crossfade_frame
            callbacks.vc.extra_frame = extra_frame

        cfg["silent_threshold"] = callbacks_kwargs.get("silent_threshold", -90)

        cfg["vad_enabled"] = callbacks_kwargs.get("vad_enabled", True)
        cfg["vad_sensitivity"] = callbacks_kwargs.get("vad_sensitivity", 3)
        cfg["vad_frame_ms"] = callbacks_kwargs.get("vad_frame_ms", 30)

        cfg["formant_shifting"] = callbacks_kwargs.get("formant_shifting", False)
        cfg["formant_qfrency"] = callbacks_kwargs.get("formant_qfrency", 1.0)
        cfg["formant_timbre"] = callbacks_kwargs.get("formant_timbre", 1.0)

        cfg["model_path"] = callbacks_kwargs.get("model_path", "")
        cfg["sid"] = callbacks_kwargs.get("sid", 0)
        cfg["index_path"] = callbacks_kwargs.get("index_path", "")
        cfg["f0_method"] = callbacks_kwargs.get("f0_method", "")
        cfg["embedder_model"] = callbacks_kwargs.get("embedder_model", "")
        cfg["embedder_model_custom"] = callbacks_kwargs.get("embedder_model_custom", "")

        callbacks.vc.send_config(cfg)

        callbacks.audio.f0_up_key = callbacks_kwargs.get("f0_up_key", 0)
        callbacks.audio.index_rate = callbacks_kwargs.get("index_rate", 0.75)
        callbacks.audio.protect = callbacks_kwargs.get("protect", 0.5)
        callbacks.audio.volume_envelope = callbacks_kwargs.get("volume_envelope", 1)

        callbacks.audio.f0_autotune = callbacks_kwargs.get("f0_autotune", False)
        callbacks.audio.f0_autotune_strength = callbacks_kwargs.get(
            "f0_autotune_strength", 1.0
        )
        callbacks.audio.proposed_pitch = callbacks_kwargs.get("proposed_pitch", False)
        callbacks.audio.proposed_pitch_threshold = callbacks_kwargs.get(
            "proposed_pitch_threshold", 155.0
        )

        callbacks.audio.input_audio_gain = callbacks_kwargs.get("input_audio_gain", 1.0)
        callbacks.audio.output_audio_gain = callbacks_kwargs.get(
            "output_audio_gain", 1.0
        )
        callbacks.audio.monitor_audio_gain = callbacks_kwargs.get(
            "monitor_audio_gain", 1.0
        )


def change_config(value, key, if_kwargs=False):
    global callbacks_kwargs

    if running and audio_manager is not None and callbacks is not None:
        if if_kwargs and value is not None:
            callbacks_kwargs.setdefault("kwargs", {})[key] = value
        elif value is not None:
            callbacks_kwargs[key] = value

        change_callbacks_config()


def stop_realtime():
    global running, callbacks, audio_manager
    if running and audio_manager is not None and callbacks is not None:
        _cleanup_realtime()
        time.sleep(0.1)

        print(f"Realtime stopped.")
        return (
            i18n("Realtime stopped."),
            interactive_true,
            interactive_false,
        )
    else:
        _cleanup_realtime()
        return i18n("Realtime pipeline not found!"), interactive_true, interactive_false


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
        return {}, {}


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
        return [
            gr.update(interactive=True, visible=True),
            gr.update(interactive=False, visible=False),
        ]

    return [
        gr.update(
            interactive=data.get("start_button", True),
            visible=data.get("start_button", True),
        ),
        gr.update(
            interactive=data.get("stop_button", False),
            visible=data.get("stop_button", False),
        ),
    ]


def update_value_from_json(data):
    if not data:
        return [gr.update(), gr.update(value=None)]

    return [
        data.get("button", "Start"),
        data.get("path", None),
    ]


def soundfile_record_audio(
    record_button, record_audio_path: str = None, export_format: str = "WAV"
):
    global callbacks

    if running and audio_manager is not None and callbacks is not None:
        if record_button == "Start":
            gr.Info(i18n("Start recording..."))

            if not record_audio_path:
                record_audio_path = os.path.join(
                    now_dir, "assets", "audios", "record_audio.wav"
                )

            callbacks.vc.send_config(
                {
                    "record_start": True,
                    "record_audio_path": record_audio_path,
                    "export_format": export_format,
                }
            )

            return "Stop", None
        else:
            gr.Info(i18n("Stop recording!"))

            callbacks.vc.send_config({"record_stop": True})

            return "Start", record_audio_path

    gr.Warning(i18n("Realtime pipeline not found!"))
    return "Start", None


def realtime_tab():
    input_devices, output_devices = [], []
    saved_settings = load_realtime_settings()

    if not client_mode:
        input_devices, output_devices = get_audio_devices_formatted()
        input_devices, output_devices = list(input_devices.keys()), list(
            output_devices.keys()
        )
    else:
        input_devices = (
            [saved_settings["client_input_device"]]
            if saved_settings["client_input_device"]
            else []
        )
        output_devices = (
            [saved_settings["client_output_device"]]
            if saved_settings["client_output_device"]
            else []
        )

    # Load saved settings

    with gr.Blocks() as ui:
        with step_card(1, "Audio Devices", i18n, description="Choose input, output and optional monitor devices for live voice."):
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
                                saved_settings[
                                    (
                                        "client_input_device"
                                        if client_mode
                                        else "input_device"
                                    )
                                ],
                                input_devices,
                            ),
                            interactive=True,
                        )
                        input_audio_gain = gr.Slider(
                            minimum=0,
                            maximum=200,
                            value=saved_settings.get("input_audio_gain", 100),
                            label=i18n("Input Gain (%)"),
                            info=i18n(
                                "Adjusts the input volume before processing. Prevents clipping or boosts a quiet mic."
                            ),
                            interactive=True,
                        )
                        input_asio_channels = gr.Slider(
                            minimum=0,
                            maximum=16,
                            value=0,
                            step=1,
                            label=i18n("Input ASIO Channel"),
                            interactive=True,
                            visible=saved_settings["asio_enabled"],
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
                                saved_settings[
                                    (
                                        "client_output_device"
                                        if client_mode
                                        else "output_device"
                                    )
                                ],
                                output_devices,
                            ),
                            interactive=True,
                        )
                        output_audio_gain = gr.Slider(
                            minimum=0,
                            maximum=200,
                            value=saved_settings.get("output_audio_gain", 100),
                            label=i18n("Output Gain (%)"),
                            info=i18n(
                                "Adjusts the final volume of the converted voice after processing."
                            ),
                            interactive=True,
                        )
                        output_asio_channels = gr.Slider(
                            minimum=0,
                            maximum=16,
                            value=0,
                            step=1,
                            label=i18n("Output ASIO Channel"),
                            interactive=True,
                            visible=saved_settings["asio_enabled"],
                        )
                        asio_output_stereo = gr.Checkbox(
                            label=i18n("Stereo"),
                            value=True,
                            interactive=True,
                            visible=saved_settings["asio_enabled"],
                        )
            with gr.Accordion(i18n("Monitor Device (Optional)"), open=False):
                with gr.Column():
                    use_monitor_device = gr.Checkbox(
                        label=i18n("Use Monitor Device"),
                        value=saved_settings.get("use_monitor_device", False),
                        interactive=True,
                    )
                    monitor_output_device = gr.Dropdown(
                        label=i18n("Monitor Device"),
                        info=i18n(
                            "Select the device for monitoring your voice (e.g., your headphones)."
                        ),
                        choices=output_devices,
                        value=get_safe_dropdown_value(
                            saved_settings[
                                (
                                    "client_monitor_device"
                                    if client_mode
                                    else "monitor_device"
                                )
                            ],
                            output_devices,
                        ),
                        interactive=True,
                    )
                    monitor_audio_gain = gr.Slider(
                        minimum=0,
                        maximum=200,
                        value=saved_settings.get("monitor_audio_gain", 100),
                        label=i18n("Monitor Gain (%)"),
                        info=i18n(
                            "Adjusts the volume of the monitor feed, independent of the main output."
                        ),
                        interactive=True,
                    )
                    monitor_asio_channels = gr.Slider(
                        minimum=0,
                        maximum=16,
                        value=0,
                        step=1,
                        label=i18n("Monitor ASIO Channel"),
                        interactive=True,
                        visible=saved_settings["asio_enabled"],
                    )
            with gr.Accordion(i18n("Record Audio (Optional)"), open=False):
                with gr.Column():
                    record_audio_path = gr.Textbox(
                        label=i18n("Output Path"),
                        placeholder=i18n("Enter output path"),
                        info=i18n(
                            "The path where the output audio will be saved, by default in assets/audios/output.wav"
                        ),
                        value=os.path.join(
                            now_dir, "assets", "audios", "output.wav"
                        ),
                        interactive=True,
                    )
                export_format = gr.Dropdown(
                    filterable=False,
                    label=i18n("Export Format"),
                    info=i18n("Select the format to export the audio."),
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                record_audio = gr.Button(i18n("Start"))
                record_output = gr.Audio(label=i18n("Export Audio"))
            with gr.Row():
                exclusive_mode = gr.Checkbox(
                    label=i18n("Exclusive Mode"),
                    info=i18n(
                        "For WASAPI (Windows), gives the app exclusive control for potentially lower latency."
                        if not client_mode
                        else "Gives the app exclusive control for potentially lower latency."
                    ),
                    value=False,
                    interactive=True,
                    visible=client_mode,  # This is quite bad because it's prone to errors when used locally.
                )
                vad_enabled = gr.Checkbox(
                    label=i18n("Enable VAD"),
                    info=i18n(
                        "Enables Voice Activity Detection to only process audio when you are speaking, saving CPU."
                    ),
                    value=saved_settings.get("vad_enabled", True),
                    interactive=True,
                )
        with step_card(2, "Voice Model", i18n, description="Select the voice model and index used for realtime conversion."):
            with gr.Row():
                model_choices = (
                    sorted(names, key=extract_model_and_epoch) if names else []
                )
                model_file = gr.Dropdown(
                    label=i18n("Voice Model"),
                    choices=path_choices(model_choices),
                    interactive=True,
                    value=get_safe_dropdown_value(
                        saved_settings["model_file"], model_choices, default_weight
                    ),
                    allow_custom_value=True,
                )
                index_choices = sorted(get_files("index"))
                index_file = gr.Dropdown(
                    label=i18n("Index File"),
                    choices=path_choices(index_choices),
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
                refresh_button = gr.Button(i18n("Refresh models and indexes"))
        with step_card(3, "Conversion Parameters", i18n, description="Tune pitch, voice matching and pitch algorithms."):
            pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                label=i18n("Pitch"),
                info=i18n(
                    "Set the pitch of the audio, the higher the value, the higher the pitch."
                ),
                value=saved_settings.get("pitch", 0),
                interactive=True,
            )
            with gr.Accordion(i18n("Advanced Settings"), open=False):
                with gr.Accordion(i18n("Voice Matching"), open=True):
                    index_rate = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Search Feature Ratio"),
                        info=i18n(
                            "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                        ),
                        value=saved_settings.get("index_rate", 0),
                        interactive=True,
                    )
                    volume_envelope = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=saved_settings.get("volume_envelope", 1),
                        label=i18n("Volume Envelope"),
                        info=i18n(
                            "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                        ),
                        interactive=True,
                    )
                    protect = gr.Slider(
                        minimum=0,
                        maximum=0.5,
                        value=saved_settings.get("protect", 0.33),
                        label=i18n("Protect Voiceless Consonants"),
                        info=i18n(
                            "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                        ),
                        interactive=True,
                    )
                with gr.Accordion(i18n("Models & Algorithms"), open=False):
                    sid = gr.Dropdown(
                        label=i18n("Speaker ID"),
                        choices=(
                            get_speakers_id(default_weight) if default_weight else [0]
                        ),
                        value=0,
                        interactive=True,
                    )
                    f0_method = gr.Dropdown(
                        filterable=False,
                        choices=["rmvpe", "fcpe", "crepe", "swift"],
                        value="fcpe",
                        label=i18n("Pitch extraction algorithm"),
                        info=i18n(
                            "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                        ),
                        interactive=True,
                    )
                    embedder_model = gr.Dropdown(
                        filterable=False,
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
                                    choices=path_choices(refresh_embedders_folders()),
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
                with gr.Accordion(i18n("Advanced Pitch & Tuning"), open=False):
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
                with gr.Accordion(i18n("Formant Shifting"), open=False):
                    formant_shifting = gr.Checkbox(
                        label=i18n("Formant Shifting"),
                        info=i18n(
                            "Enable formant shifting. Used for male to female and vice-versa convertions."
                        ),
                        visible=True,
                        value=False,
                        interactive=True,
                    )
                    formant_qfrency = gr.Slider(
                        minimum=0,
                        maximum=16,
                        step=0.1,
                        label=i18n("Quefrency for formant shifting"),
                        info=i18n("Modifies formants width and spectral envelope."),
                        visible=False,
                        value=1.0,
                        interactive=True,
                    )
                    formant_timbre = gr.Slider(
                        minimum=0,
                        maximum=16,
                        step=0.1,
                        label=i18n("Timbre for formant shifting"),
                        info=i18n("Adjusts vocal tract length / timbre brightness."),
                        visible=False,
                        value=1.0,
                        interactive=True,
                    )
        with step_card(4, "Performance Settings", i18n, description="Balance latency, CPU usage and silence detection."):
            with gr.Accordion(i18n("Advanced Settings"), open=False):
                with gr.Accordion(i18n("Buffer & Latency"), open=True):
                    chunk_size = gr.Slider(
                        minimum=2.7,
                        maximum=2730.7,
                        value=saved_settings.get("chunk_size", 250),
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
                        value=saved_settings.get("cross_fade_overlap_size", 0.05),
                        step=0.01,
                        label=i18n("Crossfade Overlap Size (s)"),
                        info=i18n(
                            "Duration of the fade between audio chunks to prevent clicks. Higher values create smoother transitions but may increase latency."
                        ),
                        interactive=True,
                    )
                with gr.Accordion(i18n("Context & Detection"), open=False):
                    extra_convert_size = gr.Slider(
                        minimum=0.1,
                        maximum=5,
                        value=saved_settings.get("extra_convert_size", 2.5),
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
                        value=saved_settings.get("silent_threshold", -60),
                        step=1,
                        label=i18n("Silence Threshold (dB)"),
                        info=i18n(
                            "Volume level below which audio is treated as silence and not processed. Helps to save CPU resources and reduce background noise."
                        ),
                        interactive=True,
                    )
        with step_card(5, "Start Realtime", i18n, description="Agree to the terms and start or stop the live session."):
            terms_checkbox = gr.Checkbox(
                label=i18n("I agree to the terms of use"),
                info=i18n(
                    "Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your realtime."
                ),
                value=saved_settings.get("terms_accepted", False),
                interactive=True,
            )
            with gr.Row():
                start_button = gr.Button(i18n("Start"), variant="primary")
                stop_button = gr.Button(i18n("Stop"), interactive=False, visible=False)
            latency_info = gr.Label(
                label=i18n("Status"),
                value=i18n("Realtime not started."),
                elem_id="realtime-status-info",
            )

        json_audio_hidden = gr.JSON(visible=False)
        json_button_hidden = gr.JSON(visible=False)
        json_value_hidden = gr.JSON(visible=False)

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = i18n("You must agree to the Terms of Use to proceed.")
                gr.Info(message)
                yield message, interactive_true, interactive_false
                return
            yield from start_realtime(*args)

        def _realtime_with_toast(terms_accepted, *args):
            if not terms_accepted:
                # enforce_terms explains the rejection with its own toast.
                yield from enforce_terms(terms_accepted, *args)
                return
            gr.Info(i18n("Starting real-time conversion..."))
            # start_realtime reports failures as yielded status strings
            # instead of raising, so warn about them for screen reader
            # users. The status text is translated, so failures are
            # detected structurally instead of by matching English words:
            # every failure yield re-enables the start button, while
            # progress updates keep it disabled.
            try:
                for update in enforce_terms(terms_accepted, *args):
                    status = update[0] if isinstance(update, tuple) else update
                    if (
                        isinstance(update, tuple)
                        and len(update) > 1
                        and update[1] is interactive_true
                    ):
                        gr.Warning(status)
                    yield update
            except Exception:
                gr.Warning(
                    i18n(
                        "An error occurred during real-time conversion. Please check the console logs for more details."
                    )
                )
                raise

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
                choices=path_choices(new_index_choices), value=safe_index_value
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
                outputs=[
                    input_audio_device,
                    output_audio_device,
                    monitor_output_device,
                ],
            )

            json_button_hidden.change(
                fn=update_button_from_json,
                inputs=[json_button_hidden],
                outputs=[start_button, stop_button],
            )

            json_value_hidden.change(
                fn=update_value_from_json,
                inputs=[json_value_hidden],
                outputs=[record_audio, record_output],
            )
        else:
            refresh_devices_button.click(
                fn=refresh_devices,
                outputs=[
                    input_audio_device,
                    output_audio_device,
                    monitor_output_device,
                ],
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
            fn=lambda: gr.update(choices=path_choices(refresh_embedders_folders())),
            inputs=[],
            outputs=[embedder_model_custom],
        )

        def update_visibility(checkbox, count):
            return [gr.update(visible=checkbox) for _ in range(count)]


        def formant_visible(checkbox):
            return update_visibility(checkbox, 2)

        formant_shifting.change(
            fn=formant_visible,
            inputs=[formant_shifting],
            outputs=[formant_qfrency, formant_timbre],
        )


        if client_mode:

            def client_start_handler(
                input_device,
                input_gain,
                output_device,
                output_gain,
                monitor_device,
                monitor_gain,
                use_monitor,
                vad,
                chunk,
                cross_fade,
                extra_convert,
                silent_thresh,
                pitch_val,
                index_r,
                vol_env,
                prot,
                model,
                index,
                terms,
            ):
                save_realtime_settings(
                    input_device=input_device,
                    output_device=output_device,
                    monitor_device=monitor_device,
                    model_file=model,
                    index_file=index,
                    use_monitor_device=use_monitor,
                    input_audio_gain=int(input_gain) if input_gain is not None else 100,
                    output_audio_gain=(
                        int(output_gain) if output_gain is not None else 100
                    ),
                    monitor_audio_gain=(
                        int(monitor_gain) if monitor_gain is not None else 100
                    ),
                    vad_enabled=vad,
                    chunk_size=chunk,
                    cross_fade_overlap_size=cross_fade,
                    extra_convert_size=extra_convert,
                    silent_threshold=silent_thresh,
                    pitch=pitch_val,
                    index_rate=index_r,
                    volume_envelope=vol_env,
                    protect=prot,
                    terms_accepted=terms,
                )
                return [interactive_false, interactive_visible]

            start_button.click(
                fn=client_start_handler,
                inputs=[
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
                    model_file,
                    index_file,
                    terms_checkbox,
                ],
                outputs=[start_button, stop_button],
            )

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
                    formant_shifting,
                    formant_qfrency,
                    formant_timbre,
                ],
                outputs=[json_button_hidden],
            )

            stop_button.click(
                fn=None, js="StopAudioStream", outputs=[json_button_hidden]
            )

            record_audio.click(
                fn=None,
                js="SoundfileRecordAudio",
                inputs=[record_audio, record_audio_path, export_format],
                outputs=[json_value_hidden],
            )
        else:
            start_button.click(
                fn=_realtime_with_toast,
                inputs=[
                    terms_checkbox,
                    input_audio_device,
                    input_audio_gain,
                    input_asio_channels,
                    output_audio_device,
                    output_audio_gain,
                    output_asio_channels,
                    asio_output_stereo,
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
                    formant_shifting,
                    formant_qfrency,
                    formant_timbre,
                ],
                outputs=[latency_info, start_button, stop_button],
            )

            stop_button.click(
                fn=stop_realtime, outputs=[latency_info, start_button, stop_button]
            )  # .then(
            #     fn=lambda: (
            #         yield gr.update(value="Stopped"),
            #         interactive_true,
            #         interactive_false,
            #     ),
            #     inputs=None,
            #     outputs=[latency_info, start_button, stop_button],
            # )

            record_audio.click(
                fn=soundfile_record_audio,
                inputs=[record_audio, record_audio_path, export_format],
                outputs=[record_audio, record_output],
            )

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
        terms_checkbox.change(
            fn=lambda val: save_realtime_settings(terms_accepted=val),
            inputs=[terms_checkbox],
            outputs=[],
        )

        input_audio_device.change(
            fn=save_realtime_settings,
            inputs=[
                input_audio_device,
                output_audio_device,
                monitor_output_device,
                model_file,
                index_file,
            ],
            outputs=[],
        )

        output_audio_device.change(
            fn=save_realtime_settings,
            inputs=[
                input_audio_device,
                output_audio_device,
                monitor_output_device,
                model_file,
                index_file,
            ],
            outputs=[],
        )

        monitor_output_device.change(
            fn=save_realtime_settings,
            inputs=[
                input_audio_device,
                output_audio_device,
                monitor_output_device,
                model_file,
                index_file,
            ],
            outputs=[],
        )

        model_file.change(
            fn=save_realtime_settings,
            inputs=[
                input_audio_device,
                output_audio_device,
                monitor_output_device,
                model_file,
                index_file,
            ],
            outputs=[],
        )

        index_file.change(
            fn=save_realtime_settings,
            inputs=[
                input_audio_device,
                output_audio_device,
                monitor_output_device,
                model_file,
                index_file,
            ],
            outputs=[],
        )

        if client_mode:

            def refresh_all():
                new_names = get_files("model")
                new_indexes = sorted(get_files("index"))
                return (
                    gr.update(
                        choices=path_choices(
                            sorted(new_names, key=extract_model_and_epoch)
                        )
                    ),
                    gr.update(choices=path_choices(new_indexes)),
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
                new_indexes = sorted(get_files("index"))
                input_choices, output_choices = get_audio_devices_formatted()
                input_choices, output_choices = list(input_choices.keys()), list(
                    output_choices.keys()
                )
                return (
                    gr.update(
                        choices=path_choices(
                            sorted(new_names, key=extract_model_and_epoch)
                        )
                    ),
                    gr.update(choices=path_choices(new_indexes)),
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

        model_file.change(
            js=(
                "(value) => window.ChangeConfig(value, 'model_path')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "model_path"))
                if not client_mode
                else None
            ),
            inputs=[model_file],
            outputs=[],
        )
        index_file.change(
            js=(
                "(value) => window.ChangeConfig(value, 'index_path')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "index_path"))
                if not client_mode
                else None
            ),
            inputs=[index_file],
            outputs=[],
        )
        f0_method.change(
            js=(
                "(value) => window.ChangeConfig(value, 'f0_method')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "f0_method"))
                if not client_mode
                else None
            ),
            inputs=[f0_method],
            outputs=[],
        )
        embedder_model.change(
            js=(
                "(value) => window.ChangeConfig(value, 'embedder_model')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "embedder_model"))
                if not client_mode
                else None
            ),
            inputs=[embedder_model],
            outputs=[],
        )
        embedder_model_custom.change(
            js=(
                "(value) => window.ChangeConfig(value, 'embedder_model_custom')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "embedder_model_custom"))
                if not client_mode
                else None
            ),
            inputs=[embedder_model_custom],
            outputs=[],
        )

        input_audio_gain.change(
            js=(
                "(value) => window.ChangeConfig(value, 'input_audio_gain')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value / 100.0, "input_audio_gain"))
                if not client_mode
                else None
            ),
            inputs=[input_audio_gain],
            outputs=[],
        )
        output_audio_gain.change(
            js=(
                "(value) => window.ChangeConfig(value, 'output_audio_gain')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value / 100.0, "output_audio_gain"))
                if not client_mode
                else None
            ),
            inputs=[output_audio_gain],
            outputs=[],
        )
        monitor_audio_gain.change(
            js=(
                "(value) => window.ChangeConfig(value, 'monitor_audio_gain')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value / 100.0, "monitor_audio_gain"))
                if not client_mode
                else None
            ),
            inputs=[monitor_audio_gain],
            outputs=[],
        )
        vad_enabled.change(
            js=(
                "(value) => window.ChangeConfig(value, 'vad_enabled')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "vad_enabled"))
                if not client_mode
                else None
            ),
            inputs=[vad_enabled],
            outputs=[],
        )
        cross_fade_overlap_size.change(
            js=(
                "(value) => window.ChangeConfig(value, 'cross_fade_overlap_size')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "cross_fade_overlap_size"))
                if not client_mode
                else None
            ),
            inputs=[cross_fade_overlap_size],
            outputs=[],
        )
        extra_convert_size.change(
            js=(
                "(value) => window.ChangeConfig(value, 'extra_convert_size')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "extra_convert_size"))
                if not client_mode
                else None
            ),
            inputs=[extra_convert_size],
            outputs=[],
        )
        silent_threshold.change(
            js=(
                "(value) => window.ChangeConfig(value, 'silent_threshold')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "silent_threshold"))
                if not client_mode
                else None
            ),
            inputs=[silent_threshold],
            outputs=[],
        )

        # chunk_size.change(fn=lambda value: change_config(int(value * AUDIO_SAMPLE_RATE / 1000) if not client_mode else None, "block_frame"), inputs=[chunk_size], outputs=[])

        pitch.change(
            js=(
                "(value) => window.ChangeConfig(value, 'f0_up_key')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "f0_up_key"))
                if not client_mode
                else None
            ),
            inputs=[pitch],
            outputs=[],
        )
        index_rate.change(
            js=(
                "(value) => window.ChangeConfig(value, 'index_rate')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "index_rate"))
                if not client_mode
                else None
            ),
            inputs=[index_rate],
            outputs=[],
        )
        volume_envelope.change(
            js=(
                "(value) => window.ChangeConfig(value, 'volume_envelope')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "volume_envelope"))
                if not client_mode
                else None
            ),
            inputs=[volume_envelope],
            outputs=[],
        )
        protect.change(
            js=(
                "(value) => window.ChangeConfig(value, 'protect')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "protect"))
                if not client_mode
                else None
            ),
            inputs=[protect],
            outputs=[],
        )

        autotune.change(
            js=(
                "(value) => window.ChangeConfig(value, 'autotune')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "autotune"))
                if not client_mode
                else None
            ),
            inputs=[autotune],
            outputs=[],
        )
        autotune_strength.change(
            js=(
                "(value) => window.ChangeConfig(value, 'autotune_strength')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "autotune_strength"))
                if not client_mode
                else None
            ),
            inputs=[autotune_strength],
            outputs=[],
        )
        proposed_pitch.change(
            js=(
                "(value) => window.ChangeConfig(value, 'proposed_pitch')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "proposed_pitch"))
                if not client_mode
                else None
            ),
            inputs=[proposed_pitch],
            outputs=[],
        )
        proposed_pitch_threshold.change(
            js=(
                "(value) => window.ChangeConfig(value, 'proposed_pitch_threshold')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "proposed_pitch_threshold"))
                if not client_mode
                else None
            ),
            inputs=[proposed_pitch_threshold],
            outputs=[],
        )
        formant_shifting.change(
            js=(
                "(value) => window.ChangeConfig(value, 'formant_shifting')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "formant_shifting"))
                if not client_mode
                else None
            ),
            inputs=[formant_shifting],
            outputs=[],
        )
        formant_qfrency.change(
            js=(
                "(value) => window.ChangeConfig(value, 'formant_qfrency')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "formant_qfrency"))
                if not client_mode
                else None
            ),
            inputs=[formant_qfrency],
            outputs=[],
        )
        formant_timbre.change(
            js=(
                "(value) => window.ChangeConfig(value, 'formant_timbre')"
                if client_mode
                else None
            ),
            fn=(
                (lambda value: change_config(value, "formant_timbre"))
                if not client_mode
                else None
            ),
            inputs=[formant_timbre],
            outputs=[],
        )

