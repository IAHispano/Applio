import datetime
import json
import os
import shutil
import sys
import traceback

import gradio as gr
import regex as re
import torch

from assets.i18n.i18n import I18nAuto
from assets.app_shell import step_header
from core import run_batch_infer_script, run_infer_script
from rvc.lib.utils import format_title
from tabs.settings.sections.filter import get_filter_trigger, load_config_filter
from tabs.settings.sections.restart import stop_infer

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")
custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)

PRESETS_DIR = os.path.join(now_dir, "assets", "presets")
FORMANTSHIFT_DIR = os.path.join(now_dir, "assets", "formant_shift")

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

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


def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/").lower()


# BASE model/index folder names for many latin languages (legacy: zips = models)
MODEL_FOLDER = re.compile(r"^(?:model.{0,4}|mdl(?:s)?|weight.{0,4}|zip(?:s)?)$")
INDEX_FOLDER = re.compile(r"^(?:ind.{0,4}|idx(?:s)?)$")


def is_mdl_alias(name: str) -> bool:
    return bool(MODEL_FOLDER.match(name))


def is_idx_alias(name: str) -> bool:
    return bool(INDEX_FOLDER.match(name))


def alias_score(path: str, want_model: bool) -> int:
    """
    Handles duplicate files, compare file type to path and assign a score:
    2 = Path contains correct alias  (e.g., model file in 'modelos/' folder)
    1 = Path contains opposite alias (e.g., model file in 'index/' folder)
    0 = Path contains no recognized aliases
    """
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

    for root, directories, files in os.walk(model_root_relative, followlinks=True):
        directories[:] = [
            name
            for name in directories
            if name.lower() != ".onnx-cache"
            and not name.lower().startswith(".onnx-export-")
        ]
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


default_weight = next(iter(get_files("model")), None)

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext))
    and root == audio_root_relative
    and "_output" not in name
]

custom_embedders = [
    os.path.join(dirpath, dirname)
    for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
    for dirname in dirnames
]


def update_sliders(preset):
    with open(
        os.path.join(PRESETS_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["pitch"],
        values["index_rate"],
        values["rms_mix_rate"],
        values["protect"],
        values.get("cleanup_mode", "off"),
        values.get("cleanup_strength", 12),
    )


def update_sliders_formant(preset):
    with open(
        os.path.join(FORMANTSHIFT_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["formant_qfrency"],
        values["formant_timbre"],
    )


def export_presets(presets, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(presets, json_file, ensure_ascii=False, indent=4)


def import_presets(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        presets = json.load(json_file)
    return presets


def get_presets_data(
    pitch, index_rate, rms_mix_rate, protect, cleanup_mode="off", cleanup_strength=12
):
    return {
        "pitch": pitch,
        "index_rate": index_rate,
        "rms_mix_rate": rms_mix_rate,
        "protect": protect,
        "cleanup_mode": cleanup_mode,
        "cleanup_strength": cleanup_strength,
    }


def export_presets_button(
    preset_name,
    pitch,
    index_rate,
    rms_mix_rate,
    protect,
    cleanup_mode="off",
    cleanup_strength=12,
):
    if preset_name:
        file_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        presets_data = get_presets_data(
            pitch, index_rate, rms_mix_rate, protect, cleanup_mode, cleanup_strength
        )
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(presets_data, json_file, ensure_ascii=False, indent=4)
        return "Export successful"
    return "Export cancelled"


def import_presets_button(file_path):
    if file_path:
        imported_presets = import_presets(file_path.name)
        return (
            list(imported_presets.keys()),
            imported_presets,
            "Presets imported successfully!",
        )
    return [], {}, "No file selected for import."


def list_json_files(directory):
    return [f.rsplit(".", 1)[0] for f in os.listdir(directory) if f.endswith(".json")]


def refresh_presets():
    json_files = list_json_files(PRESETS_DIR)
    return gr.update(choices=json_files)


def output_path_fn(input_audio_path):
    original_name_without_extension = os.path.basename(input_audio_path).rsplit(".", 1)[
        0
    ]
    new_name = original_name_without_extension + "_output.wav"
    output_path = os.path.join(os.path.dirname(input_audio_path), new_name)
    return output_path


def change_choices(model):
    if model:
        speakers = get_speakers_id(model)
    else:
        speakers = [0]

    models_list = sorted(get_files("model"))
    indexes_list = sorted(get_files("index"))

    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and root == audio_root_relative
        and "_output" not in name
    ]

    return (
        {"choices": path_choices(models_list), "__type__": "update"},
        {"choices": path_choices(indexes_list), "__type__": "update"},
        {"choices": path_choices(sorted(audio_paths)), "__type__": "update"},
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
    )


def extract_model_and_epoch(path):
    base_name = os.path.basename(path)
    match = re.match(r"(.+?)_(\d+)e_", base_name)
    if match:
        model, epoch = match.groups()
        return model, int(epoch)
    return "", 0


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(audio_root_relative, os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path, output_path_fn(target_path)


def save_to_wav2(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    shutil.copy(file_path, target_path)
    return target_path, output_path_fn(target_path)


def delete_outputs():
    gr.Info(i18n("Outputs cleared!"))
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and name.__contains__("_output"):
                os.remove(os.path.join(root, name))


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


def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)

    normalize_pathd_target_folder = os.path.abspath(target_folder)
    normalize_pathd_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalize_pathd_target_folder.startswith(
        normalize_pathd_custom_embedder_root
    ):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))
    if config_file:
        shutil.copy(
            config_file, os.path.join(target_folder, os.path.basename(config_file))
        )

    return f"Files moved to folder {target_folder}"


def refresh_formant():
    json_files = list_json_files(FORMANTSHIFT_DIR)
    return gr.update(choices=json_files)


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders


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


def filter_dropdowns(filter_text):
    ft = (filter_text or "").lower()
    all_models = sorted(get_files("model"), key=extract_model_and_epoch)
    all_indexes = sorted(get_files("index"))
    filtered_models = [m for m in all_models if ft in m.lower()]
    filtered_indexes = [i for i in all_indexes if ft in i.lower()]
    return (
        gr.update(choices=path_choices(filtered_models)),
        gr.update(choices=path_choices(filtered_indexes)),
    )


def update_filter_visibility(_):
    en = load_config_filter()
    if not en:
        box = gr.update(visible=False, value="")
        m_upd, i_upd = filter_dropdowns("")
        return box, m_upd, i_upd
    return gr.update(visible=True), gr.skip(), gr.skip()


# Inference tab
def inference_tab():
    trigger = get_filter_trigger()
    with gr.Column(scale=0, min_width=0, elem_id="applio-conversion-top"):
        with gr.Column(scale=0, min_width=0, elem_classes="applio-card", elem_id="applio-voice-model"):
            step_header(1, "Voice Model", i18n, description="Select the voice model and index file used for conversion.")
            with gr.Row(elem_id="applio-model-selectors"):
                with gr.Column(scale=1, min_width=240):
                    model_file = gr.Dropdown(
                        label=i18n("Voice Model"),
                        info=i18n("Select the voice model to use for the conversion."),
                        choices=path_choices(
                            sorted(get_files("model"), key=extract_model_and_epoch)
                        ),
                        value=default_weight,
                        interactive=True,
                        allow_custom_value=True,
                    )
                with gr.Column(scale=1, min_width=240):
                    index_file = gr.Dropdown(
                        label=i18n("Index File"),
                        info=i18n("Select the index file to use for the conversion."),
                        choices=path_choices(sorted(get_files("index"))),
                        value=match_index(default_weight),
                        interactive=True,
                        allow_custom_value=True,
                    )
            filter_box_inf = gr.Textbox(
                label=i18n("Filter"),
                info=i18n("Path must contain:"),
                placeholder=i18n("Type to filter..."),
                interactive=True,
                scale=0,
                visible=load_config_filter(),
            )
            filter_box_inf.blur(
                fn=filter_dropdowns,
                inputs=[filter_box_inf],
                outputs=[model_file, index_file],
            )
            trigger.change(
                fn=update_filter_visibility,
                inputs=[trigger],
                outputs=[filter_box_inf, model_file, index_file],
                show_progress=False,
            )
            with gr.Row(elem_id="applio-model-actions"):
                unload_button = gr.Button(i18n("Unload Voice"), size="sm", min_width=100)
                refresh_button = gr.Button(
                    i18n("Refresh models and indexes"), size="sm", min_width=100
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
                    fn=lambda model_file_value: match_index(model_file_value),
                    inputs=[model_file],
                    outputs=[index_file],
                )
        with gr.Column(scale=0, min_width=0, elem_classes="applio-card"):
            step_header(2, "Audio Source", i18n, description="Upload a single file or point to a folder for batch conversion.")
            with gr.Tabs(selected="single"):
                with gr.Tab(i18n("Single"), id="single") as single_source:
                    with gr.Column():
                        upload_audio = gr.Audio(
                            label=i18n("Upload Audio"), type="filepath", editable=False
                        )
                        with gr.Row():
                            audio = gr.Dropdown(
                                label=i18n("Select Audio"),
                                info=i18n("Select the audio to convert."),
                                choices=path_choices(sorted(audio_paths)),
                                value=audio_paths[0] if audio_paths else "",
                                interactive=True,
                                allow_custom_value=True,
                            )
                with gr.Tab(i18n("Batch"), id="batch") as batch_source:
                    with gr.Row():
                        with gr.Column():
                            input_folder_batch = gr.Textbox(
                                label=i18n("Input Folder"),
                                info=i18n("Select the folder containing the audios to convert."),
                                placeholder=i18n("Enter input path"),
                                value=os.path.join(now_dir, "assets", "audios"),
                                interactive=True,
                            )
                            output_folder_batch = gr.Textbox(
                                label=i18n("Output Folder"),
                                info=i18n(
                                    "Select the folder where the output audios will be saved."
                                ),
                                placeholder=i18n("Enter output path"),
                                value=os.path.join(now_dir, "assets", "audios"),
                                interactive=True,
                            )

    with gr.Column(scale=0, min_width=0, elem_classes="applio-card", elem_id="applio-parameters-card"):
        step_header(3, "Conversion Parameters", i18n, description="Tune pitch, voice matching, algorithms and output settings.")
        with gr.Tabs(selected="single", elem_id="applio-conversion-parameters") as parameters:
            with gr.Tab(i18n("Single"), id="single"):
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
                with gr.Accordion(i18n("Advanced Settings"), open=False):
                    with gr.Accordion(i18n("Voice Matching"), open=True):
                        index_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Search Feature Ratio"),
                            info=i18n(
                                "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                            ),
                            value=0.75,
                            interactive=True,
                        )
                        rms_mix_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Volume Envelope"),
                            info=i18n(
                                "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                            ),
                            value=1,
                            interactive=True,
                        )
                        protect = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n("Protect Voiceless Consonants"),
                            info=i18n(
                                "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                            ),
                            value=0.5,
                            interactive=True,
                        )
                    with gr.Accordion(i18n("Models & Algorithms"), open=False):
                        sid = gr.Dropdown(
                            label=i18n("Speaker ID"),
                            info=i18n("Select the speaker ID to use for the conversion."),
                            choices=get_speakers_id(model_file.value),
                            value=0,
                            interactive=True,
                        )
                        f0_method = gr.Radio(
                            label=i18n("Pitch extraction algorithm"),
                            info=i18n(
                                "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                            ),
                            choices=[
                                "crepe",
                                "rmvpe",
                                "fcpe",
                                "swift",
                            ],
                            value="rmvpe",
                            interactive=True,
                        )
                        embedder_model = gr.Radio(
                            label=i18n("Embedder Model"),
                            info=i18n("Model used for learning speaker embedding."),
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
                    with gr.Accordion(i18n("Audio cleanup"), open=False):
                        cleanup_mode = gr.Dropdown(
                            label=i18n("Audio cleanup"),
                            choices=[
                                (i18n("Off"), "off"),
                                (i18n("Light cleanup (FFmpeg)"), "light"),
                                (i18n("Speech enhancement (DeepFilterNet3)"), "speech"),
                            ],
                            value="off",
                            interactive=True,
                            info=i18n(
                                "Cleans input before conversion. Speech mode downloads about 35 MB on first use and may suppress singing. Processing stays local."
                            ),
                        )
                        cleanup_strength = gr.Slider(
                            minimum=1,
                            maximum=30,
                            step=1,
                            value=12,
                            label=i18n("Noise reduction limit (dB)"),
                            interactive=True,
                            info=i18n(
                                "Higher values remove more noise and can change the voice."
                            ),
                        )
                    with gr.Accordion(i18n("Formant Shifting"), open=False):
                        formant_shifting = gr.Checkbox(
                            label=i18n("Formant Shifting"),
                            info=i18n(
                                "Enable formant shifting. Used for male to female and vice-versa convertions."
                            ),
                            value=False,
                            visible=True,
                            interactive=True,
                        )
                        with gr.Row(visible=False) as formant_row:
                            formant_preset = gr.Dropdown(
                                label=i18n("Browse presets for formanting"),
                                info=i18n(
                                    "Presets are located in /assets/formant_shift folder"
                                ),
                                choices=list_json_files(FORMANTSHIFT_DIR),
                                visible=False,
                                interactive=True,
                            )
                            formant_refresh_button = gr.Button(
                                value=i18n("Refresh formant presets"),
                                visible=False,
                            )
                        formant_qfrency = gr.Slider(
                            value=1.0,
                            info=i18n("Default value is 1.0"),
                            label=i18n("Quefrency for formant shifting"),
                            minimum=0.0,
                            maximum=16.0,
                            step=0.1,
                            visible=False,
                            interactive=True,
                        )
                        formant_timbre = gr.Slider(
                            value=1.0,
                            info=i18n("Default value is 1.0"),
                            label=i18n("Timbre for formant shifting"),
                            minimum=0.0,
                            maximum=16.0,
                            step=0.1,
                            visible=False,
                            interactive=True,
                        )
                    with gr.Accordion(i18n("Preset Settings"), open=False):
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label=i18n("Select Custom Preset"),
                                choices=list_json_files(PRESETS_DIR),
                                interactive=True,
                            )
                            presets_refresh_button = gr.Button(i18n("Refresh Presets"))
                        import_file = gr.File(
                            label=i18n("Select file to import"),
                            file_count="single",
                            type="filepath",
                            interactive=True,
                        )
                        import_file.change(
                            import_presets_button,
                            inputs=import_file,
                            outputs=[preset_dropdown],
                        )
                        presets_refresh_button.click(
                            refresh_presets, outputs=preset_dropdown
                        )
                        with gr.Row():
                            preset_name_input = gr.Textbox(
                                label=i18n("Preset Name"),
                                placeholder=i18n("Enter preset name"),
                            )
                            export_button = gr.Button(i18n("Export Preset"))
                    with gr.Accordion(i18n("Output Settings"), open=False):
                        output_path = gr.Textbox(
                            label=i18n("Output Path"),
                            placeholder=i18n("Enter output path"),
                            info=i18n(
                                "The path where the output audio will be saved, by default in assets/audios/output.wav"
                            ),
                            value=(
                                output_path_fn(audio_paths[0])
                                if audio_paths
                                else os.path.join(now_dir, "assets", "audios", "output.wav")
                            ),
                            interactive=True,
                        )
                        export_format = gr.Radio(
                            label=i18n("Export Format"),
                            info=i18n("Select the format to export the audio."),
                            choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                            value="WAV",
                            interactive=True,
                        )
                        clear_outputs_infer = gr.Button(
                            i18n("Clear Outputs (Deletes all audios in assets/audios)")
                        )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch,
                        index_rate,
                        rms_mix_rate,
                        protect,
                        cleanup_mode,
                        cleanup_strength,
                    ],
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        index_rate,
                        rms_mix_rate,
                        protect,
                        cleanup_mode,
                        cleanup_strength,
                    ],
                )

                def enforce_terms(terms_accepted, *args):
                    if not terms_accepted:
                        message = i18n("You must agree to the Terms of Use to proceed.")
                        gr.Info(message)
                        return message, None
                    try:
                        gr.Info(i18n("Converting audio..."))
                        result = run_infer_script(
                            *args[:-2], cleanup_mode=args[-2], cleanup_strength=args[-1]
                        )
                        gr.Info(result[0])
                        return result
                    except Exception:
                        traceback.print_exc()
                        gr.Warning(
                            i18n(
                                "An error occurred during audio conversion. Please check the console logs for more details."
                            )
                        )
                        return (
                            i18n(
                                "An error occurred during audio conversion. Please check the console logs for more details."
                            ),
                            None,
                        )

                def enforce_terms_batch(terms_accepted, *args):
                    if not terms_accepted:
                        message = i18n("You must agree to the Terms of Use to proceed.")
                        gr.Info(message)
                        return message
                    try:
                        return run_batch_infer_script(
                            *args[:-2], cleanup_mode=args[-2], cleanup_strength=args[-1]
                        )
                    except Exception:
                        traceback.print_exc()
                        return i18n(
                            "An error occurred during audio batch conversion. Please check the console logs for more details."
                        )

                gr.Markdown(value=i18n("## Conversion"))

                terms_checkbox = gr.Checkbox(
                    label=i18n("I agree to the terms of use"),
                    info=i18n(
                        "Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your inference."
                    ),
                    value=False,
                    interactive=True,
                )

                convert_button1 = gr.Button(i18n("Convert"), variant="primary")

                with gr.Row(equal_height=True, elem_classes="applio-output-row"):
                    vc_output1 = gr.Textbox(
                        label=i18n("Conversion output"),
                        info=i18n("The output information will be displayed here."),
                        lines=3,
                    )
                    vc_output2 = gr.Audio(label=i18n("Export Audio"))

            # Batch inference tab
            with gr.Tab(i18n("Batch"), id="batch"):
                pitch_batch = gr.Slider(
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
                with gr.Accordion(i18n("Advanced Settings"), open=False):
                    with gr.Accordion(i18n("Voice Matching"), open=True):
                        index_rate_batch = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Search Feature Ratio"),
                            info=i18n(
                                "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                            ),
                            value=0.75,
                            interactive=True,
                        )
                        rms_mix_rate_batch = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Volume Envelope"),
                            info=i18n(
                                "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                            ),
                            value=1,
                            interactive=True,
                        )
                        protect_batch = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n("Protect Voiceless Consonants"),
                            info=i18n(
                                "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                            ),
                            value=0.5,
                            interactive=True,
                        )
                    with gr.Accordion(i18n("Models & Algorithms"), open=False):
                        sid_batch = gr.Dropdown(
                            label=i18n("Speaker ID"),
                            info=i18n("Select the speaker ID to use for the conversion."),
                            choices=get_speakers_id(model_file.value),
                            value=0,
                            interactive=True,
                        )
                        f0_method_batch = gr.Radio(
                            label=i18n("Pitch extraction algorithm"),
                            info=i18n(
                                "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                            ),
                            choices=[
                                "crepe",
                                "rmvpe",
                                "fcpe",
                                "swift",
                            ],
                            value="rmvpe",
                            interactive=True,
                        )
                        embedder_model_batch = gr.Radio(
                            label=i18n("Embedder Model"),
                            info=i18n("Model used for learning speaker embedding."),
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
                            interactive=True,
                        )
                        with gr.Column(visible=False) as embedder_custom_batch:
                            with gr.Accordion(i18n("Custom Embedder"), open=True):
                                with gr.Row():
                                    embedder_model_custom_batch = gr.Dropdown(
                                        label=i18n("Select Custom Embedder"),
                                        choices=path_choices(refresh_embedders_folders()),
                                        interactive=True,
                                        allow_custom_value=True,
                                    )
                                    refresh_embedders_button_batch = gr.Button(
                                        i18n("Refresh embedders")
                                    )
                                folder_name_input_batch = gr.Textbox(
                                    label=i18n("Folder Name"), interactive=True
                                )
                                with gr.Row():
                                    bin_file_upload_batch = gr.File(
                                        label=i18n("Upload .bin"),
                                        type="filepath",
                                        interactive=True,
                                    )
                                    config_file_upload_batch = gr.File(
                                        label=i18n("Upload .json"),
                                        type="filepath",
                                        interactive=True,
                                    )
                                move_files_button_batch = gr.Button(
                                    i18n("Move files to custom embedder folder")
                                )
                    with gr.Accordion(i18n("Advanced Pitch & Tuning"), open=False):
                        autotune_batch = gr.Checkbox(
                            label=i18n("Autotune"),
                            info=i18n(
                                "Apply a soft autotune to your inferences, recommended for singing conversions."
                            ),
                            visible=True,
                            value=False,
                            interactive=True,
                        )
                        autotune_strength_batch = gr.Slider(
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
                        proposed_pitch_batch = gr.Checkbox(
                            label=i18n("Proposed Pitch"),
                            info=i18n(
                                "Adjust the input audio pitch to match the voice model range."
                            ),
                            visible=True,
                            value=False,
                            interactive=True,
                        )
                        proposed_pitch_threshold_batch = gr.Slider(
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
                    with gr.Accordion(i18n("Audio cleanup"), open=False):
                        cleanup_mode_batch = gr.Dropdown(
                            label=i18n("Audio cleanup"),
                            choices=[
                                (i18n("Off"), "off"),
                                (i18n("Light cleanup (FFmpeg)"), "light"),
                                (i18n("Speech enhancement (DeepFilterNet3)"), "speech"),
                            ],
                            value="off",
                            interactive=True,
                            info=i18n(
                                "Cleans input before conversion. Speech mode downloads about 35 MB on first use and may suppress singing. Processing stays local."
                            ),
                        )
                        cleanup_strength_batch = gr.Slider(
                            minimum=1,
                            maximum=30,
                            step=1,
                            value=12,
                            label=i18n("Noise reduction limit (dB)"),
                            interactive=True,
                            info=i18n(
                                "Higher values remove more noise and can change the voice."
                            ),
                        )
                    with gr.Accordion(i18n("Formant Shifting"), open=False):
                        formant_shifting_batch = gr.Checkbox(
                            label=i18n("Formant Shifting"),
                            info=i18n(
                                "Enable formant shifting. Used for male to female and vice-versa convertions."
                            ),
                            value=False,
                            visible=True,
                            interactive=True,
                        )
                        with gr.Row(visible=False) as formant_row_batch:
                            formant_preset_batch = gr.Dropdown(
                                label=i18n("Browse presets for formanting"),
                                info=i18n(
                                    "Presets are located in /assets/formant_shift folder"
                                ),
                                choices=list_json_files(FORMANTSHIFT_DIR),
                                visible=False,
                                interactive=True,
                            )
                            formant_refresh_button_batch = gr.Button(
                                value=i18n("Refresh formant presets"),
                                visible=False,
                            )
                        formant_qfrency_batch = gr.Slider(
                            value=1.0,
                            info=i18n("Default value is 1.0"),
                            label=i18n("Quefrency for formant shifting"),
                            minimum=0.0,
                            maximum=16.0,
                            step=0.1,
                            visible=False,
                            interactive=True,
                        )
                        formant_timbre_batch = gr.Slider(
                            value=1.0,
                            info=i18n("Default value is 1.0"),
                            label=i18n("Timbre for formant shifting"),
                            minimum=0.0,
                            maximum=16.0,
                            step=0.1,
                            visible=False,
                            interactive=True,
                        )
                    with gr.Accordion(i18n("Preset Settings"), open=False):
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label=i18n("Select Custom Preset"),
                                interactive=True,
                            )
                            presets_batch_refresh_button = gr.Button(
                                i18n("Refresh Presets")
                            )
                        import_file = gr.File(
                            label=i18n("Select file to import"),
                            file_count="single",
                            type="filepath",
                            interactive=True,
                        )
                        import_file.change(
                            import_presets_button,
                            inputs=import_file,
                            outputs=[preset_dropdown],
                        )
                        presets_batch_refresh_button.click(
                            refresh_presets, outputs=preset_dropdown
                        )
                        with gr.Row():
                            preset_name_input = gr.Textbox(
                                label=i18n("Preset Name"),
                                placeholder=i18n("Enter preset name"),
                            )
                            export_button = gr.Button(i18n("Export Preset"))
                    with gr.Accordion(i18n("Output Settings"), open=False):
                        export_format_batch = gr.Radio(
                            label=i18n("Export Format"),
                            info=i18n("Select the format to export the audio."),
                            choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                            value="WAV",
                            interactive=True,
                        )
                        clear_outputs_batch = gr.Button(
                            i18n("Clear Outputs (Deletes all audios in assets/audios)")
                        )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch_batch,
                        index_rate_batch,
                        rms_mix_rate_batch,
                        protect_batch,
                        cleanup_mode_batch,
                        cleanup_strength_batch,
                    ],
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch_batch,
                        index_rate_batch,
                        rms_mix_rate_batch,
                        protect_batch,
                        cleanup_mode_batch,
                        cleanup_strength_batch,
                    ],
                    outputs=[],
                )

                gr.Markdown(value=i18n("## Conversion"))

                terms_checkbox_batch = gr.Checkbox(
                    label=i18n("I agree to the terms of use"),
                    info=i18n(
                        "Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your inference."
                    ),
                    value=False,
                    interactive=True,
                )
                convert_button_batch = gr.Button(i18n("Convert"), variant="primary")
                stop_button = gr.Button(i18n("Stop convert"), visible=False)
                stop_button.click(fn=stop_infer, inputs=[], outputs=[])

                with gr.Row():
                    vc_output3 = gr.Textbox(
                        label=i18n("Batch conversion output"),
                        info=i18n("The output information will be displayed here."),
                    )

    single_source.select(lambda: gr.Tabs(selected="single"), outputs=parameters, queue=False)
    batch_source.select(lambda: gr.Tabs(selected="batch"), outputs=parameters, queue=False)

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

    def enable_stop_convert_button(terms_accepted):
        if not terms_accepted:
            return {"visible": True, "__type__": "update"}, {
                "visible": False,
                "__type__": "update",
            }
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }

    def disable_stop_convert_button():
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }

    def toggle_visible_formant_shifting(checkbox):
        if checkbox:
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
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
    proposed_pitch_batch.change(
        fn=toggle_visible,
        inputs=[proposed_pitch_batch],
        outputs=[proposed_pitch_threshold_batch],
    )
    formant_shifting.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row,
            formant_preset,
            formant_refresh_button,
            formant_qfrency,
            formant_timbre,
        ],
    )
    formant_shifting_batch.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting_batch],
        outputs=[
            formant_row_batch,
            formant_preset_batch,
            formant_refresh_button_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
        ],
    )
    formant_refresh_button.click(
        fn=refresh_formant,
        inputs=[],
        outputs=[formant_preset],
    )
    formant_preset.change(
        fn=update_sliders_formant,
        inputs=[formant_preset],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
    )
    formant_preset_batch.change(
        fn=update_sliders_formant,
        inputs=[formant_preset_batch],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
    )
    autotune_batch.change(
        fn=toggle_visible,
        inputs=[autotune_batch],
        outputs=[autotune_strength_batch],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, audio, sid, sid_batch],
    ).then(
        fn=filter_dropdowns,
        inputs=[filter_box_inf],
        outputs=[model_file, index_file],
    )
    audio.change(
        fn=output_path_fn,
        inputs=[audio],
        outputs=[output_path],
    )
    upload_audio.upload(
        fn=save_to_wav2,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    upload_audio.stop_recording(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    clear_outputs_infer.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    clear_outputs_batch.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    embedder_model.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model],
        outputs=[embedder_custom],
    )
    embedder_model_batch.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model_batch],
        outputs=[embedder_custom_batch],
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
    move_files_button_batch.click(
        fn=create_folder_and_move_files,
        inputs=[
            folder_name_input_batch,
            bin_file_upload_batch,
            config_file_upload_batch,
        ],
        outputs=[],
    )
    refresh_embedders_button_batch.click(
        fn=lambda: gr.update(choices=path_choices(refresh_embedders_folders())),
        inputs=[],
        outputs=[embedder_model_custom_batch],
    )
    convert_button1.click(
        fn=enforce_terms,
        inputs=[
            terms_checkbox,
            pitch,
            index_rate,
            rms_mix_rate,
            protect,
            f0_method,
            audio,
            output_path,
            model_file,
            index_file,
            autotune,
            autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
            export_format,
            embedder_model,
            embedder_model_custom,
            formant_shifting,
            formant_qfrency,
            formant_timbre,
            sid,
            cleanup_mode,
            cleanup_strength,
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button_batch.click(
        fn=enable_stop_convert_button,
        inputs=[terms_checkbox_batch],
        outputs=[convert_button_batch, stop_button],
    ).then(
        fn=enforce_terms_batch,
        inputs=[
            terms_checkbox_batch,
            pitch_batch,
            index_rate_batch,
            rms_mix_rate_batch,
            protect_batch,
            f0_method_batch,
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            autotune_batch,
            autotune_strength_batch,
            proposed_pitch_batch,
            proposed_pitch_threshold_batch,
            export_format_batch,
            embedder_model_batch,
            embedder_model_custom_batch,
            formant_shifting_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
            sid_batch,
            cleanup_mode_batch,
            cleanup_strength_batch,
        ],
        outputs=[vc_output3],
    ).then(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
    stop_button.click(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
