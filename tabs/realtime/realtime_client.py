import gradio as gr
import os
import sys
import regex as re
import shutil
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

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


def update_dropdowns_from_json(data):
    if not data:
        return [
            gr.update(choices=[], value=None), 
            gr.update(choices=[], value=None), 
            gr.update(choices=[], value=None)
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
        gr.update(interactive=data.get("stop_button", False))
    ]

def realtime_tab():
    with gr.Blocks() as ui:
        with gr.Row():
            start_button = gr.Button(i18n("Start"), variant="primary")
            stop_button = gr.Button(i18n("Stop"), interactive=False)
        gr.Label(label=i18n("Status"), value="Realtime not started.", elem_id="realtime-status-info")
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
                                choices=[],
                                value=None,
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
                    with gr.Accordion("Output Device", open=True):
                        with gr.Column():
                            output_audio_device = gr.Dropdown(
                                label=i18n("Output Device"),
                                info=i18n(
                                    "Select the device where the final converted voice will be sent (e.g., a virtual cable)."
                                ),
                                choices=[],
                                value=None,
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
                with gr.Accordion("Monitor Device (Optional)", open=False):
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
                            choices=[],
                            value=None,
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
                with gr.Row():
                    exclusive_mode = gr.Checkbox(
                        label=i18n("Exclusive Mode"),
                        info=i18n(
                            "Gives the app exclusive control for potentially lower latency."
                        ),
                        value=True,
                        interactive=True,
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
                        value=default_weight,
                        allow_custom_value=True,
                    )
                    index_choices = get_files("index")
                    index_file = gr.Dropdown(
                        label=i18n("Index File"),
                        choices=index_choices,
                        value=match_index(default_weight) if default_weight else None,
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
                        value=0.75,
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
                    value=512,
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

        def update_on_model_change(model_path):
            new_index = match_index(model_path)
            new_sids = get_speakers_id(model_path)

            # Get updated index choices
            new_index_choices = get_files("index")
            # Use the matched index as fallback, but handle empty strings
            return gr.update(
                choices=new_index_choices, value=new_index
            ), gr.update(choices=new_sids, value=0 if new_sids else None)

        def toggle_visible(checkbox):
            return {"visible": checkbox, "__type__": "update"}

        def toggle_visible_embedder_custom(embedder_model):
            if embedder_model == "custom":
                return {"visible": True, "__type__": "update"}
            return {"visible": False, "__type__": "update"}

        refresh_devices_button.click(
            fn=None,
            js="getAudioDevices",
            outputs=[json_audio_hidden],
        )

        json_audio_hidden.change(
            fn=update_dropdowns_from_json,
            inputs=[json_audio_hidden],
            outputs=[input_audio_device, output_audio_device, monitor_output_device]
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
                clean_strength
            ],
            outputs=[json_button_hidden],
        )

        stop_button.click(
            fn=None, js="StopAudioStream", outputs=[json_button_hidden]
        )

        json_button_hidden.change(
            fn=update_button_from_json,
            inputs=[json_button_hidden],
            outputs=[start_button, stop_button]
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
