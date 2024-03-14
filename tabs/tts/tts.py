import os, sys
import gradio as gr
import regex as re
import json
import random

from core import (
    run_tts_script,
)

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
model_root_relative = os.path.relpath(model_root, now_dir)

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if (
        file.endswith((".pth", ".onnx"))
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]


def change_choices():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".onnx"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]

    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]
    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
    )


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def process_input(file_path):
    with open(file_path, "r") as file:
        file_contents = file.read()
    gr.Info(f"The text from the txt file has been loaded!")
    return file_contents, None


def match_index(model_file_value):
    if model_file_value:
        model_folder = os.path.dirname(model_file_value)
        index_files = get_indexes()
        for index_file in index_files:
            if os.path.dirname(index_file) == model_folder:
                return index_file
    return ""


def tts_tab():
    default_weight = random.choice(names) if names else ""
    with gr.Row():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model to use for the conversion."),
                choices=sorted(names, key=lambda path: os.path.getsize(path)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            best_default_index_path = match_index(model_file.value)
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the index file to use for the conversion."),
                choices=get_indexes(),
                value=best_default_index_path,
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Column():
            refresh_button = gr.Button(i18n("Refresh"))
            unload_button = gr.Button(i18n("Unload Voice"))

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

    json_path = os.path.join("rvc", "lib", "tools", "tts_voices.json")
    with open(json_path, "r") as file:
        tts_voices_data = json.load(file)

    short_names = [voice.get("ShortName", "") for voice in tts_voices_data]

    tts_voice = gr.Dropdown(
        label=i18n("TTS Voices"),
        info=i18n("Select the TTS voice to use for the conversion."),
        choices=short_names,
        interactive=True,
        value=None,
    )

    tts_text = gr.Textbox(
        label=i18n("Text to Synthesize"),
        info=i18n("Enter the text to synthesize."),
        placeholder=i18n("Enter text to synthesize"),
        lines=3,
    )

    txt_file = gr.File(
        label=i18n("Or you can upload a .txt file"),
        type="filepath",
    )

    with gr.Accordion(i18n("Advanced Settings"), open=False):
        with gr.Column():
            output_tts_path = gr.Textbox(
                label=i18n("Output Path for TTS Audio"),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_output.wav"),
                interactive=True,
            )
            output_rvc_path = gr.Textbox(
                label=i18n("Output Path for RVC Audio"),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav"),
                interactive=True,
            )
            export_format = gr.Radio(
                label=i18n("Export Format"),
                info=i18n("Select the format to export the audio."),
                choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                value="WAV",
                interactive=True,
            )
            split_audio = gr.Checkbox(
                label=i18n("Split Audio"),
                info=i18n(
                    "Split the audio into chunks for inference to obtain better results in some cases."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            autotune = gr.Checkbox(
                label=i18n("Autotune"),
                info=i18n(
                    "Apply a soft autotune to your inferences, recommended for singing conversions."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            clean_audio = gr.Checkbox(
                label=i18n("Clean Audio"),
                info=i18n(
                    "Clean your audio output using noise detection algorithms, recommended for speaking audios."
                ),
                visible=True,
                value=True,
                interactive=True,
            )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Clean Strength"),
                info=i18n(
                    "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                ),
                visible=True,
                value=0.5,
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
            filter_radius = gr.Slider(
                minimum=0,
                maximum=7,
                label=i18n("Filter Radius"),
                info=i18n(
                    "If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration."
                ),
                value=3,
                step=1,
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
            hop_length = gr.Slider(
                minimum=1,
                maximum=512,
                step=1,
                label=i18n("Hop Length"),
                info=i18n(
                    "Denotes the duration it takes for the system to transition to a significant pitch change. Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy."
                ),
                value=128,
                interactive=True,
            )
            with gr.Column():
                f0method = gr.Radio(
                    label=i18n("Pitch extraction algorithm"),
                    info=i18n(
                        "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                    ),
                    choices=[
                        "pm",
                        "harvest",
                        "dio",
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                        "hybrid[rmvpe+fcpe]",
                    ],
                    value="rmvpe",
                    interactive=True,
                )

    convert_button1 = gr.Button(i18n("Convert"))

    with gr.Row():  # Defines output info + output audio download after conversion
        vc_output1 = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
        )
        vc_output2 = gr.Audio(label=i18n("Export Audio"))

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[],
        outputs=[model_file, index_file],
    )
    txt_file.upload(
        fn=process_input,
        inputs=[txt_file],
        outputs=[tts_text, txt_file],
    )
    convert_button1.click(
        fn=run_tts_script,
        inputs=[
            tts_text,
            tts_voice,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            hop_length,
            f0method,
            output_tts_path,
            output_rvc_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            clean_audio,
            clean_strength,
            export_format,
        ],
        outputs=[vc_output1, vc_output2],
    )
