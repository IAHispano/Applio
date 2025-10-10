import os
import sys
import json
import gradio as gr
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()
config_file = os.path.join(now_dir, "assets", "config.json")

filter_trigger = None


def get_filter_trigger():
    global filter_trigger
    if filter_trigger is None:
        filter_trigger = gr.Textbox(visible=False)
    return filter_trigger


def load_config_filter():
    with open(config_file, "r", encoding="utf8") as f:
        cfg = json.load(f)
    return bool(cfg.get("model_index_filter", False))


def save_config_filter(val: bool):
    with open(config_file, "r", encoding="utf8") as f:
        cfg = json.load(f)
    cfg["model_index_filter"] = bool(val)
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)


def filter_tab():
    checkbox = gr.Checkbox(
        label=i18n("Enable model/index list filter"),
        info=i18n(
            "Adds a keyword filter for the model/index selection lists in the Inference and TTS tabs."
        ),
        value=load_config_filter(),
        interactive=True,
    )
    checkbox.change(fn=save_config_filter, inputs=[checkbox], outputs=[])
    return checkbox
