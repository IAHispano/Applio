import os
import sys
import json

now_dir = os.getcwd()
sys.path.append(now_dir)

import gradio as gr
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def set_precision(precision: str):
    with open(os.path.join(now_dir, "assets", "config.json"), "r") as f:
        config = json.load(f)

    config["precision"] = precision

    with open(os.path.join(now_dir, "assets", "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Precision set to {precision}.")
    return f"Precision set to {precision}."


def get_precision():
    with open(os.path.join(now_dir, "assets", "config.json"), "r") as f:
        config = json.load(f)

    return config["precision"] if "precision" in config else None


def precision_tab():
    precision = gr.Radio(
        label=i18n("Precision"),
        info=i18n("Select the precision you want to use for training and inference."),
        value=get_precision(),
        choices=["fp32", "fp16", "bf16"],
        interactive=True,
    )
    precision_info = gr.Textbox(
        label=i18n("Output Information"),
        info=i18n("The output information will be displayed here."),
        value="",
        max_lines=1,
    )
    button = gr.Button(i18n("Update precision"))

    button.click(
        fn=set_precision,
        inputs=[precision],
        outputs=[precision_info],
    )
