import sys

sys.path.append("..")
import os

now_dir = os.getcwd()
from lib.infer.infer_libs.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

import gradio as gr
import traceback


def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


def processing_():

        with gr.Accordion(
            label=i18n("Model fusion, can be used to test timbre fusion")
        ):
            with gr.Row():
                with gr.Column():
                    name_to_save0 = gr.Textbox(
                        label=i18n("Name:"),
                        value="",
                        max_lines=1,
                        interactive=True,
                        placeholder=i18n("Name for saving"),
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Weight for Model A:"),
                        value=0.5,
                        interactive=True,
                    )
                    if_f0_ = gr.Checkbox(
                        label=i18n("Whether the model has pitch guidance."),
                        value=True,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=i18n("Model architecture version:"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    sr_ = gr.Radio(
                        label=i18n("Target sample rate:"),
                        choices=["40k", "48k", "32k"],
                        value="40k",
                        interactive=True,
                    )
                with gr.Column():
                    ckpt_a = gr.Textbox(
                        label=i18n("Path to Model A:"),
                        value="",
                        interactive=True,
                        placeholder=i18n("Path to model"),
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("Path to Model B:"),
                        value="",
                        interactive=True,
                        placeholder=i18n("Path to model"),
                    )
                    info__ = gr.Textbox(
                        label=i18n("Model information to be placed:"),
                        value="",
                        max_lines=8,
                        interactive=True,
                        placeholder=i18n("Model information to be placed"),
                    )
                    info4 = gr.Textbox(
                        label=i18n("Output information:"), value="", max_lines=8
                    )

            but6 = gr.Button(i18n("Fusion"), variant="primary")

            but6.click(
                merge,
                [
                    ckpt_a,
                    ckpt_b,
                    alpha_a,
                    sr_,
                    if_f0_,
                    info__,
                    name_to_save0,
                    version_2,
                ],
                info4,
                api_name="ckpt_merge",
            )  # def merge(path1,path2,alpha1,sr,f0,info):

        with gr.Accordion(label=i18n("Modify model information")):
            with gr.Row():  ######
                with gr.Column():
                    ckpt_path0 = gr.Textbox(
                        label=i18n("Path to Model:"),
                        value="",
                        interactive=True,
                        placeholder=i18n("Path to model"),
                    )
                    info_ = gr.Textbox(
                        label=i18n("Model information to be modified:"),
                        value="",
                        max_lines=8,
                        interactive=True,
                        placeholder=i18n("Model information to be placed"),
                    )

                with gr.Column():
                    name_to_save1 = gr.Textbox(
                        label=i18n("Save file name:"),
                        placeholder=i18n("Name for saving"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )

                    info5 = gr.Textbox(
                        label=i18n("Output information:"), value="", max_lines=8
                    )
            but7 = gr.Button(i18n("Modify"), variant="primary")
            but7.click(change_info, [ckpt_path0, info_, name_to_save1], info5, api_name="ckpt_modify",)

        with gr.Accordion(label=i18n("View model information")):
            with gr.Row():
                with gr.Column():
                    ckpt_path1 = gr.Textbox(
                        label=i18n("Path to Model:"),
                        value="",
                        interactive=True,
                        placeholder=i18n("Path to model"),
                    )

                    info6 = gr.Textbox(
                        label=i18n("Output information:"), value="", max_lines=8
                    )
                    but8 = gr.Button(i18n("View"), variant="primary")
            but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")

        with gr.Accordion(label=i18n("Model extraction")):
            with gr.Row():
                with gr.Column():
                    save_name = gr.Textbox(
                        label=i18n("Name:"),
                        value="",
                        interactive=True,
                        placeholder=i18n("Name for saving"),
                    )
                    if_f0__ = gr.Checkbox(
                        label=i18n("Whether the model has pitch guidance."),
                        value=True,
                        interactive=True,
                    )
                    version_1 = gr.Radio(
                        label=i18n("Model architecture version:"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    sr__ = gr.Radio(
                        label=i18n("Target sample rate:"),
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )

                with gr.Column():
                    ckpt_path2 = gr.Textbox(
                        label=i18n("Path to Model:"),
                        placeholder=i18n("Path to model"),
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label=i18n("Model information to be placed:"),
                        value="",
                        max_lines=8,
                        interactive=True,
                        placeholder=i18n("Model information to be placed"),
                    )
                    info7 = gr.Textbox(
                        label=i18n("Output information:"), value="", max_lines=8
                    )

            with gr.Row():
                but9 = gr.Button(i18n("Extract"), variant="primary")
                ckpt_path2.change(
                    change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                )
            but9.click(
                extract_small_model,
                [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                info7,
                api_name="ckpt_extract",
            )
