import sys

sys.path.append("..")
import os

now_dir = os.getcwd()
from rvc.train.process_ckpt import (
    extract_small_model,
)

from rvc.lib.process.model_fusion import model_fusion
from rvc.lib.process.model_information import (
    model_information,
)

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

import gradio as gr


def processing():
    with gr.Accordion(label=i18n("Model fusion (On progress)"), open=False):
        with gr.Column():
            model_fusion_name = gr.Textbox(
                label=i18n("Model Name"),
                value="",
                max_lines=1,
                interactive=True,
                placeholder=i18n("Enter model name"),
            )
            model_fusion_a = gr.Textbox(
                label=i18n("Path to Model A"),
                value="",
                interactive=True,
                placeholder=i18n("Path to model"),
            )
            model_fusion_b = gr.Textbox(
                label=i18n("Path to Model B"),
                value="",
                interactive=True,
                placeholder=i18n("Path to model"),
            )
        model_fusion_output_info = gr.Textbox(
            label=i18n("Output Information"),
            value="",
        )

        model_fusion_button = gr.Button(
            i18n("Fusion"), variant="primary", interactive=False
        )

        model_fusion_button.click(
            model_fusion,
            [
                model_fusion_name,
                model_fusion_a,
                model_fusion_b,
            ],
            model_fusion_output_info,
            api_name="model_fusion",
        )

    with gr.Accordion(label=i18n("View model information")):
        with gr.Row():
            with gr.Column():
                model_view_model_path = gr.Textbox(
                    label=i18n("Path to Model"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Path to model"),
                )

        model_view_output_info = gr.Textbox(
            label=i18n("Output Information"), value="", max_lines=8
        )
        model_view_button = gr.Button(i18n("View"), variant="primary")
        model_view_button.click(
            model_information,
            [model_view_model_path],
            model_view_output_info,
            api_name="model_info",
        )

    with gr.Accordion(label=i18n("Model extraction")):
        with gr.Row():
            with gr.Column():
                model_extract_name = gr.Textbox(
                    label=i18n("Model Name"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Enter model name"),
                )
                model_extract_path = gr.Textbox(
                    label=i18n("Path to Model"),
                    placeholder=i18n("Path to model"),
                    interactive=True,
                )
                model_extract_info = gr.Textbox(
                    label=i18n("Model information to be placed"),
                    value="",
                    max_lines=8,
                    interactive=True,
                    placeholder=i18n("Model information to be placed"),
                )
            with gr.Column():
                model_extract_pitch_guidance = gr.Checkbox(
                    label=i18n("Pitch Guidance"),
                    value=True,
                    interactive=True,
                )
                model_extract_rvc_version = gr.Radio(
                    label=i18n("RVC Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                )
                model_extract_sampling_rate = gr.Radio(
                    label=i18n("Sampling Rate"),
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    interactive=True,
                )
        model_extract_output_info = gr.Textbox(
            label=i18n("Output Information"), value="", max_lines=8
        )

        model_extract_button = gr.Button(i18n("Extract"), variant="primary")
        model_extract_button.click(
            extract_small_model,
            [
                model_extract_path,
                model_extract_name,
                model_extract_sampling_rate,
                model_extract_pitch_guidance,
                model_extract_info,
                model_extract_rvc_version,
            ],
            model_extract_output_info,
            api_name="model_extract",
        )
