import os, sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.train.process_ckpt import (
    extract_small_model,
)

from rvc.lib.process.model_information import (
    model_information,
)

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

import gradio as gr


def processing():
    with gr.Accordion(label=i18n("View model information")):
        with gr.Row():
            with gr.Column():
                model_view_model_path = gr.Textbox(
                    label=i18n("Path to Model"),
                    info=i18n("Introduce the model pth path"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Path to model"),
                )

        model_view_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info="The output information will be displayed here.",
            value="",
            max_lines=8,
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
                    info=i18n("Name of the new model."),
                    value="",
                    interactive=True,
                    placeholder=i18n("Enter model name"),
                )
                model_extract_path = gr.Textbox(
                    label=i18n("Path to Model"),
                    info=i18n("Introduce the model pth path"),
                    placeholder=i18n("Path to model"),
                    interactive=True,
                )
                model_extract_info = gr.Textbox(
                    label=i18n("Model information to be placed"),
                    info=i18n("The information to be placed in the model (You can leave it blank or put anything)."),
                    value="",
                    max_lines=8,
                    interactive=True,
                    placeholder=i18n("Model information to be placed"),
                )
            with gr.Column():

                model_extract_rvc_version = gr.Radio(
                    label=i18n("RVC Version"),
                    choices=["v1", "v2"],
                    info=i18n("The RVC version of the model."),
                    value="v2",
                    interactive=True,
                )
                model_extract_sampling_rate = gr.Radio(
                    label=i18n("Sampling Rate"),
                    choices=["32000", "40000", "48000"],
                    info=i18n(
                        "The sampling rate of the audio files."
                    ),
                    value="40000",
                    interactive=True,
                )
                model_extract_pitch_guidance = gr.Checkbox(
                    label=i18n("Pitch Guidance"),
                    info=i18n(
                        "By employing pitch guidance, it becomes feasible to mirror the intonation of the original voice, including its pitch. This feature is particularly valuable for singing and other scenarios where preserving the original melody or pitch pattern is essential."
                    ),
                    value=True,
                    interactive=True,
                )
        model_extract_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info="The output information will be displayed here.",
            value="",
            max_lines=8,
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
