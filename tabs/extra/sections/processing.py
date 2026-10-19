import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_model_information_script
from assets.i18n.i18n import I18nAuto
from assets.app_shell import step_card

i18n = I18nAuto()


def processing_tab():
    def _model_info_with_toast(pth_path):
        result = run_model_information_script(pth_path)
        gr.Info(i18n("Model information loaded."))
        return result

    with step_card(1, "Select Model", i18n, description="Point to the model you want to inspect."):
        model_view_model_path = gr.Textbox(
            label=i18n("Path to Model"),
            info=i18n("Introduce the model pth path"),
            value="",
            interactive=True,
            placeholder=i18n("Enter path to model"),
        )

    with step_card(2, "Model Information", i18n, description="Load the metadata and training details."):
        model_view_output_info = gr.Textbox(
            label=i18n("Model information output"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=11,
        )
        model_view_button = gr.Button(i18n("View"))
        model_view_button.click(
            fn=_model_info_with_toast,
            inputs=[model_view_model_path],
            outputs=[model_view_output_info],
        )
