import gradio as gr
from core import run_model_information_script

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def model_information_tab():
    with gr.Column():
        model_name = gr.Textbox(
            label=i18n("Path to Model"),
            info=i18n("Introduce the model pth path"),
            placeholder=i18n("Introduce the model pth path"),
            interactive=True,
        )
        model_information_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=12,
            interactive=False,
        )
        model_information_button = gr.Button(i18n("See Model Information"))
        model_information_button.click(
            fn=run_model_information_script,
            inputs=[model_name],
            outputs=[model_information_output_info],
        )
