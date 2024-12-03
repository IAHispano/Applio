import gradio as gr

from rvc.configs.config import Config

config = Config()

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def precision_tab():
    with gr.Row():
        with gr.Column():

            precision = gr.Radio(
                label=i18n("Precision"),
                info=i18n(
                    "Select the precision you want to use for training and inference."
                ),
                choices=[
                    "fp16",
                    "fp32",
                ],
                value=config.get_precision(),
                interactive=True,
            )
            precision_output = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
                value="",
                max_lines=8,
                interactive=False,
            )

            update_button = gr.Button(i18n("Update precision"))
            update_button.click(
                fn=config.set_precision,
                inputs=[precision],
                outputs=[precision_output],
            )
