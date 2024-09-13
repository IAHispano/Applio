import os, sys

now_dir = os.getcwd()
sys.path.append(now_dir)

import gradio as gr
from assets.i18n.i18n import I18nAuto

from core import run_model_author_script

i18n = I18nAuto()


def model_author_tab():
    model_author_name = gr.Textbox(
        label=i18n("Model Author Name"),
        info=i18n("The name that will appear in the model information."),
        placeholder=i18n("Enter your nickname"),
        interactive=True,
    )
    model_author_output_info = gr.Textbox(
        label=i18n("Output Information"),
        info=i18n("The output information will be displayed here."),
        value="",
        max_lines=1,
    )
    button = gr.Button(i18n("Set name"), variant="primary")

    button.click(
        fn=run_model_author_script,
        inputs=[model_author_name],
        outputs=[model_author_output_info],
    )
