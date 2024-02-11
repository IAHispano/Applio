import gradio as gr
import os
import sys

def restart_applio():
    if os.name == "nt": 
        os.system("cls")
    python = sys.executable
    os.execl(python, python, *sys.argv)

from assets.version_checker import compare_version
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def version_tab():
    with gr.Row():
        with gr.Column():
            version_check = gr.Textbox(
                label=i18n("Version checker"),
                interactive=False,
            )
            version_button = gr.Button(i18n("Check for updates"))
            version_button.click(
                fn=compare_version,
                inputs=[],
                outputs=[version_check],
            )
            restart_button = gr.Button(i18n("Restart Applio"))
            restart_button.click(
                fn=restart_applio,
                inputs=[],
                outputs=[],
            )
