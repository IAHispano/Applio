import gradio as gr
import os
import sys


def restart_applio():
    if os.name == "nt":
        os.system("cls")
    gr.Info("Applio has been restarted please press enter in the console to continue")
    python = sys.executable
    os.execl(python, python, *sys.argv)


from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def restart_tab():
    with gr.Row():
        with gr.Column():
            restart_button = gr.Button(i18n("Restart Applio"))
            restart_button.click(
                fn=restart_applio,
                inputs=[],
                outputs=[],
            )
