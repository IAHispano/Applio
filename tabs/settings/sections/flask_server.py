import os
import sys
import gradio as gr
from assets.i18n.i18n import I18nAuto
import requests

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.flask.server import start_flask, load_config_flask, save_config

i18n = I18nAuto()


def flask_server_tab():
    with gr.Row():
        with gr.Column():
            flask_checkbox = gr.Checkbox(
                label=i18n(
                    "Enable Applio integration with applio.org/models using flask"
                ),
                info=i18n(
                    "It will activate the possibility of downloading models with a click from the website."
                ),
                interactive=True,
                value=load_config_flask(),
            )
            flask_checkbox.change(
                fn=toggle,
                inputs=[flask_checkbox],
                outputs=[],
            )


def toggle(checkbox):
    save_config(bool(checkbox))
    if load_config_flask() == True:
        start_flask()
    else:
        try:
            requests.post("http://localhost:8000/shutdown")
        except requests.exceptions.ConnectionError:
            pass
