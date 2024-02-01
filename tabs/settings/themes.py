import os
import sys
import base64
import pathlib
import tempfile
import gradio as gr

from assets.i18n.i18n import I18nAuto
import rvc.lib.tools.loadThemes as loadThemes
now_dir = os.getcwd()
sys.path.append("..")

i18n = I18nAuto()

def theme_tab():
    with gr.Row():
        with gr.Column():
            themes_select = gr.Dropdown(
                loadThemes.get_list(),
                value=loadThemes.read_json(),
                label=i18n("Select Theme:"),
                visible=True,
            )
            themes_select.change(
                fn=loadThemes.select_theme,
                inputs=themes_select,
                outputs=[],
            )

                