import os
import sys
import base64
import pathlib
import tempfile
import gradio as gr

from assets.i18n.i18n import I18nAuto
import assets.themes.loadThemes as loadThemes

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()


def theme_tab():
    with gr.Row():
        with gr.Column():
            themes_select = gr.Dropdown(
                loadThemes.get_list(),
                value=loadThemes.read_json(),
                label=i18n("Theme"),
                info=i18n(
                    "Select the theme you want to use. (Requires restarting Applio)"
                ),
                visible=True,
            )
            themes_select.change(
                fn=loadThemes.select_theme,
                inputs=themes_select,
                outputs=[],
            )
