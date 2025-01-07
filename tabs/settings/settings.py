import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.settings.sections.presence import presence_tab
from tabs.settings.sections.themes import theme_tab
from tabs.settings.sections.version import version_tab
from tabs.settings.sections.lang import lang_tab
from tabs.settings.sections.restart import restart_tab
from tabs.settings.sections.model_author import model_author_tab


def settings_tab():
    with gr.TabItem(label="General"):
        presence_tab()
        theme_tab()
        version_tab()
        lang_tab()
        restart_tab()
    with gr.TabItem(label="Training"):
        model_author_tab()
