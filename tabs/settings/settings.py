import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

from tabs.settings.sections.presence import presence_tab
from tabs.settings.sections.themes import theme_tab
from tabs.settings.sections.version import version_tab
from tabs.settings.sections.lang import lang_tab
from tabs.settings.sections.restart import restart_tab
from tabs.settings.sections.model_author import model_author_tab
from tabs.settings.sections.precision import precision_tab


def settings_tab():
    with gr.TabItem(label=i18n("General")):
        presence_tab()
        theme_tab()
        version_tab()
        lang_tab()
        restart_tab()
    with gr.TabItem(label=i18n("Training")):
        model_author_tab()
        precision_tab()
