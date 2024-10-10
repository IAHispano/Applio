import gradio as gr

import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.settings.advanced.advanced import adv_tab
from tabs.settings.lang import lang_tab
from tabs.settings.flask_server import flask_server_tab
from tabs.settings.model_author import model_author_tab
from tabs.settings.presence import presence_tab
from tabs.settings.restart import restart_tab
from tabs.settings.themes import theme_tab
from tabs.settings.version import version_tab

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def settings_tab():
    with gr.TabItem(i18n("General")):
        model_author_tab()
        lang_tab()
        flask_server_tab()
        presence_tab()
        theme_tab()
        version_tab()
        restart_tab()


    with gr.TabItem(i18n("Advanced Settings")):
        adv_tab()
