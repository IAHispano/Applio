import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.extra.sections.processing import processing_tab
from tabs.extra.sections.analyzer import analyzer_tab
from tabs.extra.sections.f0_extractor import f0_extractor_tab

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def extra_tab():
    with gr.TabItem(i18n("Model information")):
        processing_tab()

    with gr.TabItem(i18n("F0 Curve")):
        f0_extractor_tab()

    with gr.TabItem(i18n("Audio Analyzer")):
        analyzer_tab()
