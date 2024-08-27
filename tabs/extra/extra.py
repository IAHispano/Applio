import gradio as gr

import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.extra.processing.processing import processing_tab
from tabs.extra.analyzer.analyzer import analyzer_tab
from tabs.extra.f0_extractor.f0_extractor import f0_extractor_tab

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def extra_tab():
    gr.Markdown(
        value=i18n(
            "This section contains some extra utilities that often may be in experimental phases."
        )
    )
    with gr.TabItem(i18n("Model information")):
        processing_tab()

    with gr.TabItem(i18n("F0 Curve")):
        f0_extractor_tab()

    with gr.TabItem(i18n("Audio Analyzer")):
        analyzer_tab()
