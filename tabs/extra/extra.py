import gradio as gr

import tabs.extra.processing.processing as processing
import tabs.extra.analyzer.analyzer as analyzer

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def extra_tab():
    gr.Markdown(
        value=i18n(
            "This section contains some extra utilities that often may be in experimental phases."
        )
    )

    with gr.TabItem(i18n("Processing")):
        processing.processing()

    with gr.TabItem(i18n("Audio Analyzer")):
        analyzer.analyzer()
