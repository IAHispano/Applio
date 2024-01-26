import gradio as gr
import sys
import os

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.settings.presence import presence_tab

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

from assets.discord_presence import RPCManager

RPCManager.start_presence()

with gr.Blocks(theme="ParityError/Interstellar", title="Applio") as Applio:
    gr.Markdown("# Applio")
    gr.Markdown(
        i18n(
            "Ultimate voice cloning tool, meticulously optimized for unrivaled power, modularity, and user-friendly experience."
        )
    )
    gr.Markdown(
        i18n(
            "[Support](https://discord.gg/IAHispano) — [Discord Bot](https://discord.com/oauth2/authorize?client_id=1144714449563955302&permissions=1376674695271&scope=bot%20applications.commands) — [Find Voices](https://applio.org/models) — [GitHub](https://github.com/IAHispano/Applio)"
        )
    )
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("Train")):
        train_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()

    with gr.Tab(i18n("Settings")):
        presence_tab()


if __name__ == "__main__":
    Applio.launch(
        favicon_path="assets/ICON.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=6969,
    )
