import gradio as gr
import sys
import os
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)

# Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.settings.presence import presence_tab, load_config_presence
from tabs.settings.flask_server import flask_server_tab
from tabs.settings.themes import theme_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.version import version_tab
from tabs.settings.lang import lang_tab
from tabs.settings.restart import restart_tab

# Assets
import assets.themes.loadThemes as loadThemes
from assets.i18n.i18n import I18nAuto
import assets.installation_checker as installation_checker
from assets.discord_presence import RPCManager
from assets.flask.server import start_flask, load_config_flask

i18n = I18nAuto()
if load_config_presence() == True:
    RPCManager.start_presence()
installation_checker.check_installation()
logging.getLogger("uvicorn").disabled = True
logging.getLogger("fairseq").disabled = True
if load_config_flask() == True:
    print("Starting Flask server")
    start_flask()

my_applio = loadThemes.load_json()
if my_applio:
    pass
else:
    my_applio = "ParityError/Interstellar"

with gr.Blocks(theme=my_applio, title="Applio") as Applio:
    gr.Markdown("# Applio")
    gr.Markdown(
        i18n(
            "Ultimate voice cloning tool. Meticulously optimized for unrivaled power, modularity & user-friendly experience."
        )
    )
    gr.Markdown(
        i18n(
            "[Join Us](https://discord.gg/IAHispano) — [Discord Bot](https://discord.com/oauth2/authorize?client_id=1144714449563955302&permissions=1376674695271&scope=bot%20applications.commands) — [Voice Models](https://applio.org/models) — [GitHub](https://github.com/IAHispano/Applio)"
        )
    )
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("Train")):
        train_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report Bugs")):
        report_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Settings")):
        presence_tab()
        flask_server_tab()
        theme_tab()
        version_tab()
        lang_tab()
        restart_tab()


if __name__ == "__main__":
    Applio.launch(
        favicon_path="assets/ICON.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port="--port" in sys.argv or 6969,
    )
