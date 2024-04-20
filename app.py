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
from tabs.settings.fake_gpu import fake_gpu_tab, gpu_available, load_fake_gpu
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
from core import run_prerequisites_script

run_prerequisites_script("False", "True", "True", "True")

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
        if gpu_available() or load_fake_gpu():
            train_tab()
        else:
            gr.Markdown(
                i18n(
                    "Training is currently unsupported due to the absence of a GPU. To activate the training tab, navigate to the settings tab and enable the 'Fake GPU' option."
                )
            )

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Settings")):
        presence_tab()
        flask_server_tab()
        if not gpu_available():
            fake_gpu_tab()
        theme_tab()
        version_tab()
        lang_tab()
        restart_tab()


def launch_gradio(port):
    Applio.launch(
        favicon_path="assets/ICON.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=port,
    )


if __name__ == "__main__":
    port = 6969
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            port = int(sys.argv[port_index])

        launch_gradio(port)

    else:
        # if launch fails, decrement port number and try again (up to 10 times)
        for i in range(10):
            try:
                launch_gradio(port)
                break
            except OSError:
                print("Failed to launch on port", port, "- trying again...")
                port -= 1
            except Exception as e:
                print(f"Unexpected error during launch: {e}")
                break
