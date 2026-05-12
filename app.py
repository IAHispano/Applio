# Plataform config
from rvc.lib.platform import platform_config

# Make sure config file exists
import os  # We will also use this later, but we need it now
import shutil

# TODO: This path is regenerated all over the place in Applio
# should probably be in a static module for everything to reference
CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")

# The base config file to start from
CONFIG_TEMPLATE_PATH = os.path.join(now_dir, "assets", "config_template.json")

if not os.path.exists(CONFIG_PATH):
    print("Config file not found. Creating fresh from template.")
    shutil.copy(CONFIG_TEMPLATE_PATH, CONFIG_PATH)

platform_config()

import gradio as gr
import sys
import pathlib
import logging

from typing import Any

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add current directory to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Suppress ConnectionResetError on Windows when a remote peer forcibly closes the
# connection during asyncio shutdown (WinError 10054 / ProactorBasePipeTransport).
if sys.platform == "win32":
    import asyncio.proactor_events as _pe

    _orig_ccl = _pe._ProactorBasePipeTransport._call_connection_lost

    def _ccl_patched(self, exc):
        try:
            _orig_ccl(self, exc)
        except ConnectionResetError:
            pass

    _pe._ProactorBasePipeTransport._call_connection_lost = _ccl_patched

# detect gradio
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab
from tabs.realtime.realtime import realtime_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
)

# Initialize i18n
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Start Discord presence if enabled
from tabs.settings.sections.presence import load_config_presence

if load_config_presence():
    from assets.discord_presence import RPCManager

    RPCManager.start_presence()

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

my_applio = loadThemes.load_theme() or "ParityError/Interstellar"
client_mode = "--client" in sys.argv

# Define Gradio interface
with gr.Blocks(
    title="Applio",
    **(
        {
            "theme": my_applio,
            "css": "footer{display:none !important}",
            "js": (
                (
                    "() => {\n"
                    + pathlib.Path(
                        os.path.join(now_dir, "tabs", "realtime", "main.js")
                    ).read_text()
                    + "\n}"
                )
                if client_mode
                else None
            ),
        }
        if not GRADIO_6
        else {}
    ),
) as Applio:
    gr.Markdown("# Applio")
    gr.Markdown(
        i18n(
            "A simple, high-quality voice conversion tool focused on ease of use and performance."
        )
    )
    gr.Markdown(
        i18n(
            "[Support](https://discord.gg/urxFjYmYYh) — [GitHub](https://github.com/IAHispano/Applio)"
        )
    )
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("Training")):
        train_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Realtime")):
        realtime_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Settings")):
        settings_tab()

    gr.Markdown("""
    <div style="text-align: center; font-size: 0.9em; text-color: a3a3a3;">
    By using Applio, you agree to comply with ethical and legal standards, respect intellectual property and privacy rights, avoid harmful or prohibited uses, and accept full responsibility for any outcomes, while Applio disclaims liability and reserves the right to amend these terms.
    </div>
    """)


def launch_gradio(server_name: str, server_port: int) -> None:
    app, _, _ = Applio.launch(
        favicon_path="assets/ICON.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=client_mode,
        **(
            {
                "theme": my_applio,
                "css": "footer{display:none !important}",
                "js": (
                    pathlib.Path(
                        os.path.join(now_dir, "tabs", "realtime", "main.js")
                    ).read_text()
                    if client_mode
                    else None
                ),
            }
            if GRADIO_6
            else {}
        ),
    )

    if client_mode:
        import time
        from rvc.realtime.client import app as fastapi_app

        app.mount("/api", fastapi_app)

        while True:
            time.sleep(5)


def get_value_from_args(key: str, default: Any = None) -> Any:
    if key in sys.argv:
        index = sys.argv.index(key) + 1
        if index < len(sys.argv):
            return sys.argv[index]
    return default


if __name__ == "__main__":
    port = int(get_value_from_args("--port", DEFAULT_PORT))
    server = get_value_from_args("--server-name", DEFAULT_SERVER_NAME)

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break
