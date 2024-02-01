import gradio as gr
import sys
import os

now_dir = os.getcwd()
sys.path.append(now_dir)


class InstallationError(Exception):
    def __init__(self, message="InstallationError"):
        self.message = message
        super().__init__(self.message)


try:
    system_drive = os.getenv("SystemDrive")
    current_drive = os.path.splitdrive(now_dir)[0]
    if current_drive.upper() != system_drive.upper():
        raise InstallationError(
            f"Error: Current working directory is not on the default system drive ({system_drive}). Please move Applio in the correct drive."
        )
except:
    pass
else:
    if "OneDrive" in now_dir:
        raise InstallationError(
            "Error: Current working directory is on OneDrive. Please move Applio in another folder."
        )
    elif " " in now_dir:
        raise InstallationError(
            "Error: Current working directory contains spaces. Please move Applio in another folder."
        )
    try:
        now_dir.encode("ascii")
    except UnicodeEncodeError:
        raise InstallationError(
            "Error: Current working directory contains non-ASCII characters. Please move Applio in another folder."
        )

from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.settings.presence import presence_tab
from tabs.settings.themes import theme_tab
import rvc.lib.tools.loadThemes as loadThemes

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

from assets.discord_presence import RPCManager

RPCManager.start_presence()

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
        theme_tab()


if __name__ == "__main__":
    Applio.launch(
        favicon_path="assets/ICON.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=6969,
    )
