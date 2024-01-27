import os
import sys
import base64
import pathlib
import tempfile
import gradio as gr
import threading
from assets.i18n.i18n import I18nAuto
from assets.discord_presence import RPCManager

now_dir = os.getcwd()
sys.path.append("..")

i18n = I18nAuto()


def presence_tab():
    with gr.Row():
        with gr.Column():
            presence = gr.Checkbox(
                label=i18n("Enable Applio integration with Discord presence"),
                interactive=True,
                value=True,
            )
            presence.change(
                fn=toggle,
                inputs=[presence],
                outputs=[],
            )


def toggle(checkbox):

    if bool(checkbox):
        # print("Start Presence")
        try:
            RPCManager.start_presence()
        except KeyboardInterrupt:
            RPCManager.stop_presence()
    else:
        # print("Stop presence")
        RPCManager.stop_presence()
