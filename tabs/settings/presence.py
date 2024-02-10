import os
import sys
import base64
import pathlib
import tempfile
import gradio as gr
import threading
import json
from assets.i18n.i18n import I18nAuto
from assets.discord_presence import RPCManager

now_dir = os.getcwd()
sys.path.append("..")

i18n = I18nAuto()

CONFIG_FILE = "config.json" 


def load_config():
    default_config = {"enable_presence": True}  
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    return default_config


def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file)


config = load_config()


def presence_tab():
    with gr.Row():
        with gr.Column():
            presence = gr.Checkbox(
                label=i18n("Enable Applio integration with Discord presence"),
                interactive=True,
                value=config["enable_presence"],
            )
            presence.change(
                fn=toggle,
                inputs=[presence],
                outputs=[],
            )


def toggle(checkbox):
    config["enable_presence"] = bool(checkbox) 
    save_config(config)
    if config["enable_presence"]:
        try:
            RPCManager.start_presence()
        except KeyboardInterrupt:
            RPCManager.stop_presence()
    else:
        RPCManager.stop_presence()
