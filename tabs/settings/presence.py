import os
import sys
import gradio as gr
import json
from assets.i18n.i18n import I18nAuto
from assets.discord_presence import RPCManager

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()
config_file = os.path.join(now_dir, "assets", "config.json")


def load_config_presence():
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        return config["discord_presence"]


def save_config(value):
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        config["discord_presence"] = value
    with open(config_file, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)


def presence_tab():
    with gr.Row():
        with gr.Column():
            presence = gr.Checkbox(
                label=i18n("Enable Applio integration with Discord presence"),
                info=i18n(
                    "It will activate the possibility of displaying the current Applio activity in Discord."
                ),
                interactive=True,
                value=load_config_presence(),
            )
            presence.change(
                fn=toggle,
                inputs=[presence],
                outputs=[],
            )


def toggle(checkbox):
    save_config(bool(checkbox))
    if load_config_presence() == True:
        try:
            RPCManager.start_presence()
        except KeyboardInterrupt:
            RPCManager.stop_presence()
    else:
        RPCManager.stop_presence()
