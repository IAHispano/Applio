import os, sys
import torch
import json
import gradio as gr
from assets.i18n.i18n import I18nAuto
from tabs.settings.restart import restart_applio

now_dir = os.getcwd()
sys.path.append(now_dir)
i18n = I18nAuto()

ngpu = torch.cuda.device_count()
config_file = os.path.join(now_dir, "assets", "config.json")


def gpu_available():
    if torch.cuda.is_available() or ngpu != 0:
        return True


def load_fake_gpu():
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        return config["fake_gpu"]


def save_config(value):
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        config["fake_gpu"] = value
    with open(config_file, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)


def fake_gpu_tab():
    with gr.Row():
        with gr.Column():
            presence = gr.Checkbox(
                label=i18n("Enable fake GPU"),
                info=i18n(
                    "Activates the train tab. However, please note that this device lacks GPU capabilities, hence training is not supported. This option is only for testing purposes. (This option will restart Applio)"
                ),
                interactive=True,
                value=load_fake_gpu(),
            )
            presence.change(
                fn=toggle,
                inputs=[presence],
                outputs=[],
            )


def toggle(checkbox):
    save_config(bool(checkbox))
    restart_applio()
