import os
import sys
import json
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from rvc.realtime.core import AUDIO_SAMPLE_RATE

i18n = I18nAuto()

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")
IS_WINDOWS = sys.platform == "win32"


def _load():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            rt = cfg.get("realtime", {})
            return (
                rt.get("asio_enabled", False),
                rt.get("audio_sample_rate", AUDIO_SAMPLE_RATE),
            )
    except Exception as e:
        print(f"Error loading realtime audio settings: {e}")
    return False, AUDIO_SAMPLE_RATE


def _save_asio_enabled(value):
    try:
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg.setdefault("realtime", {})["asio_enabled"] = value
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving asio_enabled: {e}")


def _save_audio_sample_rate(value):
    try:
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg.setdefault("realtime", {})["audio_sample_rate"] = value
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving audio_sample_rate: {e}")


def realtime_audio_tab():
    saved_asio, saved_sr = _load()
    with gr.Row():
        with gr.Column():
            asio_enabled = gr.Checkbox(
                label=i18n("Enable ASIO"),
                info=i18n(
                    "Enable ASIO driver support. (Requires restarting Applio)"
                ),
                value=saved_asio,
                interactive=True,
                visible=IS_WINDOWS,
            )
            audio_sample_rate = gr.Dropdown(
                label=i18n("Sample Rate"),
                info=i18n(
                    "Sample rate used for audio streaming. Specify the actual sample rate configured on your device, as drivers may report incorrect values."
                ),
                choices=[44100, 48000, 88200, 96000, 176400, 192000],
                value=saved_sr,
                interactive=True,
                visible=saved_asio and IS_WINDOWS,
            )
            if IS_WINDOWS:
                asio_enabled.change(
                    fn=_save_asio_enabled,
                    inputs=[asio_enabled],
                    outputs=[],
                )
                asio_enabled.change(
                    fn=lambda v: gr.update(visible=v),
                    inputs=[asio_enabled],
                    outputs=[audio_sample_rate],
                )
                audio_sample_rate.change(
                    fn=_save_audio_sample_rate,
                    inputs=[audio_sample_rate],
                    outputs=[],
                )
