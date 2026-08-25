import os
import sys
import json
import gradio as gr
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()
config_file = os.path.join(now_dir, "assets", "config.json")

# Keep in sync with HIGH_REGISTER_DEFAULTS in rvc/lib/predictors/f0.py
DEFAULTS = {"enabled": False, "mode": "true_pitch", "f0_ceil": 1250}
MODES = ["true_pitch", "fold"]
F0_CEIL_RANGE = (1000, 2000)


def load_config_high_register():
    with open(config_file, "r", encoding="utf8") as f:
        cfg = json.load(f)
    section = cfg.get("rmvpe_high_register")
    settings = {**DEFAULTS, **(section if isinstance(section, dict) else {})}
    # Sanitize hand-edited values so the UI always gets something valid.
    if settings["mode"] not in MODES:
        settings["mode"] = DEFAULTS["mode"]
    try:
        settings["f0_ceil"] = min(
            max(float(settings["f0_ceil"]), F0_CEIL_RANGE[0]), F0_CEIL_RANGE[1]
        )
    except (TypeError, ValueError):
        settings["f0_ceil"] = DEFAULTS["f0_ceil"]
    settings["enabled"] = bool(settings["enabled"])
    return settings


def save_config_high_register(enabled, mode, f0_ceil):
    with open(config_file, "r", encoding="utf8") as f:
        cfg = json.load(f)
    cfg["rmvpe_high_register"] = {
        "enabled": bool(enabled),
        "mode": mode if mode in MODES else DEFAULTS["mode"],
        "f0_ceil": float(f0_ceil),
    }
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=2)


def rmvpe_high_register_tab():
    settings = load_config_high_register()
    with gr.Accordion(i18n("RMVPE high-register correction"), open=False):
        enabled = gr.Checkbox(
            label=i18n("Enable RMVPE high-register correction"),
            info=i18n(
                "Fixes the octave errors rmvpe.pt makes above ~1040 Hz (flutes, high strings, whistle register) with a second pass on octave-down-resampled audio. Off by default; when disabled RMVPE behaves exactly as before."
            ),
            value=settings["enabled"],
            interactive=True,
        )
        mode = gr.Radio(
            label=i18n("Correction mode"),
            info=i18n(
                "true_pitch writes the real pitch, capped at the F0 ceiling. fold only repairs wrong-octave frames using half-pitch drives; use it with models trained on uncorrected labels to keep their timbre."
            ),
            choices=MODES,
            value=settings["mode"],
            interactive=True,
        )
        f0_ceil = gr.Slider(
            label=i18n("F0 ceiling (Hz)"),
            info=i18n(
                "Highest pitch written in true_pitch mode. Models trained on stock RMVPE labels cannot synthesize above ~1250 Hz; raise it only for models trained with the correction enabled."
            ),
            minimum=F0_CEIL_RANGE[0],
            maximum=F0_CEIL_RANGE[1],
            step=10,
            value=settings["f0_ceil"],
            interactive=True,
        )
        for component in (enabled, mode, f0_ceil):
            component.change(
                fn=save_config_high_register,
                inputs=[enabled, mode, f0_ceil],
                outputs=[],
            )
