import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from assets.app_shell import step_card

i18n = I18nAuto()

from tabs.settings.sections.presence import presence_tab
from tabs.settings.sections.realtime_audio import realtime_audio_tab
from tabs.settings.sections.themes import theme_tab
from tabs.settings.sections.version import version_tab
from tabs.settings.sections.lang import lang_tab
from tabs.settings.sections.restart import restart_tab
from tabs.settings.sections.model_author import model_author_tab
from tabs.settings.sections.precision import precision_tab
from tabs.settings.sections.filter import filter_tab, get_filter_trigger
from tabs.settings.sections.rmvpe_high_register import rmvpe_high_register_tab


def settings_tab(filter_state_trigger=None):
    if filter_state_trigger is None:
        filter_state_trigger = get_filter_trigger()

    with gr.TabItem(label=i18n("General")):
        with step_card(
            1,
            "List Filtering",
            i18n,
            description="Show a keyword filter for the model and index lists.",
        ):
            filter_component = filter_tab()

            filter_component.change(
                fn=lambda checked: gr.update(value=str(checked)),
                inputs=[filter_component],
                outputs=[filter_state_trigger],
                show_progress=False,
            )
        with step_card(
            2,
            "Pitch Correction",
            i18n,
            description="Fix high-register octave errors in RMVPE pitch.",
        ):
            rmvpe_high_register_tab()
        with step_card(
            3,
            "Discord Integration",
            i18n,
            description="Show your current Applio activity in Discord.",
        ):
            presence_tab()
        with step_card(
            4,
            "Audio Devices",
            i18n,
            description="Configure drivers and the streaming sample rate.",
        ):
            realtime_audio_tab()
        with step_card(
            5,
            "Appearance & Language",
            i18n,
            description="Pick a theme and language. Applies after restart.",
        ):
            theme_tab()
            lang_tab()
        with step_card(
            6,
            "Updates & System",
            i18n,
            description="Check for updates or restart the interface.",
        ):
            version_tab()
            restart_tab()
    with gr.TabItem(label=i18n("Training")):
        with step_card(
            1,
            "Model Author",
            i18n,
            description="Set the name shown in exported model information.",
        ):
            model_author_tab()
        with step_card(
            2,
            "Training Precision",
            i18n,
            description="Choose the numeric precision for training and inference.",
        ):
            precision_tab()
