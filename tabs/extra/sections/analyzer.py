import os, sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_audio_analyzer_script
from assets.i18n.i18n import I18nAuto
from assets.app_shell import step_card

i18n = I18nAuto()


def analyzer_tab():
    with step_card(
        1, "Audio Source", i18n, description="Upload the audio you want to analyze."
    ):
        audio_input = gr.Audio(type="filepath")
    with step_card(
        2,
        "Analysis Results",
        i18n,
        description="Run the analysis and inspect the output.",
    ):
        get_info_button = gr.Button(value=i18n("Get information about the audio"))
        output_info = gr.Textbox(
            label=i18n("Audio analysis output"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        image_output = gr.Image(type="filepath", interactive=False)

    get_info_button.click(
        fn=run_audio_analyzer_script,
        inputs=[audio_input],
        outputs=[output_info, image_output],
    )
