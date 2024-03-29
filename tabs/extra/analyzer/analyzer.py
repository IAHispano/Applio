import os, sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_audio_analyzer_script
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def analyzer():
    with gr.Column():
        audio_input = gr.Audio(type="filepath")
        get_info_button = gr.Button(
            value=i18n("Get information about the audio"), variant="primary"
        )
    with gr.Column():
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    value=i18n("Information about the audio file"),
                    visible=True,
                )
                output_markdown = gr.Markdown(
                    value=i18n("Waiting for information..."), visible=True
                )
            image_output = gr.Image(type="filepath", interactive=False)

    get_info_button.click(
        fn=run_audio_analyzer_script,
        inputs=[audio_input],
        outputs=[output_markdown, image_output],
    )
