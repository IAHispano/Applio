import os, sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_audio_analyzer_script
from tabs.inference.inference import sup_audioext
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

def validate_upload(path):
    if not path:
        return gr.update(interactive=False)
    if all((
        path,
        os.path.isfile(path),
        path.lower().endswith(tuple(sup_audioext))
    )):
        return gr.update(interactive=True)
    return gr.update(interactive=False)

def analyzer_tab():
    with gr.Column():
        audio_input = gr.Audio(type="filepath")
        output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        get_info_button = gr.Button(value=i18n("Get information about the audio"), interactive=False)
        image_output = gr.Image(type="filepath", interactive=False)

    get_info_button.click(
        fn=run_audio_analyzer_script,
        inputs=[audio_input],
        outputs=[output_info, image_output],
    )

    audio_input.change(
        fn=validate_upload,
        inputs=[audio_input],
        outputs=[get_info_button],
    )

    audio_input.clear(
        lambda: gr.update(interactive=False),
        outputs=[get_info_button],
    )
