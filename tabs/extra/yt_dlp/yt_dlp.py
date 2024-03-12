import os, sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_model_information_script
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

import gradio as gr
import subprocess

def get_video_title(url):
    
    result = subprocess.run(["yt-dlp", "--get-title", url], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return "Unknown Video"

def fetch(url, custom_name, ext):
    title = get_video_title(url)
    #  
    max_length = 50  # 
    truncated_title = title[:max_length].strip()
    
    filename = f"{custom_name}.{ext}" if custom_name else f"{truncated_title}.{ext}"
    opts = {
        "wav": ["-f", "ba", "-x", "--audio-format", "wav"],

    }[ext]
    command = ["yt-dlp"] + wav + [url, "-o", filename]
    subprocess.run(command)

    return filename


def processing():
    with gr.Accordion(label=i18n("download youtube wav using Yt-dlp")):
        with gr.Row():
            with gr.Column():
                url = gr.Textbox(
                    label=i18n("url youtube"),
                    info=i18n("paste url youtube here"),
                    value="",
                    interactive=True,
                    placeholder=i18n("example : https://youtu.be/iN0-dRNsmRM?si=42PgawH73GIrvYLs"),
                )

                filename = gr.Textbox(
                    label=i18n("custom file name"),
                    info=i18n("custom name for file"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Default video title"),
                )

         outputs = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
        )
        fetch_button = gr.Button(i18n("View"), variant="primary")
        fetch_button.click(
            fetch,
            [url, filename],
            outputs,
            api_name="fetch",
                     )
