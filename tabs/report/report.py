import os
import sys
import base64
import pathlib
import tempfile
import gradio as gr

from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append("..")

i18n = I18nAuto()

recorder_js_path = os.path.join(now_dir, "tabs", "report", "recorder.js")
main_js_path = os.path.join(now_dir, "tabs", "report", "main.js")
record_button_js_path = os.path.join(now_dir, "tabs", "report", "record_button.js")

recorder_js = pathlib.Path(recorder_js_path).read_text()
main_js = pathlib.Path(main_js_path).read_text()
record_button_js = (
    pathlib.Path(record_button_js_path)
    .read_text()
    .replace("let recorder_js = null;", recorder_js)
    .replace("let main_js = null;", main_js)
)


def save_base64_video(base64_string):
    try:
        video_data = base64.b64decode(base64_string)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(video_data)
        return temp_filename
    except Exception as error:
        raise gr.Error(f"Failed to convert video to mp4:\n{error}")


def report_tab():
    instructions = [
        i18n("# How to Report an Issue on GitHub"),
        i18n(
            "1. Click on the 'Record Screen' button below to start recording the issue you are experiencing."
        ),
        i18n(
            "2. Once you have finished recording the issue, click on the 'Stop Recording' button (the same button, but the label changes depending on whether you are actively recording or not)."
        ),
        i18n(
            "3. Go to [GitHub Issues](https://github.com/IAHispano/Applio/issues) and click on the 'New Issue' button."
        ),
        i18n(
            "4. Complete the provided issue template, ensuring to include details as needed, and utilize the assets section to upload the recorded file from the previous step."
        ),
    ]
    components = [gr.Markdown(value=instruction) for instruction in instructions]

    start_button = gr.Button(i18n("Record Screen"))
    video_component = gr.Video(interactive=False)

    def toggle_button_label(returned_string):
        if returned_string.startswith(i18n("Record")):
            return gr.Button(value=i18n("Stop Recording")), None
        else:
            try:
                temp_filename = save_base64_video(returned_string)
                return gr.Button(value=i18n("Record Screen")), gr.Video(
                    value=temp_filename, interactive=False
                )
            except gr.Error as error:
                return gr.Button(value=i18n("Record Screen")), error

    start_button.click(
        toggle_button_label,
        start_button,
        [start_button, video_component],
        js=record_button_js,
    )

    components.extend([start_button, video_component])
