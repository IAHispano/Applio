import os
import sys
import base64
import pathlib
import tempfile
import gradio as gr

from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

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
    base64_video = base64_string
    video_data = base64.b64decode(base64_video)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(video_data)
    print(f"Temporary MP4 file saved as: {temp_filename}")
    return temp_filename


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

    start_button = gr.Button("Record Screen")
    video_component = gr.Video(interactive=False)

    def toggle_button_label(returned_string):
        if returned_string.startswith("Record"):
            return gr.Button(value="Stop Recording"), None
        else:
            try:
                temp_filename = save_base64_video(returned_string)
            except Exception as error:
                print(f"An error occurred converting video to mp4: {error}")
                return gr.Button(value="Record Screen"), gr.Warning(
                    f"Failed to convert video to mp4:\n{error}"
                )
            return gr.Button(value="Record Screen"), gr.Video(
                value=temp_filename, interactive=False
            )

    start_button.click(
        fn=toggle_button_label,
        inputs=[start_button],
        outputs=[start_button, video_component],
        js=record_button_js,
    )
