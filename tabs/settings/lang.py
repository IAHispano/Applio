import os, sys
import json
import gradio as gr
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

json_file_path = os.path.join(now_dir, "assets", "i18n", "override_lang.json")

def get_language_settings():
    with open(json_file_path, "r") as f:
        config = json.load(f)

    if config["override"] == False:
        return "False"
    else:
        return config["language"]

def save_lang_settings(select_language):
    json_file_path = os.path.join(now_dir, "assets", "i18n", "override_lang.json")
    
    with open(json_file_path, "r") as f:
        config = json.load(f)

    if select_language == "False":
        config["override"] = False
    else:
        config["override"] = True
        config["language"] = select_language
    
    gr.Info("Language settings have been saved. Restart the app to apply the changes.")

    with open(json_file_path, "w") as f:
        json.dump(config, f, indent=2)  

def lang_tab():
    with gr.Column():
        select_language = gr.Dropdown(
            label=i18n("Override language settings (Restart required)"),
            value=get_language_settings(),
            choices=["False"] + i18n._get_available_languages(),
            interactive=True,
        )

        select_language.change(
            fn=save_lang_settings,
            inputs=[select_language],
            outputs=[],
        )            