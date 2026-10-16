import os, sys
import json
import gradio as gr
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

config_file = os.path.join(now_dir, "assets", "config.json")

AUTO_LANGUAGE = "Language automatically detected in the system"

# Locales officially supported in the language dropdown.
SUPPORTED_LANGUAGES = [
    "ar_AR",
    "de_DE",
    "en_US",
    "es_ES",
    "fr_FR",
    "hi_IN",
    "id_ID",
    "it_IT",
    "ja_JA",
    "ko_KO",
    "pt_BR",
    "ru_RU",
    "tr_TR",
    "vi_VI",
    "zh_CN",
]

# Native names of the supported locales, so the
# dropdown can show something friendlier than the raw code.
LANGUAGE_DISPLAY_NAMES = {
    "ar_AR": "العربية",
    "de_DE": "Deutsch",
    "en_US": "English",
    "es_ES": "Español",
    "fr_FR": "Français",
    "hi_IN": "हिन्दी",
    "id_ID": "Bahasa Indonesia",
    "it_IT": "Italiano",
    "ja_JA": "日本語",
    "ko_KO": "한국어",
    "pt_BR": "Português (Brasil)",
    "ru_RU": "Русский",
    "tr_TR": "Türkçe",
    "vi_VI": "Tiếng Việt",
    "zh_CN": "简体中文",
}


def get_language_choices():
    # Each entry is a (label, value) pair: the label is what the user sees,
    # the value is what gets saved and must stay the raw code so
    # I18nAuto can load "<code>.json". Only officially supported
    # languages are listed, even if extra files exist on disk.
    available = set(i18n._get_available_languages())
    named = [
        (f"{LANGUAGE_DISPLAY_NAMES.get(code, code)} ({code})", code)
        for code in SUPPORTED_LANGUAGES
        if code in available
    ]
    named.sort(key=lambda choice: choice[0].lower())
    return [(i18n(AUTO_LANGUAGE), AUTO_LANGUAGE)] + named


def get_language_settings():
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)

    if config["lang"]["override"] == False:
        return AUTO_LANGUAGE
    else:
        return config["lang"]["selected_lang"]


def save_lang_settings(selected_language):
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)

    if selected_language == AUTO_LANGUAGE:
        config["lang"]["override"] = False
    else:
        config["lang"]["override"] = True
        config["lang"]["selected_lang"] = selected_language

    gr.Info(i18n("Language have been saved. Restart Applio to apply the changes."))

    with open(config_file, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)


def lang_tab():
    with gr.Column():
        selected_language = gr.Dropdown(
            label=i18n("Language"),
            info=i18n(
                "Select the language you want to use. (Requires restarting Applio)"
            ),
            value=get_language_settings(),
            choices=get_language_choices(),
            interactive=True,
        )

        selected_language.change(
            fn=save_lang_settings,
            inputs=[selected_language],
            outputs=[],
        )
