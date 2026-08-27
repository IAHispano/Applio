import os, sys
import json
import gradio as gr
from assets.i18n.i18n import I18nAuto

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

config_file = os.path.join(now_dir, "assets", "config.json")

AUTO_LANGUAGE = "Language automatically detected in the system"

# Native names of the locales shipped in assets/i18n/languages, so the
# dropdown can show something friendlier than the raw code. Unknown codes
# (e.g. a newly added language file) fall back to the code itself.
LANGUAGE_DISPLAY_NAMES = {
    "af_AF": "Afrikaans",
    "am_AM": "አማርኛ",
    "ar_AR": "العربية",
    "az_AZ": "Azərbaycan",
    "ba_BA": "Башҡортса",
    "be_BE": "Беларуская",
    "bn_BN": "বাংলা",
    "bs_BS": "Bosanski",
    "ca_CA": "Català",
    "ceb_CEB": "Cebuano",
    "cs_CS": "Čeština",
    "de_DE": "Deutsch",
    "el_EL": "Ελληνικά",
    "en_US": "English",
    "es_ES": "Español",
    "eu_EU": "Euskara",
    "fa_FA": "فارسی",
    "fj_FJ": "Na Vosa Vakaviti",
    "fr_FR": "Français",
    "ga_GA": "Gaeilge",
    "gu_GU": "ગુજરાતી",
    "he_HE": "עברית",
    "hi_IN": "हिन्दी",
    "hr_HR": "Hrvatski",
    "ht_HT": "Kreyòl Ayisyen",
    "hu_HU": "Magyar",
    "id_ID": "Bahasa Indonesia",
    "it_IT": "Italiano",
    "ja_JA": "日本語",
    "jv_JV": "Basa Jawa",
    "ko_KO": "한국어",
    "lt_LT": "Lietuvių",
    "lv_LV": "Latviešu",
    "mg_MG": "Malagasy",
    "ml_IN": "മലയാളം",
    "mr_MR": "मराठी",
    "ms_MS": "Bahasa Melayu",
    "mt_MT": "Malti",
    "nl_NL": "Nederlands",
    "otq_OTQ": "Hñähñu",
    "pa_PA": "ਪੰਜਾਬੀ",
    "pl_PL": "Polski",
    "pt_BR": "Português (Brasil)",
    "pt_PT": "Português (Portugal)",
    "ro_RO": "Română",
    "ru_RU": "Русский",
    "sk_SK": "Slovenčina",
    "sm_SM": "Gagana Sāmoa",
    "sr_RS": "Српски",
    "sw_SW": "Kiswahili",
    "ta_IN": "தமிழ்",
    "te_TE": "తెలుగు",
    "th_TH": "ไทย",
    "to_TO": "Lea faka-Tonga",
    "tr_TR": "Türkçe",
    "uk_UK": "Українська",
    "ur_UR": "اردو",
    "vi_VI": "Tiếng Việt",
    "wu_WU": "吴语",
    "zh_CN": "简体中文",
}


def get_language_choices():
    # Each entry is a (label, value) pair: the label is what the user sees,
    # the value is what gets saved and must stay the raw code so
    # I18nAuto can load "<code>.json".
    named = [
        (f"{LANGUAGE_DISPLAY_NAMES.get(code, code)} ({code})", code)
        for code in i18n._get_available_languages()
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
