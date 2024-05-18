import os, sys, shutil
import tempfile
import gradio as gr
import pandas as pd
import requests
import wget
from core import run_download_script

from assets.i18n.i18n import I18nAuto

from rvc.lib.utils import format_title

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

gradio_temp_dir = os.path.join(tempfile.gettempdir(), "gradio")

if os.path.exists(gradio_temp_dir):
    shutil.rmtree(gradio_temp_dir)


def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )
    else:
        file_name = format_title(os.path.basename(dropbox))
        if ".pth" in dropbox:
            model_name = format_title(file_name.split(".pth")[0])
        else:
            if "v2" not in dropbox:
                model_name = format_title(
                    file_name.split("_nprobe_1_")[1].split("_v1")[0]
                )
            else:
                model_name = format_title(
                    file_name.split("_nprobe_1_")[1].split("_v2")[0]
                )
        model_path = os.path.join(now_dir, "logs", model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if os.path.exists(os.path.join(model_path, file_name)):
            os.remove(os.path.join(model_path, file_name))
        shutil.move(dropbox, os.path.join(model_path, file_name))
        print(f"{file_name} saved in {model_path}")
        gr.Info(f"{file_name} saved in {model_path}")
    return None


def search_models(name):
    url = f"https://cjtfqzjfdimgpvpwhzlv.supabase.co/rest/v1/models?name=ilike.%25{name}%25&order=created_at.desc&limit=15"
    headers = {
        "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNqdGZxempmZGltZ3B2cHdoemx2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTUxNjczODgsImV4cCI6MjAxMDc0MzM4OH0.7z5WMIbjR99c2Ooc0ma7B_FyGq10G8X-alkCYTkKR10"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    if len(data) == 0:
        gr.Info(i18n("We couldn't find models by that name."))
        return None
    else:
        df = pd.DataFrame(data)[["name", "link", "epochs", "type"]]
        df["link"] = df["link"].apply(
            lambda x: f'<a href="{x}" target="_blank">{x}</a>'
        )
        return df


json_url = "https://huggingface.co/IAHispano/Applio/raw/main/pretrains.json"


def fetch_pretrained_data():
    response = requests.get(json_url)
    response.raise_for_status()
    return response.json()


def get_pretrained_list():
    data = fetch_pretrained_data()
    return list(data.keys())


def get_pretrained_sample_rates(model):
    data = fetch_pretrained_data()
    return list(data[model].keys())


def download_pretrained_model(model, sample_rate):
    data = fetch_pretrained_data()
    paths = data[model][sample_rate]
    pretraineds_custom_path = os.path.join("rvc", "pretraineds", "pretraineds_custom")
    os.makedirs(pretraineds_custom_path, exist_ok=True)

    d_url = f"https://huggingface.co/{paths['D']}"
    g_url = f"https://huggingface.co/{paths['G']}"

    gr.Info("Downloading Pretrained Model...")
    print("Downloading Pretrained Model...")
    wget.download(d_url, out=pretraineds_custom_path)
    wget.download(g_url, out=pretraineds_custom_path)


def update_sample_rate_dropdown(model):
    return {
        "choices": get_pretrained_sample_rates(model),
        "value": get_pretrained_sample_rates(model)[0],
        "__type__": "update",
    }


def download_tab():
    with gr.Column():
        gr.Markdown(value=i18n("## Download Model"))
        model_link = gr.Textbox(
            label=i18n("Model Link"),
            placeholder=i18n("Introduce the model link"),
            interactive=True,
        )
        model_download_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        model_download_button = gr.Button(i18n("Download Model"))
        model_download_button.click(
            fn=run_download_script,
            inputs=[model_link],
            outputs=[model_download_output_info],
            api_name="model_download",
        )
        gr.Markdown(value=i18n("## Drop files"))
        dropbox = gr.File(
            label=i18n(
                "Drag your .pth file and .index file into this space. Drag one and then the other."
            ),
            type="filepath",
        )

        dropbox.upload(
            fn=save_drop_model,
            inputs=[dropbox],
            outputs=[dropbox],
        )
        gr.Markdown(value=i18n("## Search Model"))
        search_name = gr.Textbox(
            label=i18n("Model Name"),
            placeholder=i18n("Introduce the model name to search."),
            interactive=True,
        )
        search_table = gr.Dataframe(datatype="markdown")
        search = gr.Button(i18n("Search"))
        search.click(
            fn=search_models,
            inputs=[search_name],
            outputs=[search_table],
        )
        search_name.submit(search_models, [search_name], search_table)
        gr.Markdown(value=i18n("## Download Pretrained Models"))
        pretrained_model = gr.Dropdown(
            label=i18n("Pretrained"),
            info=i18n("Select the pretrained model you want to download."),
            choices=get_pretrained_list(),
            value="Titan",
            interactive=True,
        )
        pretrained_sample_rate = gr.Dropdown(
            label=i18n("Sampling Rate"),
            info=i18n("And select the sampling rate."),
            choices=get_pretrained_sample_rates(pretrained_model.value),
            value="40k",
            interactive=True,
            allow_custom_value=True,
        )
        pretrained_model.change(
            update_sample_rate_dropdown,
            inputs=[pretrained_model],
            outputs=[pretrained_sample_rate],
        )
        download_pretrained = gr.Button(i18n("Download"))
        download_pretrained.click(
            fn=download_pretrained_model,
            inputs=[pretrained_model, pretrained_sample_rate],
            outputs=[],
        )
