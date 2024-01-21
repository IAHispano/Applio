import os
import subprocess
import sys
import gradio as gr
from assets.i18n.i18n import I18nAuto
from core import (
    run_preprocess_script,
    run_extract_script,
    run_train_script,
    run_index_script,
)
from rvc.configs.config import max_vram_gpu, get_gpu_info

i18n = I18nAuto()
now_dir = os.getcwd()
sys.path.append(now_dir)
pretraineds_custom_path = os.path.join(
    now_dir, "rvc", "pretraineds", "pretraineds_custom"
)

if not os.path.exists(pretraineds_custom_path):
    os.makedirs(pretraineds_custom_path)


def get_pretrained_list(suffix):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pretraineds_custom_path)
        for filename in filenames
        if filename.endswith(".pth") and suffix in filename
    ]


pretraineds_list_d = get_pretrained_list("D")
pretraineds_list_g = get_pretrained_list("G")


def refresh_custom_pretraineds():
    return (
        {"choices": sorted(get_pretrained_list("G")), "__type__": "update"},
        {"choices": sorted(get_pretrained_list("D")), "__type__": "update"},
    )


def run_train(
    model_name,
    rvc_version,
    save_every_epoch,
    save_only_latest,
    save_every_weights,
    total_epoch,
    sampling_rate,
    batch_size,
    gpu,
    pitch_guidance,
    pretrained,
    custom_pretrained,
    g_pretrained_path,
    d_pretrained_path,
):
    core = os.path.join("core.py")
    command = [
        "python",
        core,
        "train",
        str(model_name),
        str(rvc_version),
        str(save_every_epoch),
        str(save_only_latest),
        str(save_every_weights),
        str(total_epoch),
        str(sampling_rate),
        str(batch_size),
        str(gpu),
        str(pitch_guidance),
        str(pretrained),
        str(custom_pretrained),
        str(g_pretrained_path),
        str(d_pretrained_path),
    ]
    subprocess.run(command)


def save_drop_model(dropbox):
    if ".pth" not in dropbox:
        gr.Info(
            i18n(
                "The file you dropped is not a valid pretrained file. Please try again."
            )
        )
    else:
        file_name = os.path.basename(dropbox)
        pretrained_path = os.path.join(pretraineds_custom_path, file_name)
        if os.path.exists(pretrained_path):
            os.remove(pretrained_path)
        os.rename(dropbox, pretrained_path)
        gr.Info(
            i18n(
                "Click the refresh button to see the pretrained file in the dropdown menu."
            )
        )
    return None


def train_tab():
    with gr.Accordion(i18n("Preprocess")):
        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(
                    label=i18n("Model Name"),
                    placeholder=i18n("Enter model name"),
                    value="my-project",
                    interactive=True,
                )
                dataset_path = gr.Textbox(
                    label=i18n("Dataset Path"),
                    placeholder=i18n("Enter dataset path"),
                    interactive=True,
                )
            with gr.Column():
                sampling_rate = gr.Radio(
                    label=i18n("Sampling Rate"),
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    interactive=True,
                )

                rvc_version = gr.Radio(
                    label=i18n("RVC Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                )

        preprocess_output_info = gr.Textbox(
            label=i18n("Output Information"),
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            preprocess_button = gr.Button(i18n("Preprocess Dataset"))
            preprocess_button.click(
                run_preprocess_script,
                [model_name, dataset_path, sampling_rate],
                preprocess_output_info,
                api_name="preprocess_dataset",
            )

    with gr.Accordion(i18n("Extract")):
        with gr.Row():
            hop_length = gr.Slider(
                1, 512, 128, step=1, label=i18n("Hop Length"), interactive=True
            )
        with gr.Row():
            with gr.Column():
                f0method = gr.Radio(
                    label=i18n("Pitch extraction algorithm"),
                    choices=["pm", "dio", "crepe", "crepe-tiny", "harvest", "rmvpe"],
                    value="rmvpe",
                    interactive=True,
                )

        extract_output_info = gr.Textbox(
            label=i18n("Output Information"),
            value="",
            max_lines=8,
            interactive=False,
        )
        extract_button = gr.Button(i18n("Extract Features"))
        extract_button.click(
            run_extract_script,
            [model_name, rvc_version, f0method, hop_length, sampling_rate],
            extract_output_info,
            api_name="extract_features",
        )

    with gr.Accordion(i18n("Train")):
        with gr.Row():
            batch_size = gr.Slider(
                1,
                50,
                max_vram_gpu(0),
                step=1,
                label=i18n("Batch Size"),
                interactive=True,
            )
            save_every_epoch = gr.Slider(
                1, 100, 10, step=1, label=i18n("Save Every Epoch"), interactive=True
            )
            total_epoch = gr.Slider(
                1, 1000, 500, step=1, label=i18n("Total Epoch"), interactive=True
            )
        with gr.Row():
            pitch_guidance = gr.Checkbox(
                label=i18n("Pitch Guidance"), value=True, interactive=True
            )
            pretrained = gr.Checkbox(
                label=i18n("Pretrained"), value=True, interactive=True
            )
            save_only_latest = gr.Checkbox(
                label=i18n("Save Only Latest"), value=False, interactive=True
            )
            save_every_weights = gr.Checkbox(
                label=i18n("Save Every Weights"), value=True, interactive=True, 
            )
            custom_pretrained = gr.Checkbox(
                label=i18n("Custom Pretrained"), value=False, interactive=True
            )
            multiple_gpu = gr.Checkbox(
                label=i18n("GPU Settings"), value=False, interactive=True
            )

        with gr.Row():
            with gr.Column(visible=False) as pretrained_custom_settings:
                with gr.Accordion("Pretrained Custom Settings"):
                    upload_pretrained = gr.File(
                        label=i18n("Upload Pretrained Model"),
                        type="filepath",
                        interactive=True,
                    )
                    refresh_custom_pretaineds_button = gr.Button(
                        i18n("Refresh Custom Pretraineds")
                    )
                    g_pretrained_path = gr.Dropdown(
                        label=i18n("Custom Pretrained G"),
                        choices=sorted(pretraineds_list_g),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    d_pretrained_path = gr.Dropdown(
                        label=i18n("Custom Pretrained D"),
                        choices=sorted(pretraineds_list_d),
                        interactive=True,
                        allow_custom_value=True,
                    )
            with gr.Column(visible=False) as gpu_custom_settings:
                with gr.Accordion("GPU Settings"):
                    gpu = gr.Textbox(
                        label=i18n("GPU Number"),
                        placeholder=i18n("0 to ∞ separated by -"),
                        value="0",
                        interactive=True,
                    )
                    gr.Textbox(
                        label=i18n("GPU Information"),
                        value=get_gpu_info(),
                        interactive=False,
                    )

        with gr.Row():
            train_output_info = gr.Textbox(
                label=i18n("Output Information"),
                value="",
                max_lines=8,
                interactive=False,
            )

        with gr.Row():
            train_button = gr.Button(i18n("Start Training"))
            train_button.click(
                run_train,
                [
                    model_name,
                    rvc_version,
                    save_every_epoch,
                    save_only_latest,
                    save_every_weights,
                    total_epoch,
                    sampling_rate,
                    batch_size,
                    gpu,
                    pitch_guidance,
                    pretrained,
                    custom_pretrained,
                    g_pretrained_path,
                    d_pretrained_path,
                ],
                train_output_info,
                api_name="start_training",
            )

            index_button = gr.Button(i18n("Generate Index"))
            index_button.click(
                run_index_script,
                [model_name, rvc_version],
                train_output_info,
                api_name="generate_index",
            )

            def toggle_visible(checkbox):
                return {"visible": checkbox, "__type__": "update"}

            custom_pretrained.change(
                fn=toggle_visible,
                inputs=[custom_pretrained],
                outputs=[pretrained_custom_settings],
            )

            refresh_custom_pretaineds_button.click(
                fn=refresh_custom_pretraineds,
                inputs=[],
                outputs=[g_pretrained_path, d_pretrained_path],
            )

            upload_pretrained.upload(
                fn=save_drop_model,
                inputs=[upload_pretrained],
                outputs=[upload_pretrained],
            )

            multiple_gpu.change(
                fn=toggle_visible,
                inputs=[multiple_gpu],
                outputs=[gpu_custom_settings],
            )
