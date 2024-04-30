import os
import subprocess
import sys
import shutil
import gradio as gr
from assets.i18n.i18n import I18nAuto
from core import (
    run_preprocess_script,
    run_extract_script,
    run_train_script,
    run_index_script,
    run_prerequisites_script,
)
from rvc.configs.config import max_vram_gpu, get_gpu_info
from rvc.lib.utils import format_title
from tabs.settings.restart import restart_applio

i18n = I18nAuto()
now_dir = os.getcwd()
sys.path.append(now_dir)

pretraineds_v1 = [
    (
        "pretrained_v1/",
        [
            "D32k.pth",
            "D40k.pth",
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    ),
]

folder_mapping = {
    "pretrained_v1/": "rvc/pretraineds/pretrained_v1/",
}

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

# Custom Pretraineds
pretraineds_custom_path = os.path.join(
    now_dir, "rvc", "pretraineds", "pretraineds_custom"
)

pretraineds_custom_path_relative = os.path.relpath(pretraineds_custom_path, now_dir)

if not os.path.exists(pretraineds_custom_path_relative):
    os.makedirs(pretraineds_custom_path_relative)


def get_pretrained_list(suffix):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pretraineds_custom_path_relative)
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


# Dataset Creator
datasets_path = os.path.join(now_dir, "assets", "datasets")

if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

datasets_path_relative = os.path.relpath(datasets_path, now_dir)


def get_datasets_list():
    return [
        dirpath
        for dirpath, _, filenames in os.walk(datasets_path_relative)
        if any(filename.endswith(tuple(sup_audioext)) for filename in filenames)
    ]


def refresh_datasets():
    return {"choices": sorted(get_datasets_list()), "__type__": "update"}


# Model Names
models_path = os.path.join(now_dir, "logs")


def get_models_list():
    return [
        os.path.basename(dirpath)
        for dirpath in os.listdir(models_path)
        if os.path.isdir(os.path.join(models_path, dirpath))
        and all(excluded not in dirpath for excluded in ["zips", "mute"])
    ]


def refresh_models():
    return {"choices": sorted(get_models_list()), "__type__": "update"}


# Refresh Models and Datasets
def refresh_models_and_datasets():
    return (
        {"choices": sorted(get_models_list()), "__type__": "update"},
        {"choices": sorted(get_datasets_list()), "__type__": "update"},
    )


# Drop Model
def save_drop_model(dropbox):
    if ".pth" not in dropbox:
        gr.Info(
            i18n(
                "The file you dropped is not a valid pretrained file. Please try again."
            )
        )
    else:
        file_name = os.path.basename(dropbox)
        pretrained_path = os.path.join(pretraineds_custom_path_relative, file_name)
        if os.path.exists(pretrained_path):
            os.remove(pretrained_path)
        os.rename(dropbox, pretrained_path)
        gr.Info(
            i18n(
                "Click the refresh button to see the pretrained file in the dropdown menu."
            )
        )
    return None


# Drop Dataset
def save_drop_dataset_audio(dropbox, dataset_name):
    if not dataset_name:
        gr.Info("Please enter a valid dataset name. Please try again.")
        return None, None
    else:
        file_extension = os.path.splitext(dropbox)[1][1:].lower()
        if file_extension not in sup_audioext:
            gr.Info("The file you dropped is not a valid audio file. Please try again.")
        else:
            dataset_name = format_title(dataset_name)
            audio_file = format_title(os.path.basename(dropbox))
            dataset_path = os.path.join(now_dir, "assets", "datasets", dataset_name)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            destination_path = os.path.join(dataset_path, audio_file)
            if os.path.exists(destination_path):
                os.remove(destination_path)
            os.rename(dropbox, destination_path)
            gr.Info(
                i18n(
                    "The audio file has been successfully added to the dataset. Please click the preprocess button."
                )
            )
            dataset_path = os.path.dirname(destination_path)
            relative_dataset_path = os.path.relpath(dataset_path, now_dir)

            return None, relative_dataset_path


# Export
## Get Pth and Index Files
def get_pth_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".pth")
    ]


def get_index_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]


def refresh_pth_and_index_list():
    return (
        {"choices": sorted(get_pth_list()), "__type__": "update"},
        {"choices": sorted(get_index_list()), "__type__": "update"},
    )


## Export Pth and Index Files
def export_pth(pth_path):
    if pth_path and os.path.exists(pth_path):
        return pth_path
    return None


def export_index(index_path):
    if index_path and os.path.exists(index_path):
        return index_path
    return None


## Upload to Google Drive
def upload_to_google_drive(pth_path, index_path):
    def upload_file(file_path):
        if file_path:
            try:
                gr.Info(f"Uploading {pth_path} to Google Drive...")
                google_drive_folder = "/content/drive/MyDrive/ApplioExported"
                if not os.path.exists(google_drive_folder):
                    os.makedirs(google_drive_folder)
                google_drive_file_path = os.path.join(
                    google_drive_folder, os.path.basename(file_path)
                )
                if os.path.exists(google_drive_file_path):
                    os.remove(google_drive_file_path)
                shutil.copy2(file_path, google_drive_file_path)
                gr.Info("File uploaded successfully.")
            except Exception as error:
                print(error)
                gr.Info("Error uploading to Google Drive")

    upload_file(pth_path)
    upload_file(index_path)


# Train Tab
def train_tab():
    with gr.Accordion(i18n("Preprocess")):
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label=i18n("Model Name"),
                    info=i18n("Name of the new model."),
                    choices=get_models_list(),
                    value="my-project",
                    interactive=True,
                    allow_custom_value=True,
                )
                dataset_path = gr.Dropdown(
                    label=i18n("Dataset Path"),
                    info=i18n("Path to the dataset folder."),
                    # placeholder=i18n("Enter dataset path"),
                    choices=get_datasets_list(),
                    allow_custom_value=True,
                    interactive=True,
                )
                refresh = gr.Button(i18n("Refresh"))
                dataset_creator = gr.Checkbox(
                    label=i18n("Dataset Creator"),
                    value=False,
                    interactive=True,
                    visible=True,
                )

                with gr.Column(visible=False) as dataset_creator_settings:
                    with gr.Accordion(i18n("Dataset Creator")):
                        dataset_name = gr.Textbox(
                            label=i18n("Dataset Name"),
                            info=i18n("Name of the new dataset."),
                            placeholder=i18n("Enter dataset name"),
                            interactive=True,
                        )
                        upload_audio_dataset = gr.File(
                            label=i18n("Upload Audio Dataset"),
                            type="filepath",
                            interactive=True,
                        )

            with gr.Column():
                sampling_rate = gr.Radio(
                    label=i18n("Sampling Rate"),
                    info=i18n("The sampling rate of the audio files."),
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    interactive=True,
                )

                rvc_version = gr.Radio(
                    label=i18n("RVC Version"),
                    info=i18n("The RVC version of the model."),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                )

        preprocess_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
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
                1,
                512,
                128,
                step=1,
                label=i18n("Hop Length"),
                info=i18n(
                    "Denotes the duration it takes for the system to transition to a significant pitch change. Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy."
                ),
                visible=False,
                interactive=True,
            )
        with gr.Row():
            with gr.Column():
                f0method = gr.Radio(
                    label=i18n("Pitch extraction algorithm"),
                    info=i18n(
                        "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                    ),
                    choices=["pm", "dio", "crepe", "crepe-tiny", "harvest", "rmvpe"],
                    value="rmvpe",
                    interactive=True,
                )

        extract_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
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
                info=i18n(
                    "It's advisable to align it with the available VRAM of your GPU. A setting of 4 offers improved accuracy but slower processing, while 8 provides faster and standard results."
                ),
                interactive=True,
            )
            save_every_epoch = gr.Slider(
                1,
                100,
                10,
                step=1,
                label=i18n("Save Every Epoch"),
                info=i18n("Determine at how many epochs the model will saved at."),
                interactive=True,
            )
            total_epoch = gr.Slider(
                1,
                10000,
                500,
                step=1,
                label=i18n("Total Epoch"),
                info=i18n(
                    "Specifies the overall quantity of epochs for the model training process."
                ),
                interactive=True,
            )
        with gr.Row():
            pitch_guidance = gr.Checkbox(
                label=i18n("Pitch Guidance"),
                info=i18n(
                    "By employing pitch guidance, it becomes feasible to mirror the intonation of the original voice, including its pitch. This feature is particularly valuable for singing and other scenarios where preserving the original melody or pitch pattern is essential."
                ),
                value=True,
                interactive=True,
            )
            pretrained = gr.Checkbox(
                label=i18n("Pretrained"),
                info=i18n(
                    "Utilize pretrained models when training your own. This approach reduces training duration and enhances overall quality."
                ),
                value=True,
                interactive=True,
            )
            save_only_latest = gr.Checkbox(
                label=i18n("Save Only Latest"),
                info=i18n(
                    "Enabling this setting will result in the G and D files saving only their most recent versions, effectively conserving storage space."
                ),
                value=False,
                interactive=True,
            )
            save_every_weights = gr.Checkbox(
                label=i18n("Save Every Weights"),
                info=i18n(
                    "This setting enables you to save the weights of the model at the conclusion of each epoch."
                ),
                value=True,
                interactive=True,
            )
            custom_pretrained = gr.Checkbox(
                label=i18n("Custom Pretrained"),
                info=i18n(
                    "Utilizing custom pretrained models can lead to superior results, as selecting the most suitable pretrained models tailored to the specific use case can significantly enhance performance."
                ),
                value=False,
                interactive=True,
            )
            multiple_gpu = gr.Checkbox(
                label=i18n("GPU Settings"),
                info=(
                    i18n(
                        "Sets advanced GPU settings, recommended for users with better GPU architecture."
                    )
                ),
                value=False,
                interactive=True,
            )
            overtraining_detector = gr.Checkbox(
                label=i18n("Overtraining Detector"),
                info=i18n(
                    "Detect overtraining to prevent the model from learning the training data too well and losing the ability to generalize to new data."
                ),
                value=False,
                interactive=True,
            )

        with gr.Row():
            with gr.Column(visible=False) as pretrained_custom_settings:
                with gr.Accordion(i18n("Pretrained Custom Settings")):
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
                        info=i18n(
                            "Select the custom pretrained model for the generator."
                        ),
                        choices=sorted(pretraineds_list_g),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    d_pretrained_path = gr.Dropdown(
                        label=i18n("Custom Pretrained D"),
                        info=i18n(
                            "Select the custom pretrained model for the discriminator."
                        ),
                        choices=sorted(pretraineds_list_d),
                        interactive=True,
                        allow_custom_value=True,
                    )
            with gr.Column(visible=False) as gpu_custom_settings:
                with gr.Accordion(i18n("GPU Settings")):
                    gpu = gr.Textbox(
                        label=i18n("GPU Number"),
                        info=i18n(
                            "Specify the number of GPUs you wish to utilize for training by entering them separated by hyphens (-)."
                        ),
                        placeholder=i18n("0 to ∞ separated by -"),
                        value="0",
                        interactive=True,
                    )
                    gr.Textbox(
                        label=i18n("GPU Information"),
                        info=i18n("The GPU information will be displayed here."),
                        value=get_gpu_info(),
                        interactive=False,
                    )
            with gr.Column(visible=False) as overtraining_settings:
                with gr.Accordion(i18n("Overtraining Detector Settings")):
                    overtraining_threshold = gr.Slider(
                        1,
                        100,
                        50,
                        step=1,
                        label=i18n("Overtraining Threshold"),
                        info=i18n(
                            "Set the maximum number of epochs you want your model to stop training if no improvement is detected."
                        ),
                        interactive=True,
                    )

        with gr.Row():
            train_output_info = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
                value="",
                max_lines=8,
                interactive=False,
            )

        with gr.Row():
            train_button = gr.Button(i18n("Start Training"))
            train_button.click(
                run_train_script,
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
                    overtraining_detector,
                    overtraining_threshold,
                    pretrained,
                    custom_pretrained,
                    g_pretrained_path,
                    d_pretrained_path,
                ],
                train_output_info,
                api_name="start_training",
            )

            stop_train_button = gr.Button(
                i18n("Stop Training & Restart Applio"), visible=False
            )
            stop_train_button.click(
                fn=restart_applio,
                inputs=[],
                outputs=[],
            )

            index_button = gr.Button(i18n("Generate Index"))
            index_button.click(
                run_index_script,
                [model_name, rvc_version],
                train_output_info,
                api_name="generate_index",
            )

    with gr.Accordion(i18n("Export Model"), open=False):
        if not os.name == "nt":
            gr.Markdown(
                i18n(
                    "The button 'Upload' is only for google colab: Uploads the exported files to the ApplioExported folder in your Google Drive."
                )
            )
        with gr.Row():
            with gr.Column():
                pth_file_export = gr.File(
                    label=i18n("Exported Pth file"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                pth_dropdown_export = gr.Dropdown(
                    label=i18n("Pth file"),
                    info=i18n("Select the pth file to be exported"),
                    choices=get_pth_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
            with gr.Column():
                index_file_export = gr.File(
                    label=i18n("Exported Index File"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                index_dropdown_export = gr.Dropdown(
                    label=i18n("Index File"),
                    info=i18n("Select the index file to be exported"),
                    choices=get_index_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
        with gr.Row():
            with gr.Column():
                refresh_export = gr.Button(i18n("Refresh"))
                if not os.name == "nt":
                    upload_exported = gr.Button(i18n("Upload"), variant="primary")
                    upload_exported.click(
                        fn=upload_to_google_drive,
                        inputs=[pth_dropdown_export, index_dropdown_export],
                        outputs=[],
                    )

            def toggle_visible(checkbox):
                return {"visible": checkbox, "__type__": "update"}

            def toggle_visible_hop_length(f0method):
                if f0method == "crepe" or f0method == "crepe-tiny":
                    return {"visible": True, "__type__": "update"}
                return {"visible": False, "__type__": "update"}

            def toggle_pretrained(pretrained, custom_pretrained):
                if custom_pretrained == False:
                    return {"visible": pretrained, "__type__": "update"}, {
                        "visible": False,
                        "__type__": "update",
                    }
                else:
                    return {"visible": pretrained, "__type__": "update"}, {
                        "visible": pretrained,
                        "__type__": "update",
                    }

            def enable_stop_train_button():
                return {"visible": False, "__type__": "update"}, {
                    "visible": True,
                    "__type__": "update",
                }

            def disable_stop_train_button():
                return {"visible": True, "__type__": "update"}, {
                    "visible": False,
                    "__type__": "update",
                }

            def download_prerequisites(version):
                for remote_folder, file_list in pretraineds_v1:
                    local_folder = folder_mapping.get(remote_folder, "")
                    missing = False
                    for file in file_list:
                        destination_path = os.path.join(local_folder, file)
                        if not os.path.exists(destination_path):
                            missing = True
                if version == "v1" and missing == True:
                    gr.Info(
                        "Downloading prerequisites... Please wait till it finishes to start preprocessing."
                    )
                    run_prerequisites_script("True", "False", "True", "True")
                    gr.Info(
                        "Prerequisites downloaded successfully, you may now start preprocessing."
                    )

            rvc_version.change(
                fn=download_prerequisites,
                inputs=[rvc_version],
                outputs=[],
            )

            refresh.click(
                fn=refresh_models_and_datasets,
                inputs=[],
                outputs=[model_name, dataset_path],
            )

            dataset_creator.change(
                fn=toggle_visible,
                inputs=[dataset_creator],
                outputs=[dataset_creator_settings],
            )

            upload_audio_dataset.upload(
                fn=save_drop_dataset_audio,
                inputs=[upload_audio_dataset, dataset_name],
                outputs=[upload_audio_dataset, dataset_path],
            )

            f0method.change(
                fn=toggle_visible_hop_length,
                inputs=[f0method],
                outputs=[hop_length],
            )

            pretrained.change(
                fn=toggle_pretrained,
                inputs=[pretrained, custom_pretrained],
                outputs=[custom_pretrained, pretrained_custom_settings],
            )

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

            overtraining_detector.change(
                fn=toggle_visible,
                inputs=[overtraining_detector],
                outputs=[overtraining_settings],
            )

            multiple_gpu.change(
                fn=toggle_visible,
                inputs=[multiple_gpu],
                outputs=[gpu_custom_settings],
            )

            train_button.click(
                fn=enable_stop_train_button,
                inputs=[],
                outputs=[train_button, stop_train_button],
            )

            train_output_info.change(
                fn=disable_stop_train_button,
                inputs=[],
                outputs=[train_button, stop_train_button],
            )

            pth_dropdown_export.change(
                fn=export_pth,
                inputs=[pth_dropdown_export],
                outputs=[pth_file_export],
            )

            index_dropdown_export.change(
                fn=export_index,
                inputs=[index_dropdown_export],
                outputs=[index_file_export],
            )

            refresh_export.click(
                fn=refresh_pth_and_index_list,
                inputs=[],
                outputs=[pth_dropdown_export, index_dropdown_export],
            )
