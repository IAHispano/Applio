import os
import shutil
import sys
from multiprocessing import cpu_count

import gradio as gr

from assets.i18n.i18n import I18nAuto
from core import (
    run_extract_script,
    run_index_script,
    run_preprocess_script,
    run_prerequisites_script,
    run_train_script,
)
from rvc.configs.config import get_gpu_info, get_number_of_gpus, max_vram_gpu
from rvc.lib.utils import format_title
from tabs.settings.sections.restart import stop_train

i18n = I18nAuto()
now_dir = os.getcwd()
sys.path.append(now_dir)


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
    now_dir, "rvc", "models", "pretraineds", "custom"
)

pretraineds_custom_path_relative = os.path.relpath(pretraineds_custom_path, now_dir)

custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)
custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)

os.makedirs(custom_embedder_root, exist_ok=True)
os.makedirs(pretraineds_custom_path_relative, exist_ok=True)


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
        and all(excluded not in dirpath for excluded in ["zips", "mute", "reference"])
    ]


def refresh_models():
    return {"choices": sorted(get_models_list()), "__type__": "update"}


# Refresh Models and Datasets
def refresh_models_and_datasets():
    return (
        {"choices": sorted(get_models_list()), "__type__": "update"},
        {"choices": sorted(get_datasets_list()), "__type__": "update"},
    )


# Refresh Custom Embedders
def get_embedder_custom_list():
    return [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]


def refresh_custom_embedder_list():
    return {"choices": sorted(get_embedder_custom_list()), "__type__": "update"}


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
        shutil.copy(dropbox, pretrained_path)
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
            shutil.copy(dropbox, destination_path)
            gr.Info(
                i18n(
                    "The audio file has been successfully added to the dataset. Please click the preprocess button."
                )
            )
            dataset_path = os.path.dirname(destination_path)
            relative_dataset_path = os.path.relpath(dataset_path, now_dir)

            return None, relative_dataset_path


# Drop Custom Embedder
def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)

    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))
    if config_file:
        shutil.copy(
            config_file, os.path.join(target_folder, os.path.basename(config_file))
        )

    return f"Files moved to folder {target_folder}"


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders


# Export
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


# Export Pth and Index Files
def export_pth(pth_path):
    allowed_paths = get_pth_list()
    normalized_allowed_paths = [
        os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths
    ]
    normalized_pth_path = os.path.abspath(os.path.join(now_dir, pth_path))

    if normalized_pth_path in normalized_allowed_paths:
        return pth_path
    else:
        print(f"Attempted to export invalid pth path: {pth_path}")
        return None


def export_index(index_path):
    allowed_paths = get_index_list()
    normalized_allowed_paths = [
        os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths
    ]
    normalized_index_path = os.path.abspath(os.path.join(now_dir, index_path))

    if normalized_index_path in normalized_allowed_paths:
        return index_path
    else:
        print(f"Attempted to export invalid index path: {index_path}")
        return None


# Upload to Google Drive
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
                print(f"An error occurred uploading to Google Drive: {error}")
                gr.Info("Error uploading to Google Drive")

    upload_file(pth_path)
    upload_file(index_path)


def auto_enable_checkpointing():
    try:
        return max_vram_gpu(0) < 6
    except:
        return False


# Train Tab
def train_tab():
    # Model settings section
    with gr.Accordion(i18n("Model Settings")):
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
                architecture = gr.Radio(
                    label=i18n("Architecture"),
                    info=i18n(
                        "Choose the model architecture:\n- **RVC (V2)**: Default option, compatible with all clients.\n- **Applio**: Advanced quality with improved vocoders and higher sample rates, Applio-only."
                    ),
                    choices=["RVC", "Applio"],
                    value="RVC",
                    interactive=True,
                    visible=False,  # to be visible once pretraineds are ready
                )
            with gr.Column():
                sampling_rate = gr.Radio(
                    label=i18n("Sampling Rate"),
                    info=i18n("The sampling rate of the audio files."),
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    interactive=True,
                )
                vocoder = gr.Radio(
                    label=i18n("Vocoder"),
                    info=i18n(
                        "Choose the vocoder for audio synthesis:\n- **HiFi-GAN**: Default option, compatible with all clients.\n- **MRF HiFi-GAN**: Higher fidelity, Applio-only.\n- **RefineGAN**: Superior audio quality, Applio-only."
                    ),
                    choices=["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
                    value="HiFi-GAN",
                    interactive=False,
                    visible=False,  # to be visible once pretraineds are ready
                )
        with gr.Accordion(
            i18n("Advanced Settings"),
            open=False,
        ):
            with gr.Row():
                with gr.Column():
                    cpu_cores = gr.Slider(
                        1,
                        min(cpu_count(), 32),  # max 32 parallel processes
                        min(cpu_count(), 32),
                        step=1,
                        label=i18n("CPU Cores"),
                        info=i18n(
                            "The number of CPU cores to use in the extraction process. The default setting are your cpu cores, which is recommended for most cases."
                        ),
                        interactive=True,
                    )

                with gr.Column():
                    gpu = gr.Textbox(
                        label=i18n("GPU Number"),
                        info=i18n(
                            "Specify the number of GPUs you wish to utilize for extracting by entering them separated by hyphens (-)."
                        ),
                        placeholder=i18n("0 to ∞ separated by -"),
                        value=str(get_number_of_gpus()),
                        interactive=True,
                    )
                    gr.Textbox(
                        label=i18n("GPU Information"),
                        info=i18n("The GPU information will be displayed here."),
                        value=get_gpu_info(),
                        interactive=False,
                    )
    # Preprocess section
    with gr.Accordion(i18n("Preprocess")):
        dataset_path = gr.Dropdown(
            label=i18n("Dataset Path"),
            info=i18n("Path to the dataset folder."),
            # placeholder=i18n("Enter dataset path"),
            choices=get_datasets_list(),
            allow_custom_value=True,
            interactive=True,
        )
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
        refresh = gr.Button(i18n("Refresh"))

        with gr.Accordion(i18n("Advanced Settings"), open=False):
            cut_preprocess = gr.Radio(
                label=i18n("Audio cutting"),
                info=i18n(
                    "Audio file slicing method: Select 'Skip' if the files are already pre-sliced, 'Simple' if excessive silence has already been removed from the files, or 'Automatic' for automatic silence detection and slicing around it."
                ),
                choices=["Skip", "Simple", "Automatic"],
                value="Automatic",
                interactive=True,
            )
            with gr.Row():
                chunk_len = gr.Slider(
                    0.5,
                    5.0,
                    3.0,
                    step=0.1,
                    label=i18n("Chunk length (sec)"),
                    info=i18n("Length of the audio slice for 'Simple' method."),
                    interactive=True,
                )
                overlap_len = gr.Slider(
                    0.0,
                    0.4,
                    0.3,
                    step=0.1,
                    label=i18n("Overlap length (sec)"),
                    info=i18n(
                        "Length of the overlap between slices for 'Simple' method."
                    ),
                    interactive=True,
                )

            with gr.Row():
                process_effects = gr.Checkbox(
                    label=i18n("Noise filter"),
                    info=i18n(
                        "It's recommended to deactivate this option if your dataset has already been processed."
                    ),
                    value=True,
                    interactive=True,
                    visible=True,
                )

                normalization_mode = gr.Radio(
                    label=i18n("Normalization mode"),
                    info=i18n(
                        "Audio normalization: Select 'none' if the files are already normalized, 'pre' to normalize the entire input file at once, or 'post' to normalize each slice individually."
                    ),
                    choices=["none", "pre", "post"],
                    value="none",
                    interactive=True,
                    visible=True,
                )

                noise_reduction = gr.Checkbox(
                    label=i18n("Noise Reduction"),
                    info=i18n(
                        "It's recommended keep deactivate this option if your dataset has already been processed."
                    ),
                    value=False,
                    interactive=True,
                    visible=True,
                )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Noise Reduction Strength"),
                info=i18n(
                    "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                ),
                visible=False,
                value=0.5,
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
                fn=run_preprocess_script,
                inputs=[
                    model_name,
                    dataset_path,
                    sampling_rate,
                    cpu_cores,
                    cut_preprocess,
                    process_effects,
                    noise_reduction,
                    clean_strength,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                ],
                outputs=[preprocess_output_info],
            )

    # Extract section
    with gr.Accordion(i18n("Extract")):
        with gr.Row():
            f0_method = gr.Radio(
                label=i18n("Pitch extraction algorithm"),
                info=i18n(
                    "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                ),
                choices=[
                    "crepe",
                    "crepe-tiny",
                    "rmvpe",
                    # "fcpe"
                ],
                value="rmvpe",
                interactive=True,
            )

            embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Model used for learning speaker embedding."),
                choices=[
                    "contentvec",
                    # "spin",
                    "spin-v2",
                    # "chinese-hubert-base",
                    # "japanese-hubert-base",
                    # "korean-hubert-base",
                    "custom",
                ],
                value="contentvec",
                interactive=True,
            )
        include_mutes = gr.Slider(
            0,
            10,
            2,
            step=1,
            label=i18n("Silent training files"),
            info=i18n(
                "Adding several silent files to the training set enables the model to handle pure silence in inferred audio files. Select 0 if your dataset is clean and already contains segments of pure silence."
            ),
            value=True,
            interactive=True,
        )
        with gr.Row(visible=False) as embedder_custom:
            with gr.Accordion(i18n("Custom Embedder"), open=True):
                with gr.Row():
                    embedder_model_custom = gr.Dropdown(
                        label=i18n("Select Custom Embedder"),
                        choices=refresh_embedders_folders(),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    refresh_embedders_button = gr.Button(i18n("Refresh embedders"))
                folder_name_input = gr.Textbox(
                    label=i18n("Folder Name"), interactive=True
                )
                with gr.Row():
                    bin_file_upload = gr.File(
                        label=i18n("Upload .bin"), type="filepath", interactive=True
                    )
                    config_file_upload = gr.File(
                        label=i18n("Upload .json"), type="filepath", interactive=True
                    )
                move_files_button = gr.Button(
                    i18n("Move files to custom embedder folder")
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
            fn=run_extract_script,
            inputs=[
                model_name,
                f0_method,
                cpu_cores,
                gpu,
                sampling_rate,
                embedder_model,
                embedder_model_custom,
                include_mutes,
            ],
            outputs=[extract_output_info],
        )

    # Training section
    with gr.Accordion(i18n("Training")):
        with gr.Row():
            batch_size = gr.Slider(
                1,
                64,
                4,
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
                200,
                step=1,
                label=i18n("Total Epoch"),
                info=i18n(
                    "Specifies the overall quantity of epochs for the model training process."
                ),
                interactive=True,
            )
        with gr.Accordion(i18n("Advanced Settings"), open=False):
            with gr.Row():
                with gr.Column():
                    save_only_latest = gr.Checkbox(
                        label=i18n("Save Only Latest"),
                        info=i18n(
                            "Enabling this setting will result in the G and D files saving only their most recent versions, effectively conserving storage space."
                        ),
                        value=True,
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
                    pretrained = gr.Checkbox(
                        label=i18n("Pretrained"),
                        info=i18n(
                            "Utilize pretrained models when training your own. This approach reduces training duration and enhances overall quality."
                        ),
                        value=True,
                        interactive=True,
                    )
                with gr.Column():
                    cleanup = gr.Checkbox(
                        label=i18n("Fresh Training"),
                        info=i18n(
                            "Enable this setting only if you are training a new model from scratch or restarting the training. Deletes all previously generated weights and tensorboard logs."
                        ),
                        value=False,
                        interactive=True,
                    )
                    cache_dataset_in_gpu = gr.Checkbox(
                        label=i18n("Cache Dataset in GPU"),
                        info=i18n(
                            "Cache the dataset in GPU memory to speed up the training process."
                        ),
                        value=False,
                        interactive=True,
                    )
                    checkpointing = gr.Checkbox(
                        label=i18n("Checkpointing"),
                        info=i18n(
                            "Enables memory-efficient training. This reduces VRAM usage at the cost of slower training speed. It is useful for GPUs with limited memory (e.g., <6GB VRAM) or when training with a batch size larger than what your GPU can normally accommodate."
                        ),
                        value=auto_enable_checkpointing(),
                        interactive=True,
                    )
            with gr.Row():
                custom_pretrained = gr.Checkbox(
                    label=i18n("Custom Pretrained"),
                    info=i18n(
                        "Utilizing custom pretrained models can lead to superior results, as selecting the most suitable pretrained models tailored to the specific use case can significantly enhance performance."
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
            index_algorithm = gr.Radio(
                label=i18n("Index Algorithm"),
                info=i18n(
                    "KMeans is a clustering algorithm that divides the dataset into K clusters. This setting is particularly useful for large datasets."
                ),
                choices=["Auto", "Faiss", "KMeans"],
                value="Auto",
                interactive=True,
            )

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message
            return run_train_script(*args)

        terms_checkbox = gr.Checkbox(
            label=i18n("I agree to the terms of use"),
            info=i18n(
                "Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your training."
            ),
            value=False,
            interactive=True,
        )
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
                fn=enforce_terms,
                inputs=[
                    terms_checkbox,
                    model_name,
                    save_every_epoch,
                    save_only_latest,
                    save_every_weights,
                    total_epoch,
                    sampling_rate,
                    batch_size,
                    gpu,
                    overtraining_detector,
                    overtraining_threshold,
                    pretrained,
                    cleanup,
                    index_algorithm,
                    cache_dataset_in_gpu,
                    custom_pretrained,
                    g_pretrained_path,
                    d_pretrained_path,
                    vocoder,
                    checkpointing,
                ],
                outputs=[train_output_info],
            )

            stop_train_button = gr.Button(i18n("Stop Training"), visible=False)
            stop_train_button.click(
                fn=stop_train,
                inputs=[model_name],
                outputs=[],
            )

            index_button = gr.Button(i18n("Generate Index"))
            index_button.click(
                fn=run_index_script,
                inputs=[model_name, index_algorithm],
                outputs=[train_output_info],
            )

    # Export Model section
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
                    upload_exported = gr.Button(i18n("Upload"))
                    upload_exported.click(
                        fn=upload_to_google_drive,
                        inputs=[pth_dropdown_export, index_dropdown_export],
                        outputs=[],
                    )

            def toggle_visible(checkbox):
                return {"visible": checkbox, "__type__": "update"}

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

            def download_prerequisites():
                gr.Info(
                    "Checking for prerequisites with pitch guidance... Missing files will be downloaded. If you already have them, this step will be skipped."
                )
                run_prerequisites_script(
                    pretraineds_hifigan=True,
                    models=False,
                    exe=False,
                )
                gr.Info(
                    "Prerequisites check complete. Missing files were downloaded, and you may now start preprocessing."
                )

            def toggle_visible_embedder_custom(embedder_model):
                if embedder_model == "custom":
                    return {"visible": True, "__type__": "update"}
                return {"visible": False, "__type__": "update"}

            def toggle_architecture(architecture):
                if architecture == "Applio":
                    return {
                        "choices": ["32000", "40000", "48000"],
                        "__type__": "update",
                    }, {
                        "interactive": True,
                        "__type__": "update",
                    }
                else:
                    return {
                        "choices": ["32000", "40000", "48000"],
                        "__type__": "update",
                        "value": "40000",
                    }, {"interactive": False, "__type__": "update", "value": "HiFi-GAN"}

            def update_slider_visibility(noise_reduction):
                return gr.update(visible=noise_reduction)

            noise_reduction.change(
                fn=update_slider_visibility,
                inputs=noise_reduction,
                outputs=clean_strength,
            )
            architecture.change(
                fn=toggle_architecture,
                inputs=[architecture],
                outputs=[sampling_rate, vocoder],
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
            embedder_model.change(
                fn=toggle_visible_embedder_custom,
                inputs=[embedder_model],
                outputs=[embedder_custom],
            )
            embedder_model.change(
                fn=toggle_visible_embedder_custom,
                inputs=[embedder_model],
                outputs=[embedder_custom],
            )
            move_files_button.click(
                fn=create_folder_and_move_files,
                inputs=[folder_name_input, bin_file_upload, config_file_upload],
                outputs=[],
            )
            refresh_embedders_button.click(
                fn=refresh_embedders_folders, inputs=[], outputs=[embedder_model_custom]
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
