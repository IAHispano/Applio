import sys

sys.path.append("..")
import os
import shutil

now_dir = os.getcwd()
import soundfile as sf
import librosa
from lib.tools import audioEffects
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()
import gradio as gr
import tabs.resources as resources


def save_to_wav2(dropbox):
    file_path = dropbox.name
    target_path = os.path.join("assets","audios", os.path.basename(file_path))

    if os.path.exists(target_path):
        os.remove(target_path)
        print("Replacing old dropdown file...")

    shutil.move(file_path, target_path)
    return target_path


audio_root = "assets/audios"
audio_others_root = "assets/audios/audio-others"
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
audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_root
]

audio_others_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_others_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_others_root
]


def change_choices3():
    audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_root
    ]
    audio_others_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_others_root, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_others_root
    ]

    return (
        {"choices": sorted(audio_others_paths), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
    )


def generate_output_path(output_folder, base_name, extension):
    index = 1
    while True:
        output_path = os.path.join(output_folder, f"{base_name}_{index}.{extension}")
        if not os.path.exists(output_path):
            return output_path
        index += 1


def combine_and_save_audios(
    audio1_path, audio2_path, output_path, volume_factor_audio1, volume_factor_audio2
):
    audio1, sr1 = librosa.load(audio1_path, sr=None)
    audio2, sr2 = librosa.load(audio2_path, sr=None)

    # Alinear las tasas de muestreo
    if sr1 != sr2:
        if sr1 > sr2:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        else:
            audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=sr2)

    # Ajustar los audios para que tengan la misma longitud
    target_length = min(len(audio1), len(audio2))
    audio1 = librosa.util.fix_length(audio1, target_length)
    audio2 = librosa.util.fix_length(audio2, target_length)

    # Ajustar el volumen de los audios multiplicando por el factor de ganancia
    if volume_factor_audio1 != 1.0:
        audio1 *= volume_factor_audio1
    if volume_factor_audio2 != 1.0:
        audio2 *= volume_factor_audio2

    # Combinar los audios
    combined_audio = audio1 + audio2

    sf.write(output_path, combined_audio, sr1)


def audio_combined(
    audio1_path,
    audio2_path,
    volume_factor_audio1=1.0,
    volume_factor_audio2=1.0,
    reverb_enabled=False,
    compressor_enabled=False,
    noise_gate_enabled=False,
):
    output_folder = os.path.join(now_dir,"assets", "audios", "audio-outputs")
    os.makedirs(output_folder, exist_ok=True)

    # Generar nombres únicos para los archivos de salida
    base_name = "combined_audio"
    extension = "wav"
    output_path = generate_output_path(output_folder, base_name, extension)
    print(reverb_enabled)
    print(compressor_enabled)
    print(noise_gate_enabled)

    if reverb_enabled or compressor_enabled or noise_gate_enabled:
        # Procesa el primer audio con los efectos habilitados
        base_name = "effect_audio"
        output_path = generate_output_path(output_folder, base_name, extension)
        processed_audio_path = audioEffects.process_audio(
            audio2_path,
            output_path,
            reverb_enabled,
            compressor_enabled,
            noise_gate_enabled,
        )
        base_name = "combined_audio"
        output_path = generate_output_path(output_folder, base_name, extension)
        # Combina el audio procesado con el segundo audio usando audio_combined
        combine_and_save_audios(
            audio1_path,
            processed_audio_path,
            output_path,
            volume_factor_audio1,
            volume_factor_audio2,
        )

        return i18n("Conversion complete!"), output_path
    else:
        base_name = "combined_audio"
        output_path = generate_output_path(output_folder, base_name, extension)
        # No hay efectos habilitados, combina directamente los audios sin procesar
        combine_and_save_audios(
            audio1_path,
            audio2_path,
            output_path,
            volume_factor_audio1,
            volume_factor_audio2,
        )

        return i18n("Conversion complete!"), output_path


def merge_audios():
    with gr.Group():
        gr.Markdown(
            value="## " + i18n("Merge your generated audios with the instrumental")
        )
        gr.Markdown(value="", visible=True)
        gr.Markdown(value="", visible=True)
        with gr.Row():
            with gr.Column():
                dropbox = gr.File(label=i18n("Drag your audio here:"))
                gr.Markdown(value=i18n("### Instrumental settings:"))
                input_audio1 = gr.Dropdown(
                    label=i18n("Choose your instrumental:"),
                    choices=sorted(audio_others_paths),
                    value="",
                    interactive=True,
                )
                input_audio1_scale = gr.Slider(
                    minimum=0,
                    maximum=10,
                    label=i18n("Volume of the instrumental audio:"),
                    value=1.00,
                    interactive=True,
                )
                gr.Markdown(value=i18n("### Audio settings:"))
                input_audio3 = gr.Dropdown(
                    label=i18n("Select the generated audio"),
                    choices=sorted(audio_paths),
                    value="",
                    interactive=True,
                )
                with gr.Row():
                    input_audio3_scale = gr.Slider(
                        minimum=0,
                        maximum=10,
                        label=i18n("Volume of the generated audio:"),
                        value=1.00,
                        interactive=True,
                    )

                gr.Markdown(value=i18n("### Add the effects:"))
                reverb_ = gr.Checkbox(
                    label=i18n("Reverb"),
                    value=False,
                    interactive=True,
                )
                compressor_ = gr.Checkbox(
                    label=i18n("Compressor"),
                    value=False,
                    interactive=True,
                )
                noise_gate_ = gr.Checkbox(
                    label=i18n("Noise Gate"),
                    value=False,
                    interactive=True,
                )
                with gr.Row():
                    butnone = gr.Button(i18n("Merge"), variant="primary").style(
                        full_width=True
                    )
                    refresh_button = gr.Button(
                        i18n("Refresh"), variant="primary"
                    ).style(full_width=True)

                vc_output1 = gr.Textbox(label=i18n("Output information:"))
                vc_output2 = gr.Audio(
                    label=i18n(
                        "Export audio (click on the three dots in the lower right corner to download)"
                    ),
                    type="filepath",
                )

                dropbox.upload(
                    fn=save_to_wav2, inputs=[dropbox], outputs=[input_audio1]
                )
                dropbox.upload(
                    fn=resources.change_choices2, inputs=[], outputs=[input_audio1]
                )

                refresh_button.click(
                    fn=lambda: change_choices3(),
                    inputs=[],
                    outputs=[input_audio1, input_audio3],
                )

                butnone.click(
                    fn=audio_combined,
                    inputs=[
                        input_audio1,
                        input_audio3,
                        input_audio1_scale,
                        input_audio3_scale,
                        reverb_,
                        compressor_,
                        noise_gate_,
                    ],
                    outputs=[vc_output1, vc_output2],
                )
