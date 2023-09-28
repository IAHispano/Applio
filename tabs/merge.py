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
import numpy as np
from scipy.signal import resample

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

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import glob
import re
def combine_and_save_audios(
    audio1_path, audio2_path, output_path, volume_factor_audio1, volume_factor_audio2
):
    audio1 = AudioSegment.from_file(audio1_path)
    audio2 = AudioSegment.from_file(audio2_path)
    
    # Verificar cuál audio tiene mayor longitud
    if len(audio1) > len(audio2):
        # Calcular la diferencia en duración en segundos
        diff_duration_seconds = (len(audio1) - len(audio2)) / 1000.0  # Convertir a segundos
        print(f"diff_duration_seconds: {diff_duration_seconds} seconds")
        # Crear el segmento de silencio en Pydub
        silence = AudioSegment.silent(duration=int(diff_duration_seconds))  # Convertir a milisegundos

        # Agregar el silencio al audio2 para igualar la duración
        audio2 = audio2 + silence
    else:
        # Calcular la diferencia en duración en segundos
        diff_duration_seconds = (len(audio2) - len(audio1)) / 1000.0  # Convertir a segundos
        print(f"diff_duration_seconds: {diff_duration_seconds} seconds")
        # Crear el segmento de silencio en Pydub
        silence = AudioSegment.silent(duration=int(diff_duration_seconds))  # Convertir a milisegundos

        # Agregar el silencio al audio1 para igualar la duración
        audio1 = audio1 + silence

    # Ajustar el volumen de los audios multiplicando por el factor de ganancia
    if volume_factor_audio1 != 1.0:
        audio1 *= volume_factor_audio1
    if volume_factor_audio2 != 1.0:
        audio2 *= volume_factor_audio2

    # Combinar los audios
    combined_audio = audio1.overlay(audio2)

    # Guardar el audio combinado en el archivo de salida
    combined_audio.export(output_path, format="wav")


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

def process_audio(file_path):
    try:
        # load audio file
        song = AudioSegment.from_file(file_path)

        print(f"Ignore the warning if you saw any...")

        # set silence threshold and duration
        silence_thresh = -70  # dB
        min_silence_len = 750  # ms, adjust as needed

        # detect nonsilent parts
        nonsilent_parts = detect_nonsilent(song, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

        # Create a new directory to store chunks
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).split('.')[0]
        new_dir_path = os.path.join(file_dir, file_name)
        os.makedirs(new_dir_path, exist_ok=True)

        # Check if timestamps file exists, if so delete it
        timestamps_file = os.path.join(file_dir, f"{file_name}_timestamps.txt")
        if os.path.isfile(timestamps_file):
            os.remove(timestamps_file)

        # export chunks and save start times
        segment_count = 0
        for i, (start_i, end_i) in enumerate(nonsilent_parts):
            chunk = song[start_i:end_i]
            chunk_file_path = os.path.join(new_dir_path, f"chunk{i}.wav")
            chunk.export(chunk_file_path, format="wav")

            print(f"Segment {i} created!")
            segment_count += 1
            
            # write start times to file
            with open(timestamps_file, "a", encoding="utf-8") as f:
                f.write(f"{chunk_file_path} starts at {start_i} ms\n")
        
        print(f"Total segments created: {segment_count}")
        print(f"Split all chunks for {file_path} successfully!")

        return "Finish", new_dir_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", None


def merge_audio(timestamps_file):
    try:
        # Extract prefix from the timestamps filename
        prefix = os.path.basename(timestamps_file).replace('_timestamps.txt', '')
        timestamps_dir = os.path.dirname(timestamps_file)
        print(timestamps_dir)
        print(prefix)

        # Open the timestamps file
        with open(timestamps_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Initialize empty list to hold audio segments
        audio_segments = []
        last_end_time = 0

        print(f"Processing file: {timestamps_file}")

        for line in lines:
            # Extract filename and start time from line
            match = re.search(r"(chunk\d+.wav) starts at (\d+) ms", line)
            if match:
                filename, start_time = match.groups()
                start_time = int(start_time)

                # Construct the complete path to the chunk file
                chunk_file = os.path.join(timestamps_dir, prefix, filename)

                # Add silence from last_end_time to start_time
                silence_duration = max(start_time - last_end_time, 0)
                silence = AudioSegment.silent(duration=silence_duration)
                audio_segments.append(silence)

                # Load audio file and append to list
                audio = AudioSegment.from_wav(chunk_file)
                audio_segments.append(audio)

                # Update last_end_time
                last_end_time = start_time + len(audio)

                print(f"Processed chunk: {chunk_file}")

        # Concatenate all audio_segments and export
        merged_filename = f"{prefix}_merged.wav"
        merged_audio = sum(audio_segments)
        merged_audio.export(os.path.join(timestamps_dir, "audio-outputs", merged_filename), format="wav")

        print(f"Exported merged file: {merged_filename}\n")

    except Exception as e:
        print(f"An error occurred: {e}")




def merge_audios():
        gr.Markdown(
            value="## " + i18n("Merge your generated audios with the instrumental")
        )
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
