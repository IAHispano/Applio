import gradio as gr
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os


def generate_spectrogram(audio_data, sample_rate, file_name):
    plt.clf()

    plt.specgram(
        audio_data,
        Fs=sample_rate / 1,
        NFFT=4096,
        sides="onesided",
        cmap="Reds_r",
        scale_by_freq=True,
        scale="dB",
        mode="magnitude",
        window=np.hanning(4096),
    )

    plt.title(file_name)
    plt.savefig("spectrogram.png")


def get_audio_info(audio_file):
    audio_data, sample_rate = sf.read(audio_file)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    generate_spectrogram(audio_data, sample_rate, os.path.basename(audio_file))

    audio_info = sf.info(audio_file)
    bit_depth = {"PCM_16": 16, "FLOAT": 32}.get(audio_info.subtype, 0)

    minutes, seconds = divmod(audio_info.duration, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds *= 1000

    speed_in_kbps = audio_info.samplerate * bit_depth / 1000

    info_table = f"""
    - **File Name:** {os.path.basename(audio_file)}
    - **Duration:** {int(minutes)} minutes, {int(seconds)} seconds, {int(milliseconds)} milliseconds
    - **Bitrate:** {speed_in_kbps} kbp/s
    - **Audio Channels:** {audio_info.channels}
    - **Sampling rate:** {audio_info.samplerate} Hz
    - **Bit per second:** {audio_info.samplerate * audio_info.channels * bit_depth} bit/s
    """

    return info_table, "spectrogram.png"


def analyzer():
    with gr.Column():
        audio_input = gr.Audio(type="filepath")
        get_info_button = gr.Button(
            value="Get information about the audio", variant="primary"
        )
    with gr.Column():
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    value="**Information about the audio file:**",
                    visible=True,
                )
                output_markdown = gr.Markdown(
                    value="Waiting for information...", visible=True
                )
            image_output = gr.Image(type="filepath", interactive=False)

    get_info_button.click(
        fn=get_audio_info,
        inputs=[audio_input],
        outputs=[output_markdown, image_output],
    )
