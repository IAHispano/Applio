import os
import librosa
import gradio as gr
from matplotlib import pyplot as plt

from rvc.lib.predictors.F0Extractor import F0Extractor

from assets.i18n.i18n import I18nAuto
from assets.app_shell import step_card

i18n = I18nAuto()


def extract_f0_curve(audio_path: str, method: str):
    """Export a pitch plot and curve without decoding the input twice."""
    print("Extracting F0 Curve...")
    image_path = os.path.join("logs", "f0_plot.png")
    txt_path = os.path.join("logs", "f0_curve.txt")
    sr = librosa.get_samplerate(audio_path)
    hop_length = 160

    librosa.note_to_hz("C1")
    librosa.note_to_hz("C8")

    f0_extractor = F0Extractor(audio_path, sample_rate=sr, method=method)
    f0 = f0_extractor.extract_f0()

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title(method)
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (Hz)")
    plt.savefig(image_path)
    plt.close()

    with open(txt_path, "w") as txtfile:
        for i, f0_value in enumerate(f0):
            frequency = i * sr / hop_length
            txtfile.write(f"{frequency},{f0_value}\n")

    print("F0 Curve extracted successfully!")
    return image_path, txt_path


def f0_extractor_tab():
    with step_card(1, "Audio Source", i18n, description="Upload audio and pick the pitch algorithm."):
        audio = gr.Audio(label=i18n("Upload Audio"), type="filepath")
        f0_method = gr.Dropdown(
            filterable=False,
            label=i18n("Pitch extraction algorithm"),
            info=i18n(
                "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
            ),
            choices=["crepe", "fcpe", "rmvpe"],
            value="rmvpe",
        )
    with step_card(2, "Extract & Results", i18n, description="Run the extraction and inspect the curve."):
        button = gr.Button(i18n("Extract F0 Curve"))

        with gr.Row(equal_height=True, elem_classes="applio-f0-results"):
            txt_output = gr.File(label=i18n("F0 Curve"), type="filepath", height=280)
            image_output = gr.Image(
                label=i18n("F0 Plot"), type="filepath", interactive=False, height=280
            )

    button.click(
        fn=extract_f0_curve,
        inputs=[
            audio,
            f0_method,
        ],
        outputs=[image_output, txt_output],
    )
