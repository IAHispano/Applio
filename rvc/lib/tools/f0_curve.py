"""F0 curve extraction (engine utility, no UI dependencies)."""

import os

import librosa
from matplotlib import pyplot as plt

from rvc.lib.predictors.F0Extractor import F0Extractor


def extract_f0_curve(
    audio_path: str,
    method: str,
    image_path: str = None,
    txt_path: str = None,
):
    print("Extracting F0 Curve...")
    image_path = image_path or os.path.join("logs", "f0_plot.png")
    txt_path = txt_path or os.path.join("logs", "f0_curve.txt")
    y, sr = librosa.load(audio_path, sr=None)
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
