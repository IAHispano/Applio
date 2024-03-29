import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa


def calculate_features(y, sr):
    stft = np.abs(librosa.stft(y))
    duration = librosa.get_duration(y=y, sr=sr)
    cent = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
    return stft, duration, cent, bw, rolloff

def plot_title(title):
    plt.suptitle(title, fontsize=16, fontweight="bold")

def plot_spectrogram(y, sr, stft, duration, cmap='inferno'):
    plt.subplot(3, 1, 1)
    plt.imshow(
        librosa.amplitude_to_db(stft, ref=np.max),
        origin="lower",
        extent=[0, duration, 0, sr / 1000],
        aspect="auto",
        cmap=cmap  # Change the colormap here
    )
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Spectrogram")


def plot_waveform(y, sr, duration):
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")


def plot_features(times, cent, bw, rolloff, duration):
    plt.subplot(3, 1, 3)
    plt.plot(times, cent, label="Spectral Centroid (kHz)", color="b")
    plt.plot(times, bw, label="Spectral Bandwidth (kHz)", color="g")
    plt.plot(times, rolloff, label="Spectral Rolloff (kHz)", color="r")
    plt.xlabel("Time (s)")
    plt.title("Spectral Features")
    plt.legend()


def analyze_audio(audio_file, save_plot_path="logs/audio_analysis.png"):
    y, sr = librosa.load(audio_file)
    stft, duration, cent, bw, rolloff = calculate_features(y, sr)

    plt.figure(figsize=(12, 10))
    
    plot_title("Audio Analysis" + " - " + audio_file.split("/")[-1])
    plot_spectrogram(y, sr, stft, duration)
    plot_waveform(y, sr, duration)
    plot_features(librosa.times_like(cent), cent, bw, rolloff, duration)

    plt.tight_layout()

    if save_plot_path:
        plt.savefig(save_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    audio_info = f"""
    - Sample Rate: {sr},
    - Duration: {(
            str(round(duration, 2)) + " seconds"
            if duration < 60
            else str(round(duration / 60, 2)) + " minutes"
    )},
    - Number of Samples: {len(y)},
    - Bits per Sample: {librosa.get_samplerate(audio_file)},
    - Channels: {"Mono (1)" if y.ndim == 1 else "Stereo (2)"}
    """
    return audio_info, save_plot_path
