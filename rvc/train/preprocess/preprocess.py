import os
import sys
import time
import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
from multiprocessing import cpu_count, Pool

now_directory = os.getcwd()
sys.path.append(now_directory)

from rvc.lib.utils import load_audio
from rvc.train.slicer import Slicer

# Parse command line arguments
experiment_directory = str(sys.argv[1])
input_root = str(sys.argv[2])
sample_rate = int(sys.argv[3])
percentage = float(sys.argv[4])
num_processes = int(sys.argv[5]) if len(sys.argv) > 5 else cpu_count()

# Define constants
OVERLAP = 0.3
TAIL = percentage + OVERLAP
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000

# Define directory paths
GT_WAVS_DIR = os.path.join(experiment_directory, "sliced_audios")
WAVS16K_DIR = os.path.join(experiment_directory, "sliced_audios_16k")


class PreProcess:
    def __init__(self, sr: int, exp_dir: str, per: float):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr
        )
        self.per = per
        self.exp_dir = exp_dir

    def _normalize_audio(self, audio: np.ndarray):
        """Normalizes the audio to the desired amplitude."""
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None  # Indicate audio should be filtered out
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def _write_audio(self, audio: np.ndarray, filename: str, sr: int):
        """Writes the audio to a WAV file."""
        wavfile.write(filename, sr, audio.astype(np.float32))

    def process_audio_segment(self, audio_segment: np.ndarray, idx0: int, idx1: int):
        """Processes a single audio segment."""
        normalized_audio = self._normalize_audio(audio_segment)
        if normalized_audio is None:
            print(f"{idx0}-{idx1}-filtered")
            return

        # Write original sample rate audio
        gt_wav_path = os.path.join(GT_WAVS_DIR, f"{idx0}_{idx1}.wav")
        self._write_audio(normalized_audio, gt_wav_path, self.sr)

        # Resample and write 16kHz audio
        audio_16k = librosa.resample(
            normalized_audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K
        )
        wav_16k_path = os.path.join(WAVS16K_DIR, f"{idx0}_{idx1}.wav")
        self._write_audio(audio_16k, wav_16k_path, SAMPLE_RATE_16K)

    def process_audio(self, path: str, idx0: int):
        """Processes a single audio file."""
        try:
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(self.b_high, self.a_high, audio)

            idx1 = 0
            for audio_segment in self.slicer.slice(audio):
                i = 0
                while True:
                    start = int(self.sr * (self.per - OVERLAP) * i)
                    i += 1
                    if len(audio_segment[start:]) > TAIL * self.sr:
                        tmp_audio = audio_segment[
                            start : start + int(self.per * self.sr)
                        ]
                        self.process_audio_segment(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio_segment[start:]
                        self.process_audio_segment(tmp_audio, idx0, idx1)
                        idx1 += 1
                        break
        except Exception as error:
            print(f"An error occurred on {path} path: {error}")

    def process_audio_file(self, file_path_idx):
        file_path, idx0 = file_path_idx
        self.process_audio(file_path, idx0)

    def process_audio_multiprocessing_input_directory(
        self, input_root: str, num_processes: int
    ):
        # Get list of files
        files = [
            (os.path.join(input_root, f), idx)
            for idx, f in enumerate(os.listdir(input_root))
            if f.endswith(".wav")
        ]

        # Create the directories if they don't exist
        os.makedirs(GT_WAVS_DIR, exist_ok=True)
        os.makedirs(WAVS16K_DIR, exist_ok=True)

        # Use multiprocessing to process files
        with Pool(processes=num_processes) as pool:
            pool.map(self.process_audio_file, files)


def preprocess_training_set(
    input_root: str, sr: int, num_processes: int, exp_dir: str, per: float
):
    start_time = time.time()
    pp = PreProcess(sr, exp_dir, per)
    print(f"Starting preprocess with {num_processes} cores...")
    pp.process_audio_multiprocessing_input_directory(input_root, num_processes)
    elapsed_time = time.time() - start_time
    print(f"Preprocess completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    preprocess_training_set(
        input_root, sample_rate, num_processes, experiment_directory, percentage
    )
