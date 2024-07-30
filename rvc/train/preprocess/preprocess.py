from multiprocessing import cpu_count
import os
import sys
import time
from typing import List, Tuple
import multiprocessing
from scipy import signal
from scipy.io import wavfile
import librosa
import numpy as np

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

# Define directory paths
GT_WAVS_DIR = f"{experiment_directory}/sliced_audios"
WAVS16K_DIR = f"{experiment_directory}/sliced_audios_16k"

# Create directories if they don't exist
os.makedirs(experiment_directory, exist_ok=True)
os.makedirs(GT_WAVS_DIR, exist_ok=True)
os.makedirs(WAVS16K_DIR, exist_ok=True)


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

    def normalize_and_write(self, tmp_audio: np.ndarray, idx0: int, idx1: int):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return
        tmp_audio = (tmp_audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (
            1 - ALPHA
        ) * tmp_audio
        wavfile.write(
            f"{GT_WAVS_DIR}/{idx0}_{idx1}.wav",
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(
            f"{WAVS16K_DIR}/{idx0}_{idx1}.wav",
            16000,
            tmp_audio.astype(np.float32),
        )

    def process_audio(self, path: str, idx0: int):
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
                        self.normalize_and_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio_segment[start:]
                        idx1 += 1
                        break
                self.normalize_and_write(tmp_audio, idx0, idx1)
        except Exception as error:
            print(f"An error occurred on {path} path: {error}")

    def process_audio_multiprocessing(self, infos: List[Tuple[str, int]]):
        for path, idx0 in infos:
            self.process_audio(path, idx0)

    def process_audio_multiprocessing_input_directory(
        self, input_root: str, num_processes: int
    ):
        try:
            infos = [
                (f"{input_root}/{name}", idx)
                for idx, name in enumerate(sorted(list(os.listdir(input_root))))
            ]
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.map(
                    self.process_audio_multiprocessing,
                    [infos[i::num_processes] for i in range(num_processes)],
                )
        except Exception as error:
            print(f"An error occurred on {input_root} path: {error}")


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
