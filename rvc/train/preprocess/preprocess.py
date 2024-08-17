import os
import sys
import time
import torchaudio
import torch
from scipy import signal
from scipy.io import wavfile
import numpy as np
import multiprocessing
from pydub import AudioSegment
from distutils.util import strtobool

multiprocessing.set_start_method("spawn", force=True)

now_directory = os.getcwd()
sys.path.append(now_directory)

from rvc.lib.utils import load_audio
from rvc.train.preprocess.slicer import Slicer

# Remove colab logs
import logging

logging.getLogger("pydub").setLevel(logging.WARNING)
logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

# Constants
OVERLAP = 0.3
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000


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
        self.device = "cpu"
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio(self, audio: torch.Tensor):
        tmp_max = torch.abs(audio).max()
        if tmp_max > 2.5:
            return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def _write_audio(self, audio: torch.Tensor, filename: str, sr: int):
        audio = audio.cpu().numpy()
        wavfile.write(filename, sr, audio.astype(np.float32))

    def process_audio_segment(
        self,
        audio_segment: torch.Tensor,
        idx0: int,
        idx1: int,
        process_effects: bool,
    ):
        if process_effects == False:
            normalized_audio = audio_segment
        else:
            normalized_audio = self._normalize_audio(audio_segment)
        if normalized_audio is None:
            print(f"{idx0}-{idx1}-filtered")
            return

        gt_wav_path = os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav")
        self._write_audio(normalized_audio, gt_wav_path, self.sr)

        resampler = torchaudio.transforms.Resample(
            orig_freq=self.sr, new_freq=SAMPLE_RATE_16K
        ).to(self.device)
        audio_16k = resampler(normalized_audio.float())
        wav_16k_path = os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav")
        self._write_audio(audio_16k, wav_16k_path, SAMPLE_RATE_16K)

    def process_audio(
        self, path: str, idx0: int, cut_preprocess: bool, process_effects: bool
    ):
        try:
            audio = load_audio(path, self.sr)
            if process_effects == False:
                audio = torch.tensor(audio, device=self.device).float()
            else:
                audio = torch.tensor(
                    signal.lfilter(self.b_high, self.a_high, audio), device=self.device
                ).float()
            idx1 = 0
            if cut_preprocess:
                for audio_segment in self.slicer.slice(audio.cpu().numpy()):
                    audio_segment = torch.tensor(
                        audio_segment, device=self.device
                    ).float()
                    i = 0
                    while True:
                        start = int(self.sr * (self.per - OVERLAP) * i)
                        i += 1
                        if len(audio_segment[start:]) > (self.per + OVERLAP) * self.sr:
                            tmp_audio = audio_segment[
                                start : start + int(self.per * self.sr)
                            ]
                            self.process_audio_segment(
                                tmp_audio, idx0, idx1, process_effects
                            )
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(
                                tmp_audio, idx0, idx1, process_effects
                            )
                            idx1 += 1
                            break
            else:
                self.process_audio_segment(audio, idx0, idx1, process_effects)
        except Exception as error:
            print(f"An error occurred on {path} path: {error}")

    def process_audio_file(self, file_path_idx, cut_preprocess, process_effects):
        file_path, idx0 = file_path_idx
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".wav"]:
            audio = AudioSegment.from_file(file_path)
            file_path = os.path.join("/tmp", f"{idx0}.wav")
            audio.export(file_path, format="wav")
        self.process_audio(file_path, idx0, cut_preprocess, process_effects)


def process_file(args):
    pp, file, cut_preprocess, process_effects = args
    pp.process_audio_file(file, cut_preprocess, process_effects)


def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    per: float,
    cut_preprocess: bool,
    process_effects: bool,
):
    start_time = time.time()

    pp = PreProcess(sr, exp_dir, per)
    print(f"Starting preprocess with {num_processes} processes...")

    files = [
        (os.path.join(input_root, f), idx)
        for idx, f in enumerate(os.listdir(input_root))
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    ]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_processes) as pool:
        pool.map(
            process_file,
            [(pp, file, cut_preprocess, process_effects) for file in files],
        )

    elapsed_time = time.time() - start_time
    print(f"Preprocess completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    percentage = float(sys.argv[4])
    num_processes = sys.argv[5]
    if num_processes.lower() == "none":
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = int(num_processes)
    cut_preprocess = strtobool(sys.argv[6])
    process_effects = strtobool(sys.argv[7])

    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        percentage,
        cut_preprocess,
        process_effects,
    )
