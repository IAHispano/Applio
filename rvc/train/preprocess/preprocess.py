import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
from fractions import Fraction


def strtobool(val):
    """Convert a string representation of truth to a bool."""
    return val.lower() in ("yes", "true", "t", "y", "1")


import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import soxr
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

now_directory = os.getcwd()
sys.path.append(now_directory)

import logging

from rvc.lib.utils import load_audio
from rvc.train.preprocess.slicer import Slicer

logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
RES_TYPE = "soxr_vhq"

# shortest tail worth keeping as its own slice in "Simple" cutting
MIN_TAIL_SECONDS = 1.0

# wav is float32 with no level ceiling, flac is half the size but 24-bit PCM
DATASET_FORMATS = ("wav", "flac")
FLAC_COMPRESSION_LEVEL = 5 / 8  # libsndfile takes 0.0-1.0; this is FLAC level 5


def secs_to_samples(seconds: float, sr: int) -> int:
    """
    Converts seconds to an exact sample count, refusing fractional results.

    `int(sr * seconds)` truncates, so a chunk and its overlap can disagree by a
    sample and the stride drifts across a long file.

    Args:
        seconds (float): Duration to convert.
        sr (int): Sampling rate.
    """
    frac = Fraction(str(seconds)) * sr
    if frac.denominator != 1:
        raise ValueError(f"{seconds}s at {sr} Hz is not a whole number of samples.")
    return frac.numerator


def save_audio(
    directory: str,
    name: str,
    sample_rate: int,
    dataset_format: str,
    audio: np.ndarray,
):
    """
    Writes one slice in the requested dataset format.

    Args:
        directory (str): Directory to write into.
        name (str): File name without the extension.
        sample_rate (int): Sampling rate of the audio.
        dataset_format (str): Either "wav" or "flac".
        audio (np.ndarray): Audio data array.
    """
    if dataset_format == "flac":
        sf.write(
            os.path.join(directory, f"{name}.flac"),
            audio,
            sample_rate,
            format="FLAC",
            subtype="PCM_24",
            compression_level=FLAC_COMPRESSION_LEVEL,
        )
    else:
        wavfile.write(
            os.path.join(directory, f"{name}.wav"),
            sample_rate,
            audio.astype(np.float32),
        )


class PreProcess:
    def __init__(self, sr: int, exp_dir: str, dataset_format: str = "wav"):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.dataset_format = dataset_format
        # sos rather than ba: at 48 Hz the poles sit at |p| = 0.998, which makes
        # the transfer-function form badly conditioned
        self.hp_sos = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr, output="sos"
        )
        self.hp_zi = signal.sosfilt_zi(self.hp_sos)
        self.exp_dir = exp_dir
        self.device = "cpu"
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def high_pass(self, audio: np.ndarray) -> np.ndarray:
        """
        Removes DC offset and subsonic rumble.

        The filter state is primed with the mean of the opening window instead
        of starting at zero, which would read a DC offset as a step and leave a
        ~45 ms thump at the head of the first slice of every file.

        Args:
            audio (np.ndarray): Audio data array.
        """
        if audio.size == 0:
            return audio
        # the pole time constant is 3.3 ms, so 20 ms is enough to estimate DC
        window = min(audio.size, int(0.02 * self.sr))
        prime = float(np.mean(audio[:window]))
        filtered, _ = signal.sosfilt(self.hp_sos, audio, zi=self.hp_zi * prime)
        # sosfilt promotes to float64, the rest of the pipeline is float32
        return filtered.astype(np.float32, copy=False)

    def _normalize_audio(self, audio: np.ndarray):
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None
        if tmp_max == 0:
            # digital silence, dividing by the peak would fill the slice with NaN
            return audio
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def _fit_to_flac(self, audio: np.ndarray, audio_16k: np.ndarray, label: str):
        """
        Scales a slice pair down so 24-bit PCM can hold it.

        FLAC clips outside [-1, 1] and the peak blend above can reach 1.3. Both
        copies take the same factor so they stay level-matched.

        Args:
            audio (np.ndarray): Full rate slice.
            audio_16k (np.ndarray): The 16 kHz copy of the same slice.
            label (str): Slice name, for the warning.
        """
        peak = max(float(np.abs(audio).max()), float(np.abs(audio_16k).max()))
        if peak <= 1.0:
            return audio, audio_16k
        print(f"{label}: peak {peak:.2f} scaled down to fit 24-bit FLAC.")
        return audio / peak, audio_16k / peak

    def process_audio_segment(
        self,
        normalized_audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        normalization_mode: str,
    ):
        if normalized_audio is None or normalized_audio.size == 0:
            print(f"{sid}-{idx0}-{idx1}-filtered")
            return
        if normalization_mode == "post":
            normalized_audio = self._normalize_audio(normalized_audio)
            if normalized_audio is None:
                print(f"{sid}-{idx0}-{idx1}-filtered")
                return
        audio_16k = librosa.resample(
            normalized_audio,
            orig_sr=self.sr,
            target_sr=SAMPLE_RATE_16K,
            res_type=RES_TYPE,
        )
        name = f"{sid}_{idx0}_{idx1}"
        if self.dataset_format == "flac":
            normalized_audio, audio_16k = self._fit_to_flac(
                normalized_audio, audio_16k, name
            )
        # full SR for training
        save_audio(
            self.gt_wavs_dir, name, self.sr, self.dataset_format, normalized_audio
        )
        # 16KHz for feature extraction
        save_audio(
            self.wavs16k_dir, name, SAMPLE_RATE_16K, self.dataset_format, audio_16k
        )

    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
    ):
        chunk_length = secs_to_samples(chunk_len, self.sr)
        overlap_length = secs_to_samples(overlap_len, self.sr)
        stride = chunk_length - overlap_length
        if stride <= 0:
            # a non-positive stride writes slices until the disk fills
            raise ValueError(
                "Simple cutting needs overlap_len < chunk_len, got "
                f"chunk_len={chunk_len}s overlap_len={overlap_len}s."
            )

        total = len(audio)
        min_tail = secs_to_samples(MIN_TAIL_SECONDS, self.sr)
        slice_idx = 0
        last_start = None

        for start in range(0, total, stride):
            end = start + chunk_length
            if end > total:
                # slide the last window back so it ends on the final sample,
                # rather than dropping the tail or padding it with silence
                remainder = total - start
                if remainder < min_tail:
                    break
                if total >= chunk_length:
                    start = total - chunk_length
                    if last_start is not None and start <= last_start:
                        break  # the re-anchored window would repeat the previous one
                    chunk = audio[start:]
                else:
                    # shorter than one chunk, and variable lengths are fine downstream
                    chunk = audio
            else:
                chunk = audio[start:end]

            self.process_audio_segment(chunk, sid, idx0, slice_idx, normalization_mode)
            last_start = start
            slice_idx += 1

            if start + chunk_length >= total:
                break

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
    ):
        audio_length = 0
        try:
            audio = load_audio(path, self.sr)
            audio_length = librosa.get_duration(y=audio, sr=self.sr)

            if process_effects:
                audio = self.high_pass(audio)
            if normalization_mode == "pre":
                normalized = self._normalize_audio(audio)
                if normalized is None:
                    # letting the None travel on becomes a TypeError that the
                    # handler below reports as a generic processing error
                    print(f"Skipping {path}: peak too high to normalize.")
                    return 0
                audio = normalized
            if noise_reduction:
                audio = nr.reduce_noise(
                    y=audio, sr=self.sr, prop_decrease=reduction_strength
                )
            if cut_preprocess == "Skip":
                # no cutting
                self.process_audio_segment(
                    audio,
                    sid,
                    idx0,
                    0,
                    normalization_mode,
                )
            elif cut_preprocess == "Simple":
                # simple
                self.simple_cut(
                    audio,
                    sid,
                    idx0,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                )
            elif cut_preprocess == "Automatic":
                idx1 = 0
                # legacy
                for audio_segment in self.slicer.slice(audio):
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if (
                            len(audio_segment[start:])
                            > (PERCENTAGE + OVERLAP) * self.sr
                        ):
                            tmp_audio = audio_segment[
                                start : start + int(PERCENTAGE * self.sr)
                            ]
                            self.process_audio_segment(
                                tmp_audio,
                                sid,
                                idx0,
                                idx1,
                                normalization_mode,
                            )
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(
                                tmp_audio,
                                sid,
                                idx0,
                                idx1,
                                normalization_mode,
                            )
                            idx1 += 1
                            break

        except Exception as error:
            print(f"Error processing audio: {error}")
        return audio_length


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def save_dataset_duration(file_path, dataset_duration, dataset_format="wav"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    formatted_duration = format_duration(dataset_duration)
    new_data = {
        "total_dataset_duration": formatted_duration,
        "total_seconds": dataset_duration,
        "dataset_format": dataset_format,
    }
    data.update(new_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def speaker_id_from_directory(root: str, input_root: str) -> int:
    """
    Returns the speaker id for a dataset subfolder.

    The root folder itself is speaker 0. A subfolder names its id with a leading
    integer, so both "3" and "3_maria" work.

    Args:
        root (str): The subfolder being read.
        input_root (str): Root of the dataset.
    """
    if os.path.normpath(root) == os.path.normpath(input_root):
        return 0
    name = os.path.basename(os.path.normpath(root))
    return int(name.split("_")[0])


def process_audio_wrapper(args):
    (
        pp,
        file,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
    ) = args
    file_path, idx0, sid = file
    return pp.process_audio(
        file_path,
        idx0,
        sid,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
    )


def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    dataset_format: str = "wav",
):
    if not os.path.exists(input_root):
        print(f"The dataset path does not exist: '{input_root}'.")
        sys.exit(1)

    if not os.path.isdir(input_root):
        print(f"The dataset path is not a directory: '{input_root}'.")
        sys.exit(1)

    if dataset_format not in DATASET_FORMATS:
        print(
            f"Unknown dataset format '{dataset_format}'. "
            f"Expected one of: {', '.join(DATASET_FORMATS)}."
        )
        sys.exit(1)

    if cut_preprocess == "Simple":
        if overlap_len >= chunk_len:
            print(
                "Simple cutting needs overlap_len < chunk_len, got "
                f"chunk_len={chunk_len}s overlap_len={overlap_len}s."
            )
            sys.exit(1)
        try:
            # checked once here so a bad length is not reported per file
            secs_to_samples(chunk_len, sr)
            secs_to_samples(overlap_len, sr)
        except ValueError as error:
            print(error)
            sys.exit(1)

    start_time = time.time()
    pp = PreProcess(sr, exp_dir, dataset_format)
    print(f"Starting preprocess with {num_processes} processes...")

    files = []
    idx = 0
    speaker_ids = set()

    for root, _, filenames in os.walk(input_root):
        audio_files = [
            f
            for f in filenames
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
        ]
        if not audio_files:
            continue
        try:
            sid = speaker_id_from_directory(root, input_root)
        except ValueError:
            print(
                "Speaker ID folders must start with an integer, got "
                f'"{os.path.basename(os.path.normpath(root))}" instead. '
                "Name them '0', '1', ... or '0_name', '1_name', ..."
            )
            sys.exit(1)
        speaker_ids.add(sid)
        for f in audio_files:
            files.append((os.path.join(root, f), idx, sid))
            idx += 1

    if len(files) == 0:
        print(
            f"No audio files found in the dataset path: '{input_root}'. Please check that the path is correct and contains valid audio files."
        )
        sys.exit(1)

    # the speaker embedding is sized from the number of distinct ids, so a gap
    # indexes past the end of that table once training starts
    expected_ids = set(range(len(speaker_ids)))
    if speaker_ids != expected_ids:
        missing = sorted(expected_ids - speaker_ids)
        print(
            f"Speaker IDs must be contiguous and start at 0. "
            f"Found: {sorted(speaker_ids)}. Missing: {missing}."
        )
        sys.exit(1)

    if len(speaker_ids) > 1:
        print(f"Found {len(speaker_ids)} speakers.")

    audio_length = []
    with tqdm(total=len(files)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            futures = [
                executor.submit(
                    process_audio_wrapper,
                    (
                        pp,
                        file,
                        cut_preprocess,
                        process_effects,
                        noise_reduction,
                        reduction_strength,
                        chunk_len,
                        overlap_len,
                        normalization_mode,
                    ),
                )
                for file in files
            ]
            for future in concurrent.futures.as_completed(futures):
                audio_length.append(future.result())
                pbar.update(1)

    audio_length = sum(audio_length)
    save_dataset_duration(
        os.path.join(exp_dir, "model_info.json"),
        dataset_duration=audio_length,
        dataset_format=dataset_format,
    )
    elapsed_time = time.time() - start_time
    print(
        f"Preprocess completed in {elapsed_time:.2f} seconds on {format_duration(audio_length)} seconds of audio."
    )


if __name__ == "__main__":
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    num_processes = sys.argv[4]
    if num_processes.lower() == "none":
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = int(num_processes)
    cut_preprocess = str(sys.argv[5])
    process_effects = strtobool(sys.argv[6])
    noise_reduction = strtobool(sys.argv[7])
    reduction_strength = float(sys.argv[8])
    chunk_len = float(sys.argv[9])
    overlap_len = float(sys.argv[10])
    normalization_mode = str(sys.argv[11])
    dataset_format = str(sys.argv[12]) if len(sys.argv) > 12 else "wav"
    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        dataset_format,
    )
