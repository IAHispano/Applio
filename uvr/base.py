"""Shared base for Applio UVR architecture separators (slim, soundfile-only output)."""

import gc
import os
import re
from contextlib import contextmanager
from logging import Logger

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from tqdm import tqdm

from uvr import spectral as spec_utils


class AudioExportError(Exception):
    def __init__(self, message, path=None, backend=None):
        super().__init__(message)
        self.path = path
        self.backend = backend


class InvalidAudioDataError(Exception):
    pass


def validate_audio_source(stem_source):
    stem_source = np.asarray(stem_source)
    if stem_source.ndim not in (1, 2) or (
        stem_source.ndim == 2 and stem_source.shape[1] not in (1, 2)
    ):
        raise InvalidAudioDataError(
            f"Audio data has invalid shape {stem_source.shape}; expected mono or stereo frames"
        )
    return stem_source


@contextmanager
def atomic_output_path(target_path, backend="soundfile"):
    """Yield a same-directory temp path and atomically publish it on success."""
    import tempfile

    temp_fd = None
    temp_path = None
    error = None
    try:
        target_dir = os.path.dirname(target_path) or "."
        suffix = os.path.splitext(target_path)[1]
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(target_path)}.", suffix=suffix, dir=target_dir
        )
        os.close(temp_fd)
        temp_fd = None
        yield temp_path
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise OSError("Audio backend produced an empty output file")
        os.replace(temp_path, target_path)
    except Exception as exc:
        error = exc
    finally:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    if error is not None:
        if isinstance(error, AudioExportError):
            raise error
        raise AudioExportError(
            f"Failed to publish audio output {target_path} with {backend}: {error}",
            path=target_path,
            backend=backend,
        ) from error


def select_torch_device(use_gpu: bool):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_onnx_providers(use_gpu: bool):
    if use_gpu:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def download_file(url: str, output_path: str, logger=None, timeout=300):
    """Download a URL to disk with a progress bar; skip when already present."""
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        if logger:
            logger.debug(f"Model file already present: {output_path}")
        return output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".part"
    with requests.get(url, stream=True, timeout=timeout) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download {url}, response code: {response.status_code}"
            )
        total = int(response.headers.get("content-length", 0))
        with (
            open(tmp_path, "wb") as f,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(output_path),
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp_path, output_path)
    return output_path


class BaseSeparator:
    """Common mix preparation, soundfile writing and stem path logic."""

    VOCAL_STEM = "Vocals"
    INST_STEM = "Instrumental"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    NO_STEM = "No "

    STEM_PAIR_MAPPER = {VOCAL_STEM: INST_STEM, INST_STEM: VOCAL_STEM}
    NON_ACCOM_STEMS = (VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM)

    def __init__(self, config: dict):
        self.logger: Logger = config.get("logger")
        self.torch_device = config.get("torch_device", torch.device("cpu"))
        self.onnx_execution_provider = config.get(
            "onnx_execution_provider", ["CPUExecutionProvider"]
        )
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data", {})
        self.output_dir = config.get("output_dir")
        self.output_format = (config.get("output_format") or "wav").lower()
        self.normalization_threshold = config.get("normalization_threshold", 0.9)
        self.amplification_threshold = config.get("amplification_threshold", 0.0)
        self.output_single_stem = config.get("output_single_stem")
        self.sample_rate = config.get("sample_rate", 44100)
        self.wav_subtype = "PCM_16"
        self.input_bit_depth = 16
        self.input_subtype = "PCM_16"

        self.primary_stem_name = self.model_data.get("primary_stem", "Vocals")
        self.secondary_stem_name = self.secondary_stem(self.primary_stem_name)

        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None
        self.secondary_source = None

    def secondary_stem(self, primary_stem: str):
        primary_stem = primary_stem if primary_stem else self.NO_STEM
        if primary_stem in self.STEM_PAIR_MAPPER:
            return self.STEM_PAIR_MAPPER[primary_stem]
        if self.NO_STEM in primary_stem:
            return primary_stem.replace(self.NO_STEM, "")
        return f"{self.NO_STEM}{primary_stem}"

    def prepare_mix(self, mix):
        if not isinstance(mix, np.ndarray):
            try:
                info = sf.info(mix)
                self.input_subtype = info.subtype
                self.input_bit_depth = (
                    24
                    if "24" in info.subtype
                    else 32 if "32" in info.subtype or "FLOAT" in info.subtype else 16
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not read audio file info, defaulting to 16-bit output: {e}"
                )
            mix, _sr = librosa.load(mix, mono=False, sr=self.sample_rate)
        if mix.size == 0 or not np.isfinite(mix).all():
            raise InvalidAudioDataError(
                "Input audio is empty or contains non-finite samples"
            )
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
        return mix

    def write_audio(self, stem_path: str, stem_source):
        stem_source = validate_audio_source(stem_source)
        stem_source = spec_utils.normalize(
            wave=stem_source,
            max_peak=self.normalization_threshold,
            min_peak=self.amplification_threshold,
        )
        if self.output_dir:
            stem_path = os.path.join(self.output_dir, stem_path)
            os.makedirs(self.output_dir, exist_ok=True)
        if self.input_bit_depth == 24:
            subtype = "PCM_24"
        elif self.input_bit_depth == 32:
            subtype = "PCM_32"
        else:
            subtype = "PCM_16"
        if stem_source.ndim == 2 and stem_source.flags["F_CONTIGUOUS"]:
            stem_source = np.ascontiguousarray(stem_source)
        with atomic_output_path(stem_path, "soundfile") as temp_path:
            sf.write(temp_path, stem_source, self.sample_rate, subtype=subtype)

    def final_process(self, stem_path, source, stem_name):
        self.write_audio(stem_path, source)
        return {stem_name: source}

    def sanitize_filename(self, filename):
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
        sanitized = re.sub(r"_+", "_", sanitized)
        return sanitized.strip("_. ")

    def get_stem_output_path(self, stem_name, custom_output_names=None):
        if custom_output_names:
            lowered = {k.lower(): v for k, v in custom_output_names.items()}
            if stem_name.lower() in lowered:
                return f"{self.sanitize_filename(lowered[stem_name.lower()])}.{self.output_format}"
        base = self.sanitize_filename(self.audio_file_base or "audio")
        stem = self.sanitize_filename(stem_name)
        model = self.sanitize_filename(self.model_name or "uvr")
        return f"{base}_({stem})_{model}.{self.output_format}"

    def clear_gpu_cache(self):
        gc.collect()
        if self.torch_device == torch.device("cuda"):
            torch.cuda.empty_cache()
