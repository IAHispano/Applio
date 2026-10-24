"""Optional file denoising shared by Gradio, the worker and standalone CLI.

DeepFilterNet runs as a pinned native CPU executable, keeping its NumPy <2
Python dependency out of the engine environment. Assets are cached on first
use; no speech is uploaded.
"""

from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from rvc.lib.tools.http_download import download_http
from rvc.lib.user_config import config_path

VERSION = "0.5.6"
MODEL_REVISION = "d375b2d8309e0935d165700c91da9de862a99c31"
MODEL_HASH = "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616"
RUNTIMES = {
    ("Windows", "x86_64"): (
        "deep-filter-0.5.6-x86_64-pc-windows-msvc.exe",
        "75e11fa16445f560cb6b021521ddb89e89270d13b83089705d98776f58fd7915",
    ),
    ("Darwin", "aarch64"): (
        "deep-filter-0.5.6-aarch64-apple-darwin",
        "4601e7f4e4c03e59a4c5b5000216ef3add3e808799cfccd95e14e83ea4611081",
    ),
    ("Darwin", "x86_64"): (
        "deep-filter-0.5.6-x86_64-apple-darwin",
        "d3be84003acb7c23e738ad7f70a158ec779a8d233a82e7fa3e717d112eb5b50f",
    ),
    ("Linux", "x86_64"): (
        "deep-filter-0.5.6-x86_64-unknown-linux-musl",
        "70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da",
    ),
    ("Linux", "aarch64"): (
        "deep-filter-0.5.6-aarch64-unknown-linux-gnu",
        "14e02a1c0028f3ca0bdf83b62b3336e56ba0556894ef295a95e8573f06557166",
    ),
    ("Linux", "armv7l"): (
        "deep-filter-0.5.6-armv7-unknown-linux-gnueabihf",
        "29feffdfacbaa30ff7811bd275f252d405d43ad6e5523ffcedeaf3ff5bb8335b",
    ),
}


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def _asset_lock(root):
    # OS-owned locks release even if a cancelled job kills the downloader.
    # Keep the lock file in place: unlinking it can let new callers lock a
    # different inode while another process is still using the original one.
    with (root / "download.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + 300
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Audio cleanup assets are downloading. Please retry."
                    )
                time.sleep(0.1)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _runtime_assets():
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else machine
    arch = "x86_64" if arch in ("amd64", "x64") else arch
    key = (platform.system(), arch)
    if key not in RUNTIMES:
        raise RuntimeError(
            f"Speech cleanup is unavailable for {key[0]} {key[1]}. Use Light cleanup."
        )
    filename, checksum = RUNTIMES[key]
    root = (
        Path(
            os.environ.get("APPLIO_CLEANUP_CACHE")
            or config_path().parent / "audio-cleanup"
        )
        / VERSION
    )
    root.mkdir(parents=True, exist_ok=True)
    executable, model = root / filename, root / "DeepFilterNet3_onnx.tar.gz"
    with _asset_lock(root):
        for path, url, digest in (
            (
                executable,
                f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{VERSION}/{filename}",
                checksum,
            ),
            (
                model,
                f"https://raw.githubusercontent.com/Rikorose/DeepFilterNet/{MODEL_REVISION}/models/DeepFilterNet3_onnx.tar.gz",
                MODEL_HASH,
            ),
        ):
            if not path.is_file() or _digest(path) != digest:
                print(f"Downloading audio cleanup asset: {path.name}", flush=True)
                download_http(url, path, sha256=digest)
        if os.name != "nt":
            executable.chmod(executable.stat().st_mode | 0o100)
    return executable, model


def _ffmpeg():
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for root in (
        os.environ.get("APPLIO_ROOT"),
        os.environ.get("APPLIO_CODE_ROOT"),
        str(Path(__file__).resolve().parents[3]),
    ):
        if root and (Path(root) / name).is_file():
            return str(Path(root) / name)
    found = shutil.which(name)
    if not found:
        raise RuntimeError("Light cleanup requires FFmpeg. Install it or select Off.")
    return found


def _run(arguments, duration):
    result = subprocess.run(
        [str(value) for value in arguments],
        capture_output=True,
        text=True,
        timeout=max(120, duration * 2 + 60),
        creationflags=(
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        ),
    )
    if result.returncode:
        raise RuntimeError(
            f"Audio cleanup failed: {(result.stderr or result.stdout)[-2000:]}"
        )


def cleanup_audio(audio, sample_rate, mode="off", strength=12, *, temporary_directory=None):
    """Return normalized float audio with the original rate, channels and length.

    Off is an exact bypass. Light uses FFmpeg spectral reduction; Speech uses
    DeepFilterNet3 with a user-selected attenuation limit, not the aggressive
    unlimited setting. Speech models can remove music and sustained vocals.
    """
    if mode not in ("off", "light", "speech"):
        raise ValueError("Unknown audio cleanup mode")
    if mode == "off":
        return audio
    strength = float(strength)
    if not math.isfinite(strength) or not 1 <= strength <= 30:
        raise ValueError("Cleanup strength must be between 1 and 30 dB")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Invalid cleanup sample rate")
    original = np.asarray(audio)
    if original.ndim not in (1, 2) or (original.ndim == 2 and original.shape[1] < 1):
        raise ValueError("Cleanup requires mono or multichannel audio")
    data = original.astype(np.float32)
    if np.issubdtype(original.dtype, np.signedinteger):
        data /= float(-np.iinfo(original.dtype).min)
    if not np.isfinite(data).all():
        raise ValueError("Cleanup input contains non-finite samples")
    if not data.size or not data.any():
        return data
    print(f"Audio cleanup: {mode}, reduction limit {strength:g} dB", flush=True)
    rate = 48000
    common = math.gcd(int(sample_rate), rate)
    work = (
        resample_poly(data, rate // common, int(sample_rate) // common, axis=0)
        if sample_rate != rate
        else data
    )
    length = len(work)
    with tempfile.TemporaryDirectory(
        prefix="applio-cleanup-", dir=temporary_directory
    ) as directory:
        folder = Path(directory)
        source, target = folder / "input.wav", folder / "output.wav"
        if mode == "light":
            sf.write(source, work, rate, subtype="FLOAT")
            # Flush the filter's final block without shifting the speech timeline.
            filters = f"apad=pad_dur=0.05,afftdn=nr={strength:g}:nf=-50:tn=1,atrim=end_sample={length},asetpts=PTS-STARTPTS"
            _run(
                [
                    _ffmpeg(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    source,
                    "-af",
                    filters,
                    "-c:a",
                    "pcm_f32le",
                    target,
                ],
                length / rate,
            )
        else:
            executable, model = _runtime_assets()
            padding = ((0, 4800),) + ((0, 0),) * (work.ndim - 1)
            # Float input avoids the native CLI's 24-bit integer WAV panic.
            sf.write(source, np.pad(work, padding), rate, subtype="FLOAT")
            output = folder / "enhanced"
            output.mkdir()
            _run(
                [
                    executable,
                    "-m",
                    model,
                    "-D",
                    "-a",
                    "100",
                    "-o",
                    output,
                    source,
                ],
                length / rate,
            )
            target = output / source.name
        cleaned, output_rate = sf.read(
            target, dtype="float32", always_2d=work.ndim == 2
        )
        if (
            output_rate != rate
            or cleaned.shape[1:] != work.shape[1:]
            or len(cleaned) < length
        ):
            raise RuntimeError("Audio cleanup returned incompatible or truncated audio")
        cleaned = cleaned[:length]
        if mode == "speech":
            # Keep the reduction limit outside the model's speech gate, which
            # can zero non-speech frames even with its native limit enabled.
            original_mix = 10 ** (-strength / 20)
            cleaned = work * original_mix + cleaned * (1 - original_mix)
    if sample_rate != rate:
        cleaned = resample_poly(
            cleaned, int(sample_rate) // common, rate // common, axis=0
        )
    cleaned = cleaned[: len(data)]
    if cleaned.shape != data.shape or not np.isfinite(cleaned).all():
        raise RuntimeError(
            "Audio cleanup changed the audio shape or produced invalid samples"
        )
    return cleaned


def cleanup_file(path, mode="off", strength=12, *, temporary_directory=None):
    """Clean a synthesized WAV atomically, retaining its original rate and length."""
    if mode == "off":
        return
    audio, rate = sf.read(path, dtype="float32")
    cleaned = cleanup_audio(
        audio, rate, mode, strength, temporary_directory=temporary_directory
    )
    descriptor, temporary = tempfile.mkstemp(
        suffix=".wav", dir=Path(path).resolve().parent
    )
    os.close(descriptor)
    try:
        sf.write(temporary, cleaned, rate, format="WAV", subtype="PCM_16")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
