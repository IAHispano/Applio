"""Speech providers shared by the UI, worker and CLI; ML imports stay lazy."""

import asyncio
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROVIDERS = ("edge", "gtts", "piper")


def catalog():
    return json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))


def voices(provider="edge"):
    if provider == "edge":
        entries = json.loads(
            Path(__file__).with_name("tts_voices.json").read_text(encoding="utf-8")
        )
        return [
            {
                "shortName": v["ShortName"],
                "friendlyName": v.get("FriendlyName", v["ShortName"]),
            }
            for v in entries
        ]
    if provider not in PROVIDERS:
        raise ValueError("Invalid params")
    return catalog()[provider]


def _run(args, **kwargs):
    return subprocess.run(
        args,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        **kwargs,
    )


def _speed(rate):
    # Reciprocal slowdown keeps -100 finite without flattening the lower half
    # of the slider, while positive percentages remain ordinary speed increases.
    return 1 + rate / 100 if rate >= 0 else 1 / (1 - rate / 100)


def ensure_piper_runtime():
    from rvc.lib.tools.audio_cleanup import _asset_lock
    from rvc.lib.user_config import config_path

    root = Path(
        os.environ.get("APPLIO_TTS_CACHE") or config_path().parent / "tts" / "piper"
    )
    root.mkdir(parents=True, exist_ok=True)
    # This lock also covers runtime installation across Gradio/CLI processes.
    with _asset_lock(root):
        packages = {"piper-tts": "1.8.0", "pathvalidate": "3.3.1"}
        missing = []
        for name, version in packages.items():
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                installed = None
            if installed != version:
                missing.append(f"{name}=={version}")
        if missing:
            # NumPy and the platform-selected ONNX runtime belong to engine setup.
            # Letting pip resolve Piper's dependencies would also install CPU ORT
            # over Windows GPU/ROCm installations.
            _run([sys.executable, "-m", "pip", "install", "--no-deps", *missing])
    return root


def _piper(text, voice, rate, output):
    from rvc.lib.tools.audio_cleanup import _asset_lock
    from rvc.lib.tools.http_download import download_http

    root = ensure_piper_runtime()
    with _asset_lock(root):
        data = catalog()
        entry = next(v for v in data["piper"] if v["shortName"] == voice)
        directory = root / voice
        directory.mkdir(exist_ok=True)
        for asset in entry["files"]:
            destination = directory / asset["name"]
            valid = False
            if destination.is_file():
                with destination.open("rb") as stream:
                    valid = (
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        == asset["sha256"]
                    )
            if not valid:
                print("APPLIO_STAGE:Downloading model…", flush=True)
                url = f"https://huggingface.co/rhasspy/piper-voices/resolve/{data['piperRevision']}/{asset['path']}"
                download_http(url, destination, sha256=asset["sha256"])
    model = directory / f"{voice}.onnx"
    _run(
        [
            sys.executable,
            "-m",
            "piper",
            "--model",
            str(model),
            "--output_file",
            str(output),
            "--length_scale",
            str(1 / _speed(rate)),
        ],
        input=text,
        text=True,
        encoding="utf-8",
    )


def _google(text, voice, rate, output):
    from gtts import gTTS

    # gTTS exposes only normal/slow speech. Apply the shared continuous rate
    # through FFmpeg so the percentage control retains its meaning.
    with tempfile.TemporaryDirectory(prefix="applio-tts-") as directory:
        source = Path(directory) / "speech.mp3"
        gTTS(text=text, lang=voice, lang_check=False, timeout=(15, 60)).save(
            str(source)
        )
        _run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-af",
                f"atempo={_speed(rate)}",
                "-f",
                "wav",
                str(output),
            ]
        )


async def synthesize(text, voice, rate, output, provider="edge"):
    if provider not in PROVIDERS or not -100 <= rate <= 100:
        raise ValueError("Invalid params")
    voice = (
        voice
        or {"edge": "en-US-AnaNeural", "gtts": "en", "piper": "en_US-lessac-medium"}[
            provider
        ]
    )
    if not text.strip():
        raise ValueError("Enter text or upload a .txt file.")
    if provider != "edge" and voice not in {v["shortName"] for v in voices(provider)}:
        raise ValueError("Invalid params")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if provider == "edge":
        import edge_tts

        await edge_tts.Communicate(text, voice, rate=f"{rate:+d}%").save(str(output))
    else:
        await asyncio.to_thread(
            _google if provider == "gtts" else _piper, text, voice, rate, output
        )
