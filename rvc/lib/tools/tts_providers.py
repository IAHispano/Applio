"""Speech providers shared by the UI, worker and CLI; ML imports stay lazy."""

from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PROVIDERS = ("edge",)


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
    if provider == "rvc":
        return []
    raise ValueError("Invalid params")


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


async def synthesize(text, voice, rate, output, provider="edge"):
    if provider != "edge" or not -100 <= rate <= 100:
        raise ValueError("Invalid params")
    if not text.strip():
        raise ValueError("Enter text or upload a .txt file.")
    import edge_tts

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    await edge_tts.Communicate(
        text, voice or "en-US-AnaNeural", rate=f"{rate:+d}%"
    ).save(str(output))
