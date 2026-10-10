"""Per-user settings shared with the API, without importing ML dependencies."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile


def config_path():
    """Match app/api/src/config.ts, including headless/portable overrides."""
    file_override = os.environ.get("APPLIO_CONFIG_FILE")
    if file_override:
        return Path(file_override).resolve()
    override = os.environ.get("APPLIO_CONFIG_DIR")
    if override:
        directory = Path(override).resolve()
    elif sys.platform == "win32":
        directory = (
            Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
            / "Applio"
        )
    elif sys.platform == "darwin":
        directory = Path.home() / "Library" / "Application Support" / "Applio"
    else:
        directory = (
            Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
            / "Applio"
        )
    return directory / "config.json"


def get_logs_dir():
    """Model/training storage, pinned by the API for its child processes."""
    override = os.environ.get("APPLIO_LOGS_DIR")
    if override:
        return str(Path(override).resolve())
    config = load_config()
    # A saved move is applied by the API at its next startup. Standalone Python
    # and currently running sessions keep using the source until that commits.
    configured = config.get("logs_move_from") or config.get("logs_dir")
    root = Path(os.environ.get("APPLIO_ROOT") or Path(__file__).resolve().parents[2])
    return str(Path(configured).resolve() if configured else root / "logs")


def resolve_logs_path(value):
    """Resolve the UI's logical logs/ paths for realtime model loading."""
    normalized = (value or "").replace("\\", "/")
    if normalized == "logs" or normalized.startswith("logs/"):
        root = Path(get_logs_dir()).resolve()
        resolved = (root / normalized[5:]).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Path escapes logs folder: {value}")
        return str(resolved)
    return value


def _read_object(file):
    with file.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a settings object in {file}")
    return value


def _merge(base, overrides):
    for key, value in overrides.items():
        if key in ("__proto__", "constructor", "prototype"):
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config():
    root = Path(os.environ.get("APPLIO_ROOT") or Path(__file__).resolve().parents[2])
    code = Path(os.environ.get("APPLIO_CODE_ROOT") or root)
    defaults = _read_object(code / "assets" / "config_template.json")
    file = config_path()
    if not file.exists():
        legacy = root / "assets" / "config.json"
        preferences = _read_object(legacy) if legacy.exists() else {}
        preferences.pop("version", None)
        file.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=file.parent, delete=False
            ) as stream:
                temp = Path(stream.name)
                json.dump(preferences, stream, indent=2)
                stream.write("\n")
            try:
                # Publish atomically; an existing per-user file always wins.
                os.link(temp, file)
            except FileExistsError:
                pass
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    preferences = _read_object(file)
    preferences.pop("version", None)
    return _merge(defaults, preferences)
