"""Stamp icon + version metadata onto the unpacked Windows exe.

Used for local/unprivileged builds where electron-builder's winCodeSign step is
skipped (it needs symlink privileges). Uses the rcedit binary from the
electron-builder tool cache — no admin rights required.

Usage (from app/desktop):
    python stamp.py
"""

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def find_rcedit() -> str:
    node_rcedit = os.path.join(ROOT, "node_modules", "rcedit", "bin", "rcedit-x64.exe")
    if os.path.exists(node_rcedit):
        return node_rcedit
    cache = os.path.join(os.environ.get("LOCALAPPDATA", ""), "electron-builder", "Cache", "winCodeSign")
    candidates = glob.glob(os.path.join(cache, "*", "rcedit-x64.exe"))
    if candidates:
        return sorted(candidates, key=os.path.getmtime)[-1]
    raise FileNotFoundError("rcedit-x64.exe not found in node_modules or electron-builder cache")


def app_version() -> str:
    pkg = os.path.join(HERE, "package.json")
    if os.path.exists(pkg):
        with open(pkg, encoding="utf-8") as f:
            v = json.load(f).get("version")
            if v:
                return str(v)
    with open(os.path.join(ROOT, "assets", "config_template.json"), encoding="utf-8") as f:
        return str(json.load(f).get("version", "1.0.0"))


def main() -> None:
    exe = os.path.join(HERE, "dist-installers", "win-unpacked", "Applio.exe")
    if not os.path.exists(exe):
        raise FileNotFoundError(f"unpack first: {exe}")
    icon = os.path.join(ROOT, "assets", "ICON.ico")
    version = app_version()
    rcedit = find_rcedit()
    print(f"rcedit: {rcedit}")
    print(f"exe: {exe} (v{version})")
    subprocess.run(
        [
            rcedit,
            exe,
            "--set-icon",
            icon,
            "--set-file-version",
            version,
            "--set-product-version",
            version,
            "--set-version-string",
            "CompanyName",
            "Applio",
            "--set-version-string",
            "FileDescription",
            "Applio",
            "--set-version-string",
            "ProductName",
            "Applio",
        ],
        check=True,
    )
    print("stamped OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - CLI surface, report and exit
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
