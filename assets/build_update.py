"""Create the code-only release asset and inventory for compiled installations."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from updater import digest, file_mode, validate_manifest, write_json


def build(root, output, paths, seed_install=False):
    policy = json.loads(
        (root / "assets/update_policy.json").read_text(encoding="utf-8")
    )
    version = json.loads(
        (root / "assets/config_template.json").read_text(encoding="utf-8")
    )["version"]
    files = {}
    for name in sorted(paths):
        mode = file_mode(name)
        path = root / name
        if mode and path.is_file():
            if path.is_symlink():
                raise ValueError(f"Release files must not be links: {name}")
            files[name] = {"sha256": digest(path), "mode": mode}
    manifest = validate_manifest({**policy, "version": version, "files": files})
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"ApplioUpdate-{version}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("update-manifest.json", json.dumps(manifest, indent=2))
        for name in files:
            package.write(root / name, "payload/" + name)
    archive.with_suffix(".zip.sha256").write_text(
        f"{digest(archive)}  {archive.name}\n", encoding="ascii"
    )
    # Existing extracted releases need these two files once to bootstrap updates.
    with zipfile.ZipFile(
        output / "ApplioUpdater.zip", "w", zipfile.ZIP_DEFLATED
    ) as package:
        for name in ("run-updater.bat", "assets/updater.py"):
            package.write(root / name, name)
    if seed_install:
        write_json(root / ".applio-update/installed.json", manifest)
    print(f"Built {archive.name}: {len(files)} managed/default files.")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-install", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    # No Git is needed on users' machines; only release packaging uses it.
    listing = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode(
        "utf-8"
    )
    build(
        root,
        args.output.resolve(),
        [p for p in listing.split("\0") if p],
        args.seed_install,
    )


if __name__ == "__main__":
    main()
