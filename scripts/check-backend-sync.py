"""Check backend parity against another local checkout, without ML dependencies."""

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess


def content(file):
    text = file.read_text(encoding="utf-8")
    # Each repository runs its own formatter. Compare Python behavior, not layout.
    if file.suffix == ".py":
        return ast.dump(ast.parse(text), include_attributes=False)
    if file.suffix == ".json":
        return json.loads(text)
    return text


def requirements(root, excluded):
    result = set()
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0].lower().replace("_", "-")
        if name not in excluded:
            result.add(line)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("other", type=Path, help="Path to the other Applio repository")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    other = args.other.resolve()
    manifest = json.loads((root / "backend-sync.json").read_text(encoding="utf-8"))
    differences = []
    for name in manifest["files"]:
        try:
            if content(root / name) != content(other / name):
                differences.append(name)
        except (OSError, SyntaxError, ValueError) as error:
            differences.append(f"{name}: {error}")

    # Detect newly added modules that otherwise could be silently left out.
    for checkout in (root, other):
        tracked = subprocess.check_output(
            ["git", "-C", str(checkout), "ls-files", "rvc", "uvr"], text=True
        ).splitlines()
        for name in tracked:
            if name not in manifest["files"] and not Path(name).name.startswith(
                "test_"
            ):
                differences.append(f"Missing from manifest: {name}")

    try:
        if requirements(root, manifest["requirements_exclude"]) != requirements(
            other, manifest["requirements_exclude"]
        ):
            differences.append("requirements.txt (engine dependencies)")
        defaults = [
            json.loads(
                (checkout / "assets/config_template.json").read_text(encoding="utf-8")
            )
            for checkout in (root, other)
        ]
        for key in manifest["config_keys"]:
            if defaults[0].get(key) != defaults[1].get(key):
                differences.append(f"assets/config_template.json: {key}")
    except (OSError, ValueError) as error:
        differences.append(str(error))

    if differences:
        for difference in sorted(set(differences)):
            print(f"DIFF {difference}")
        return 1
    print(
        f"Backend parity verified: {len(manifest['files'])} files, dependencies, and engine settings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
