"""Updates extracted Windows releases without Git or third-party Python imports.

Release archives contain a hashed inventory, never an environment or user data.
Keep path policy here (rather than trusting the downloaded inventory).
"""

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

REPOSITORY = "IAHispano/Applio"
API = f"https://api.github.com/repos/{REPOSITORY}"
STATE = ".applio-update"
MAX_DOWNLOAD = 256 * 1024 * 1024
MAX_UNPACKED = 512 * 1024 * 1024
DATA_ROOTS = {
    "logs",
    "env",
    "venv",
    ".venv",
    ".git",
    STATE,
    "audios",
    "datasets",
    "models",
    "weights",
    "outputs",
    "downloads",
    "cache",
    "__pycache__",
}
ASSET_CODE = {
    "config_template.json",
    "ecosystem.json",
    "update_policy.json",
    "applio-logo.svg",
    "ICON.ico",
}


class UpdateError(Exception):
    pass


def safe_name(name):
    """Reject ZIP traversal, Windows aliases, ADS, and case/path ambiguity."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise UpdateError(f"Invalid package path: {name!r}")
    parts = name.split("/")
    for part in parts:
        if (
            not part
            or part in {".", ".."}
            or part.endswith((".", " "))
            or re.search(r'[<>:"|?*\x00-\x1f]', part)
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?", part)
        ):
            raise UpdateError(f"Unsafe package path: {name!r}")
    return PurePosixPath(name)


def file_mode(name):
    """Only shipped code is replaceable; user-owned folders are never pruned."""
    parts = safe_name(name).parts
    lower = tuple(p.lower() for p in parts)
    if lower[0] in DATA_ROOTS or "__pycache__" in lower:
        return None
    if lower[:3] == ("tabs", "plugins", "installed"):
        return None
    if lower[0] == "assets":
        if len(parts) < 2:
            return None
        if lower[1] in {"presets", "formant_shift"}:
            return "seed" if Path(name).suffix.lower() == ".json" else None
        if lower[1] in {"ui", "i18n", "themes", "zluda"}:
            # Never accept model weights or audio through an update payload.
            allowed = {
                ".py",
                ".json",
                ".css",
                ".js",
                ".svg",
                ".ttf",
                ".woff2",
                ".md",
                ".bat",
                ".sh",
            }
            return "replace" if Path(name).suffix.lower() in allowed else None
        return (
            "replace"
            if len(parts) == 2 and (name.endswith(".py") or parts[1] in ASSET_CODE)
            else None
        )
    if lower[0] == "docs":
        return "replace" if Path(name).suffix.lower() == ".md" else None
    if lower[0] in {"rvc", "tabs"}:
        if lower[:2] == ("rvc", "models"):
            return None
        return (
            "replace"
            if Path(name).suffix.lower()
            in {".py", ".json", ".css", ".js", ".html", ".svg"}
            else None
        )
    if len(parts) == 1:
        if name.startswith("requirements") and name.endswith(".txt"):
            return "replace"
        if name in {
            "app.py",
            "core.py",
            "run-updater.bat",
            "run-applio.bat",
            "run-applio.sh",
            "run-tensorboard.bat",
            "run-tensorboard.sh",
            "LICENSE",
            "README.md",
            "UPDATING.md",
            "TERMS_OF_USE.md",
        }:
            return "replace"
    return None


def destination(root, name):
    path = root.joinpath(*safe_name(name).parts)
    # Refuse symlinks and Windows junctions even if their final target is in root.
    for node in (path, *path.parents):
        if node == root:
            break
        if node.is_symlink() or (hasattr(node, "is_junction") and node.is_junction()):
            raise UpdateError(f"Update path contains a link: {name}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise UpdateError(f"Update path escapes the installation: {name}")
    return path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def version_key(value):
    if not isinstance(value, str) or not re.fullmatch(r"v?\d+\.\d+\.\d+", value):
        raise UpdateError(f"Unsupported release version: {value!r}")
    return tuple(map(int, value.lstrip("v").split(".")))


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise UpdateError("Invalid release manifest.")
    if manifest.get("schema") != 1:
        raise UpdateError(
            "This package requires a newer updater. Download the new compiled release."
        )
    version_key(manifest.get("version"))
    if manifest.get("python") != list(sys.version_info[:2]):
        raise UpdateError(
            "This release needs a different Python runtime. Download its full compiled package; your current files have not been changed."
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise UpdateError("The update contains no file inventory.")
    seen = set()
    for name, entry in files.items():
        mode = file_mode(name)
        if not isinstance(entry, dict) or mode is None or entry.get("mode") != mode:
            raise UpdateError(f"The update attempts to change a protected file: {name}")
        if name.casefold() in seen or not re.fullmatch(
            r"[0-9a-f]{64}", entry.get("sha256", "")
        ):
            raise UpdateError(f"Invalid or duplicate inventory entry: {name}")
        seen.add(name.casefold())
    for name in files:
        if any(
            str(p).casefold() in seen
            for p in PurePosixPath(name).parents
            if str(p) != "."
        ):
            raise UpdateError(f"A package file is also a directory: {name}")
    for required in (
        "app.py",
        "assets/config_template.json",
        "assets/updater.py",
        "run-updater.bat",
    ):
        if required not in files:
            raise UpdateError(f"Incomplete release package: {required} is missing.")
    requirements = manifest.get("requirements")
    if (
        not isinstance(requirements, list)
        or not requirements
        or any(
            name not in files or not name.startswith("requirements") or "/" in name
            for name in requirements
        )
    ):
        raise UpdateError("Invalid dependency file inventory.")
    if not isinstance(manifest.get("migrations", []), list):
        raise UpdateError("Invalid migration inventory.")
    for migration in manifest.get("migrations", []):
        if not isinstance(migration, dict):
            raise UpdateError("Invalid data migration.")
        for key in ("from", "to"):
            name = migration.get(key)
            safe_name(name)
            # Explicit data copies, never moves, executable scripts, or overwrites.
            if not user_data_path(name):
                raise UpdateError(f"Invalid data migration path: {name}")
        a, b = PurePosixPath(migration["from"]), PurePosixPath(migration["to"])
        if a == b or a in b.parents or b in a.parents:
            raise UpdateError("Data migration paths must not overlap.")
    return manifest


def user_data_path(name):
    parts = tuple(p.lower() for p in safe_name(name).parts)
    return parts[0] in {"logs", "audios", "datasets", "outputs"} or (
        len(parts) > 1
        and parts[:2]
        in {
            ("assets", "audios"),
            ("assets", "datasets"),
            ("assets", "presets"),
            ("assets", "formant_shift"),
            ("rvc", "models"),
        }
    )


def download(url, path, limit=MAX_DOWNLOAD):
    if not url.startswith("https://"):
        raise UpdateError("Only HTTPS downloads are supported.")
    print(f"Downloading {path.name}...", flush=True)
    request = Request(
        url,
        headers={
            "User-Agent": "Applio-Updater/1",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urlopen(request, timeout=60) as response, path.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise UpdateError("Download exceeds the package size limit.")
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def latest_release(directory):
    metadata = directory / "release.json"
    download(f"{API}/releases/latest", metadata, 2 * 1024 * 1024)
    release = json.loads(metadata.read_text(encoding="utf-8"))
    tag = release.get("tag_name")
    version_key(tag)
    if release.get("prerelease") or release.get("draft"):
        raise UpdateError("The latest release is not a stable public release.")
    name = f"ApplioUpdate-{tag.lstrip('v')}.zip"
    assets = {
        asset["name"]: asset["browser_download_url"]
        for asset in release.get("assets", [])
    }
    if name not in assets or name + ".sha256" not in assets:
        raise UpdateError(
            f"Release {tag} is published, but its update package is not ready. Try again later; nothing has been changed."
        )
    prefix = f"https://github.com/{REPOSITORY}/releases/download/"
    if any(not assets[key].startswith(prefix) for key in (name, name + ".sha256")):
        raise UpdateError("Unexpected release download location.")
    return tag, assets[name], assets[name + ".sha256"]


def unpack_package(archive, stage):
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        if len(entries) > 30000 or sum(i.file_size for i in entries) > MAX_UNPACKED:
            raise UpdateError("Update archive exceeds the extraction limits.")
        if len({i.filename.casefold() for i in entries}) != len(entries):
            raise UpdateError("Duplicate archive entries.")
        for entry in entries:
            safe_name(entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
                raise UpdateError("Unexpected directory or link in the package.")
        manifest = validate_manifest(json.loads(package.read("update-manifest.json")))
        expected = {
            "update-manifest.json",
            *("payload/" + n for n in manifest["files"]),
        }
        if {i.filename for i in entries} != expected:
            raise UpdateError("Archive contents do not match the inventory.")
        for name, entry in manifest["files"].items():
            target = destination(stage, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open("payload/" + name) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if digest(target) != entry["sha256"]:
                raise UpdateError(f"Checksum mismatch: {name}")
    template = json.loads(
        (stage / "assets/config_template.json").read_text(encoding="utf-8")
    )
    if version_key(template["version"]) != version_key(manifest["version"]):
        raise UpdateError("Package version does not match its configuration template.")
    return manifest


@contextmanager
def update_lock(root):
    lockfile = destination(root, STATE + "/update.lock")
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    with lockfile.open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise UpdateError(
                "Another update is in progress. Wait for it to finish."
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def ensure_not_updating(root):
    try:
        root = Path(root).resolve()
        with update_lock(root):
            if any(
                json.loads(p.read_text(encoding="utf-8"))["status"] == "applying"
                for p in backups(root)
            ):
                raise UpdateError(
                    "An update was interrupted. Run run-updater.bat to recover before starting Applio."
                )
    except UpdateError as error:
        raise SystemExit(str(error)) from error


def assert_app_closed(root):
    if os.name != "nt":
        return
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    processes = (wintypes.DWORD * 65536)()
    size = wintypes.DWORD()
    if not psapi.EnumProcesses(processes, ctypes.sizeof(processes), ctypes.byref(size)):
        raise UpdateError(
            "Could not check running processes. Close Applio and try again."
        )
    env = (root / "env").resolve()
    for pid in processes[: size.value // ctypes.sizeof(wintypes.DWORD)]:
        if pid == os.getpid():
            continue
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            continue
        try:
            name = ctypes.create_unicode_buffer(32768)
            length = wintypes.DWORD(len(name))
            if kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(length)):
                if Path(name.value).resolve().is_relative_to(env):
                    raise UpdateError(
                        "Applio or one of its workers is still running. Close Applio, training, and TensorBoard before updating."
                    )
        finally:
            kernel.CloseHandle(handle)


def installed_version(root):
    return json.loads(
        (root / "assets/config_template.json").read_text(encoding="utf-8")
    )["version"]


def legacy_inventory(root, directory):
    """Bootstrap old HF builds from their exact GitHub tag, never from main."""
    version = installed_version(root)
    version_key(version)
    archive = directory / "previous-release.zip"
    download(
        f"https://codeload.github.com/{REPOSITORY}/zip/refs/tags/{quote(version, safe='')}",
        archive,
    )
    files = {}
    with zipfile.ZipFile(archive) as package:
        if sum(i.file_size for i in package.infolist()) > MAX_UNPACKED:
            raise UpdateError("Previous-release inventory is too large.")
        for entry in package.infolist():
            if entry.is_dir():
                continue
            name = entry.filename.partition("/")[2]
            mode = file_mode(name)
            if mode:
                files[name] = {
                    "mode": mode,
                    "sha256": hashlib.sha256(package.read(entry)).hexdigest(),
                }
    if "app.py" not in files:
        raise UpdateError(
            "Could not establish the previous release inventory. No files were changed."
        )
    return {"schema": 1, "version": version, "files": files}


def merge_defaults(current, defaults):
    result = dict(current)
    for key, value in defaults.items():
        if key not in result:
            result[key] = value
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_defaults(result[key], value)
    return result


class Transaction:
    def __init__(self, root, version):
        self.root = root
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.folder = destination(root, f"{STATE}/backups/{stamp}")
        self.folder.mkdir(parents=True)
        self.journal = {"status": "applying", "version": version, "entries": []}
        self.save()

    def save(self):
        write_json(self.folder / "journal.json", self.journal)

    def record(self, name):
        if any(e["path"] == name for e in self.journal["entries"]):
            return destination(self.root, name)
        path = destination(self.root, name)
        if path.exists() and not path.is_file():
            # A code directory can become a file only if nothing user-owned remains.
            try:
                path.rmdir()
            except OSError as error:
                raise UpdateError(
                    f"A nonempty directory blocks the update: {name}"
                ) from error
        existed = path.is_file()
        if existed:
            backup = destination(self.folder / "files", name)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        self.journal["entries"].append({"path": name, "existed": existed})
        self.save()  # Persist recovery information before modifying the installation.
        return path

    def replace(self, name, source):
        target = self.record(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".applio-new", dir=target.parent
        )
        os.close(handle)
        temporary = Path(name)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def remove(self, name):
        target = destination(self.root, name)
        if not target.is_file():
            return
        self.record(name).unlink()
        parent = target.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def complete(self):
        self.journal["status"] = "complete"
        self.save()


def rollback(root, folder):
    journal = json.loads((folder / "journal.json").read_text(encoding="utf-8"))
    for entry in reversed(journal["entries"]):
        name = entry["path"]
        # Journals can restore app files, configuration, state, and copied user data only.
        if (
            file_mode(name) is None
            and name not in {"assets/config.json", STATE + "/installed.json"}
            and not user_data_path(name)
        ):
            raise UpdateError(f"Unsafe recovery entry: {name}")
        target = destination(root, name)
        if entry["existed"]:
            backup = destination(folder / "files", name)
            if not backup.is_file():
                raise UpdateError(f"Missing recovery file: {name}")
            if target.is_file() and digest(target) == digest(backup):
                continue
            if target.is_dir():
                target.rmdir()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)
        else:
            target.unlink(missing_ok=True)
            parent = target.parent
            while parent != root:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
    journal["status"] = "rolled-back"
    write_json(folder / "journal.json", journal)


def backups(root):
    directory = destination(root, STATE + "/backups")
    return sorted(directory.glob("*/journal.json"), reverse=True)


def recover_pending(root):
    for path in backups(root):
        if json.loads(path.read_text(encoding="utf-8"))["status"] == "applying":
            print("Recovering an interrupted code update...", flush=True)
            rollback(root, path.parent)


def copy_migrations(root, manifest, transaction, completed):
    completed = set(completed)
    for migration in manifest.get("migrations", []):
        identifier = migration["from"] + " -> " + migration["to"]
        if identifier in completed:
            continue
        source = destination(root, migration["from"])
        if not source.exists():
            completed.add(identifier)
            continue
        paths = sorted(source.rglob("*")) if source.is_dir() else [source]
        for item in paths:
            relative = item.relative_to(source) if source.is_dir() else Path()
            source_name = item.relative_to(root).as_posix()
            destination(root, source_name)
            if item.is_dir():
                continue
            name = (PurePosixPath(migration["to"]) / relative.as_posix()).as_posix()
            target = destination(root, name)
            if target.exists():
                if not target.is_file() or digest(target) != digest(item):
                    raise UpdateError(
                        f"Data migration would overwrite an existing file: {name}"
                    )
            else:
                transaction.replace(name, item)
        completed.add(identifier)
    return sorted(completed)


def apply_files(root, stage, previous, manifest):
    transaction = Transaction(root, manifest["version"])
    try:
        obsolete = set(previous["files"]) - set(manifest["files"])
        for name in sorted(obsolete, key=lambda n: (-len(PurePosixPath(n).parts), n)):
            if file_mode(name) == "replace":
                transaction.remove(name)
        for name, entry in manifest["files"].items():
            target = destination(root, name)
            if entry["mode"] == "seed" and target.exists():
                continue
            transaction.replace(name, stage / name)
        config = destination(root, "assets/config.json")
        template = json.loads(
            (stage / "assets/config_template.json").read_text(encoding="utf-8")
        )
        if config.exists():
            current = json.loads(config.read_text(encoding="utf-8"))
            merged = merge_defaults(current, template)
            merged["version"] = manifest["version"]
            transaction.record("assets/config.json")
            write_json(config, merged)
        completed = copy_migrations(
            root, manifest, transaction, previous.get("completed_migrations", [])
        )
        transaction.record(STATE + "/installed.json")
        write_json(
            destination(root, STATE + "/installed.json"),
            {**manifest, "completed_migrations": completed},
        )
        transaction.complete()
        return transaction.folder
    except BaseException:
        print(
            "Update did not finish. Restoring the previous code and settings...",
            flush=True,
        )
        rollback(root, transaction.folder)
        raise


def install_dependencies(root, stage, previous, manifest):
    requirements = manifest["requirements"]
    if previous.get("requirements") == requirements and all(
        previous["files"].get(n, {}).get("sha256") == manifest["files"][n]["sha256"]
        for n in requirements
    ):
        print("Dependencies are unchanged.", flush=True)
        return
    # Legacy inventories still identify the exact shipped requirement hashes.
    if not previous.get("requirements") and all(
        previous["files"].get(n, {}).get("sha256") == manifest["files"][n]["sha256"]
        for n in requirements
    ):
        print("Dependencies are unchanged.", flush=True)
        return
    python = destination(root, "env/python.exe")
    if not python.is_file() or Path(sys.executable).resolve() != python.resolve():
        raise UpdateError(
            "Run run-updater.bat with Applio's bundled env/python.exe to update dependencies."
        )
    record = destination(root, STATE + "/dependencies-before.txt")
    with record.open("w", encoding="utf-8") as stream:
        subprocess.run(
            [str(python), "-m", "pip", "freeze"], stdout=stream, check=True, cwd=root
        )
    # Use the same CUDA wheel index as the compiled installer, with explicit Python.
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu128",
    ]
    for name in requirements:
        command.extend(["-r", str(stage / name)])
    print(
        "Installing release dependencies into the bundled Python environment...",
        flush=True,
    )
    try:
        subprocess.run(command, check=True, cwd=stage)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as error:
        raise UpdateError(
            "Dependency installation did not finish. Application code and user data have not been changed. Some Python packages may have changed; rerun the updater to finish. The previous package list is in .applio-update/dependencies-before.txt."
        ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check", action="store_true", help="Check for a release without installing it"
    )
    action.add_argument(
        "--rollback",
        action="store_true",
        help="Restore the most recent code/config backup (does not downgrade Python packages)",
    )
    parser.add_argument(
        "--package", type=Path, help="Install a locally downloaded official update ZIP"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    args = parser.parse_args(argv)
    if args.package and args.rollback:
        parser.error("--rollback cannot be combined with --package")
    root = args.root.resolve()
    try:
        with update_lock(root):
            if not args.check:
                assert_app_closed(root)
                recover_pending(root)
            version_key(installed_version(root))
            if args.rollback:
                candidates = [
                    p
                    for p in backups(root)
                    if json.loads(p.read_text(encoding="utf-8"))["status"] == "complete"
                ]
                if not candidates:
                    raise UpdateError("There is no completed update to roll back.")
                rollback(root, candidates[0].parent)
                print(
                    "Previous code and settings restored. Python packages have not been downgraded."
                )
                return 0
            with tempfile.TemporaryDirectory(prefix="applio-update-") as temporary:
                work = Path(temporary)
                if args.package:
                    archive = args.package.resolve()
                    if archive.stat().st_size > MAX_DOWNLOAD:
                        raise UpdateError(
                            "Update archive exceeds the download size limit."
                        )
                    stage = work / "payload"
                    manifest = unpack_package(archive, stage)
                else:
                    tag, url, checksum_url = latest_release(work)
                    if version_key(tag) <= version_key(installed_version(root)):
                        print(f"Applio {installed_version(root)} is up to date.")
                        return 0
                    if args.check:
                        print(
                            f"Applio {tag} is available. Close Applio and run run-updater.bat to install it."
                        )
                        return 0
                    archive = work / "update.zip"
                    checksum = work / "update.sha256"
                    download(checksum_url, checksum, 4096)
                    expected = checksum.read_text(encoding="utf-8").split()[0]
                    if not re.fullmatch(r"[0-9a-f]{64}", expected):
                        raise UpdateError("Invalid release checksum.")
                    download(url, archive)
                    if digest(archive) != expected:
                        raise UpdateError(
                            "The downloaded update checksum does not match. Nothing has been installed."
                        )
                    stage = work / "payload"
                    manifest = unpack_package(archive, stage)
                    if version_key(manifest["version"]) != version_key(tag):
                        raise UpdateError(
                            "The update package does not match the GitHub release."
                        )
                if version_key(manifest["version"]) <= version_key(
                    installed_version(root)
                ):
                    raise UpdateError(
                        "The package is not newer than the installed release."
                    )
                if args.check:
                    print(f"Validated update package for {manifest['version']}.")
                    return 0
                inventory = destination(root, STATE + "/installed.json")
                previous = (
                    json.loads(inventory.read_text(encoding="utf-8"))
                    if inventory.exists()
                    else legacy_inventory(root, work)
                )
                # All paths are checked before dependencies or installation files change.
                for name in (
                    set(previous["files"])
                    | set(manifest["files"])
                    | {"assets/config.json", STATE + "/installed.json"}
                ):
                    destination(root, name)
                if (root / "assets/config.json").exists():
                    config = json.loads(
                        (root / "assets/config.json").read_text(encoding="utf-8")
                    )
                    if not isinstance(config, dict):
                        raise UpdateError(
                            "The local configuration is not a JSON object."
                        )
                install_dependencies(root, stage, previous, manifest)
                backup = apply_files(root, stage, previous, manifest)
                print(
                    f"Updated Applio to {manifest['version']}. Start run-applio.bat.\nCode and settings backup: {backup}\nModels, logs, audio, datasets, and installed plugins were preserved."
                )
                return 0
    except (
        UpdateError,
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as error:
        print(f"Update stopped: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("Update cancelled.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
