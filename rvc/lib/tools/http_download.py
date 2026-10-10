"""Retry and resume HTTP transfers; publish only complete, verified files."""

import hashlib
import json
import os
from pathlib import Path
import re
import time

import requests


def download_http(url, destination, progress=None, sha256=None, retries=3):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + ".part")
    metadata_file = Path(str(partial) + ".json")
    for attempt in range(retries + 1):
        offset = 0
        metadata = {}
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                metadata = {}
            if (
                metadata.get("url") == url
                and isinstance(metadata.get("validator"), str)
                and metadata["validator"]
            ):
                offset = partial.stat().st_size
        except (OSError, ValueError, TypeError):
            pass
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers.update(
                {"Range": f"bytes={offset}-", "If-Range": metadata["validator"]}
            )
        try:
            with requests.get(
                url, stream=True, headers=headers, timeout=(30, 60)
            ) as response:
                if response.status_code == 416:
                    metadata_file.unlink(missing_ok=True)
                response.raise_for_status()
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise ValueError("Unexpected compression in model transfer")
                etag = response.headers.get("etag", "")
                validator = (
                    etag
                    if etag and not etag.startswith("W/")
                    else response.headers.get("last-modified", "")
                )
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)",
                    response.headers.get("content-range", ""),
                )
                if response.status_code == 206:
                    if (
                        not match
                        or not offset
                        or int(match[1]) != offset
                        or validator != metadata.get("validator")
                    ):
                        metadata_file.unlink(missing_ok=True)
                        raise ValueError("Invalid download resume response")
                else:
                    offset = 0
                length = int(response.headers.get("content-length", 0))
                total = (
                    int(match[3]) if match and response.status_code == 206 else length
                )
                received = offset
                if progress:
                    progress(received, total)
                with partial.open("ab" if offset else "wb") as file:
                    metadata_file.write_text(
                        json.dumps(
                            {"url": url, "validator": validator, "total": total}
                        ),
                        encoding="utf-8",
                    )
                    for block in response.iter_content(1024 * 256):
                        if block:
                            file.write(block)
                            received += len(block)
                            if progress:
                                progress(received, total)
                    if (
                        not received
                        or (total and received != total)
                        or (length and received - offset != length)
                    ):
                        raise ValueError("Incomplete model download")
                    file.flush()
                    os.fsync(file.fileno())
                if sha256:
                    with partial.open("rb") as file:
                        hash = hashlib.sha256()
                        for block in iter(lambda: file.read(1024 * 1024), b""):
                            hash.update(block)
                        digest = hash.hexdigest()
                    if digest != sha256.lower():
                        metadata_file.unlink(missing_ok=True)
                        raise ValueError("Model checksum does not match")
                os.replace(partial, destination)
                metadata_file.unlink(missing_ok=True)
                return str(destination)
        except (requests.RequestException, OSError, ValueError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            permanent = (
                status is not None and status < 500 and status not in (408, 416, 429)
            )
            if attempt == retries or permanent:
                raise
            print(f"Retrying download ({attempt + 1}/{retries})…", flush=True)
            time.sleep(0.25 * 2**attempt)
