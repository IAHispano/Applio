import os
import re
import sys
import json
import time
import shutil
import tempfile
import warnings
from typing import Optional, Union, IO
import requests
from urllib.parse import urlparse, unquote
from tqdm import tqdm

CHUNK_SIZE = 512 * 1024
HOME = os.path.expanduser("~")


def indent(text: str, prefix: str):
    """Indent each non-empty line of text with the given prefix."""
    return "".join(
        (prefix + line if line.strip() else line) for line in text.splitlines(True)
    )


class FileURLRetrievalError(Exception):
    """Custom exception for issues retrieving file URLs."""


def _extract_download_url_from_confirmation(contents: str, url_origin: str):
    """Extract the download URL from a Google Drive confirmation page."""
    patterns = [
        r'href="(\/uc\?export=download[^"]+)',
        r'href="/open\?id=([^"]+)"',
        r'"downloadUrl":"([^"]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, contents)
        if match:
            url = match.group(1)
            if pattern == r'href="/open\?id=([^"]+)"':
                uuid_match = re.search(
                    r'<input\s+type="hidden"\s+name="uuid"\s+value="([^"]+)"',
                    contents,
                )
                if uuid_match:
                    uuid = uuid_match.group(1)
                    return (
                        "https://drive.usercontent.google.com/download?id="
                        + url
                        + "&confirm=t&uuid="
                        + uuid
                    )
                raise FileURLRetrievalError(
                    f"Could not find UUID for download from {url_origin}"
                )
            elif pattern == r'"downloadUrl":"([^"]+)':
                return url.replace("\\u003d", "=").replace("\\u0026", "&")
            else:
                return "https://docs.google.com" + url.replace("&", "&")

    error_match = re.search(r'<p class="uc-error-subcaption">(.*)</p>', contents)
    if error_match:
        error = error_match.group(1)
        raise FileURLRetrievalError(error)

    raise FileURLRetrievalError(
        "Cannot retrieve the public link of the file. "
        "You may need to change the permission to "
        "'Anyone with the link', or have had many accesses."
    )


def _create_session(
    proxy: Optional[str] = None,
    use_cookies: bool = True,
    return_cookies_file: bool = False,
):
    """Create a requests session with optional proxy and cookie handling."""
    sess = requests.session()
    sess.headers.update(
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"}
    )

    if proxy:
        sess.proxies = {"http": proxy, "https": proxy}

    cookies_file = os.path.join(HOME, ".cache/gdown/cookies.json")
    if os.path.exists(cookies_file) and use_cookies:
        try:
            with open(cookies_file) as f:
                cookies = json.load(f)
            for k, v in cookies:
                sess.cookies[k] = v
        except json.JSONDecodeError:
            warnings.warn("Corrupted Cookies file")

    return (sess, cookies_file) if return_cookies_file else sess


def download(
    output: Optional[str] = None,
    quiet: bool = False,
    proxy: Optional[str] = None,
    speed: Optional[float] = None,
    use_cookies: bool = True,
    verify: Union[bool, str] = True,
    id: Optional[str] = None,
    fuzzy: bool = True,
    resume: bool = False,
    format: Optional[str] = None,
    url: Optional[str] = None,
):
    """Download a file from a URL, supporting Google Drive links.

    Args:
        output: Output filepath. Default is basename of URL.
        quiet: Suppress terminal output.
        proxy: HTTP/HTTPS proxy.
        speed: Download speed limit (bytes per second).
        use_cookies: Flag to use cookies.
        verify: Verify TLS certificates.
        id: Google Drive's file ID.
        fuzzy: Fuzzy Google Drive ID extraction.
        resume: Resume download from a tmp file.
        format: Format for Google Docs/Sheets/Slides.
        url: URL to download from.

    Returns:
        Output filename, or None on error.
    """
    if not (id is None) ^ (url is None):
        raise ValueError("Either url or id has to be specified")

    if id is not None:
        url = f"https://drive.google.com/uc?id={id}"

    url_origin = url
    sess, cookies_file = _create_session(
        proxy=proxy, use_cookies=use_cookies, return_cookies_file=True
    )

    while True:
        res = sess.get(url, stream=True, verify=verify)
        res.raise_for_status()

        if url == url_origin and res.status_code == 500:
            url = f"https://drive.google.com/open?id={id}"
            continue

        if res.headers.get("Content-Type", "").startswith("text/html"):
            title_match = re.search("<title>(.+)</title>", res.text)
            if title_match:
                title = title_match.group(1)
                if title.endswith(" - Google Docs"):
                    url = f"https://docs.google.com/document/d/{id}/export?format={'docx' if format is None else format}"
                    continue
                if title.endswith(" - Google Sheets"):
                    url = f"https://docs.google.com/spreadsheets/d/{id}/export?format={'xlsx' if format is None else format}"
                    continue
                if title.endswith(" - Google Slides"):
                    url = f"https://docs.google.com/presentation/d/{id}/export?format={'pptx' if format is None else format}"
                    continue
            if (
                "Content-Disposition" in res.headers
                and res.headers["Content-Disposition"].endswith("pptx")
                and format not in (None, "pptx")
            ):
                url = f"https://docs.google.com/presentation/d/{id}/export?format={'pptx' if format is None else format}"
                continue

        if use_cookies:
            os.makedirs(os.path.dirname(cookies_file), exist_ok=True)
            cookies = [
                (k, v)
                for k, v in sess.cookies.items()
                if not k.startswith("download_warning_")
            ]
            with open(cookies_file, "w") as f:
                json.dump(cookies, f, indent=2)

        if "Content-Disposition" in res.headers:
            break

        parsed_url = urlparse(url)
        is_gdrive = parsed_url.hostname in ("drive.google.com", "docs.google.com")
        is_download_link = parsed_url.path.endswith("/uc")

        if not (is_gdrive and is_download_link and fuzzy):
            break

        try:
            url = _extract_download_url_from_confirmation(res.text, url_origin)
        except FileURLRetrievalError as e:
            raise FileURLRetrievalError(e)

    content_disposition = res.headers.get("Content-Disposition", "")
    filename_match = re.search(
        r"filename\*=UTF-8''(.*)", content_disposition
    ) or re.search(r'filename=["\']?(.*?)["\']?$', content_disposition)
    filename_from_url = (
        unquote(filename_match.group(1)) if filename_match else os.path.basename(url)
    )
    download_path = output or filename_from_url

    if isinstance(download_path, str) and download_path.endswith(os.path.sep):
        os.makedirs(download_path, exist_ok=True)
        download_path = os.path.join(download_path, filename_from_url)

    temp_dir = os.path.dirname(download_path) or "."
    prefix = os.path.basename(download_path)

    if isinstance(download_path, str):
        existing_tmp_files = [
            os.path.join(temp_dir, file)
            for file in os.listdir(temp_dir)
            if file.startswith(prefix)
        ]
        if resume and existing_tmp_files:
            if len(existing_tmp_files) > 1:
                print(
                    "There are multiple temporary files to resume:",
                    file=sys.stderr,
                )
                for file in existing_tmp_files:
                    print(f"\t{file}", file=sys.stderr)
                print(
                    "Please remove them except one to resume downloading.",
                    file=sys.stderr,
                )
                return None
            temp_file_path = existing_tmp_files[0]
        else:
            resume = False
            temp_file_path = tempfile.mktemp(
                suffix=tempfile.template, prefix=prefix, dir=temp_dir
            )

        try:
            file_obj: IO = open(temp_file_path, "ab")
        except Exception as e:
            print(
                f"Could not open the temporary file {temp_file_path}: {e}",
                file=sys.stderr,
            )
            return None
    else:
        temp_file_path = None
        file_obj = download_path

    if temp_file_path is not None and file_obj.tell() != 0:
        headers = {"Range": f"bytes={file_obj.tell()}-"}
        res = sess.get(url, headers=headers, stream=True, verify=verify)
        res.raise_for_status()

    try:
        total = int(res.headers.get("Content-Length", 0))
        if total > 0:
            if not quiet:
                pbar = tqdm(
                    total=total, unit="B", unit_scale=True, desc=filename_from_url
                )
        else:
            if not quiet:
                pbar = tqdm(unit="B", unit_scale=True, desc=filename_from_url)

        t_start = time.time()
        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            file_obj.write(chunk)
            if not quiet:
                pbar.update(len(chunk))
            if speed is not None:
                elapsed_time_expected = 1.0 * pbar.n / speed
                elapsed_time = time.time() - t_start
                if elapsed_time < elapsed_time_expected:
                    time.sleep(elapsed_time_expected - elapsed_time)
        if not quiet:
            pbar.close()

        if temp_file_path:
            file_obj.close()
            shutil.move(temp_file_path, download_path)
    finally:
        sess.close()

    return download_path
