import os
import re
import six
import sys
import json
import tqdm
import time
import shutil
import warnings
import tempfile
import textwrap
import requests
from six.moves import urllib_parse


def indent(text, prefix):
    """Indent each non-empty line of text with the given prefix."""
    return "".join(
        (prefix + line if line.strip() else line) for line in text.splitlines(True)
    )


class FileURLRetrievalError(Exception):
    pass


class FolderContentsMaximumLimitError(Exception):
    pass


def parse_url(url, warning=True):
    """Parse URLs especially for Google Drive links.

    Args:
        url: URL to parse.
        warning: Whether to warn if the URL is not a download link.

    Returns:
        A tuple (file_id, is_download_link), where file_id is the ID of the
        file on Google Drive, and is_download_link is a flag indicating
        whether the URL is a download link.
    """
    parsed = urllib_parse.urlparse(url)
    query = urllib_parse.parse_qs(parsed.query)
    is_gdrive = parsed.hostname in ("drive.google.com", "docs.google.com")
    is_download_link = parsed.path.endswith("/uc")

    if not is_gdrive:
        return None, is_download_link

    file_id = query.get("id", [None])[0]
    if file_id is None:
        for pattern in (
            r"^/file/d/(.*?)/(edit|view)$",
            r"^/file/u/[0-9]+/d/(.*?)/(edit|view)$",
            r"^/document/d/(.*?)/(edit|htmlview|view)$",
            r"^/document/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
            r"^/presentation/d/(.*?)/(edit|htmlview|view)$",
            r"^/presentation/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
            r"^/spreadsheets/d/(.*?)/(edit|htmlview|view)$",
            r"^/spreadsheets/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
        ):
            match = re.match(pattern, parsed.path)
            if match:
                file_id = match.group(1)
                break

    if warning and not is_download_link:
        warnings.warn(
            "You specified a Google Drive link that is not the correct link "
            "to download a file. You might want to try `--fuzzy` option "
            f"or the following url: https://drive.google.com/uc?id={file_id}"
        )

    return file_id, is_download_link


CHUNK_SIZE = 512 * 1024  # 512KB
HOME = os.path.expanduser("~")


def get_url_from_gdrive_confirmation(contents):
    """Extract the download URL from a Google Drive confirmation page."""
    for pattern in (
        r'href="(\/uc\?export=download[^"]+)',
        r'href="/open\?id=([^"]+)"',
        r'"downloadUrl":"([^"]+)',
    ):
        match = re.search(pattern, contents)
        if match:
            url = match.group(1)
            if pattern == r'href="/open\?id=([^"]+)"':
                uuid = re.search(
                    r'<input\s+type="hidden"\s+name="uuid"\s+value="([^"]+)"',
                    contents,
                ).group(1)
                url = (
                    "https://drive.usercontent.google.com/download?id="
                    + url
                    + "&confirm=t&uuid="
                    + uuid
                )
            elif pattern == r'"downloadUrl":"([^"]+)':
                url = url.replace("\\u003d", "=").replace("\\u0026", "&")
            else:
                url = "https://docs.google.com" + url.replace("&", "&")
            return url

    match = re.search(r'<p class="uc-error-subcaption">(.*)</p>', contents)
    if match:
        error = match.group(1)
        raise FileURLRetrievalError(error)

    raise FileURLRetrievalError(
        "Cannot retrieve the public link of the file. "
        "You may need to change the permission to "
        "'Anyone with the link', or have had many accesses."
    )


def _get_session(proxy, use_cookies, return_cookies_file=False):
    """Create a requests session with optional proxy and cookie handling."""
    sess = requests.session()
    sess.headers.update(
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"}
    )

    if proxy is not None:
        sess.proxies = {"http": proxy, "https": proxy}
        print("Using proxy:", proxy, file=sys.stderr)

    cookies_file = os.path.join(HOME, ".cache/gdown/cookies.json")
    if os.path.exists(cookies_file) and use_cookies:
        with open(cookies_file) as f:
            cookies = json.load(f)
        for k, v in cookies:
            sess.cookies[k] = v

    return (sess, cookies_file) if return_cookies_file else sess


def download(
    url=None,
    output=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    verify=True,
    id=None,
    fuzzy=True,
    resume=False,
    format=None,
):
    """Download file from URL.

    Parameters
    ----------
    url: str
        URL. Google Drive URL is also supported.
    output: str
        Output filename. Default is basename of URL.
    quiet: bool
        Suppress terminal output. Default is False.
    proxy: str
        Proxy.
    speed: float
        Download byte size per second (e.g., 256KB/s = 256 * 1024).
    use_cookies: bool
        Flag to use cookies. Default is True.
    verify: bool or string
        Either a bool, in which case it controls whether the server's TLS
        certificate is verified, or a string, in which case it must be a path
        to a CA bundle to use. Default is True.
    id: str
        Google Drive's file ID.
    fuzzy: bool
        Fuzzy extraction of Google Drive's file Id. Default is False.
    resume: bool
        Resume the download from existing tmp file if possible.
        Default is False.
    format: str, optional
        Format of Google Docs, Spreadsheets and Slides. Default is:
            - Google Docs: 'docx'
            - Google Spreadsheet: 'xlsx'
            - Google Slides: 'pptx'

    Returns
    -------
    output: str
        Output filename.
    """
    if not (id is None) ^ (url is None):
        raise ValueError("Either url or id has to be specified")
    if id is not None:
        url = f"https://drive.google.com/uc?id={id}"

    url_origin = url

    sess, cookies_file = _get_session(
        proxy=proxy, use_cookies=use_cookies, return_cookies_file=True
    )

    gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=not fuzzy)

    if fuzzy and gdrive_file_id:
        # overwrite the url with fuzzy match of a file id
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        url_origin = url
        is_gdrive_download_link = True

    while True:
        res = sess.get(url, stream=True, verify=verify)

        if url == url_origin and res.status_code == 500:
            # The file could be Google Docs or Spreadsheets.
            url = f"https://drive.google.com/open?id={gdrive_file_id}"
            continue

        if res.headers["Content-Type"].startswith("text/html"):
            title = re.search("<title>(.+)</title>", res.text)
            if title:
                title = title.group(1)
                if title.endswith(" - Google Docs"):
                    url = f"https://docs.google.com/document/d/{gdrive_file_id}/export?format={'docx' if format is None else format}"
                    continue
                if title.endswith(" - Google Sheets"):
                    url = f"https://docs.google.com/spreadsheets/d/{gdrive_file_id}/export?format={'xlsx' if format is None else format}"
                    continue
                if title.endswith(" - Google Slides"):
                    url = f"https://docs.google.com/presentation/d/{gdrive_file_id}/export?format={'pptx' if format is None else format}"
                    continue
        elif (
            "Content-Disposition" in res.headers
            and res.headers["Content-Disposition"].endswith("pptx")
            and format not in (None, "pptx")
        ):
            url = f"https://docs.google.com/presentation/d/{gdrive_file_id}/export?format={'pptx' if format is None else format}"
            continue

        if use_cookies:
            os.makedirs(os.path.dirname(cookies_file), exist_ok=True)
            with open(cookies_file, "w") as f:
                cookies = [
                    (k, v)
                    for k, v in sess.cookies.items()
                    if not k.startswith("download_warning_")
                ]
                json.dump(cookies, f, indent=2)

        if "Content-Disposition" in res.headers:
            # This is the file
            break
        if not (gdrive_file_id and is_gdrive_download_link):
            break

        # Need to redirect with confirmation
        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except FileURLRetrievalError as e:
            message = (
                "Failed to retrieve file url:\n\n"
                "{}\n\n"
                "You may still be able to access the file from the browser:"
                f"\n\n\t{url_origin}\n\n"
                "but Gdown can't. Please check connections and permissions."
            ).format(indent("\n".join(textwrap.wrap(str(e))), prefix="\t"))
            raise FileURLRetrievalError(message)

    if gdrive_file_id and is_gdrive_download_link:
        content_disposition = urllib_parse.unquote(res.headers["Content-Disposition"])
        filename_from_url = (
            re.search(r"filename\*=UTF-8''(.*)", content_disposition)
            or re.search(r'filename=["\']?(.*?)["\']?$', content_disposition)
        ).group(1)
        filename_from_url = filename_from_url.replace(os.path.sep, "_")
    else:
        filename_from_url = os.path.basename(url)

    output = output or filename_from_url

    output_is_path = isinstance(output, six.string_types)
    if output_is_path and output.endswith(os.path.sep):
        os.makedirs(output, exist_ok=True)
        output = os.path.join(output, filename_from_url)

    if output_is_path:
        temp_dir = os.path.dirname(output) or "."
        prefix = os.path.basename(output)
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
                return
            tmp_file = existing_tmp_files[0]
        else:
            resume = False
            tmp_file = tempfile.mktemp(
                suffix=tempfile.template, prefix=prefix, dir=temp_dir
            )
        f = open(tmp_file, "ab")
    else:
        tmp_file = None
        f = output

    if tmp_file is not None and f.tell() != 0:
        headers = {"Range": f"bytes={f.tell()}-"}
        res = sess.get(url, headers=headers, stream=True, verify=verify)

    if not quiet:
        if resume:
            print("Resume:", tmp_file, file=sys.stderr)
        print(
            "To:",
            os.path.abspath(output) if output_is_path else output,
            file=sys.stderr,
        )

    try:
        total = int(res.headers.get("Content-Length", 0))
        if not quiet:
            pbar = tqdm.tqdm(total=total, unit="B", unit_scale=True)
        t_start = time.time()
        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            if not quiet:
                pbar.update(len(chunk))
            if speed is not None:
                elapsed_time_expected = 1.0 * pbar.n / speed
                elapsed_time = time.time() - t_start
                if elapsed_time < elapsed_time_expected:
                    time.sleep(elapsed_time_expected - elapsed_time)
        if not quiet:
            pbar.close()
        if tmp_file:
            f.close()
            shutil.move(tmp_file, output)
    finally:
        sess.close()

    return output
