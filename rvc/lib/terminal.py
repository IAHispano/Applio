"""Session output capture shared by Gradio and its spawned workers."""

import atexit
import codecs
import os
from pathlib import Path
import re
import sys
import tempfile
import threading

_ENV = "APPLIO_TERMINAL_LOG"
_MAX_VIEW = 128 * 1024
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_capture = None
_worker_pid = None


class _TerminalWriter:
    """Write Unicode to Windows consoles, and original bytes to pipes/files."""

    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.console = None
        if sys.platform == "win32":
            import ctypes
            import msvcrt

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            handle = msvcrt.get_osfhandle(descriptor)
            mode = ctypes.c_ulong()
            if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel.WriteConsoleW.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                    ctypes.c_void_p, ctypes.c_void_p,
                ]
                self.console = (kernel, handle)
                self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def write(self, data, final=False):
        if self.console is None:
            view = memoryview(data)
            while view:
                view = view[os.write(self.descriptor, view) :]
            return
        import ctypes

        # A progress-bar glyph may straddle two reads from the session log.
        text = self.decoder.decode(data, final=final)
        encoded = text.encode("utf-16-le")
        buffer = ctypes.create_string_buffer(encoded)
        position = 0
        kernel, handle = self.console
        while position < len(encoded):
            written = ctypes.c_ulong()
            count = min((len(encoded) - position) // 2, 16000)
            if not kernel.WriteConsoleW(
                handle, ctypes.byref(buffer, position), count,
                ctypes.byref(written), None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not written.value:
                raise OSError("Console write made no progress")
            position += written.value * 2


def _open_append(path):
    if sys.platform != "win32":
        return os.open(path, os.O_WRONLY | os.O_APPEND)
    import ctypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    # FILE_APPEND_DATA makes each write append atomically across worker handles;
    # CRT O_APPEND alone can race while seeking to EOF in separate processes.
    handle = kernel.CreateFileW(str(path), 0x4, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)
        raise


def _redirect(path):
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    descriptor = _open_append(path)
    try:
        for target in (1, 2):
            os.dup2(descriptor, target)
            if sys.platform == "win32":
                import ctypes
                import msvcrt

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
                if not kernel.SetStdHandle(
                    (-11 if target == 1 else -12) & 0xFFFFFFFF,
                    msvcrt.get_osfhandle(target),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
    finally:
        os.close(descriptor)
    # Rebuild the streams: WindowsConsoleIO retains its old console handle even
    # after dup2, whereas FileIO follows the redirected descriptors.
    sys.stdout = os.fdopen(
        os.dup(1), "w", encoding="utf-8", errors="backslashreplace", buffering=1
    )
    sys.stderr = os.fdopen(
        os.dup(2), "w", encoding="utf-8", errors="backslashreplace", buffering=1
    )


def capture_worker_output():
    """Opt spawned workers into an existing UI session; CLI runs stay unchanged."""
    global _worker_pid
    path = os.environ.get(_ENV)
    if path and _worker_pid != os.getpid():
        _redirect(path)
        _worker_pid = os.getpid()


class TerminalCapture:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(
            prefix="applio-terminal-", ignore_cleanup_errors=True
        )
        self.path = Path(self.directory.name) / "session.log"
        self.path.touch()
        self.original_streams = (sys.stdout, sys.stderr)
        self.original_fds = (os.dup(1), os.dup(2))
        self.previous_env = {
            key: os.environ.get(key)
            for key in (_ENV, "PYTHONUNBUFFERED", "PYTHONIOENCODING")
        }
        self.stop = threading.Event()
        self.closed = False
        os.environ[_ENV] = str(self.path)
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["PYTHONIOENCODING"] = "utf-8"
        _redirect(self.path)
        self.thread = threading.Thread(
            target=self._echo, name="terminal-mirror", daemon=True
        )
        self.thread.start()
        atexit.register(self.close)

    def _echo(self):
        writer = _TerminalWriter(self.original_fds[0])
        with self.path.open("rb", buffering=0) as stream:
            while True:
                chunk = stream.read(64 * 1024)
                if chunk:
                    try:
                        writer.write(chunk)
                    except OSError:
                        pass  # A closed terminal must not break the UI log.
                elif self.stop.is_set():
                    try:
                        writer.write(b"", final=True)
                    except OSError:
                        pass
                    return
                else:
                    self.stop.wait(0.05)

    def read(self, start=0):
        with self.path.open("rb") as stream:
            end = stream.seek(0, os.SEEK_END)
            offset = max(int(start), end - _MAX_VIEW)
            stream.seek(offset)
            data = stream.read(_MAX_VIEW)
        text = _ANSI.sub("", data.decode("utf-8", errors="replace"))
        # tqdm uses CR to replace the current line. Keep only its latest update.
        text = "\n".join(
            line.split("\r")[-1] for line in text.replace("\r\n", "\n").split("\n")
        )
        prefix = "[Earlier output omitted from this view]\n" if offset > int(start) else ""
        return prefix + text

    def position(self):
        return self.path.stat().st_size

    def close(self):
        if self.closed:
            return
        self.closed = True
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        for target, original in zip((1, 2), self.original_fds):
            os.dup2(original, target)
            if sys.platform == "win32":
                import ctypes
                import msvcrt

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
                kernel.SetStdHandle(
                    (-11 if target == 1 else -12) & 0xFFFFFFFF,
                    msvcrt.get_osfhandle(target),
                )
        redirected = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = self.original_streams
        for stream in redirected:
            stream.close()
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.stop.set()
        self.thread.join(timeout=2)
        if not self.thread.is_alive():
            for descriptor in self.original_fds:
                os.close(descriptor)
            self.directory.cleanup()


def start_capture():
    global _capture
    if _capture is None or _capture.closed:
        _capture = TerminalCapture()
    return _capture


def get_capture():
    return _capture
