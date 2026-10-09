"""Stop requests that never truncate a checkpoint.

The launcher asks for a stop with SIGTERM / CTRL_BREAK and only kills after a
grace period.  These handlers never terminate the process themselves: they
record the request and return, so a ``torch.save`` already in flight always
runs to completion and no checkpoint is ever left truncated.  The training
loop then acts on the flag at its next safe point.
"""

import os
import signal
import threading

from contextlib import contextmanager

from rvc.vocoders.terminal import info, warning


_stop_requested = threading.Event()
_saving_depth = 0
_saving_lock = threading.Lock()


def stop_was_requested():
    return _stop_requested.is_set()


@contextmanager
def uninterruptible_save(description):
    """Mark a region that must reach the disk before the process may exit."""
    global _saving_depth
    with _saving_lock:
        _saving_depth += 1
    try:
        yield
    finally:
        with _saving_lock:
            _saving_depth -= 1
        if _stop_requested.is_set():
            info(f"{description} finished; the stop can proceed now.", tag="[TRAIN]")


def _handle_stop_signal(signum, frame):
    """Record a stop request. Deliberately does not exit."""
    if _stop_requested.is_set():
        return
    _stop_requested.set()
    with _saving_lock:
        mid_write = _saving_depth > 0
    if mid_write:
        warning(
            "Stop requested while writing a checkpoint - "
            "finishing the write first, then exiting.",
            tag="[TRAIN]",
        )
    else:
        warning("Stop requested - exiting at the next safe point.", tag="[TRAIN]")


def install_stop_handlers():
    """Route the launcher's stop signals into the flag above."""
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_stop_signal)
        except (ValueError, OSError):
            # Not the main thread, or unsupported on this platform.
            pass


def finish_stop(writer=None):
    """Leave once nothing is being written."""
    info("Stopping cleanly.", tag="[TRAIN]")
    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
    os._exit(0)
