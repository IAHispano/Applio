"""Progress reporting: per-epoch timing, and the line a GUI front-end parses."""

import datetime

from time import time as ttime

from rvc.vocoders.terminal import reports_to_a_pipe


class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        return f"Current time: {current_time} | Time per epoch: {elapsed_time_str}"


#: Wall-clock seconds between machine-readable progress lines.  Rich's bar is
#: for a terminal; this is for whatever is reading the pipe.
_MACHINE_PROGRESS_INTERVAL = 1.0
_last_machine_progress = 0.0


def emit_machine_progress(
    epoch, total_epochs, batch, total_batches, step, metrics, rank
):
    """Print one parseable progress line for a GUI front-end, when Rich's bar
    is not drawn. Throttled by wall clock rather than batch count, since batch
    rate varies widely by configuration.
    """
    global _last_machine_progress
    if rank != 0 or not reports_to_a_pipe():
        return

    now = ttime()
    finished = batch >= total_batches
    if not finished and now - _last_machine_progress < _MACHINE_PROGRESS_INTERVAL:
        return
    _last_machine_progress = now

    print(
        f"[PROGRESS] epoch={epoch}/{total_epochs} "
        f"batch={batch}/{total_batches} step={step} "
        f"{metrics or ''}".rstrip(),
        flush=True,
    )
