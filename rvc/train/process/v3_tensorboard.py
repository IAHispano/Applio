"""Automatic TensorBoard summaries for every V3 training stage.

Only rank zero yields trainer progress, so only that rank creates a writer. The
wrapper keeps logging outside the optimization loop and closes both generator
and writer on completion, cancellation or failure. Resume purges events after
the restored checkpoint instead of displaying abandoned future updates.
"""

import inspect
import json
import math
from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path


class TrainingSummary:
    """Write scalar progress alongside one stage's checkpoints and JSONL log."""

    def __init__(self, output, first_step, configuration):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            log_dir=str(Path(output) / "tensorboard"),
            purge_step=first_step,
            flush_secs=10,
        )
        self.writer.add_text(
            "run/configuration",
            "```json\n" + json.dumps(configuration, indent=2, default=str) + "\n```",
            first_step - 1,
        )

    def write(self, progress):
        for name, value in progress.items():
            if name == "step" or not isinstance(value, (int, float, bool)):
                continue
            if not math.isfinite(value):
                continue
            tag = (
                "validation/mel_l1" if name == "validation_mel_l1" else "train/" + name
            )
            self.writer.add_scalar(tag, float(value), progress["step"])
        if "validation_mel_l1" in progress or progress.get("stopped"):
            self.writer.flush()

    def close(self):
        self.writer.close()


def log_training(function):
    """Preserve the public generator signature while logging its progress.

    Writer initialization is deferred until the first yielded update: invalid
    configurations and non-writing distributed ranks do not create empty runs.
    Scalar logging performs no model forward passes or sampling.
    """

    signature = inspect.signature(function)

    @wraps(function)
    def logged(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        iterator = function(*args, **kwargs)
        summary = None
        try:
            for progress in iterator:
                if summary is None:
                    configuration = {
                        name: asdict(value) if is_dataclass(value) else value
                        for name, value in bound.arguments.items()
                        if name not in {"stop_requested", "output"}
                    }
                    configuration.update(
                        kind=progress["kind"],
                        phase=progress["phase"],
                        precision=progress["precision"],
                    )
                    summary = TrainingSummary(
                        bound.arguments["output"], progress["step"], configuration
                    )
                summary.write(progress)
                yield progress
        finally:
            try:
                iterator.close()
            finally:
                if summary is not None:
                    summary.close()

    return logged
