"""Automatic TensorBoard summaries for every V3 training stage.

Only rank zero yields trainer progress, so only that rank creates a writer. The
wrapper keeps logging outside the optimization loop and closes both generator
and writer on completion, cancellation or failure. Resume purges events after
the restored checkpoint instead of displaying abandoned future updates.
"""

import inspect
import json
import math
import re
from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path


def write_listening(writer, clips, step, prefix="listening"):
    """Use a shared peak gain so comparisons preserve relative loudness."""
    import numpy as np

    gain = max(1.0, *(float(np.max(np.abs(wave))) for wave, _ in clips.values()))
    for label, (wave, rate) in clips.items():
        writer.add_audio(f"{prefix}/{label}", wave / gain, step, sample_rate=rate)


def sync_evaluation_audio(project):
    """Import completed evaluations for a running job without reloading its model.

    The persisted source timestamps prevent repeated imports after UI restarts.
    Two fixed examples per checkpoint keep TensorBoard lightweight. This is a
    dashboard reader, never a training worker or an additional GPU workload.
    """
    import soundfile as sf

    project = Path(project)
    output = project / "tensorboard" / "listening"
    index = output / "imported.json"
    known = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    reports = sorted((project / "evaluation").rglob("report.json"))
    pending = []
    for report in reports:
        match = re.fullmatch(r"vocoder_step_(\d+)", report.parent.name)
        key = str(report.relative_to(project))
        stamp = report.stat().st_mtime_ns
        if known.get(key) == stamp:
            continue
        details = json.loads(report.read_text(encoding="utf-8"))
        if match:
            step, prefix = int(match[1]), "listening"
        elif (
            details.get("vocoder_backend") == "bigvgan-v2"
            and report.parent.name == "matched"
        ):
            # Keep matched recordings and stage labels stable across evaluations.
            step = int(details["acoustic_step"])
            prefix = f"listening/bigvgan/{details['acoustic_phase']}"
        else:
            continue
        pending.append((step, report.parent, key, stamp, prefix))
    if not pending:
        return
    from torch.utils.tensorboard import SummaryWriter

    output.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(str(output)) as writer:
        writer.add_text(
            "listening/guide",
            "Reference = original held-out recording. Reconstruction = acoustic predictor + "
            "vocoder. Reference-mel vocoder = vocoder with the original mel features, to "
            "isolate vocoder quality. Step is the saved evaluation's update label. Compare "
            "the same example across updates; this is self-reconstruction, not voice conversion.",
            0,
        )
        for step, directory, key, stamp, prefix in sorted(pending):
            for example in range(2):
                clips = {}
                for label, suffix in (
                    ("01_reference", "reference"),
                    ("02_reconstruction", "budget_0"),
                    ("03_reference_mel_vocoder", "budget_ceiling"),
                ):
                    path = directory / f"example_{example:04d}_{suffix}.wav"
                    if path.exists():
                        wave, rate = sf.read(path, dtype="float32")
                        if wave.ndim != 1:
                            raise ValueError("Listening previews must be mono")
                        clips[label] = (wave[: rate * 8], rate)
                if clips:
                    write_listening(
                        writer, clips, step, f"{prefix}/example_{example + 1}"
                    )
            known[key] = stamp
    # The launcher serializes imports; replace keeps restart metadata complete.
    temporary = index.with_suffix(".tmp")
    temporary.write_text(json.dumps(known, indent=2), encoding="utf-8")
    temporary.replace(index)


class TrainingSummary:
    """Write scalar progress alongside one stage's checkpoints and JSONL log."""

    def __init__(self, output, first_step, configuration):
        from torch.utils.tensorboard import SummaryWriter

        self.output = Path(output)
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
            self.write_audio(progress["step"])
            self.writer.flush()

    def write_audio(self, step):
        import soundfile as sf

        clips = {}
        for label in ("reference", "reconstruction"):
            path = self.output / "listening" / f"{label}.wav"
            if path.exists():
                wave, rate = sf.read(path, dtype="float32")
                clips[label] = (wave[: rate * 8], rate)
        if clips:
            write_listening(self.writer, clips, step)

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
