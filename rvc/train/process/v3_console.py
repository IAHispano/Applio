"""Classic-style terminal progress; detailed metrics stay in JSONL/TensorBoard.

Animated bars bypass a campaign's file tee so console.log contains readable
stage/checkpoint summaries rather than thousands of carriage-return frames.
"""

import sys
import time
from tqdm import tqdm


class ConsoleProgress:
    def __init__(self, model_name=""):
        self.model_name = model_name
        self.bar = None
        self.key = None
        self.started = None
        self.last_summary = None
        self.label = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def message(self, text):
        if self.bar:
            self.bar.clear()
        print(text, flush=True)
        if self.bar:
            self.bar.refresh()

    def close(self):
        if self.bar is not None:
            bar = self.bar
            bar.close()
            self.bar = None
            if bar.n == bar.total:
                self.message(
                    f"{self.label} completed in {time.monotonic() - self.started:.2f} seconds."
                )
        self.key = None

    def update(self, key, label, count, total, unit="files", initial=0, loss=None):
        if self.key != key:
            self.close()
            self.started = time.monotonic()
            self.label = label
            self.message(f"{label}...")
            # The original terminal owns rendering; its tee still records summaries.
            stream = getattr(sys.stderr, "terminal", sys.stderr)
            self.bar = tqdm(
                total=total,
                initial=initial,
                desc=label,
                unit=unit,
                dynamic_ncols=True,
                mininterval=0.5,
                leave=False,
                file=stream,
                disable=not stream.isatty(),
            )
            self.key = key
        if loss is not None:
            self.bar.set_postfix(loss=f"{loss:.4f}", refresh=False)
        # Disabled tqdm bars do not update their counter; retain completion semantics.
        if self.bar.disable:
            self.bar.n = count
        else:
            self.bar.update(count - self.bar.n)

    def preparation(self, item):
        operation = item.get("operation", "preprocess")
        if operation == "scan":
            self.message("Finding audio recordings...")
            return
        labels = {
            "hash": "Checking recordings",
            "reserve_validation": "Reserving validation",
            "preprocess": "Preprocessing",
            "extract": "Feature extraction",
        }
        self.update(
            operation,
            labels.get(operation, operation.capitalize()),
            item["recording"],
            item["total"],
            unit="segments" if operation == "extract" else "files",
        )

    def training(self, item, budget, additional=False):
        if "step" not in item:
            if item.get("status") == "exported":
                self.close()
                self.message(f"Exported acoustic model: {item['acoustic']}")
                self.message(f"Vocoder: {item['vocoder']}")
            return
        phase, step = item["phase"], item["step"]
        if item.get("status") == "skipped":
            self.close()
            self.message(f"{phase.capitalize()} already complete at step {step:,}.")
            return
        initial = max(0, step - 1)
        total = (
            (initial + budget if additional else budget)
            if self.key != phase
            else self.bar.total
        )
        self.update(
            phase,
            phase.capitalize(),
            step,
            total,
            unit="it",
            initial=initial,
            loss=item.get("loss"),
        )
        if "validation_mel_l1" in item and self.last_summary != (phase, step):
            self.message(
                f"{self.model_name} | stage={phase} | step={step:,} | "
                f"loss={item['loss']:.4f} | validation={item['validation_mel_l1']:.4f}"
            )
            self.last_summary = (phase, step)
