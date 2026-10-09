"""Console output of the vocoder trainer: tagged lines, the two start-up
tables and the epoch bar."""

import sys
from contextlib import contextmanager

from tqdm import tqdm


def _emit(level, message, tag):
    print(f"{tag or ''} {level}{message}".strip(), flush=True)


def info(message, *, tag=None):
    _emit("", message, tag)


def success(message, *, tag=None):
    _emit("", message, tag)


def warning(message, *, tag=None):
    _emit("WARNING: ", message, tag)


def reports_to_a_pipe() -> bool:
    """Whether stdout is read by a program, which gets ``[PROGRESS]`` lines
    instead of the bar."""
    return not sys.stdout.isatty()


class _Bar:
    """tqdm behind the ``update(task, advance=, metrics=)`` call the loop makes."""

    def __init__(self, bar):
        self.bar = bar

    def update(self, task, advance=0, metrics=None):
        if metrics:
            self.bar.set_postfix_str(metrics, refresh=False)
        self.bar.update(advance)


@contextmanager
def progress_task(total, description, *, training=False, disable=False):
    with tqdm(
        total=total, desc=description, leave=False, dynamic_ncols=True,
        disable=disable or reports_to_a_pipe(),
    ) as bar:
        yield _Bar(bar), None


def _parameter_counts(module):
    module = getattr(module, "module", module)
    parameters = {id(p): p for p in module.parameters()}.values()
    return (
        sum(p.numel() for p in parameters),
        sum(p.numel() for p in parameters if p.requires_grad),
    )


def print_model_summary(models, *, title="Model Summary"):
    print(title)
    for name, module in models:
        total, trainable = _parameter_counts(module)
        print(f"  {name}: {total:,} parameters ({trainable:,} trainable)")


def print_settings_panel(rows, *, title=None):
    rows = [(str(label), str(value)) for label, value in rows]
    width = max(len(label) for label, _ in rows)
    if title:
        print(title)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")
