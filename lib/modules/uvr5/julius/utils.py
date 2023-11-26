# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Non signal processing related utilities.
"""

import inspect
import typing as tp
import sys
import time


def simple_repr(obj, attrs: tp.Optional[tp.Sequence[str]] = None,
                overrides: dict = {}):
    """
    Return a simple representation string for `obj`.
    If `attrs` is not None, it should be a list of attributes to include.
    """
    params = inspect.signature(obj.__class__).parameters
    attrs_repr = []
    if attrs is None:
        attrs = list(params.keys())
    for attr in attrs:
        display = False
        if attr in overrides:
            value = overrides[attr]
        elif hasattr(obj, attr):
            value = getattr(obj, attr)
        else:
            continue
        if attr in params:
            param = params[attr]
            if param.default is inspect._empty or value != param.default:  # type: ignore
                display = True
        else:
            display = True

        if display:
            attrs_repr.append(f"{attr}={value}")
    return f"{obj.__class__.__name__}({','.join(attrs_repr)})"


class MarkdownTable:
    """
    Simple MarkdownTable generator. The column titles should be large enough
    for the lines content. This will right align everything.

    >>> import io  # we use io purely for test purposes, default is sys.stdout.
    >>> file = io.StringIO()
    >>> table = MarkdownTable(["Item Name", "Price"], file=file)
    >>> table.header(); table.line(["Honey", "5"]); table.line(["Car", "5,000"])
    >>> print(file.getvalue().strip())  # Strip for test purposes
    | Item Name | Price |
    |-----------|-------|
    |     Honey |     5 |
    |       Car | 5,000 |
    """
    def __init__(self, columns, file=sys.stdout):
        self.columns = columns
        self.file = file

    def _writeln(self, line):
        self.file.write("|" + "|".join(line) + "|\n")

    def header(self):
        self._writeln(f" {col} " for col in self.columns)
        self._writeln("-" * (len(col) + 2) for col in self.columns)

    def line(self, line):
        out = []
        for val, col in zip(line, self.columns):
            val = format(val, '>' + str(len(col)))
            out.append(" " + val + " ")
        self._writeln(out)


class Chrono:
    """
    Measures ellapsed time, calling `torch.cuda.synchronize` if necessary.
    `Chrono` instances can be used as context managers (e.g. with `with`).
    Upon exit of the block, you can access the duration of the block in seconds
    with the `duration` attribute.

    >>> with Chrono() as chrono:
    ...     _ = sum(range(10_000))
    ...
    >>> print(chrono.duration < 10)  # Should be true unless on a really slow computer.
    True
    """
    def __init__(self):
        self.duration = None

    def __enter__(self):
        self._begin = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tracebck):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.duration = time.time() - self._begin
