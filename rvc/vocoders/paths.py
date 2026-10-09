"""Filesystem layout, derived from this file's location rather than the
working directory."""

import os

PACKAGE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(PACKAGE))
#: Applio's ``logs``; a run lives in ``logs/<name>/vocoder``.
LOGS_DIR = os.path.join(ROOT, "logs")
CONFIG_DIR = os.path.join(PACKAGE, "configs")
