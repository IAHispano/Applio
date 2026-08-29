r"""
Copied into the environment's site-packages by run-install.bat.

The Windows installer builds `env` with `conda create --prefix`, which is a
conda prefix rather than a virtual environment. A real venv sets
`site.ENABLE_USER_SITE = False`, but a conda prefix does not, so Python inserts
the per-user site-packages directory (%APPDATA%\Python\PythonXY\site-packages)
into sys.path *ahead* of the environment's own site-packages. Packages another
project installed with `pip install --user` then shadow Applio's versions, and
the mismatch surfaces far from its cause: an old numba refusing Applio's numpy,
an old safetensors refusing Applio's transformers.

site.py runs this module after building sys.path but before any third-party
import, so removing the directory here isolates every entry point, including
subprocesses and a bare `env\python.exe`. Linux and macOS install into a real
venv and are unaffected.

This runs on every interpreter start: it must never raise.
"""

import os


def _disable_user_site():
    import site
    import sys

    os.environ["PYTHONNOUSERSITE"] = "1"

    if not getattr(site, "ENABLE_USER_SITE", False):
        return

    user_paths = site.getusersitepackages()
    if isinstance(user_paths, str):
        user_paths = [user_paths]
    targets = {os.path.normcase(os.path.abspath(p)) for p in user_paths}

    for path in list(sys.path):
        if os.path.normcase(os.path.abspath(path)) in targets:
            sys.path.remove(path)

    site.ENABLE_USER_SITE = False


try:
    _disable_user_site()
except Exception:
    # a broken sitecustomize would break every interpreter start
    pass
