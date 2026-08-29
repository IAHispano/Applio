import os
import site
import sys


def disable_user_site():
    """
    Removes the per-user site-packages directory from this process.

    Applio's env is a conda prefix rather than a venv, and conda honours the
    per-user site-packages directory, so packages installed there for an
    unrelated project shadow Applio's own versions. The environment variable
    carries the same isolation into the subprocesses spawned for training.
    """
    os.environ["PYTHONNOUSERSITE"] = "1"
    if not getattr(site, "ENABLE_USER_SITE", False):
        return []

    try:
        user_paths = site.getusersitepackages()
    except Exception:
        return []
    if isinstance(user_paths, str):
        user_paths = [user_paths]
    targets = {os.path.normcase(os.path.abspath(p)) for p in user_paths}

    removed = [p for p in sys.path if os.path.normcase(os.path.abspath(p)) in targets]
    for path in removed:
        sys.path.remove(path)
    site.ENABLE_USER_SITE = False

    return removed
