"""Root launcher for Applio's Gradio Web UI (delegates to frontend/app.py)."""

import os
import runpy
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if __name__ == "__main__":
    runpy.run_path(
        os.path.join(ROOT_DIR, "frontend", "app.py"),
        run_name="__main__",
    )
