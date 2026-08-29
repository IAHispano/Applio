import os
import sys
import json
import platform


def platform_config():
    if sys.platform == "darwin" and platform.machine() == "arm64":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if sys.platform == "win32":
        try:
            config_path = os.path.join(os.getcwd(), "assets", "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if config.get("realtime", {}).get("asio_enabled", False):
                os.environ["SD_ENABLE_ASIO"] = "1"
        except Exception:
            pass
