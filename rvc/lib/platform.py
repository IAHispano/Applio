import os
import sys
import platform
from rvc.lib.user_config import load_config


def platform_config():
    if sys.platform == "darwin" and platform.machine() == "arm64":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if sys.platform == "win32":
        try:
            config = load_config()
            if config.get("realtime", {}).get("asio_enabled", False):
                os.environ["SD_ENABLE_ASIO"] = "1"
        except Exception:
            pass
