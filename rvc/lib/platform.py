import os
import sys
import platform


def platform_config():
    if sys.platform == "darwin" and platform.machine() == "arm64":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
