"""
Platform-specific configuration for macOS ARM64 compatibility.
This module configures OpenMP environment variables to prevent FAISS/OpenMP crashes.
Must be imported before any libraries that use FAISS or OpenMP.
"""

import os
import sys
import platform


def configure_macos_arm64():
    """
    Configure OpenMP settings for macOS ARM64 to prevent FAISS crashes.
    
    On Apple Silicon Macs, OpenMP threading causes segmentation faults and hangs
    in FAISS index searches. This function sets environment variables to force
    single-threaded operation, which is slower but stable.
    
    This function is idempotent and safe to call multiple times.
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Auto-configure on import
configure_macos_arm64()

