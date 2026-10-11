"""Explicit, process-wide precision opt-ins for dedicated realtime workers."""

import os

import torch


def enable_experimental_tf32(device, gpu_name):
    if (
        os.environ.get("APPLIO_EXPERIMENTAL_TF32") != "1"
        or not str(device).startswith("cuda")
        or "NVIDIA" not in (gpu_name or "").upper()
        or "ZLUDA" in (gpu_name or "").upper()
        or getattr(torch.version, "hip", None)
        or torch.cuda.get_device_capability(device)[0] < 8
    ):
        return False
    # Configure before model warmup/capture. This affects all CUDA matmul in
    # this process; it must not be toggled around asynchronous graph replay.
    torch.backends.cuda.matmul.allow_tf32 = True
    print("[Experimental] NVIDIA TF32 matmul enabled for this realtime process.")
    return True
