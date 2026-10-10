# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities to save and load models.
"""

from contextlib import contextmanager

import functools
import hashlib
import inspect
import io
from pathlib import Path
import warnings

try:
    from diffq import DiffQuantizer, UniformQuantizer, restore_quantized_state
except ImportError:
    DiffQuantizer = None
    UniformQuantizer = None
    restore_quantized_state = None
import torch


def get_quantizer(model, args, optimizer=None):
    """Return the quantizer given the XP quantization args."""
    quantizer = None
    if getattr(args, "diffq", False):
        if DiffQuantizer is None:
            raise ImportError("diffq is required for DiffQuantizer.")
        quantizer = DiffQuantizer(
            model, min_size=args.min_size, group_size=args.group_size
        )
        if optimizer is not None:
            quantizer.setup_optimizer(optimizer)
    elif getattr(args, "qat", False):
        if UniformQuantizer is None:
            raise ImportError("diffq is required for UniformQuantizer.")
        quantizer = UniformQuantizer(model, bits=args.qat, min_size=args.min_size)
    return quantizer


def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, "cpu", weights_only=False)
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)
    return model


def get_state(model, quantizer, half=False):
    """Get the state from a model, potentially with quantization applied.
    If `half` is True, model are stored as half precision, which shouldn't impact performance
    but half the state size."""
    if quantizer is None:
        dtype = torch.half if half else None
        state = {
            k: p.data.to(device="cpu", dtype=dtype)
            for k, p in model.state_dict().items()
        }
    else:
        state = quantizer.get_quantized_state()
        state["__quantized"] = True
    return state


def set_state(model, state, quantizer=None):
    """Set the state on a given model."""
    if state.get("__quantized"):
        if quantizer is not None:
            quantizer.restore_quantized_state(model, state["quantized"])
        else:
            if restore_quantized_state is None:
                raise ImportError(
                    "diffq is required to restore quantized Demucs models."
                )
            restore_quantized_state(model, state)
    else:
        model.load_state_dict(state)
    return state


def save_with_checksum(content, path):
    """Save the given value on disk, along with a sha256 hash.
    Should be used with the output of either `serialize_model` or `get_state`."""
    buf = io.BytesIO()
    torch.save(content, buf)
    sig = hashlib.sha256(buf.getvalue()).hexdigest()[:8]

    path = path.parent / (path.stem + "-" + sig + path.suffix)
    path.write_bytes(buf.getvalue())


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
