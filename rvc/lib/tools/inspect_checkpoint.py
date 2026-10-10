import io as _io
import json
import os as _os
import re as _re
import hashlib as _hashlib
import pickle as _pickle
import torch


class _TolerantUnpickler(_pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (), {})


class _TolerantPickle:
    Unpickler = _TolerantUnpickler
    load = staticmethod(lambda f, **kw: _TolerantUnpickler(f, **kw).load())
    loads = staticmethod(
        lambda s, **kw: _TolerantUnpickler(_io.BytesIO(s), **kw).load()
    )


def inspect_checkpoint(path: str) -> dict:
    data = None
    load_error = None
    try:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        except Exception:
            data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        try:
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                pickle_module=_TolerantPickle,
            )
        except TypeError:
            try:
                data = torch.load(path, map_location="cpu")
            except Exception as e:
                load_error = str(e)[:1000]
                data = None
        except Exception as e:
            load_error = str(e)[:1000]
            data = None

    if data is None:
        raise ValueError("LOAD_FAILED: " + (load_error or "unknown error"))

    d = data if isinstance(data, dict) else {}

    def _g(k, default="None"):
        try:
            v = d.get(k, default)
        except Exception:
            return default
        if v is None:
            return default
        try:
            s = str(v).strip()
            return s if s else default
        except Exception:
            return default

    # Look for sidecar model_info.json
    sidecar = {}
    try:
        info_path = _os.path.join(_os.path.dirname(path), "model_info.json")
        if _os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
    except Exception:
        pass

    # Extract epochs & steps
    epochs = _g("epoch")
    if epochs == "None" and "epoch" in sidecar:
        epochs = str(sidecar["epoch"])
    if epochs == "None" and "info" in d:
        m = _re.search(r"(\d+)\s*epoch", str(d["info"]), _re.I)
        if m:
            epochs = m.group(1)
    if epochs == "None":
        m = _re.search(r"_e(\d+)", path)
        if m:
            epochs = m.group(1)

    step = _g("step")
    if step == "None" and "step" in sidecar:
        step = str(sidecar["step"])
    if step == "None":
        m = _re.search(r"_s(\d+)", path)
        if m:
            step = m.group(1)

    # Extract sample rate
    sr = _g("sr")
    if (
        sr == "None"
        and "config" in d
        and isinstance(d["config"], (list, tuple))
        and len(d["config"]) > 0
    ):
        sr_val = d["config"][-1]
        if isinstance(sr_val, (int, float)):
            sr = f"{int(sr_val)//1000}k"

    # Extract model hash (fast sha256 of first 8MB)
    mhash = _g("model_hash")
    if mhash == "None":
        try:
            h = _hashlib.sha256()
            with open(path, "rb") as f:
                h.update(f.read(8 * 1024 * 1024))
            mhash = h.hexdigest()[:16]
        except Exception:
            mhash = "None"

    model_name = _g("model_name")
    if model_name == "None" and "model_name" in sidecar:
        model_name = str(sidecar["model_name"])
    if model_name == "None":
        model_name = _os.path.splitext(_os.path.basename(path))[0]

    embedder = _g("embedder_model")
    if embedder == "None" and "embedder_model" in sidecar:
        embedder = str(sidecar["embedder_model"])
    if embedder == "None":
        embedder = "contentvec"

    meta = {
        "model_name": model_name,
        "author": (
            _g("author")
            if _g("author") != "None"
            else str(sidecar.get("author", "None"))
        ),
        "epochs": epochs,
        "step": step,
        "sr": sr,
        "f0": _g("f0"),
        "version": _g("version", "v2"),
        "vocoder": _g("vocoder", "HiFi-GAN"),
        "embedder_model": embedder,
        "creation_date": (
            _g("creation_date")
            if _g("creation_date") != "None"
            else str(sidecar.get("creation_date", "None"))
        ),
        "model_hash": mhash,
        "dataset_length": (
            _g("dataset_length")
            if _g("dataset_length") != "None"
            else str(sidecar.get("dataset_length", "None"))
        ),
        "speakers_id": _g("speakers_id", "0"),
        "architecture": "RVC",
    }
    return meta
