import io
import pickle
import torch
from datetime import datetime


def prettify_date(date_str):
    if date_str is None:
        return "None"
    try:
        date_time_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        return date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        # Creation dates from community checkpoints vary; keep the raw value
        # instead of failing the whole inspection.
        try:
            return str(date_str)
        except Exception:
            return "None"


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler that substitutes inert placeholders for missing classes.

    Lets metadata inspection succeed on legacy/community checkpoints that
    reference classes which no longer exist in this environment.
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (), {})


class _TolerantPickle:
    Unpickler = _TolerantUnpickler
    load = staticmethod(lambda f, **kw: _TolerantUnpickler(f, **kw).load())
    loads = staticmethod(lambda s, **kw: _TolerantUnpickler(io.BytesIO(s), **kw).load())


def _load_checkpoint(path):
    """Load a checkpoint, tolerating legacy pickles and old torch versions.

    Trusted local file: fall back to unrestricted, then tolerant, unpickling
    when the safe loader rejects custom/missing globals.
    """
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # torch without the weights_only flag.
            return torch.load(path, map_location="cpu")
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_TolerantPickle,
        )


def model_information(path):
    try:
        model_data = _load_checkpoint(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not read checkpoint '{path}' (file may be corrupted or not "
            f"an RVC checkpoint): {e}"
        ) from e

    if not isinstance(model_data, dict):
        raise RuntimeError(
            f"Could not read checkpoint '{path}': unexpected checkpoint format "
            f"({type(model_data).__name__}, expected a metadata dict)."
        )

    print(f"Loaded model from {path}")

    model_name = model_data.get("model_name", "None")
    epochs = model_data.get("epoch", "None")
    steps = model_data.get("step", "None")
    sr = model_data.get("sr", "None")
    f0 = model_data.get("f0", "None")
    dataset_length = model_data.get("dataset_length", "None")
    vocoder = model_data.get("vocoder", "None")
    creation_date = model_data.get("creation_date", "None")
    model_hash = model_data.get("model_hash", None)
    model_author = model_data.get("author", "None")
    embedder_model = model_data.get("embedder_model", "None")
    speakers_id = model_data.get("speakers_id", 0)

    creation_date_str = prettify_date(creation_date) if creation_date else "None"

    return (
        f"Model Name: {model_name}\n"
        f"Model Creator: {model_author}\n"
        f"Epochs: {epochs}\n"
        f"Steps: {steps}\n"
        f"Vocoder: {vocoder}\n"
        f"Sampling Rate: {sr}\n"
        f"Dataset Length: {dataset_length}\n"
        f"Creation Date: {creation_date_str}\n"
        f"Embedder Model: {embedder_model}\n"
        f"Max Speakers ID: {speakers_id}"
        f"Hash: {model_hash}\n"
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Applio Model Information")
    parser.add_argument(
        "pth_path", nargs="?", default=None, help="Path to the model .pth file"
    )
    parser.add_argument(
        "--pth-path", dest="opt_pth", default=None, help="Path to the model .pth file"
    )
    args = parser.parse_args()

    pth = args.pth_path or args.opt_pth
    if not pth:
        print("Error: No pth path provided.")
        sys.exit(1)

    print(model_information(pth))
