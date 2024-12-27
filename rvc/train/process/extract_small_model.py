import os
import torch
import hashlib
import datetime
from collections import OrderedDict


def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part)
        updated_dict[new_key] = (
            replace_keys_in_dict(value, old_key_part, new_key_part)
            if isinstance(value, dict)
            else value
        )
    return updated_dict


def extract_small_model(
    path: str,
    name: str,
    sr: int,
    pitch_guidance: bool,
    version: str,
    epoch: int,
    step: int,
):
    try:
        ckpt = torch.load(path, map_location="cpu")
        pth_file = f"{name}.pth"
        pth_file_old_version_path = os.path.join("logs", f"{pth_file}_old_version.pth")

        if "model" in ckpt:
            ckpt = ckpt["model"]

        opt = OrderedDict(
            weight={
                key: value.half() for key, value in ckpt.items() if "enc_q" not in key
            }
        )

        config_map = {
            "40000": [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5]] * 3,
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ],
            "48000": [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5]] * 3,
                [12, 10, 2, 2] if version != "v1" else [10, 6, 2, 2, 2],
                512,
                [24, 20, 4, 4] if version != "v1" else [16, 16, 4, 4, 4],
                109,
                256,
                48000,
            ],
            "32000": [
                513,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5]] * 3,
                [10, 8, 2, 2] if version != "v1" else [10, 4, 2, 2, 2],
                512,
                [20, 16, 4, 4] if version != "v1" else [16, 16, 4, 4, 4],
                109,
                256,
                32000,
            ],
        }
        opt["config"] = config_map.get(str(sr), [])

        opt.update(
            {
                "epoch": epoch,
                "step": step,
                "sr": sr,
                "f0": int(pitch_guidance),
                "version": version,
                "creation_date": datetime.datetime.now().isoformat(),
            }
        )

        hash_input = f"{name}-{epoch}-{step}-{sr}-{version}-{opt['config']}"
        model_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        opt["model_hash"] = model_hash

        model = torch.load(pth_file_old_version_path, map_location=torch.device("cpu"))
        updated_model = replace_keys_in_dict(
            replace_keys_in_dict(
                model, ".parametrizations.weight.original1", ".weight_v"
            ),
            ".parametrizations.weight.original0",
            ".weight_g",
        )
        torch.save(updated_model, pth_file_old_version_path)
        os.remove(pth_file_old_version_path)
        os.rename(pth_file_old_version_path, pth_file)

    except Exception as error:
        print(f"An error occurred extracting the model: {error}")
