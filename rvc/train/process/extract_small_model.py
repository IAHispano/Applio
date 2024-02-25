import os
import torch
from collections import OrderedDict


def replace_keys_in_dict(d, old_key_part, new_key_part):
    # Use OrderedDict if the original is an OrderedDict
    if isinstance(d, OrderedDict):
        updated_dict = OrderedDict()
    else:
        updated_dict = {}
    for key, value in d.items():
        # Replace the key part if found
        new_key = key.replace(old_key_part, new_key_part)
        # If the value is a dictionary, apply the function recursively
        if isinstance(value, dict):
            value = replace_keys_in_dict(value, old_key_part, new_key_part)
        updated_dict[new_key] = value
    return updated_dict


def extract_small_model(path, name, sr, if_f0, version):
    try:
        ckpt = torch.load(path, map_location="cpu")
        pth_file = f"{name}.pth"
        pth_file_old_version_path = os.path.join("logs", f"{pth_file}_old_version.pth")
        opt = OrderedDict(
            weight={
                key: value.half() for key, value in ckpt.items() if "enc_q" not in key
            }
        )
        if "model" in ckpt:
            ckpt = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        if sr == "40k":
            opt["config"] = [
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
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ]
        elif sr == "48k":
            if version == "v1":
                opt["config"] = [
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
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    48000,
                ]
            else:
                opt["config"] = [
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
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2],
                    512,
                    [24, 20, 4, 4],
                    109,
                    256,
                    48000,
                ]
        elif sr == "32k":
            if version == "v1":
                opt["config"] = [
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
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    32000,
                ]
            else:
                opt["config"] = [
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
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2],
                    512,
                    [20, 16, 4, 4],
                    109,
                    256,
                    32000,
                ]
 
        opt["info"] = "Extracted model."
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        torch.save(opt, pth_file_old_version_path)

        model = torch.load(pth_file_old_version_path, map_location=torch.device("cpu"))
        torch.save(
            replace_keys_in_dict(
                replace_keys_in_dict(
                    model, ".parametrizations.weight.original1", ".weight_v"
                ),
                ".parametrizations.weight.original0",
                ".weight_g",
            ),
            pth_file_old_version_path,
        )
        os.remove(pth_file_old_version_path)
        os.rename(pth_file_old_version_path, pth_file)
    except Exception as error:
        print(error)
