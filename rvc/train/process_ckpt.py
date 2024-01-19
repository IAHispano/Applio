import os
import torch
from collections import OrderedDict

logs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")


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


def save_final(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        pth_file = f"{name}_{epoch}e.pth"
        pth_file_path = os.path.join("logs", pth_file)
        pth_file_old_version_path = os.path.join("logs", f"{pth_file}_old_version.pth")

        opt = OrderedDict(
            weight={
                key: value.half() for key, value in ckpt.items() if "enc_q" not in key
            }
        )
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"], opt["sr"], opt["f0"], opt["version"] = epoch, sr, if_f0, version
        torch.save(opt, pth_file_path)

        model = torch.load(pth_file_path, map_location=torch.device("cpu"))
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
        os.remove(pth_file_path)
        os.rename(pth_file_old_version_path, pth_file_path)

        return "Success!"
    except Exception as error:
        print(error)


def extract_small_model(path, name, sr, if_f0, info, version):
    try:
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        opt = OrderedDict(
            weight={
                key: value.half() for key, value in ckpt.items() if "enc_q" not in key
            }
        )
        opt["config"] = {
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
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ],
            "48000": {
                "v1": [
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
                ],
                "v2": [
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
                ],
            },
            "32000": {
                "v1": [
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
                ],
                "v2": [
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
                ],
            },
        }
        opt["config"] = (
            opt["config"][sr]
            if isinstance(opt["config"][sr], list)
            else opt["config"][sr][version]
        )
        if info == "":
            info = "Extracted model."
        opt["info"], opt["version"], opt["sr"], opt["f0"] = (
            info,
            version,
            sr,
            int(if_f0),
        )
        torch.save(opt, f"logs/weights/{name}.pth")
        return "Success."
    except Exception as error:
        print(error)


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        torch.save(ckpt, f"logs/weights/{name}")
        return "Success."
    except Exception as error:
        print(error)
