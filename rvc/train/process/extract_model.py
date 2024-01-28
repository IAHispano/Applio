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


def extract_model(ckpt, sr, if_f0, name, model_dir, epoch, version, hps):
    try:
        print(f"Saved model '{model_dir}' (epoch {epoch})")
        pth_file = f"{name}_{epoch}e.pth"
        pth_file_old_version_path = os.path.join(
            model_dir, f"{pth_file}_old_version.pth"
        )
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
        torch.save(opt, model_dir)

        model = torch.load(model_dir, map_location=torch.device("cpu"))
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
        os.remove(model_dir)
        os.rename(pth_file_old_version_path, model_dir)

    except Exception as error:
        print(error)
