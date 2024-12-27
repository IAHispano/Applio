import os
import sys
import torch
import hashlib
import datetime
from collections import OrderedDict
import json

now_dir = os.getcwd()
sys.path.append(now_dir)


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


def extract_model(
    ckpt,
    sr,
    pitch_guidance,
    name,
    model_dir,
    epoch,
    step,
    version,
    hps,
    overtrain_info,
    vocoder,
):
    try:
        print(f"Saved model '{model_dir}' (epoch {epoch} and step {step})")

        model_dir_path = os.path.dirname(model_dir)
        os.makedirs(model_dir_path, exist_ok=True)

        suffix = "_best_epoch" if "best_epoch" in model_dir else ""
        pth_file = f"{name}_{epoch}e_{step}s{suffix}.pth"
        pth_file_old_version_path = os.path.join(
            model_dir_path, f"{pth_file}_old_version.pth"
        )

        dataset_length, embedder_model, speakers_id = None, None, 1
        if os.path.exists(os.path.join(model_dir_path, "model_info.json")):
            with open(os.path.join(model_dir_path, "model_info.json"), "r") as f:
                data = json.load(f)
                dataset_length = data.get("total_dataset_duration")
                embedder_model = data.get("embedder_model")
                speakers_id = data.get("speakers_id", 1)

        with open(os.path.join(now_dir, "assets", "config.json"), "r") as f:
            data = json.load(f)
            model_author = data.get("model_author")

        opt = OrderedDict(
            weight={
                key: value.half() for key, value in ckpt.items() if "enc_q" not in key
            },
            config=[
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
                hps.data.sample_rate,
            ],
            epoch=epoch,
            step=step,
            sr=sr,
            f0=pitch_guidance,
            version=version,
            creation_date=datetime.datetime.now().isoformat(),
            overtrain_info=overtrain_info,
            dataset_length=dataset_length,
            model_name=name,
            author=model_author,
            embedder_model=embedder_model,
            speakers_id=speakers_id,
            vocoder=vocoder,
        )

        hash_input = f"{name}-{epoch}-{step}-{sr}-{version}-{opt['config']}"
        opt["model_hash"] = hashlib.sha256(hash_input.encode()).hexdigest()

        torch.save(opt, os.path.join(model_dir_path, pth_file))

        model = torch.load(model_dir, map_location=torch.device("cpu"))
        updated_model = replace_keys_in_dict(
            replace_keys_in_dict(
                model, ".parametrizations.weight.original1", ".weight_v"
            ),
            ".parametrizations.weight.original0",
            ".weight_g",
        )
        torch.save(updated_model, pth_file_old_version_path)
        os.remove(model_dir)
        os.rename(pth_file_old_version_path, model_dir)

    except Exception as error:
        print(f"An error occurred extracting the model: {error}")
