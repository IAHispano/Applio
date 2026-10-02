import os

import torch

from rvc.lib.algorithm.generators.pcph_bigvgan import PCPHBigVGANGenerator
from rvc.lib.tools.pretrained_selector import rectified_flow_selector

# Mel settings a vocoder must share with the flow to render its output.
MEL_KEYS = (
    "sample_rate",
    "hop_length",
    "n_fft",
    "win_length",
    "n_mels",
    "mel_fmin",
    "mel_fmax",
    "mel_mean",
    "mel_std",
)
# Arguments of PCPHBigVGANGenerator read from the `vocoder.model` config.
PCPH_BIGVGAN_KEYS = (
    "upsample_rates",
    "upsample_initial_channel",
    "resblock_kernel_sizes",
    "resblock_dilation_sizes",
    "resblock",
    "antialias",
    "filter_width",
    "rolloff",
    "filter_beta",
    "source_noise_std",
    "source_branch",
    "source_noise_eq",
    "noise_branch_bands",
    "output_gain",
    "stage_channels",
    "prenet_blocks",
    "deep_source_stages",
)


def find_vocoder(reference):
    """
    Find the vocoder a flow model was trained with. A model trained on another
    machine names a path that may not exist, so its file name is also looked
    up in the Rectified Flow pretraineds folder.

    Args:
        reference (str): Path to the vocoder as the flow model recorded it.
    """
    if reference and os.path.isfile(reference):
        return reference
    _, default_vocoder = rectified_flow_selector("")
    if reference:
        name = os.path.basename(reference.replace("\\", "/"))
        path = os.path.join("rvc", "models", "pretraineds", "rectified-flow", name)
        if os.path.isfile(path):
            return path
    return default_vocoder


def load_vocoder(path, data):
    """
    Load a PCPH-BigVGAN export as a module that takes the flow's normalised
    mel and f0.

    Args:
        path (str): Path to the vocoder.
        data (dict): The `data` section of the rectified flow config.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "rectified_vocoder" or checkpoint.get("architecture"):
        raise ValueError(f"{path} is not a PCPH-BigVGAN vocoder.")
    vocoder_data = checkpoint["config"]["data"]
    model_config = checkpoint["config"]["vocoder"]["model"]
    if model_config.get("source_type", "sine") != "pcph":
        raise ValueError(f"{path} does not have a PCPH source.")

    for key in MEL_KEYS:
        if key in vocoder_data and float(vocoder_data[key]) != float(data[key]):
            raise ValueError(
                f"{path} renders another mel than the flow's: {key} ({vocoder_data[key]} vs {data[key]})."
            )

    model = PCPHBigVGANGenerator(
        sample_rate=vocoder_data["sample_rate"],
        num_mels=vocoder_data["n_mels"],
        **{k: model_config[k] for k in PCPH_BIGVGAN_KEYS if k in model_config},
    )
    model.load_state_dict(checkpoint["model"])
    model.remove_weight_norm()
    return model.eval()
