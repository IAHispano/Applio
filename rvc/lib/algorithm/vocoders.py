import os

import torch
from torch import nn

from rvc.lib.algorithm.generators import bigvgan_v2, pupu_vocoder
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
)
# Generator of each vocoder export, by its architecture
GENERATORS = {
    pupu_vocoder.ARCHITECTURE: pupu_vocoder.PupuVocoder,
    bigvgan_v2.ARCHITECTURE: bigvgan_v2.BigVGANv2,
}


class RawMelVocoder(nn.Module):
    """
    A vocoder trained on the raw log mel, fed the flow's normalised one.

    Args:
        generator (torch.nn.Module): The vocoder's generator.
        data (dict): The `data` section of the rectified flow config.
    """

    def __init__(self, generator, data):
        super().__init__()
        self.generator = generator
        self.mel_mean = float(data["mel_mean"])
        self.mel_std = float(data["mel_std"])

    def forward(self, mel, f0):
        return self.generator(mel * self.mel_std + self.mel_mean, f0)


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
    Load a Pupu-Vocoder or BigVGAN v2 export as a module that takes the flow's
    normalised mel and f0.

    Args:
        path (str): Path to the vocoder.
        data (dict): The `data` section of the rectified flow config.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    is_export = checkpoint.get("kind") == "rectified_vocoder"
    if not is_export or checkpoint.get("architecture") not in GENERATORS:
        raise ValueError(
            f"{path} is not a Pupu-Vocoder or BigVGAN v2 export, the ones supported."
        )
    vocoder_data = checkpoint["config"]["data"]
    for key in MEL_KEYS:
        if key in vocoder_data and float(vocoder_data[key]) != float(data[key]):
            raise ValueError(
                f"{path} renders another mel than the flow's: {key} ({vocoder_data[key]} vs {data[key]})."
            )
    generator = GENERATORS[checkpoint["architecture"]](
        **checkpoint["config"]["vocoder"]["model"]
    )
    generator.load_state_dict(checkpoint["model"])
    return RawMelVocoder(generator, data).eval()
