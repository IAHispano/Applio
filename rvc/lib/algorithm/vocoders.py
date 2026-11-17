import os

import torch
from torch import nn

from rvc.lib.algorithm.generators import nagi
from rvc.lib.tools.pretrained_selector import rectified_flow_selector

# Mel settings a vocoder must share with the flow to render its output. The
# vocoders read the normalised mel, so its normalisation must match too.
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


class FlowVocoder(nn.Module):
    """
    A Nagi vocoder fed what the flow produces: the normalised mel and, from
    a flow with `contrast_band_edges`, the band contrast after its bins. The
    contrast goes to a vocoder that reads the same bands and is dropped
    otherwise.

    Args:
        generator (torch.nn.Module): The Nagi generator.
        data (dict): The `data` section of the rectified flow config.
    """

    def __init__(self, generator, data):
        super().__init__()
        self.generator = generator
        self.n_mels = int(data["n_mels"])
        self.mel_mean = float(data["mel_mean"])
        self.mel_std = float(data["mel_std"])
        contrast = generator.contrast
        self.contrast_bands = contrast.bands if contrast is not None else 0

    def forward(self, mel, f0):
        if mel.shape[1] == self.n_mels:
            return self.generator(mel, f0)
        mel, contrast = mel[:, : self.n_mels], mel[:, self.n_mels :]
        if contrast.shape[1] != self.contrast_bands:
            return self.generator(mel, f0)
        # The contrast rows went through the normalisation of the mel with the rest
        contrast = contrast * self.mel_std + self.mel_mean
        return self.generator(mel, f0, nagi.known_contrast(contrast))


def find_vocoder(reference, sample_rate=44100):
    """
    Find the vocoder a flow model was trained with. A model trained on another
    machine names a path that may not exist, so its file name is also looked
    up in the Rectified Flow pretraineds folder, and the default vocoder of the
    sampling rate is used when it is not there either.

    Args:
        reference (str): Path to the vocoder as the flow model recorded it.
        sample_rate (int, optional): Sampling rate of the flow model. Defaults to 44100.
    """
    if reference and os.path.isfile(reference):
        return reference
    _, default_vocoder = rectified_flow_selector("", sample_rate)
    if reference:
        name = os.path.basename(reference.replace("\\", "/"))
        path = os.path.join("rvc", "models", "pretraineds", "rectified-flow", name)
        if os.path.isfile(path):
            return path
    return default_vocoder


def load_vocoder(path, data):
    """
    Load a Nagi export as a module that takes the flow's output and f0.

    Args:
        path (str): Path to the vocoder.
        data (dict): The `data` section of the rectified flow config.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    is_export = checkpoint.get("kind") == "rectified_vocoder"
    if not is_export or checkpoint.get("architecture") != nagi.ARCHITECTURE:
        raise ValueError(f"{path} is not a Nagi export, the one supported.")
    vocoder_data = checkpoint["config"]["data"]
    for key in MEL_KEYS:
        if key in vocoder_data and float(vocoder_data[key]) != float(data[key]):
            raise ValueError(
                f"{path} renders another mel than the flow's: {key} ({vocoder_data[key]} vs {data[key]})."
            )
    generator = nagi.Nagi(**checkpoint["config"]["vocoder"]["model"])
    generator.load_state_dict(checkpoint["model"])
    return FlowVocoder(generator, data).eval()
