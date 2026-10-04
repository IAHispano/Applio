import os

import torch
from torch import nn

from rvc.lib.algorithm.generators.openvpi import (
    ARCHITECTURE,
    NSFHiFiGAN,
    generator_state,
    openvpi_spec,
)
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
# Options a ShiroRVC export of an OpenVPI generator may name beyond the ones
# of SingingVocoders, and the value of each that NSFHiFiGAN renders.
NSF_HIFIGAN_OPTIONS = {"activation": "leaky_relu", "upsampling": "transposed"}


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
    Load a PCPH-BigVGAN export, an OpenVPI NSF-HiFiGAN or PC-NSF-HiFiGAN
    checkpoint (SingingVocoders) or an export holding one, as a module that
    takes the flow's normalised mel and f0.

    Args:
        path (str): Path to the vocoder.
        data (dict): The `data` section of the rectified flow config.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    is_export = checkpoint.get("kind") == "rectified_vocoder"
    architecture = checkpoint.get("architecture") if is_export else ARCHITECTURE
    if architecture not in (None, ARCHITECTURE):
        raise ValueError(f"{path} is a {architecture} vocoder, which is not supported.")
    if architecture is None:
        vocoder_data = checkpoint["config"]["data"]
        model_config = checkpoint["config"]["vocoder"]["model"]
        if model_config.get("source_type", "sine") != "pcph":
            raise ValueError(f"{path} does not have a PCPH source.")
        model = PCPHBigVGANGenerator(
            sample_rate=vocoder_data["sample_rate"],
            num_mels=vocoder_data["n_mels"],
            **{k: model_config[k] for k in PCPH_BIGVGAN_KEYS if k in model_config},
        )
        model.load_state_dict(checkpoint["model"])
        model.remove_weight_norm()
    else:
        if is_export:
            hparams = dict(checkpoint["config"]["vocoder"]["model"])
            vocoder_data = checkpoint["config"]["data"]
            weights = checkpoint["model"]
            for key, value in NSF_HIFIGAN_OPTIONS.items():
                if hparams.pop(key, value) != value:
                    raise ValueError(
                        f"{path} has another {key} than SingingVocoders', which is not supported."
                    )
            for key in ("antialias", "upsample_filters"):
                hparams.pop(key, None)
        else:
            state = generator_state(checkpoint)
            if state is None:
                raise ValueError(f"{path} is not a Rectified Flow vocoder.")
            hparams, vocoder_data, weights = openvpi_spec(path, state)
        generator = NSFHiFiGAN(**hparams)
        generator.load_state_dict(weights)
        model = RawMelVocoder(generator, data)

    for key in MEL_KEYS:
        if key in vocoder_data and float(vocoder_data[key]) != float(data[key]):
            raise ValueError(
                f"{path} renders another mel than the flow's: {key} ({vocoder_data[key]} vs {data[key]})."
            )
    return model.eval()
