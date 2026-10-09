"""The vocoders trained here, each by its own recipe (``configs/<name>.json``):
building them, loading a starting point and writing the inference export."""

import os

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from rvc.vocoders.models import nsf_bigvgan, nsf_hifigan
from rvc.vocoders.paths import CONFIG_DIR

BIGVGAN = "nsf-bigvgan"
HIFIGAN = "nsf-hifigan"
ARCHITECTURES = (BIGVGAN, HIFIGAN)
#: Training architecture -> ``architecture`` of its export.
EXPORT_ARCHITECTURE = {BIGVGAN: nsf_bigvgan.ARCHITECTURE, HIFIGAN: nsf_hifigan.ARCHITECTURE}
#: Mel settings an export carries: whatever renders it has to use the same.
MEL_KEYS = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels", "mel_fmin", "mel_fmax")

_PARAMETRIZED = ".parametrizations.weight."


def shipped_config(architecture: str) -> str:
    return os.path.join(CONFIG_DIR, f"{architecture}.json")


def generator_hparams(architecture: str, config: dict) -> dict:
    """The generator's constructor arguments, from the recipe and the mel."""
    data, model = config["data"], config["vocoder"]["model"]
    hop = int(np.prod(model["upsample_rates"]))
    if hop != int(data["hop_length"]):
        raise ValueError(f"The {architecture} recipe upsamples by {hop}, but the mel hop is {data['hop_length']}.")
    return dict(sample_rate=int(data["sample_rate"]), num_mels=int(data["n_mels"]), **model)


def build_generator(architecture: str, hparams: dict) -> nn.Module:
    """The generator as its recipe trains it, under weight norm; both take the
    raw log mel and f0."""
    if architecture == BIGVGAN:
        generator = nsf_bigvgan.NSFBigVGAN(**hparams)
        blocks = [conv for block in generator.resblocks for conv in (*block.convs1, *block.convs2)]
        normed = [generator.conv_pre, *generator.ups, *blocks]
    else:
        generator = nsf_hifigan.NSFHiFiGAN(**hparams)
        blocks = [
            conv for block in generator.resblocks
            for name in ("convs1", "convs2", "convs") for conv in getattr(block, name, ())
        ]
        normed = [generator.conv_pre, generator.conv_post, *generator.ups, *blocks]
        # HiFi-GAN's normal init runs after weight norm there, where it only
        # reaches the biases; the source conv has no weight norm and takes it whole.
        for module in (generator.conv_post, *generator.ups, *blocks):
            module.bias.data.normal_(0.0, 0.01)
        if generator.mini_nsf:
            generator.source_conv.weight.data.normal_(0.0, 0.01)
            generator.source_conv.bias.data.normal_(0.0, 0.01)
    for module in normed:
        weight_norm(module)
    return generator


def build_discriminator(config: dict):
    """The v3 discriminator, as the recipe's ``discriminator`` section sets it."""
    from rvc.vocoders.setup import get_d_model

    return get_d_model(config["vocoder"]["discriminator"], config["data"]["sample_rate"])


def _row_norm(weight):
    return weight.flatten(1).norm(dim=1).view(-1, *([1] * (weight.dim() - 1)))


def training_state(state: dict, reference: dict) -> dict:
    """``state`` under the names of ``reference``, a training generator's
    state: a plain ``weight`` it holds under weight norm is split, and the
    older weight norm's ``weight_g``/``weight_v``, which SingingVocoders'
    checkpoints carry, become the parametrization's."""
    renamed = {}
    for key, value in state.items():
        base, _, leaf = key.rpartition(".")
        prefix = base + _PARAMETRIZED
        if leaf == "weight" and prefix + "original1" in reference:
            renamed[prefix + "original0"] = _row_norm(value)
            renamed[prefix + "original1"] = value
            continue
        target = {"weight_g": "original0", "weight_v": "original1"}.get(leaf)
        renamed[prefix + target if target and prefix + target in reference else key] = value
    return renamed


def pretrained_generator(path: str) -> dict:
    """Generator weights of a starting point: one of this trainer's
    checkpoints or exports, or a SingingVocoders checkpoint or export."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict")
    if isinstance(state, dict):
        return {k[len("generator."):]: v for k, v in state.items() if k.startswith("generator.")}
    if isinstance(checkpoint.get("generator"), dict):
        return checkpoint["generator"]
    return checkpoint["model"]


def export_state(state: dict) -> dict:
    """A generator's training state with weight norm folded, as inference loads it."""
    folded = {}
    for key, value in state.items():
        if key.endswith(_PARAMETRIZED + "original0"):
            continue
        if key.endswith(_PARAMETRIZED + "original1"):
            base = key[: -len(_PARAMETRIZED + "original1")]
            value = value * (state[base + _PARAMETRIZED + "original0"] / _row_norm(value))
            key = base + ".weight"
        folded[key] = value.detach().float().cpu().contiguous()
    return folded


def export_payload(architecture: str, config: dict, hparams: dict, state: dict) -> dict:
    """What the inference export holds besides its position in the run: the
    generator's own settings, its mel and its weights."""
    return {
        "kind": "rectified_vocoder",
        "architecture": EXPORT_ARCHITECTURE[architecture],
        "recipe": architecture,
        "config": {
            "data": {key: config["data"][key] for key in MEL_KEYS},
            "vocoder": {"model": hparams},
        },
        "model": export_state(state),
    }
