"""Package pinned official vocoders at the shared V3 mel boundary.

Imported vocoders are frozen inference backends, not native training resumes.
Keep upstream architecture and provenance explicit; never reinterpret weights
as the experimental spectral network merely because mel dimensions match.
"""

import json
from pathlib import Path

import torch

from rvc.configs.architectures import BACKEND, FORMAT_VERSION
from rvc.configs.neural import MelConfig
from rvc.lib.algorithm.acoustic.bigvgan import AttrDict, BigVGAN, BigVGANVocoder
from rvc.train.process.checkpoints import atomic_save

BIGVGAN_ID = "nvidia/bigvgan_v2_44khz_128band_512x"
BIGVGAN_REVISION = "95a9d1dcb12906c03edd938d77b9333d6ded7dfb"


def import_bigvgan(destination, checkpoint=None, configuration=None):
    """Import local official weights, or download the pinned compatible release."""
    from huggingface_hub import hf_hub_download
    from rvc.train.extract.features import file_hash

    if bool(checkpoint) != bool(configuration):
        raise ValueError("Provide both the generator checkpoint and its config.json")
    downloaded = not checkpoint
    if downloaded:
        configuration = hf_hub_download(
            BIGVGAN_ID, "config.json", revision=BIGVGAN_REVISION
        )
        checkpoint = hf_hub_download(
            BIGVGAN_ID, "bigvgan_generator.pt", revision=BIGVGAN_REVISION
        )
    destination = Path(destination)
    if destination.resolve() in {
        Path(checkpoint).resolve(),
        Path(configuration).resolve(),
    }:
        raise ValueError("Output must differ from the original pretrained files")
    config = json.loads(Path(configuration).read_text(encoding="utf-8"))
    BigVGANVocoder.validate_config(config)
    generator = BigVGAN(AttrDict(config))
    generator.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["generator"],
        strict=True,
    )
    generator.remove_weight_norm()
    package = dict(
        backend=BACKEND,
        format_version=FORMAT_VERSION,
        kind="vocoder",
        vocoder_backend="bigvgan-v2",
        model_config=config,
        mel=MelConfig().__dict__,
        weights={
            "generator." + key: value for key, value in generator.state_dict().items()
        },
        inference_only=True,
        adapters=None,
        capabilities={"streaming": False},
        provenance=dict(
            model_id=BIGVGAN_ID if downloaded else None,
            revision=BIGVGAN_REVISION if downloaded else None,
            implementation_commit="7d2b454564a6c7d014227f635b7423881f14bdac",
            generator_sha256=file_hash(Path(checkpoint)),
            config_sha256=file_hash(Path(configuration)),
            license="MIT with bundled upstream component notices",
        ),
    )
    atomic_save(destination, package)
    return str(destination)


def import_nsf_hifigan(destination, checkpoint):
    """Package a verified converted OpenVPI mini-NSF export without relabeling mel.

    Restricted loading, exact graph/mel checks and strict weights prevent a
    similarly named checkpoint from silently using release-default semantics.
    The upstream weight license is retained separately from the MIT source.
    """
    from rvc.lib.algorithm.acoustic import nsf_hifigan
    from rvc.lib.algorithm.acoustic.nsf_hifigan import NSFHiFiGANVocoder
    from rvc.train.extract.features import file_hash

    checkpoint, destination = Path(checkpoint), Path(destination)
    if checkpoint.resolve() == destination.resolve():
        raise ValueError("Output must differ from the original pretrained checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("kind") != "rectified_vocoder" or payload.get("architecture") != "openvpi_nsf_hifigan":
        raise ValueError("Expected a converted OpenVPI mini-NSF export")
    expected_mel = dict(sample_rate=44100, hop_length=512, n_fft=2048,
                        win_length=2048, n_mels=128, mel_fmin=40., mel_fmax=16000.)
    if payload["config"]["data"] != expected_mel:
        raise ValueError("NSF-HiFiGAN release mel settings do not match the supported contract")
    config = payload["config"]["vocoder"]["model"]
    model = NSFHiFiGANVocoder(config, f0_policy="continuous")
    weights = {"generator." + key: value for key, value in payload["model"].items()}
    model.load_state_dict(weights, strict=True)
    mel = MelConfig(fmin=40., fmax=16000., magnitude_epsilon=0.)
    package = dict(
        backend=BACKEND, format_version=FORMAT_VERSION, kind="vocoder",
        vocoder_backend="nsf-hifigan", model_config=config, mel=mel.__dict__,
        weights=weights, inference_only=True, adapters=None, f0_policy="continuous",
        capabilities={"streaming": False},
        provenance=dict(
            source=checkpoint.name, generator_sha256=file_hash(checkpoint),
            implementation_sha256=file_hash(Path(nsf_hifigan.__file__)),
            reference_revision="4d0889c4c180c75ad3000cc565864656344f8190",
            source_project="OpenVPI SingingVocoders",
            source_license="MIT; see nsf_hifigan.LICENSE",
            weight_license="CC-BY-NC-SA-4.0 per upstream OpenVPI release documentation",
            weight_license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        ),
    )
    atomic_save(destination, package)
    return str(destination)
