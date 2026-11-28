"""HSVC (Harmonic-Spectral Voice Conversion) — Next-Generation 48 kHz Voice Conversion Architecture for Applio."""

from rvc.lib.algorithm.hsvc.discriminators import (
    DualDomainDiscriminator,
    HardenedSpectralSpeakerCritic,
    UtteranceSpeakerAdversary,
)
from rvc.lib.algorithm.hsvc.losses import HSVCLoss, compute_10ms_log_rms
from rvc.lib.algorithm.hsvc.model import HSVCConfig, HSVCGenerator, is_hsvc_checkpoint
from rvc.lib.algorithm.hsvc.source import BandlimitedSourceGenerator, bandlimited_pulse
from rvc.lib.algorithm.hsvc.spectral import (
    FreqCoordSpectralBlock,
    NeuralTimbreBank,
    StreamingConvNeXtFiLMBlock,
    build_erb_interpolation_matrix,
)
from rvc.lib.algorithm.hsvc.streaming import HSVCStreamer

__all__ = [
    "BandlimitedSourceGenerator",
    "DualDomainDiscriminator",
    "FreqCoordSpectralBlock",
    "HardenedSpectralSpeakerCritic",
    "HSVCConfig",
    "HSVCGenerator",
    "HSVCLoss",
    "HSVCStreamer",
    "NeuralTimbreBank",
    "StreamingConvNeXtFiLMBlock",
    "UtteranceSpeakerAdversary",
    "bandlimited_pulse",
    "build_erb_interpolation_matrix",
    "compute_10ms_log_rms",
    "is_hsvc_checkpoint",
]
