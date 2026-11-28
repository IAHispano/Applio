"""HSVC (Harmonic-Spectral Voice Conversion) — Next-Generation 48 kHz Voice Conversion Architecture for Applio."""

from backend.architectures.hsvc.discriminators import (
    DualDomainDiscriminator,
    HardenedSpectralSpeakerCritic,
    UtteranceSpeakerAdversary,
)
from backend.architectures.hsvc.losses import HSVCLoss, compute_10ms_log_rms
from backend.architectures.hsvc.model import HSVCConfig, HSVCGenerator, is_hsvc_checkpoint
from backend.architectures.hsvc.source import BandlimitedSourceGenerator, bandlimited_pulse
from backend.architectures.hsvc.spectral import (
    FreqCoordSpectralBlock,
    NeuralTimbreBank,
    StreamingConvNeXtFiLMBlock,
    build_erb_interpolation_matrix,
)
from backend.architectures.hsvc.streaming import HSVCStreamer

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
