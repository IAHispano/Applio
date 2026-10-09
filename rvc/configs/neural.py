"""Serialized V3 contracts shared by caches, checkpoints and inference.

MelConfig fixes sample/frame/log-mel semantics. FeatureConfig identifies the
frozen encoder, pitch extractor and observation profile. AcousticConfig and
VocoderConfig specify trainable networks. Fingerprints bind semantics, not only
dimensions: encoder layer, mel floor and lookahead changes break compatibility.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

DEFAULT_BATCH_SIZE = 2


def fingerprint(value) -> str:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MelConfig:
    """Physical log-mel boundary between acoustic prediction and waveform synthesis."""

    sample_rate: int = 44100
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 22050.0
    log_floor: float = 1e-5
    basis: str = "slaney"
    padding: str = "same-reflect"
    magnitude_epsilon: float = 1e-9

    def __post_init__(self):
        if not (
            0 < self.hop_length <= self.n_fft
            and self.n_mels > 0
            and self.sample_rate > 0
        ):
            raise ValueError("Invalid sample rate, FFT, hop or mel channels")
        if (self.n_fft - self.hop_length) % 2:
            raise ValueError("FFT minus hop must be even for symmetric same padding")
        if not (
            0 <= self.fmin < self.fmax <= self.sample_rate / 2 and self.log_floor > 0
        ):
            raise ValueError("Invalid mel frequency bounds or logarithm floor")
        if self.basis != "slaney" or self.padding != "same-reflect":
            raise ValueError(
                "Unsupported mel semantics; no implicit conversion is allowed"
            )
        if not math.isfinite(self.magnitude_epsilon) or self.magnitude_epsilon < 0:
            raise ValueError("Magnitude epsilon must be finite and nonnegative")

    def frames(self, samples: int) -> int:
        if samples <= 0:
            raise ValueError("Audio must contain samples")
        return math.ceil(samples / self.hop_length)

    def timestamps(self, frames: int):
        return [(i + 0.5) * self.hop_length / self.sample_rate for i in range(frames)]


@dataclass(frozen=True)
class FeatureConfig:
    """Identity and observation timing of the frozen content/pitch frontend."""

    encoder_id: str
    encoder_hash: str
    layer: int = 12
    content_dim: int = 768
    pitch_method: str = "swift"
    pitch_id: str = ""
    pitch_threshold: float = 0.5
    profile: str = "offline"
    context_seconds: float = 1.2
    lookahead_seconds: float = 0.2
    packet_seconds: float = 0.25

    def __post_init__(self):
        if (
            not self.encoder_id
            or not self.encoder_hash
            or self.content_dim < 1
            or self.layer < 0
        ):
            raise ValueError(
                "Feature identity, content width and layer must be explicit"
            )
        if self.pitch_method not in {"swift", "rmvpe"} or not self.pitch_id:
            raise ValueError("Pitch extractor and its identity must be explicit")
        if not 0 <= self.pitch_threshold <= 1:
            raise ValueError("Pitch threshold must lie in [0, 1]")
        if self.profile not in {"offline", "bounded"}:
            raise ValueError("Feature profile must be offline or bounded")
        if (
            self.context_seconds <= 0
            or self.lookahead_seconds < 0
            or self.packet_seconds <= 0
        ):
            raise ValueError("Invalid frontend context")


@dataclass(frozen=True)
class AcousticConfig:
    """Network widths/depths, causal mode and residual versus direct-mel objective."""

    content_dim: int = 768
    mel_dim: int = 128
    speakers: int = 1
    condition_width: int = 256
    predictor_width: int = 256
    refiner_width: int = 384
    predictor_depth: int = 6
    refiner_depth: int = 8
    kernel_size: int = 7
    expansion: int = 2
    causal: bool = True
    checkpoint_blocks: bool = False
    prediction_centered: bool = True
    pitch_guidance: bool = False
    harmonic_detail: bool = False
    family: str = "residual"
    shallow_start: float = 0.4
    auxiliary_weight: float = 0.2

    def __post_init__(self):
        if self.family not in {"residual", "shallow-flow"}:
            raise ValueError("Unknown acoustic family")
        if (
            not 0 < self.shallow_start < 1
            or not math.isfinite(self.auxiliary_weight)
            or self.auxiliary_weight <= 0
        ):
            raise ValueError("Invalid shallow-flow time or auxiliary weight")
        if self.family == "shallow-flow" and (
            self.causal or self.harmonic_detail or self.prediction_centered
        ):
            raise ValueError(
                "Shallow-flow uses noncausal direct-mel coordinates without the rejected detail head"
            )
        if any(
            v < 1
            for v in (
                self.content_dim,
                self.mel_dim,
                self.speakers,
                self.condition_width,
                self.predictor_width,
                self.refiner_width,
                self.predictor_depth,
                self.refiner_depth,
                self.kernel_size,
                self.expansion,
            )
        ):
            raise ValueError("Acoustic dimensions must be positive")
        if self.kernel_size % 2 != 1:
            raise ValueError("Acoustic kernel must be odd")


@dataclass(frozen=True)
class VocoderConfig:
    """Full-rate waveform contract and lower-rate substream synthesis topology."""

    sample_rate: int = 44100
    hop_length: int = 512
    mel_dim: int = 128
    streams: int = 4
    n_fft: int = 512
    channels: int = 64
    depth: int = 8
    kernel_size: int = 7
    filter_taps: int = 63
    causal: bool = True
    checkpoint_blocks: bool = False
    backend: str = "spectral"
    frequency_kernel: int = 13
    expansion: int = 3

    def __post_init__(self):
        if self.backend not in {"spectral", "wavehax"}:
            raise ValueError("Unknown trainable vocoder backend")
        if self.backend == "wavehax" and (self.streams != 1 or self.causal):
            raise ValueError("Full-band Wavehax requires streams=1 and causal=False")
        if (
            self.frequency_kernel < 1
            or self.frequency_kernel % 2 != 1
            or self.expansion < 1
        ):
            raise ValueError(
                "Frequency kernel must be positive/odd and expansion positive"
            )
        if any(
            v < 1
            for v in (
                self.sample_rate,
                self.hop_length,
                self.mel_dim,
                self.streams,
                self.n_fft,
                self.channels,
                self.depth,
                self.kernel_size,
                self.filter_taps,
            )
        ):
            raise ValueError("Vocoder dimensions must be positive")
        if self.hop_length % self.streams or self.n_fft < 2 * (
            self.hop_length // self.streams
        ):
            raise ValueError("Substream hop must divide waveform hop and fit FFT")
        if (self.n_fft - self.hop_length // self.streams) % 2:
            raise ValueError("Substream FFT requires symmetric same padding")
        if self.kernel_size % 2 != 1 or self.filter_taps % 2 != 1:
            raise ValueError("Vocoder kernels and filter taps must be odd")


def require_contract(actual: dict, expected: dict, label: str):
    # Earlier packages omit the magnitude epsilon and use the original 1e-9.
    # Canonicalize mel defaults so adding explicit semantics does not invalidate
    # old compatible packages; epsilon=0 remains a distinct imported contract.
    mel_keys = {"sample_rate", "hop_length", "n_fft", "n_mels", "basis", "padding"}
    if mel_keys <= actual.keys() and mel_keys <= expected.keys():
        actual, expected = asdict(MelConfig(**actual)), asdict(MelConfig(**expected))
        # JSON may serialize an equivalent physical bound as 40 or 40.0.
        # Cache fingerprints remain strict; compatibility compares canonical
        # floating-point spectral values rather than their JSON spelling.
        for contract in (actual, expected):
            for key in ("fmin", "fmax", "log_floor", "magnitude_epsilon"):
                contract[key] = float(contract[key])
    if fingerprint(actual) != fingerprint(expected):
        different = [
            k
            for k in actual.keys() | expected.keys()
            if actual.get(k) != expected.get(k)
        ]
        raise ValueError(f"Incompatible {label}: {', '.join(sorted(different))}")
