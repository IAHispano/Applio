import math
import os

import torch
from torch import nn

from rvc.lib.algorithm.generators import chouwagan
from rvc.lib.tools.pretrained_selector import rectified_flow_selector

# Mel settings a vocoder must share with the flow to render its output. ChouwaGAN
# reads the normalised mel, so its normalisation must match too.
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
# The diffuser's bins turn from here up, faded in over the kHz below
DIFFUSE_HZ = 4000.0
DIFFUSE_SEED = 0


class Diffuser(nn.Module):
    """
    Above `DIFFUSE_HZ`, turns each bin of the output's STFT by a fixed random
    phase of the harmonic of f0 nearest to it, so each harmonic turns whole and
    keeps its phase as it moves. The pulse train has every harmonic in phase,
    which up there sounds like static; scattered harmonic phases spread each
    pulse's top over the cycle and leave the harmonics sharp, where a phase per
    bin blurred them. Only at inference, as a generator trained through it
    learned steady lines.

    Args:
        sample_rate (int): Sampling rate.
        n_fft (int): Size of the STFT.
        hop (int): Hop of the STFT.
        frame_hop (int): Samples per f0 frame.
    """

    def __init__(self, sample_rate, n_fft, hop, frame_hop):
        super().__init__()
        self.n_fft, self.hop, self.frame_hop = int(n_fft), int(hop), int(frame_hop)
        bins = torch.linspace(0, sample_rate / 2, self.n_fft // 2 + 1).view(1, -1, 1)
        harmonics = int(sample_rate / 2 / chouwagan.SOURCE_F0_FLOOR) + 2
        generator = torch.Generator().manual_seed(DIFFUSE_SEED)
        phases = math.pi * (2 * torch.rand(harmonics, generator=generator) - 1)
        ramp = ((bins - DIFFUSE_HZ) / 1000 + 1).clamp(0, 1)
        self.register_buffer("bins", bins, persistent=False)
        self.register_buffer("ramp", ramp, persistent=False)
        self.register_buffer("phases", phases, persistent=False)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def forward(self, audio, f0):
        """
        Args:
            audio (torch.Tensor): Audio of shape (batch, 1, samples).
            f0 (torch.Tensor): Pitch it was rendered from, shape (batch, frames), 0 where unvoiced.
        """
        x, window = audio.squeeze(1).float(), self.window.float()
        spectrum = torch.stft(
            x, self.n_fft, self.hop, window=window, return_complex=True
        )
        frame = torch.arange(spectrum.shape[-1], device=x.device)
        frame = (frame * self.hop // self.frame_hop).clamp(max=f0.shape[-1] - 1)
        f0 = f0.float()[:, frame].unsqueeze(1)
        harmonic = torch.round(self.bins / f0.clamp(min=chouwagan.SOURCE_F0_FLOOR))
        harmonic = harmonic.long().clamp(max=len(self.phases) - 1)
        phase = self.phases[harmonic] * self.ramp * (f0 > 0)
        spectrum = spectrum * torch.polar(torch.ones_like(phase), phase)
        x = torch.istft(
            spectrum, self.n_fft, self.hop, window=window, length=x.shape[-1]
        )
        return x.unsqueeze(1).to(audio.dtype)


class FlowVocoder(nn.Module):
    """
    A ChouwaGAN vocoder fed what the flow produces: the normalised mel and, from
    a flow with `contrast_band_edges`, the band contrast after its bins. The
    contrast goes to a vocoder that reads the same bands and is dropped otherwise.

    Args:
        generator (torch.nn.Module): The ChouwaGAN generator.
        data (dict): The `data` section of the rectified flow config.
        diffuse (bool, optional): Whether the output goes through a `Diffuser`, unless the generator scatters its harmonics' phases itself. Defaults to True.
    """

    def __init__(self, generator, data, diffuse=True):
        super().__init__()
        self.generator = generator
        self.diffuser = None
        if diffuse and generator.source_phase_hz is None:
            self.diffuser = Diffuser(
                generator.sample_rate,
                generator.noise_fft,
                generator.synthesis_hop,
                generator.hop_length,
            )
        self.n_mels = int(data["n_mels"])
        self.mel_mean = float(data["mel_mean"])
        self.mel_std = float(data["mel_std"])
        contrast = generator.contrast
        self.contrast_bands = contrast.bands if contrast is not None else 0

    def forward(self, mel, f0):
        audio = self.render(mel, f0)
        return audio if self.diffuser is None else self.diffuser(audio, f0)

    def render(self, mel, f0):
        if mel.shape[1] == self.n_mels:
            return self.generator(mel, f0)
        mel, contrast = mel[:, : self.n_mels], mel[:, self.n_mels :]
        if contrast.shape[1] != self.contrast_bands:
            return self.generator(mel, f0)
        # The contrast rows went through the normalisation of the mel with the rest
        contrast = contrast * self.mel_std + self.mel_mean
        return self.generator(mel, f0, chouwagan.known_contrast(contrast))


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


def load_vocoder(path, data, diffuse=True):
    """
    Load a ChouwaGAN export as a module that takes the flow's output and f0.

    Args:
        path (str): Path to the vocoder.
        data (dict): The `data` section of the rectified flow config.
        diffuse (bool, optional): Whether the output goes through a `Diffuser`. Defaults to True.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    is_export = checkpoint.get("kind") == "rectified_vocoder"
    if not is_export or checkpoint.get("architecture") != chouwagan.ARCHITECTURE:
        raise ValueError(f"{path} is not a ChouwaGAN export, the one supported.")
    vocoder_data = checkpoint["config"]["data"]
    for key in MEL_KEYS:
        if key in vocoder_data and float(vocoder_data[key]) != float(data[key]):
            raise ValueError(
                f"{path} renders another mel than the flow's: {key} ({vocoder_data[key]} vs {data[key]})."
            )
    generator = chouwagan.ChouwaGAN(**checkpoint["config"]["vocoder"]["model"])
    generator.load_state_dict(checkpoint["model"])
    return FlowVocoder(generator, data, diffuse).eval()
