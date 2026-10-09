"""Frame-local physical pitch features for the acoustic predictor.

A scalar log-F0 requires the network to learn harmonic frequency placement from
scratch and extrapolate that mapping beyond a speaker's training pitch range.
This feature supplies a fixed mel-band harmonic pattern. Its learned projection
starts at zero so upgrading weights initially preserves the original predictor.
It conditions the neural model; it never edits predicted mels or generated PCM.
"""

import torch
from torch import nn

from rvc.lib.algorithm.acoustic.spectral import MelExtractor


class HarmonicPitchFeatures(nn.Module):
    """A Hann-width harmonic magnitude template aligned to the package's mel.

    The template is independent of target timbre and source waveform phase.
    Sixty-four inverse-amplitude harmonics are deposited into FFT bins, with
    smooth fractional-bin weights and explicit Nyquist/voicing masks. Each frame
    is centered across mel bands; observed energy remains a separate control.
    All computation is frame-local, so retained-state streaming needs no halo.
    """

    def __init__(self, mel):
        super().__init__()
        self.sample_rate = mel.sample_rate
        self.n_fft = mel.n_fft
        self.log_floor = mel.log_floor
        self.register_buffer("basis", MelExtractor(mel).basis.clone())
        self.register_buffer("harmonics", torch.arange(1, 65, dtype=torch.float32))
        self.register_buffer("offsets", torch.arange(-2, 3, dtype=torch.float32))

    def forward(self, f0, voiced):
        if f0.ndim != 2 or f0.shape != voiced.shape:
            raise ValueError("Pitch features require matching [batch, frames] controls")
        with torch.autocast(device_type=f0.device.type, enabled=False):
            frequencies = f0.float()[:, None] * self.harmonics[None, :, None]
            bins = frequencies * self.n_fft / self.sample_rate
            locations = bins.floor()[:, :, None] + self.offsets[None, None, :, None]
            nyquist = self.n_fft // 2
            valid = (frequencies > 0) & (frequencies < self.sample_rate / 2)
            valid = valid[:, :, None] & (locations >= 0) & (locations <= nyquist)
            weights = torch.exp(-0.5 * ((locations - bins[:, :, None]) / 0.65).square())
            weights = weights * valid * voiced.float()[:, None, None]
            weights = weights / self.harmonics[None, :, None, None]
            spectrum = f0.new_zeros(
                (f0.shape[0], nyquist + 1, f0.shape[1]), dtype=torch.float32
            )
            spectrum.scatter_add_(
                1,
                locations.long().clamp(0, nyquist).flatten(1, 2),
                weights.flatten(1, 2),
            )
            mel = (self.basis.float() @ spectrum).clamp_min(self.log_floor).log()
            centered = mel - mel.mean(dim=1, keepdim=True)
            return centered.clamp(-8, 8) * voiced.float()[:, None] / 4
