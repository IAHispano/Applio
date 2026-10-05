"""The two V3 spectral boundaries have distinct FFT and timing contracts.

MelExtractor defines acoustic targets shared by preparation, training and
package metadata. SameSTFT analyzes vocoder substreams and inverts predicted
complex spectra. Hop, padding, window and log semantics matter even when shapes
match. FFTs/logarithms use FP32 while surrounding networks may use AMP.
"""

import torch
import torch.nn.functional as F
from torch import nn

from rvc.configs.v3 import MelConfig


class MelExtractor(nn.Module):
    """Slaney magnitude mel, natural log and ceil(samples/hop) declared frames."""

    def __init__(self, config: MelConfig):
        super().__init__()
        import librosa

        self.config = config
        basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
            htk=False,
            norm="slaney",
        )
        self.register_buffer("basis", torch.from_numpy(basis))
        self.register_buffer("window", torch.hann_window(config.n_fft))

    def forward(self, audio):
        if audio.ndim == 3:
            audio = audio[:, 0]
        if audio.ndim != 2 or not audio.shape[-1]:
            raise ValueError("Audio must have shape [batch, nonempty samples]")
        c = self.config
        frames = c.frames(audio.shape[-1])
        audio = F.pad(audio.float(), (0, frames * c.hop_length - audio.shape[-1]))
        pad = (c.n_fft - c.hop_length) // 2
        mode = "reflect" if audio.shape[-1] > pad else "replicate"
        audio = F.pad(audio[:, None], (pad, pad), mode=mode)[:, 0]
        # FFT and logarithms are kept in FP32 even under AMP.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            spec = torch.stft(
                audio,
                c.n_fft,
                c.hop_length,
                window=self.window.float(),
                center=False,
                return_complex=True,
            )
            magnitude = (spec.real.square() + spec.imag.square() + 1e-9).sqrt()
            return (self.basis.float() @ magnitude).clamp_min(c.log_floor).log()


class SameSTFT(nn.Module):
    """Explicit same padding and Hann overlap-add with boundary normalization.

    Inverse synthesis divides by the accumulated squared window and removes the
    analysis halo. FFT >= two hops guarantees the supported window overlap.
    """

    def __init__(self, n_fft, hop):
        super().__init__()
        if hop < 1 or n_fft < 2 * hop or (n_fft - hop) % 2:
            raise ValueError(
                "Hann overlap-add requires FFT >= two hops and even symmetric padding"
            )
        self.n_fft, self.hop = n_fft, hop
        self.pad = (n_fft - hop) // 2
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.pad(x.float(), (self.pad, self.pad))
            return torch.stft(
                x,
                self.n_fft,
                self.hop,
                window=self.window.float(),
                center=False,
                return_complex=True,
            )

    def inverse(self, spectrum):
        frames = spectrum.shape[-1]
        size = (frames - 1) * self.hop + self.n_fft
        with torch.autocast(device_type=spectrum.device.type, enabled=False):
            columns = (
                torch.fft.irfft(spectrum, n=self.n_fft, dim=1)
                * self.window.float()[None, :, None]
            )
            output = F.fold(
                columns, (1, size), kernel_size=(1, self.n_fft), stride=(1, self.hop)
            )[:, 0, 0]
            envelope = F.fold(
                self.window.float().square()[None, :, None].expand(1, -1, frames),
                (1, size),
                kernel_size=(1, self.n_fft),
                stride=(1, self.hop),
            )[0, 0, 0]
            sl = slice(self.pad, self.pad + frames * self.hop)
            return output[:, sl] / envelope[sl].clamp_min(1e-8)


def spectral_loss(fake, real, sizes=(256, 512, 1024, 2048)):
    loss = fake.new_zeros((), dtype=torch.float32)
    for n_fft in sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=fake.device)
        with torch.autocast(device_type=fake.device.type, enabled=False):
            length = max(fake.shape[-1], n_fft)
            f = F.pad(fake.float(), (0, length - fake.shape[-1]))
            r = F.pad(real.float(), (0, length - real.shape[-1]))
            a = torch.stft(
                f,
                n_fft,
                hop,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            ).abs()
            b = torch.stft(
                r,
                n_fft,
                hop,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            ).abs()
            loss = (
                loss
                + (a - b).abs().mean()
                + (a.clamp_min(1e-5).log() - b.clamp_min(1e-5).log()).abs().mean()
            )
    return loss / len(sizes)
