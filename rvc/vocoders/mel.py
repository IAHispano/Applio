import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F


class LogMel(nn.Module):
    """Log mel spectrogram at one frame per ``hop_length`` samples.

    The same mel as SingingVocoders' ``PitchAdjustableMelSpectrogram``.
    """

    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        basis = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        self.register_buffer("basis", torch.from_numpy(basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    @classmethod
    def from_config(cls, data: dict) -> "LogMel":
        return cls(
            data["sample_rate"],
            data["n_fft"],
            data["win_length"],
            data["hop_length"],
            data["n_mels"],
            data["mel_fmin"],
            data["mel_fmax"],
        )

    def forward(self, audio: torch.Tensor, key_shift: float = 0.0, hop_length=None) -> torch.Tensor:
        """``audio`` [batch, samples] -> [batch, n_mels, samples // hop].

        ``key_shift`` semitones scale every frequency, pitch and formants, by
        ``2 ** (key_shift / 12)``: the STFT is taken that much longer and read
        with the original bins, as SingingVocoders' ``key_shift`` does. A
        ``hop_length`` other than the configured one stretches time instead.
        """
        hop_length = int(hop_length or self.hop_length)
        factor = 2.0 ** (key_shift / 12.0)
        n_fft = int(round(self.n_fft * factor))
        win_length = int(round(self.win_length * factor))
        window = self.window
        if win_length != self.win_length:
            window = torch.hann_window(win_length, device=audio.device)
        pad = win_length - hop_length
        # Reflection needs more samples than it pads: a shorter clip gets
        # trailing silence, and its extra frames are the caller's to drop.
        short = (pad + 1) // 2 + 1 - audio.shape[-1]
        if short > 0:
            audio = F.pad(audio, (0, short))
        audio = F.pad(
            audio.float().unsqueeze(1), (pad // 2, (pad + 1) // 2), mode="reflect"
        ).squeeze(1)
        spec = torch.stft(
            audio,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        if n_fft != self.n_fft:
            bins = self.n_fft // 2 + 1
            spec = F.pad(spec, (0, 0, 0, max(0, bins - spec.shape[1])))[:, :bins]
            spec = spec * (self.win_length / win_length)
        # Out of autocast: in BF16 the log mel would come in steps of ~0.03.
        with torch.autocast(spec.device.type, enabled=False):
            return torch.log(torch.clamp(self.basis @ spec.float(), min=1e-5))
