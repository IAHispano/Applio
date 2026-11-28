"""Dual-domain discriminators and SpecAugment-hardened speaker disentanglement critics for HSVC."""

from typing import Optional, Tuple

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils.parametrizations import weight_norm


class _GradReverse(Function):
    """Autograd Gradient Reversal Layer (GRL) for adversarial speaker invariance."""

    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply identity forward and negated-scaled backward gradient reversal."""
    return _GradReverse.apply(x, lambd)


class CompactPeriodDisc(nn.Module):
    """
    Lightweight Multi-Period Discriminator sub-network operating on 2D reshaped waveforms.

    Args:
        period (int): Prime period used to fold the 1D waveform into 2D periodic columns.
    """

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chs = [1, 16, 64, 256, 512, 512]
        strides = [3, 3, 3, 3, 1]
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(chs[i], chs[i + 1], (5, 1), (strides[i], 1), padding=(2, 0)))
                for i in range(len(strides))
            ]
        )
        self.post = weight_norm(nn.Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []
        B, C, T = x.shape
        if T % self.period != 0:
            x = F.pad(x, (0, self.period - (T % self.period)), mode="reflect")
        x = x.view(B, C, -1, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class HannSTFTDisc(nn.Module):
    """
    Multi-Resolution STFT Discriminator using a Hann window on linear + log magnitudes.

    Args:
        n_fft (int): STFT FFT size.
        hop (int): STFT hop size in samples.
        win (int): Hann window length in samples.
    """

    def __init__(self, n_fft: int, hop: int, win: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, 32, (3, 9), stride=(1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(2, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 64, (3, 9), stride=(2, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(64, 64, (3, 9), stride=(2, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))),
            ]
        )
        self.post = weight_norm(nn.Conv2d(64, 1, (3, 3), 1, padding=(1, 1)))

    def forward(self, x: torch.Tensor):
        fmap = []
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            stft = torch.stft(
                x.float().squeeze(1),
                self.n_fft,
                self.hop,
                self.win,
                self.window,
                center=True,
                return_complex=True,
            )
            mag = stft.abs()
            log_mag = torch.log(mag.clamp_min(1e-5))
            h = torch.stack([mag, log_mag], dim=1).transpose(2, 3).to(x.dtype)
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            fmap.append(h)
        h = self.post(h)
        fmap.append(h)
        return torch.flatten(h, 1, -1), fmap


class DualDomainDiscriminator(nn.Module):
    """
    Combined waveform-period (MPD: 2, 3, 5, 7, 11) and 3-scale Hann STFT discriminator.

    Evaluates real and generated 48 kHz audio in both time-periodic and time-frequency
    domains, returning logits and intermediate feature maps for hinge/LSGAN + FM losses.
    """

    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList(
            [CompactPeriodDisc(p) for p in [2, 3, 5, 7, 11]]
            + [
                HannSTFTDisc(512, 128, 512),
                HannSTFTDisc(1024, 256, 1024),
                HannSTFTDisc(2048, 512, 2048),
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        """
        Discriminate real waveforms ``y`` against generated waveforms ``y_hat``.

        Args:
            y (torch.Tensor): Real 48 kHz waveforms, shape [B, 1, Samples].
            y_hat (torch.Tensor): Generated 48 kHz waveforms, shape [B, 1, Samples].

        Returns:
            Tuple[list, list, list, list]: (real_logits, gen_logits, real_fmaps, gen_fmaps).
        """
        y_rs, y_gs, f_rs, f_gs = [], [], [], []
        for d in self.discs:
            yr, fr = d(y)
            yg, fg = d(y_hat)
            y_rs.append(yr)
            y_gs.append(yg)
            f_rs.append(fr)
            f_gs.append(fg)
        return y_rs, y_gs, f_rs, f_gs


class UtteranceSpeakerAdversary(nn.Module):
    """
    Utterance-level GRL speaker classifier applied to the content bottleneck during pretraining.

    Pools masked mean and standard deviation across active speech frames so gradient reversal
    removes global speaker bias while leaving local phonetic transitions untouched.

    Args:
        in_dim (int): Bottleneck channel dimension.
        num_speakers (int): Number of training speaker classes.
        hidden (int): Hidden MLP width.
    """

    def __init__(self, in_dim: int, num_speakers: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, num_speakers),
        )

    def forward(self, z_content: torch.Tensor, active_mask: torch.Tensor, lambd: float = 0.14) -> torch.Tensor:
        z_rev = grad_reverse(z_content, lambd)
        w = active_mask.unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        mean = (z_rev * w).sum(dim=1) / denom
        var = (((z_rev - mean.unsqueeze(1)) ** 2) * w).sum(dim=1) / denom
        std = torch.sqrt(var + 1e-5)
        return self.net(torch.cat([mean, std], dim=-1))


class HardenedSpectralSpeakerCritic(nn.Module):
    """
    SpecAugment-hardened spectral speaker critic with EMA cosine speaker centroids.

    Trained on real audio during multi-speaker pretraining to guide cross-speaker swap
    timbre transfer without allowing narrowband spectral shortcuts.

    Args:
        num_speakers (int): Number of speaker classes.
        n_mels (int): Number of Mel filterbank channels (40 Hz to 16 kHz).
        sr (int): Audio sampling rate in Hz.
    """

    def __init__(self, num_speakers: int, n_mels: int = 80, sr: int = 48000):
        super().__init__()
        self.n_fft = 1024
        self.hop = 480
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        fb = torch.from_numpy(
            librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=n_mels, fmin=40.0, fmax=16000.0)
        ).float()
        self.register_buffer("mel_fb", fb, persistent=False)
        self.register_buffer("centroids", torch.zeros(num_speakers, 512), persistent=False)
        self.register_buffer("centroid_init", torch.zeros(num_speakers, dtype=torch.bool), persistent=False)

        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 192, 5, padding=2),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Conv1d(192, 192, 3, dilation=2, padding=2),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Conv1d(192, 256, 3, dilation=3, padding=3),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, 256, 1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_speakers),
        )

    def extract_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute mean-normalized 80-bin log-Mel spectrogram from [B, 1, Samples] audio."""
        with torch.amp.autocast(device_type=wav.device.type, enabled=False):
            w = wav.float().squeeze(1)
            stft = torch.stft(w, self.n_fft, self.hop, self.n_fft, self.window, center=True, return_complex=True)
            mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-7)
            mel = torch.log((self.mel_fb @ mag).clamp_min(1e-5))
            mel = mel - mel.mean(dim=(1, 2), keepdim=True)
        return mel

    def _freq_mask(self, mel: torch.Tensor, max_width: int = 14, num_masks: int = 2) -> torch.Tensor:
        _, M, _ = mel.shape
        out = mel.clone()
        for _ in range(num_masks):
            w = int(torch.randint(4, max_width + 1, (1,)).item())
            f0 = int(torch.randint(0, max(1, M - w), (1,)).item())
            out[:, f0 : f0 + w, :] = 0.0
        return out

    @torch.no_grad()
    def update_centroids(self, emb: torch.Tensor, sid: torch.Tensor, momentum: float = 0.95):
        """Update per-speaker normalized EMA centroids from clean real-audio embeddings."""
        for i in range(sid.shape[0]):
            s = int(sid[i].item())
            if not self.centroid_init[s]:
                self.centroids[s] = emb[i].detach()
                self.centroid_init[s] = True
            else:
                updated = momentum * self.centroids[s] + (1.0 - momentum) * emb[i].detach()
                self.centroids[s] = F.normalize(updated, dim=0)

    def forward(
        self,
        wav: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        apply_specaug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (speaker_logits [B, num_speakers], l2_normalized_embedding [B, 512])."""
        mel = self.extract_mel(wav)
        if apply_specaug:
            mel = self._freq_mask(mel)
        h = self.net(mel)
        if active_mask is not None:
            m = F.interpolate(active_mask.unsqueeze(1).float(), size=h.shape[-1], mode="nearest")
            denom = m.sum(dim=-1).clamp_min(1.0)
            mean = (h * m).sum(dim=-1) / denom
            var = (((h - mean.unsqueeze(-1)) ** 2) * m).sum(dim=-1) / denom
            std = torch.sqrt(var + 1e-5)
        else:
            mean = h.mean(dim=-1)
            std = h.std(dim=-1, unbiased=False)
        emb = torch.cat([mean, std], dim=-1)
        return self.head(emb), F.normalize(emb, dim=-1)
