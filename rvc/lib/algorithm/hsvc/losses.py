"""Phase-invariant multi-resolution spectral, mel, delta, dynamics, and silence losses for HSVC."""

from typing import Dict, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_10ms_log_rms(wav: torch.Tensor, hop: int = 480) -> torch.Tensor:
    """
    Compute frame-level 10 ms natural-log RMS loudness for dynamics matching.

    Args:
        wav (torch.Tensor): Audio tensor of shape [B, 1, Samples].
        hop (int): Frame hop length in samples (480 = 10 ms at 48 kHz).

    Returns:
        torch.Tensor: Per-frame log-RMS envelope of shape [B, Samples // hop].
    """
    B, _, S = wav.shape
    n_f = S // hop
    frames = wav[:, 0, : n_f * hop].view(B, n_f, hop)
    rms = torch.sqrt(frames.float().pow(2).mean(dim=-1) + 1e-10)
    return torch.log(rms.clamp_min(1e-5))


class HSVCLoss(nn.Module):
    """
    Composite reconstruction objective for 48 kHz HSVC training.

    Combines 4-scale STFT spectral convergence and log-magnitude L1, high-frequency-weighted
    (>= 4 kHz boosted 1.35x) multi-resolution log-Mel L1, time/frequency Mel derivative deltas,
    10 ms RMS loudness matching, and an explicit time-domain silence-floor penalty.

    Args:
        sr (int): Audio sampling rate in Hz (default: 48000).
        hop_in (int): Frontend frame hop in samples (default: 480).
    """

    def __init__(self, sr: int = 48000, hop_in: int = 480):
        super().__init__()
        self.sr = sr
        self.hop_in = hop_in
        self.stft_cfgs = [
            (512, 128, 512, 48),
            (1024, 256, 1024, 80),
            (2048, 480, 2048, 128),
            (4096, 960, 4096, 128),
        ]
        for i, (n_fft, hop, win, n_mels) in enumerate(self.stft_cfgs):
            self.register_buffer(f"win_{i}", torch.hann_window(win), persistent=False)
            fb = torch.from_numpy(
                librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2)
            ).float()
            self.register_buffer(f"mel_{i}", fb, persistent=False)
            mel_f = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2)
            w_hf = torch.from_numpy(np.where(mel_f >= 4000.0, 1.35, 1.0).astype(np.float32)).view(1, n_mels, 1)
            self.register_buffer(f"mel_w_{i}", w_hf, persistent=False)

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, energy_db: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all spectral, dynamics, and silence loss components.

        Args:
            y (torch.Tensor): Reference 48 kHz audio, shape [B, 1, Samples].
            y_hat (torch.Tensor): Synthesized 48 kHz audio, shape [B, 1, Samples].
            energy_db (Optional[torch.Tensor]): 100 Hz input loudness in dBFS, shape [B, T],
                used to mask frames below -62 dBFS for explicit silence suppression.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys ``mel``, ``stft_mag``, ``stft_sc``,
                ``delta``, ``rms``, and ``silence``.
        """
        y_f = y.float().squeeze(1)
        yh_f = y_hat.float().squeeze(1)

        loss_mel = 0.0
        loss_stft_mag = 0.0
        loss_stft_sc = 0.0
        loss_delta = 0.0

        for i, (n_fft, hop, win, _) in enumerate(self.stft_cfgs):
            w = getattr(self, f"win_{i}")
            fb = getattr(self, f"mel_{i}")
            mw = getattr(self, f"mel_w_{i}")
            sr_c = torch.stft(y_f, n_fft, hop, win, w, center=True, return_complex=True)
            sg_c = torch.stft(yh_f, n_fft, hop, win, w, center=True, return_complex=True)

            mr = torch.sqrt(sr_c.real.pow(2) + sr_c.imag.pow(2) + 1e-7)
            mg = torch.sqrt(sg_c.real.pow(2) + sg_c.imag.pow(2) + 1e-7)

            loss_stft_sc = loss_stft_sc + torch.norm(mr - mg, p="fro") / (torch.norm(mr, p="fro") + 1e-6)
            loss_stft_mag = loss_stft_mag + F.l1_loss(
                torch.log(mr.clamp_min(1e-5)), torch.log(mg.clamp_min(1e-5))
            )

            mel_r = torch.log((fb @ mr).clamp_min(1e-5))
            mel_g = torch.log((fb @ mg).clamp_min(1e-5))
            loss_mel = loss_mel + (torch.abs(mel_r - mel_g) * mw).mean()

            if n_fft == 2048:
                dt_r = mel_r[..., 1:] - mel_r[..., :-1]
                dt_g = mel_g[..., 1:] - mel_g[..., :-1]
                df_r = mel_r[:, 1:, :] - mel_r[:, :-1, :]
                df_g = mel_g[:, 1:, :] - mel_g[:, :-1, :]
                loss_delta = F.l1_loss(dt_r, dt_g) + F.l1_loss(df_r, df_g)

        n = float(len(self.stft_cfgs))
        loss_mel = loss_mel / n
        loss_stft_mag = loss_stft_mag / n
        loss_stft_sc = loss_stft_sc / n
        loss_rms = F.l1_loss(compute_10ms_log_rms(y_hat, self.hop_in), compute_10ms_log_rms(y, self.hop_in))

        loss_sil = y_f.new_tensor(0.0)
        if energy_db is not None:
            sil_mask_100 = (energy_db < -62.0).float()
            if sil_mask_100.sum() > 0:
                sil_mask_sr = F.interpolate(
                    sil_mask_100.unsqueeze(1), size=y_f.shape[-1], mode="nearest"
                ).squeeze(1)
                loss_sil = (torch.abs(yh_f - y_f) * sil_mask_sr).sum() / sil_mask_sr.sum().clamp_min(1.0)

        return {
            "mel": loss_mel,
            "stft_mag": loss_stft_mag,
            "stft_sc": loss_stft_sc,
            "delta": loss_delta,
            "rms": loss_rms,
            "silence": loss_sil,
        }
