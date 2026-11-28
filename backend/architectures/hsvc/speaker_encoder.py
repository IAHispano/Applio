"""Zero-Shot / One-Shot Reference Speaker & Timbre Prototype Encoder for HSVC.

Designed as an optional, frozen-decoder-compatible sidecar (~2.4M params, ~4.8 MB FP16)
that maps any 3-30s clean reference clip directly into HSVC's ``NeuralTimbreBank``
latent space:
  - ``global_emb``   : [B, spk_dim]            (global speaker FiLM vector)
  - ``proto_values`` : [B, num_protos, spk_dim] (16 phoneme/register timbre prototypes)

Because it plugs cleanly into a frozen ``HSVCGenerator`` (zero parameter or quality impact
on standard fine-tuned models) and executes only ONCE per reference clip (<5 ms on GPU),
it adds zero latency to inference or live real-time voice conversion.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
import soxr
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HSVCReferenceEncoderConfig:
    """Configuration schema for the HSVC one-shot reference timbre encoder."""

    sample_rate: int = 48000
    n_fft: int = 1024
    hop_length: int = 480
    win_length: int = 960
    n_mels: int = 80
    fmin: float = 40.0
    fmax: float = 12000.0
    channels: int = 256
    spk_dim: int = 256
    num_protos: int = 16
    n_heads: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HSVCReferenceEncoderConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class SEResBlock1D(nn.Module):
    """Multi-scale dilated Conv1D residual block with Squeeze-and-Excitation."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, reduction: int = 8):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        hidden = max(16, channels // reduction)
        self.se = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if mask is not None:
            m = mask.unsqueeze(1)
            denom = m.sum(dim=-1).clamp_min(1.0)
            pooled = (h * m).sum(dim=-1) / denom
        else:
            pooled = h.mean(dim=-1)
        scale = self.se(pooled).unsqueeze(-1)
        return F.silu(x + h * scale)


class HSVCReferenceEncoder(nn.Module):
    """
    Reference audio encoder predicting ``NeuralTimbreBank`` embeddings from a short audio clip.

    Extracts an 80-bin Log-Mel spectrogram (40 Hz - 12 kHz @ 48 kHz) covering both
    vocal-tract formants and upper breath/air timbre, encodes it with a dilated
    SE-ResNet1D backbone, and produces:
      1. ``global_emb`` [B, spk_dim] via silence-masked Attentive Statistics Pooling.
      2. ``proto_values`` [B, num_protos, spk_dim] via learned multi-head cross-attention
         queries over the encoded reference frames.
    """

    def __init__(self, config: Optional[HSVCReferenceEncoderConfig] = None):
        super().__init__()
        self.config = config or HSVCReferenceEncoderConfig()
        c = self.config

        mel_fb = librosa.filters.mel(
            sr=c.sample_rate,
            n_fft=c.n_fft,
            n_mels=c.n_mels,
            fmin=c.fmin,
            fmax=c.fmax,
        ).astype(np.float32)
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb), persistent=False)
        self.register_buffer("stft_window", torch.hann_window(c.win_length), persistent=False)

        self.in_proj = nn.Sequential(
            nn.Conv1d(c.n_mels, c.channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, c.channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                SEResBlock1D(c.channels, kernel_size=3, dilation=d)
                for d in (1, 2, 3, 4)
            ]
        )
        self.mfa = nn.Sequential(
            nn.Conv1d(c.channels * 4, c.channels, kernel_size=1),
            nn.GroupNorm(8, c.channels),
            nn.SiLU(),
        )

        # Global timbre head: Attentive Statistics Pooling (mean + std -> spk_dim)
        self.attn_pool = nn.Sequential(
            nn.Conv1d(c.channels, c.channels // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(c.channels // 2, 1, kernel_size=1),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(c.channels * 2),
            nn.Linear(c.channels * 2, c.channels),
            nn.SiLU(),
            nn.Linear(c.channels, c.spk_dim),
        )

        # Register/phoneme prototype head: num_protos learned queries attending over reference frames
        self.proto_queries = nn.Parameter(torch.randn(1, c.num_protos, c.channels) * 0.04)
        self.proto_attn = nn.MultiheadAttention(
            embed_dim=c.channels,
            num_heads=c.n_heads,
            batch_first=True,
        )
        self.proto_norm = nn.LayerNorm(c.channels)
        self.proto_head = nn.Sequential(
            nn.Linear(c.channels, c.channels),
            nn.SiLU(),
            nn.Linear(c.channels, c.spk_dim),
        )

    def extract_mel_and_mask(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute normalized 80-bin Log-Mel features and an active-frame mask ignoring silence.

        Args:
            wav (torch.Tensor): Mono waveform [B, Samples] or [B, 1, Samples] at ``config.sample_rate``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (log_mel [B, n_mels, T], active_mask [B, T]).
        """
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        c = self.config
        with torch.amp.autocast(device_type=wav.device.type, enabled=False):
            wav_f = wav.float()
            # Peak-normalize per utterance so reference loudness never biases timbre
            peak = wav_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
            wav_f = wav_f / peak * 0.9

            spec = torch.stft(
                wav_f,
                n_fft=c.n_fft,
                hop_length=c.hop_length,
                win_length=c.win_length,
                window=self.stft_window,
                center=True,
                return_complex=True,
            ).abs()
            mel = torch.matmul(self.mel_fb, spec)
            log_mel = torch.log(mel.clamp_min(1e-5))

            # Active frame mask: frames within 45 dB of utterance peak energy
            frame_energy = spec.pow(2).mean(dim=1).clamp_min(1e-10).log10() * 10.0
            peak_db = frame_energy.amax(dim=-1, keepdim=True)
            active = (frame_energy > (peak_db - 45.0)).float()
            # Ensure at least 1 active frame even on near-silent inputs
            empty = active.sum(dim=-1, keepdim=True) < 1.0
            active = torch.where(empty, torch.ones_like(active), active)

            # Per-utterance mean normalization across active frames (preserves spectral tilt!)
            m = active.unsqueeze(1)
            mean_val = (log_mel * m).sum(dim=(1, 2), keepdim=True) / (m.sum(dim=(1, 2), keepdim=True) * c.n_mels).clamp_min(1.0)
            log_mel = (log_mel - mean_val) * 0.25
        return log_mel, active

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of reference waveforms into ``(global_emb, proto_values)``.

        Args:
            wav (torch.Tensor): Mono waveform [B, Samples] or [B, 1, Samples] at ``config.sample_rate``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - global_emb: [B, spk_dim]
                - proto_values: [B, num_protos, spk_dim]
        """
        log_mel, active = self.extract_mel_and_mask(wav)
        h = self.in_proj(log_mel.to(self.in_proj[0].weight.dtype))
        feats = []
        for blk in self.blocks:
            h = blk(h, mask=active.to(h.dtype))
            feats.append(h)
        h = self.mfa(torch.cat(feats, dim=1))  # [B, C, T]

        # 1. Masked Attentive Statistics Pooling -> global_emb [B, spk_dim]
        attn_logits = self.attn_pool(h).squeeze(1)  # [B, T]
        attn_logits = attn_logits.masked_fill(active < 0.5, -1e4)
        w = F.softmax(attn_logits, dim=-1).unsqueeze(1)  # [B, 1, T]
        mean = (h * w).sum(dim=-1)
        var = ((h - mean.unsqueeze(-1)).pow(2) * w).sum(dim=-1).clamp_min(1e-6)
        std = torch.sqrt(var)
        global_emb = self.global_head(torch.cat([mean, std], dim=-1))

        # 2. Cross-Attention Prototype Extraction -> proto_values [B, num_protos, spk_dim]
        kv = h.transpose(1, 2)  # [B, T, C]
        q = self.proto_queries.expand(kv.shape[0], -1, -1).to(kv.dtype)
        key_padding_mask = active < 0.5
        # Guard against all-True padding mask
        all_masked = key_padding_mask.all(dim=-1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False
        p_out, _ = self.proto_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        proto_values = self.proto_head(self.proto_norm(p_out + q))

        return global_emb, proto_values

    @torch.inference_mode()
    def encode_audio(
        self, audio: np.ndarray, orig_sr: int, max_sec: float = 30.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an arbitrary-sample-rate numpy waveform into ``(global_emb [1, D], proto_values [1, P, D])``.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        wav = audio.astype(np.float32)
        if orig_sr != self.config.sample_rate:
            wav = soxr.resample(wav, orig_sr, self.config.sample_rate, quality="HQ").astype(np.float32)
        max_n = int(max_sec * self.config.sample_rate)
        if len(wav) > max_n:
            wav = wav[:max_n]
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(device=device, dtype=dtype)
        g, p = self.forward(wav_t)
        return g.float(), p.float()

    def export_package(self, path: str, fp16: bool = False) -> None:
        """Save a self-contained reference encoder checkpoint."""
        sd = {
            k: (v.detach().cpu().half() if (fp16 and v.is_floating_point()) else v.detach().cpu())
            for k, v in self.state_dict().items()
        }
        torch.save(
            {
                "architecture": "hsvc_ref_encoder",
                "version": "1.0",
                "config": self.config.to_dict(),
                "precision": "fp16" if fp16 else "fp32",
                "model": sd,
            },
            path,
        )

    @classmethod
    def from_package(cls, path: str, device: str = "cpu") -> "HSVCReferenceEncoder":
        """Load a saved ``HSVCReferenceEncoder`` package."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = HSVCReferenceEncoderConfig.from_dict(ckpt.get("config", {}))
        model = cls(cfg)
        sd = {
            k: (v.float() if v.is_floating_point() else v)
            for k, v in ckpt["model"].items()
        }
        model.load_state_dict(sd, strict=True)
        return model.to(device).eval()
