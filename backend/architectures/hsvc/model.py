"""Core HSVC (Harmonic-Spectral Voice Conversion) Generator model and configuration contract."""

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.architectures.hsvc.source import BandlimitedSourceGenerator
from backend.architectures.hsvc.spectral import (
    FreqCoordSpectralBlock,
    NeuralTimbreBank,
    StreamingConvNeXtFiLMBlock,
    build_erb_interpolation_matrix,
)


def is_hsvc_checkpoint(cpt: Any) -> bool:
    """
    Check whether a loaded PyTorch checkpoint dict belongs to the HSVC architecture.

    Args:
        cpt (Any): Loaded checkpoint object from ``torch.load``.

    Returns:
        bool: True if the checkpoint declares an HSVC (or legacy HarmoniX) architecture.
    """
    return isinstance(cpt, dict) and cpt.get("architecture") in ("hsvc", "harmonix")


@dataclass
class HSVCConfig:
    """
    Configuration schema for the 48 kHz HSVCGenerator.

    Attributes:
        sample_rate (int): Target audio sample rate in Hz (48000).
        hop_in (int): Frontend frame hop in samples (480 = 10 ms at 48 kHz).
        hop_dec (int): Synthesis STFT hop in samples (240 = 5 ms at 48 kHz).
        win_dec (int): Synthesis STFT Hann window length in samples (960 = 20 ms).
        n_fft_dec (int): Synthesis STFT FFT size (2048 -> 1025 frequency bins).
        n_erb_bands (int): Number of perceptual ERB/Mel output filter bands.
        erb_2d_bins (int): Frequency resolution of the 2D spectral refinement backbone.
        num_speakers (int): Speaker embedding table capacity.
        content_dim (int): Input self-supervised content feature dimension (768 for ContentVec).
        bottleneck_dim (int): Speaker-invariant content bottleneck width.
        hidden_dim (int): 1D ConvNeXt-FiLM backbone channel width.
        spk_dim (int): Global speaker embedding and timbre prototype dimension.
        num_protos (int): Number of timbre prototypes per speaker in NeuralTimbreBank.
        n_enc_layers (int): Number of 100 Hz ConvNeXt-FiLM encoder blocks.
        n_dec_1d_layers (int): Number of 200 Hz sub-frame 1D refinement blocks.
        n_spec_2d_layers (int): Number of 2D frequency-coordinate spectral blocks.
        spec_2d_channels (int): Channel width of the 2D spectral backbone.
        silence_thresh_db (float): Midpoint of the smooth energy silence gate in dBFS.
        silence_slope_db (float): Softness slope of the silence gate sigmoid in dB.
        nyquist_ratio (float): Harmonic upper frequency ratio relative to Nyquist.
        use_spec_res (bool): Optional additive STFT residual (kept False to guarantee zero 200 Hz comb lines).
    """

    sample_rate: int = 48000
    ref_sample_rate: int = 48000
    hop_in: int = 480
    hop_dec: int = 240
    win_dec: int = 960
    n_fft_dec: int = 2048
    n_erb_bands: int = 256
    erb_2d_bins: int = 128
    num_speakers: int = 256
    content_dim: int = 768
    bottleneck_dim: int = 256
    hidden_dim: int = 384
    spk_dim: int = 256
    num_protos: int = 16
    n_enc_layers: int = 8
    n_dec_1d_layers: int = 4
    n_spec_2d_layers: int = 4
    spec_2d_channels: int = 64
    silence_thresh_db: float = -64.0
    silence_slope_db: float = 3.0
    nyquist_ratio: float = 0.99
    use_spec_res: bool = False

    @staticmethod
    def stft_params_for_sr(target_sr: int, ref_sr: int = 48000) -> dict:
        """
        Compute constant-Hz-spacing STFT parameters for any target sample rate <= ``ref_sr``.

        Keeps FFT bin spacing (``ref_sr / 2048 = 23.4375 Hz``) and temporal window duration
        (20 ms Hann, 5 ms hop, 10 ms frontend) invariant so a single 48 kHz pretrained model
        adapts zero-shot or fine-tunes natively at 24k, 32k, 40k, 44.1k, or 48k.
        """
        target_sr = int(target_sr)
        if target_sr < 16000 or target_sr > ref_sr:
            raise ValueError(f"HSVC target_sr must be in [16000, {ref_sr}] Hz (got {target_sr} Hz).")
        hop_in = int(round(target_sr / 100.0))
        hop_dec = int(round(target_sr / 200.0))
        win_dec = hop_dec * 4
        n_fft_dec = int(round(2048.0 * target_sr / float(ref_sr) / 2.0)) * 2
        return {
            "sample_rate": target_sr,
            "hop_in": hop_in,
            "hop_dec": hop_dec,
            "win_dec": win_dec,
            "n_fft_dec": n_fft_dec,
        }

    def to_dict(self) -> dict:
        """Serialize configuration fields to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HSVCConfig":
        """Instantiate an HSVCConfig from a dict, ignoring unknown keys for forward compatibility."""
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class HSVCGenerator(nn.Module):
    """
    End-to-end 48 kHz Bounded-Lookahead Harmonic-Spectral Voice Conversion Generator.

    Predicts minimum-phase harmonic filters and complex aperiodic noise filters over 256
    ERB/Mel bands, applies them in the 2048-point STFT domain to alias-free bandlimited
    excitation waveforms, and reconstructs 48 kHz audio via silence-gated iSTFT.

    Args:
        config (Optional[HSVCConfig]): Model architecture configuration (defaults to HSVCConfig()).
    """

    def __init__(self, config: Optional[HSVCConfig] = None):
        super().__init__()
        self.config = config or HSVCConfig()
        c = self.config
        self.n_bins_dec = c.n_fft_dec // 2 + 1

        # 1. Speaker-invariant content bottleneck (frozen during single-speaker fine-tuning)
        self.content_norm = nn.LayerNorm(c.content_dim)
        self.content_down = nn.Sequential(
            nn.Linear(c.content_dim, c.bottleneck_dim),
            nn.LayerNorm(c.bottleneck_dim),
        )
        self.content_up = nn.Linear(c.bottleneck_dim, c.hidden_dim)

        # 2. Continuous prosody frontend (19 dims: log-F0, deltas, Fourier F0, energy, HF cues)
        self.prosody_proj = nn.Sequential(
            nn.Linear(19, c.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(c.hidden_dim // 2, c.hidden_dim),
        )
        self.in_norm = nn.LayerNorm(c.hidden_dim)

        # 3. Differentiable Neural Timbre Memory Bank
        self.timbre_bank = NeuralTimbreBank(
            c.num_speakers, c.hidden_dim, spk_dim=c.spk_dim, num_protos=c.num_protos
        )

        # 4. 100 Hz bounded-lookahead backbone (total lookahead = 2 frames = 20 ms)
        if c.n_enc_layers <= 6:
            enc_cfg = [(1, 1), (2, 1), (4, 0), (1, 0), (2, 0), (4, 0)][: c.n_enc_layers]
        else:
            base_cfg = [(1, 1), (2, 1), (4, 0), (8, 0), (1, 0), (2, 0), (4, 0), (8, 0)]
            enc_cfg = (base_cfg * ((c.n_enc_layers + 7) // 8))[: c.n_enc_layers]
        self.enc_blocks = nn.ModuleList(
            [
                StreamingConvNeXtFiLMBlock(c.hidden_dim, c.spk_dim, kernel_size=7, dilation=d, lookahead=la)
                for d, la in enc_cfg
            ]
        )

        # 5. 200 Hz sub-frame refinement (5 ms hop, 1-frame = 5 ms lookahead)
        dec_cfg = [(1, 1), (2, 0), (1, 0), (2, 0)][: c.n_dec_1d_layers]
        self.dec_1d_blocks = nn.ModuleList(
            [
                StreamingConvNeXtFiLMBlock(c.hidden_dim, c.spk_dim, kernel_size=5, dilation=d, lookahead=la)
                for d, la in dec_cfg
            ]
        )

        # 6. Alias-free bandlimited source + constant-Hz-spacing STFT/iSTFT basis
        self.source_gen = BandlimitedSourceGenerator(
            c.sample_rate,
            c.hop_in,
            nyquist_ratio=c.nyquist_ratio,
            ref_sample_rate=c.ref_sample_rate,
        )
        self.register_buffer("stft_window", torch.hann_window(c.win_dec), persistent=False)

        ref_win = int(round(c.ref_sample_rate / 200.0)) * 4
        W_erb_ref, W_phase_ref = build_erb_interpolation_matrix(
            c.n_erb_bands, 2048, ref_win, c.ref_sample_rate
        )
        self.register_buffer("W_erb", W_erb_ref[:, : self.n_bins_dec].contiguous(), persistent=False)
        self.register_buffer("W_minphase", W_phase_ref[:, : self.n_bins_dec].contiguous(), persistent=False)

        # 7. Frequency-coordinate-aware 2D spectral refinement backbone (calibrated to ref_sample_rate)
        erb_hz = librosa.mel_frequencies(n_mels=c.erb_2d_bins, fmin=0.0, fmax=c.ref_sample_rate / 2.0).astype(
            np.float32
        )
        f_lin = (erb_hz / (c.ref_sample_rate / 2.0)) * 2.0 - 1.0
        f_log = (np.log1p(erb_hz / 200.0) / np.log1p((c.ref_sample_rate / 2.0) / 200.0)) * 2.0 - 1.0
        freq_coords = torch.from_numpy(np.stack([f_lin, f_log], axis=0)).view(1, 2, c.erb_2d_bins, 1)
        self.register_buffer("freq_coords", freq_coords, persistent=False)

        self.latent_to_2d = nn.Conv1d(c.hidden_dim, c.spec_2d_channels * c.erb_2d_bins, 1)
        self.src_to_2d = nn.Conv2d(5, c.spec_2d_channels, kernel_size=3, padding=1)
        self.spec_2d_blocks = nn.ModuleList(
            [
                FreqCoordSpectralBlock(
                    c.spec_2d_channels, c.spk_dim, k_freq=7, k_time=3, lookahead=(1 if i == 0 else 0)
                )
                for i in range(c.n_spec_2d_layers)
            ]
        )

        # 8. Dual 1D + 2D spectral filter parameter heads (15 channels: 4 harmonic, 9 noise, 2 optional res)
        self.n_out_ch = 15
        self.head_1d = nn.Conv1d(c.hidden_dim, self.n_out_ch * c.n_erb_bands, 1)
        self.head_2d = nn.Conv2d(c.spec_2d_channels, self.n_out_ch, 1)
        nn.init.zeros_(self.head_2d.weight)
        nn.init.zeros_(self.head_2d.bias)
        with torch.no_grad():
            self.head_1d.weight.mul_(0.1)
            b = self.head_1d.bias.view(self.n_out_ch, c.n_erb_bands)
            b.zero_()
            b[0, :] = -1.5   # Aligned harmonic log-gain init
            b[1, :] = -2.2   # Phase-scattered harmonic log-gain init
            b[4, :] = -3.2   # Broadband noise log-gain init
            b[7, :] = -4.2   # Cosine-modulated glottal noise init
            b[10, :] = -4.2  # Sine-modulated glottal noise init

    def compute_silence_gate(self, energy_db: torch.Tensor) -> torch.Tensor:
        """Compute smooth [0, 1] energy silence gate from 10 ms RMS loudness in dBFS."""
        return torch.sigmoid((energy_db - self.config.silence_thresh_db) / self.config.silence_slope_db)

    def build_prosody_features(
        self,
        f0_100hz: torch.Tensor,
        vuv_conf: torch.Tensor,
        energy_db: torch.Tensor,
        hf_energy_db: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct the 19-dimensional continuous prosody feature vector and silence gate.

        Args:
            f0_100hz (torch.Tensor): 100 Hz pitch contour in Hz, shape [B, T].
            vuv_conf (torch.Tensor): 100 Hz voiced/unvoiced confidence, shape [B, T].
            energy_db (torch.Tensor): 100 Hz fullband log-RMS energy in dBFS, shape [B, T].
            hf_energy_db (torch.Tensor): 100 Hz >3 kHz log-RMS energy in dBFS, shape [B, T].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (prosody_feats [B, T, 19], g_sil [B, T]).
        """
        voiced = (f0_100hz > 1.0).float()
        log_f0 = ((torch.log2(f0_100hz.clamp_min(30.0)) - math.log2(200.0)) / 2.0) * voiced
        d_log_f0 = F.pad(log_f0[:, 1:] - log_f0[:, :-1], (1, 0)) * voiced

        fourier = []
        for k in range(6):
            ang = (2.0**k * math.pi) * log_f0
            fourier.append(torch.sin(ang) * voiced)
            fourier.append(torch.cos(ang) * voiced)

        energy_norm = ((energy_db + 80.0) / 80.0).clamp(0.0, 1.0)
        d_energy = F.pad(energy_norm[:, 1:] - energy_norm[:, :-1], (1, 0))
        g_sil = self.compute_silence_gate(energy_db)
        unv_hf_cue = (1.0 - voiced) * ((hf_energy_db - energy_db + 30.0) / 30.0).clamp(0.0, 1.0) * g_sil

        feats = torch.stack(
            [log_f0, d_log_f0, voiced, vuv_conf.clamp(0.0, 1.0), *fourier, energy_norm, d_energy, unv_hf_cue],
            dim=-1,
        )
        return feats, g_sil

    def _stft(self, wav_flat: torch.Tensor) -> torch.Tensor:
        """Compute complex 2048-point STFT (5 ms hop, 20 ms Hann window) in float32."""
        c = self.config
        return torch.stft(
            wav_flat.float(),
            n_fft=c.n_fft_dec,
            hop_length=c.hop_dec,
            win_length=c.win_dec,
            window=self.stft_window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Reconstruct 48 kHz time-domain waveform from complex STFT bins."""
        c = self.config
        return torch.istft(
            spec,
            n_fft=c.n_fft_dec,
            hop_length=c.hop_dec,
            win_length=c.win_dec,
            window=self.stft_window,
            center=True,
            length=length,
        )

    def encode_content(self, content_100hz: torch.Tensor) -> torch.Tensor:
        """Project 768-d ContentVec frames into the speaker-invariant 256-d content bottleneck."""
        c_in = self.content_norm(content_100hz)
        if self.training:
            B = c_in.shape[0]
            ch_scale = 1.0 + 0.04 * torch.randn(B, 1, c_in.shape[-1], device=c_in.device, dtype=c_in.dtype)
            c_in = F.dropout(c_in * ch_scale, p=0.04, training=True)
        return self.content_down(c_in)

    def decode_from_content(
        self,
        z_content: torch.Tensor,
        f0_100hz: torch.Tensor,
        vuv_conf: torch.Tensor,
        energy_db: torch.Tensor,
        hf_energy_db: torch.Tensor,
        sid: torch.Tensor,
        init_turns: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        ref_timbre: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthesize 48 kHz waveform from bottleneck content, prosody controls, and speaker ID.

        Args:
            z_content (torch.Tensor): Bottleneck content sequence, shape [B, T, bottleneck_dim].
            f0_100hz (torch.Tensor): 100 Hz F0 in Hz, shape [B, T].
            vuv_conf (torch.Tensor): 100 Hz VUV confidence, shape [B, T].
            energy_db (torch.Tensor): 100 Hz RMS energy in dBFS, shape [B, T].
            hf_energy_db (torch.Tensor): 100 Hz >3 kHz energy in dBFS, shape [B, T].
            sid (torch.Tensor): Target speaker IDs, shape [B].
            init_turns (Optional[torch.Tensor]): Initial harmonic phase [B, 1] for streaming.
            noise_scale (float): Scalar multiplier for the aperiodic noise branch (default: 1.0).
            ref_timbre (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional one-shot timbre
                override ``(global_emb [B, spk_dim], proto_values [B, num_protos, spk_dim])``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (audio [B, 1, T * hop_in], end_turns [B, 1]).
        """
        c = self.config
        B, T, _ = z_content.shape
        samples = T * c.hop_in

        # Fuse content, prosody, and frame-aware speaker timbre prototypes
        prosody, g_sil = self.build_prosody_features(f0_100hz, vuv_conf, energy_db, hf_energy_db)
        base_feat = self.in_norm(self.content_up(z_content) + self.prosody_proj(prosody))
        spk, local_timbre = self.timbre_bank(base_feat, sid, ref_timbre=ref_timbre)
        if self.training:
            local_timbre = F.dropout(local_timbre, p=0.10, training=True)
        h = (base_feat + local_timbre).transpose(1, 2)

        # 100 Hz temporal encoding + 200 Hz sub-frame decoding
        for blk in self.enc_blocks:
            h = blk(h, spk)

        T_stft = samples // c.hop_dec + 1
        h2 = F.interpolate(h, size=T_stft, mode="linear", align_corners=False)
        for blk in self.dec_1d_blocks:
            h2 = blk(h2, spk)

        # Generate alias-free source excitations and analyze them in the synthesis STFT domain
        harm_wav, noise_wav, end_turns = self.source_gen(f0_100hz, vuv_conf, g_sil, init_turns=init_turns)

        with torch.amp.autocast(device_type=h.device.type, enabled=False):
            harm_stft = self._stft(harm_wav.reshape(B * 2, samples)).view(B, 2, self.n_bins_dec, T_stft)
            noise_stft = self._stft(noise_wav.reshape(B * 3, samples)).view(B, 3, self.n_bins_dec, T_stft)
            src_mag = torch.cat(
                [
                    torch.log(harm_stft.abs().clamp_min(1e-5)),
                    torch.log(noise_stft[:, :1].abs().clamp_min(1e-5)),
                ],
                dim=1,
            )
            src_mag_ds = F.interpolate(src_mag, size=(c.erb_2d_bins, T_stft), mode="bilinear", align_corners=False)

        # 2D frequency-coordinate spectral refinement
        coords = self.freq_coords.expand(B, -1, -1, T_stft).to(h2.dtype)
        f2d = self.latent_to_2d(h2).view(B, -1, c.erb_2d_bins, T_stft)
        f2d = f2d + self.src_to_2d(torch.cat([src_mag_ds.to(h2.dtype), coords], dim=1))
        for blk2d in self.spec_2d_blocks:
            f2d = blk2d(f2d, spk)

        # Combine 1D global envelope + 2D local formant/harmonic detail in 256 ERB bands
        out_1d = self.head_1d(h2).view(B, self.n_out_ch, c.n_erb_bands, T_stft)
        out_2d = F.interpolate(self.head_2d(f2d), size=(c.n_erb_bands, T_stft), mode="bilinear", align_corners=False)
        params_erb = (out_1d + out_2d).float()

        # Apply minimum-phase harmonic filtering and complex noise filtering in float32
        with torch.amp.autocast(device_type=h.device.type, enabled=False):
            g_sil_stft = F.interpolate(g_sil.unsqueeze(1).float(), size=T_stft, mode="linear", align_corners=False)

            h_log_erb = params_erb[:, 0:2].clamp(-12.0, 4.5)
            h_dph_erb = torch.tanh(params_erb[:, 2:4]) * math.pi
            h_log_lin = torch.einsum("bcet,ef->bcft", h_log_erb, self.W_erb)
            h_ph_lin = (
                torch.einsum("bcet,ef->bcft", h_log_erb, self.W_minphase)
                + torch.einsum("bcet,ef->bcft", h_dph_erb, self.W_erb)
            )
            h_filt = torch.polar(torch.exp(h_log_lin), h_ph_lin)
            spec_harm = (harm_stft * h_filt).sum(dim=1)

            n_params = params_erb[:, 4:13].view(B, 3, 3, c.n_erb_bands, T_stft)
            n_log_lin = torch.einsum("bcet,ef->bcft", n_params[:, :, 0].clamp(-12.0, 4.5), self.W_erb)
            n_re = torch.einsum("bcet,ef->bcft", n_params[:, :, 1], self.W_erb)
            n_im = torch.einsum("bcet,ef->bcft", n_params[:, :, 2], self.W_erb)
            n_norm = torch.sqrt(n_re * n_re + n_im * n_im + 1e-6)
            n_mag = torch.exp(n_log_lin)
            n_filt = torch.complex(n_mag * (n_re / n_norm), n_mag * (n_im / n_norm))
            spec_noise = (noise_stft * n_filt).sum(dim=1) * float(noise_scale)

            if c.use_spec_res:
                res_re = torch.einsum("bet,ef->bft", torch.tanh(params_erb[:, 13]) * 0.4, self.W_erb)
                res_im = torch.einsum("bet,ef->bft", torch.tanh(params_erb[:, 14]) * 0.4, self.W_erb)
                spec_res = torch.complex(res_re, res_im)
                spec_total = (spec_harm + spec_noise + spec_res) * g_sil_stft
            else:
                spec_total = (spec_harm + spec_noise) * g_sil_stft

            audio = self._istft(spec_total, length=samples)
            sil_sr = F.interpolate(g_sil.unsqueeze(1).float(), size=samples, mode="linear", align_corners=False).squeeze(1)
            audio = (torch.tanh(audio) * sil_sr).unsqueeze(1)

        return audio, end_turns

    def forward(
        self,
        content_100hz: torch.Tensor,
        f0_100hz: torch.Tensor,
        vuv_conf: torch.Tensor,
        energy_db: torch.Tensor,
        hf_energy_db: torch.Tensor,
        sid: torch.Tensor,
        init_turns: Optional[torch.Tensor] = None,
        return_bottleneck: bool = False,
        noise_scale: float = 1.0,
        ref_timbre: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Full forward pass from 100 Hz ContentVec and prosody controls to 48 kHz audio.

        Args:
            content_100hz (torch.Tensor): Linearly upsampled ContentVec features, shape [B, T, 768].
            f0_100hz (torch.Tensor): 100 Hz F0 contour in Hz, shape [B, T].
            vuv_conf (torch.Tensor): 100 Hz VUV confidence, shape [B, T].
            energy_db (torch.Tensor): 100 Hz RMS loudness in dBFS, shape [B, T].
            hf_energy_db (torch.Tensor): 100 Hz >3 kHz loudness in dBFS, shape [B, T].
            sid (torch.Tensor): Target speaker IDs, shape [B].
            init_turns (Optional[torch.Tensor]): Optional initial streaming phase [B, 1].
            return_bottleneck (bool): If True, also return (z_content, end_turns).
            noise_scale (float): Aperiodic noise branch scaling factor (default: 1.0).
            ref_timbre (Optional[Tuple[torch.Tensor, torch.Tensor]]): Optional one-shot timbre
                override ``(global_emb [B, spk_dim], proto_values [B, num_protos, spk_dim])``.

        Returns:
            torch.Tensor or Tuple: Generated waveform [B, 1, T * hop_in] (plus bottleneck & phase if requested).
        """
        z_content = self.encode_content(content_100hz)
        audio, end_turns = self.decode_from_content(
            z_content,
            f0_100hz,
            vuv_conf,
            energy_db,
            hf_energy_db,
            sid,
            init_turns=init_turns,
            noise_scale=noise_scale,
            ref_timbre=ref_timbre,
        )
        if return_bottleneck:
            return audio, z_content, end_turns
        return audio

    @torch.no_grad()
    def inject_reference_timbre(
        self, global_emb: torch.Tensor, proto_values: torch.Tensor, sid: int = 0
    ) -> "HSVCGenerator":
        """
        Write a one-shot ``(global_emb, proto_values)`` pair into speaker slot ``sid``.

        Enables zero-overhead real-time streaming (no encoder needed at runtime) and
        instant warm-starting for rapid single-speaker fine-tuning.
        """
        g = global_emb.detach().reshape(-1).to(
            device=self.timbre_bank.global_emb.weight.device,
            dtype=self.timbre_bank.global_emb.weight.dtype,
        )
        p = proto_values.detach().reshape(-1).to(
            device=self.timbre_bank.proto_values.weight.device,
            dtype=self.timbre_bank.proto_values.weight.dtype,
        )
        self.timbre_bank.global_emb.weight[sid].copy_(g)
        self.timbre_bank.proto_values.weight[sid].copy_(p)
        return self

    def adapt_sample_rate(self, target_sr: int) -> "HSVCGenerator":
        """
        Adapt this HSVCGenerator in-place to a new output sampling rate (e.g. 24k, 32k, 40k, 44.1k, 48k).

        All learned weights remain 100% intact; only the non-learnable DSP synthesis buffers
        (source generator, Hann window, and prefix-sliced physical-Hz ERB / minimum-phase bases)
        are updated to match ``target_sr`` with exact constant-Hz bin spacing.

        Args:
            target_sr (int): Desired sampling rate in Hz (16000 <= target_sr <= ref_sample_rate).

        Returns:
            HSVCGenerator: ``self``, adapted in-place on its current device.
        """
        target_sr = int(target_sr)
        if target_sr == self.config.sample_rate:
            return self
        device = next(self.parameters()).device
        params = HSVCConfig.stft_params_for_sr(target_sr, ref_sr=self.config.ref_sample_rate)
        self.config.sample_rate = params["sample_rate"]
        self.config.hop_in = params["hop_in"]
        self.config.hop_dec = params["hop_dec"]
        self.config.win_dec = params["win_dec"]
        self.config.n_fft_dec = params["n_fft_dec"]
        self.n_bins_dec = params["n_fft_dec"] // 2 + 1

        self.source_gen = BandlimitedSourceGenerator(
            self.config.sample_rate,
            self.config.hop_in,
            nyquist_ratio=self.config.nyquist_ratio,
            ref_sample_rate=self.config.ref_sample_rate,
        ).to(device)
        self.stft_window = torch.hann_window(self.config.win_dec, device=device)

        ref_win = int(round(self.config.ref_sample_rate / 200.0)) * 4
        W_erb_ref, W_phase_ref = build_erb_interpolation_matrix(
            self.config.n_erb_bands, 2048, ref_win, self.config.ref_sample_rate
        )
        self.W_erb = W_erb_ref[:, : self.n_bins_dec].contiguous().to(device)
        self.W_minphase = W_phase_ref[:, : self.n_bins_dec].contiguous().to(device)
        return self

    def export_package(
        self,
        path: str,
        speaker_map: Optional[dict] = None,
        fp16: bool = False,
        metadata: Optional[dict] = None,
    ):
        """
        Export a self-contained inference/fine-tuning HSVC checkpoint package (.pth).

        Args:
            path (str): Output file path for the ``.pth`` package.
            speaker_map (Optional[dict]): Mapping from speaker names to integer IDs.
            fp16 (bool): If True, store floating-point weights in float16 (~39 MB vs ~78 MB FP32).
            metadata (Optional[dict]): Optional Applio metadata fields (epoch, step, model_name, etc.).
        """
        weights = {}
        for k, v in self.state_dict().items():
            t = v.detach().cpu()
            if fp16 and t.is_floating_point():
                t = t.half()
            weights[k] = t
        sr_label = "44.1k" if self.config.sample_rate == 44100 else f"{self.config.sample_rate // 1000}k"
        pkg = {
            "architecture": "hsvc",
            "version": "v1",
            "f0": 1,
            "sr": sr_label,
            "vocoder": "HSVC",
            "embedder_model": "contentvec",
            "speakers_id": len(speaker_map) if speaker_map else self.config.num_speakers,
            "config": self.config.to_dict(),
            "speaker_map": speaker_map or {},
            "weight": weights,
        }
        if metadata:
            for mk, mv in metadata.items():
                if mk not in ("weight", "config", "architecture", "vocoder") and mv is not None:
                    pkg[mk] = mv
        torch.save(pkg, path)

    @classmethod
    def from_checkpoint_dict(
        cls, pkg: dict, device: str = "cpu", target_sr: Optional[int] = None
    ) -> Tuple["HSVCGenerator", dict]:
        """
        Instantiate an eval-mode HSVCGenerator directly from an in-memory checkpoint dict.

        Args:
            pkg (dict): Checkpoint dictionary containing ``config`` and ``weight``.
            device (str): Target PyTorch device (e.g. ``"cpu"`` or ``"cuda"``).
            target_sr (Optional[int]): Optional target sampling rate to adapt the model to on load.

        Returns:
            Tuple[HSVCGenerator, dict]: Loaded model in eval mode and its ``speaker_map``.
        """
        cfg = HSVCConfig.from_dict(pkg["config"])
        model = cls(cfg)
        state = {k: v.float() if v.is_floating_point() else v for k, v in pkg["weight"].items()}
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        if target_sr is not None and int(target_sr) != model.config.sample_rate:
            model.adapt_sample_rate(int(target_sr))
        return model, pkg.get("speaker_map", {})

    @classmethod
    def from_package(
        cls, path: str, device: str = "cpu", target_sr: Optional[int] = None
    ) -> Tuple["HSVCGenerator", dict]:
        """
        Load an HSVCGenerator from a saved ``.pth`` package file on disk.

        Args:
            path (str): Path to the HSVC ``.pth`` file.
            device (str): Target PyTorch device.
            target_sr (Optional[int]): Optional target sampling rate to adapt the model to on load.

        Returns:
            Tuple[HSVCGenerator, dict]: Loaded model in eval mode and its ``speaker_map``.
        """
        pkg = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_checkpoint_dict(pkg, device=device, target_sr=target_sr)
