"""Real-time low-latency phase-continuous streaming engine for HSVC."""

from typing import Optional

import numpy as np
import torch

from rvc.lib.algorithm.hsvc.model import HSVCGenerator


class HSVCStreamer:
    """
    Bounded-context, phase-continuous overlap-add streaming wrapper for HSVCGenerator.

    Maintains exact harmonic phase continuity across chunk boundaries via ``init_turns``
    and applies a short raised-cosine crossfade over ``crossfade_samples`` to prevent
    seam discontinuities during real-time GPU or CPU execution.

    Args:
        model (HSVCGenerator): Loaded HSVC generator in eval mode.
        context_frames (int): Past context window in 10 ms frames (default: 60 = 600 ms).
        lookahead_frames (int): Future lookahead in 10 ms frames (default: 2 = 20 ms).
        crossfade_samples (int): Raised-cosine boundary crossfade length in samples (default: 240 = 5 ms).
        device (Optional[str]): Compute device (defaults to the model's current device).
    """

    def __init__(
        self,
        model: HSVCGenerator,
        context_frames: int = 60,
        lookahead_frames: int = 2,
        crossfade_samples: int = 240,
        device: Optional[str] = None,
    ):
        self.model = model.eval()
        self.device = device or str(next(model.parameters()).device)
        self.hop_in = model.config.hop_in
        self.sample_rate = model.config.sample_rate
        self.context_frames = context_frames
        self.lookahead_frames = lookahead_frames
        self.crossfade_samples = crossfade_samples

        fade = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
        self.fade_in = 0.5 * (1.0 - np.cos(np.pi * fade))
        self.fade_out = 1.0 - self.fade_in

        self.reset()

    def reset(self):
        """Reset streaming phase and overlap-add tail buffers."""
        self._running_turns = 0.0
        self._prev_tail = None

    @torch.inference_mode()
    def stream_full_features(
        self,
        content_100hz: np.ndarray,
        f0_100hz: np.ndarray,
        vuv_conf: np.ndarray,
        energy_db: np.ndarray,
        hf_energy_db: np.ndarray,
        sid: int,
        chunk_frames: int = 8,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Process a full 100 Hz feature sequence chunk-by-chunk with phase-continuous OLA.

        Args:
            content_100hz (np.ndarray): ContentVec sequence, shape [T, 768].
            f0_100hz (np.ndarray): Pitch contour in Hz, shape [T].
            vuv_conf (np.ndarray): Voiced/unvoiced confidence, shape [T].
            energy_db (np.ndarray): 10 ms RMS loudness in dBFS, shape [T].
            hf_energy_db (np.ndarray): 10 ms >3 kHz loudness in dBFS, shape [T].
            sid (int): Target speaker ID.
            chunk_frames (int): Output chunk size in 10 ms frames (e.g., 8 = 80 ms).
            noise_scale (float): Aperiodic noise scaling factor (default: 1.0).

        Returns:
            np.ndarray: Clipped 48 kHz mono waveform, shape [T * hop_in].
        """
        T = len(f0_100hz)
        sid_t = torch.tensor([sid], dtype=torch.long, device=self.device)

        c_all = torch.from_numpy(content_100hz.astype(np.float32)).unsqueeze(0).to(self.device)
        f0_all = torch.from_numpy(f0_100hz.astype(np.float32)).unsqueeze(0).to(self.device)
        vc_all = torch.from_numpy(vuv_conf.astype(np.float32)).unsqueeze(0).to(self.device)
        e_all = torch.from_numpy(energy_db.astype(np.float32)).unsqueeze(0).to(self.device)
        hf_all = torch.from_numpy(hf_energy_db.astype(np.float32)).unsqueeze(0).to(self.device)

        # Precompute continuous cumulative phase in float64 for exact chunk alignment
        f0_clean = np.where(f0_100hz > 1.0, np.clip(f0_100hz, 35.0, 2400.0), 150.0)
        f0_sr = np.interp(
            np.linspace(0, T - 1, T * self.hop_in, dtype=np.float64),
            np.arange(T, dtype=np.float64),
            f0_clean.astype(np.float64),
        )
        cum_turns = np.cumsum(f0_sr / float(self.sample_rate))

        out_chunks = []
        prev_tail = None
        extra_tail_frames = 1

        for pos in range(0, T, chunk_frames):
            chunk_end = min(T, pos + chunk_frames)
            win_start = max(0, pos - self.context_frames)
            win_end = min(T, chunk_end + self.lookahead_frames + extra_tail_frames)

            init_t = float(cum_turns[win_start * self.hop_in - 1] % 1.0) if win_start > 0 else 0.0
            init_turns = torch.tensor([[init_t]], dtype=torch.float64, device=self.device)

            wav_win = self.model(
                c_all[:, win_start:win_end],
                f0_all[:, win_start:win_end],
                vc_all[:, win_start:win_end],
                e_all[:, win_start:win_end],
                hf_all[:, win_start:win_end],
                sid_t,
                init_turns=init_turns,
                noise_scale=noise_scale,
            )[0, 0].cpu().float().numpy()

            local_start = (pos - win_start) * self.hop_in
            local_end = (chunk_end - win_start) * self.hop_in
            seg = wav_win[local_start:local_end].copy()
            next_tail = wav_win[local_end : local_end + self.crossfade_samples].copy()

            if prev_tail is not None and len(prev_tail) == self.crossfade_samples and len(seg) >= self.crossfade_samples:
                seg[: self.crossfade_samples] = (
                    prev_tail * self.fade_out + seg[: self.crossfade_samples] * self.fade_in
                )
            out_chunks.append(seg)
            prev_tail = next_tail if len(next_tail) == self.crossfade_samples else None

        return np.clip(np.concatenate(out_chunks), -1.0, 1.0)
