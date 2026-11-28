"""End-to-end audio-to-audio inference, GPU tensor feature extraction, FAISS retrieval & streaming for HSVC.

CLI example:
    python -m rvc.infer.hsvc_pipeline \
        --model logs/pretrain/hsvc_base_48k/hsvc_base_48k.pth \
        --input input.wav --output output.wav --speaker exp_ex01 --pitch 0
"""

import argparse
import os
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F

from rvc.lib.algorithm.hsvc.model import HSVCGenerator
from rvc.lib.algorithm.hsvc.streaming import HSVCStreamer
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.utils import load_embedding


@torch.no_grad()
def compute_energy_100hz_torch(
    wav: torch.Tensor, sr: int, n_frames: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ultra-fast pure-PyTorch 100 Hz RMS energy_db & >3 kHz hf_energy_db extractor (GPU or CPU).

    Works identically on 16 kHz (realtime buffer) or 48 kHz (offline) audio with <0.15 ms latency.
    Args:
        wav: [B, Samples] or [Samples] float tensor
        sr: sample rate of wav (e.g. 16000 or 48000)
        n_frames: target number of 10 ms frames
    Returns:
        (energy_db [B, n_frames], hf_energy_db [B, n_frames])
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav_f = wav.float()
    hop = max(1, sr // 100)
    win = hop * 2
    pad = hop // 2
    needed = (n_frames - 1) * hop + win
    w_pad = F.pad(wav_f, (pad, max(0, needed - wav_f.shape[-1] - pad + hop)))
    frames = w_pad.unfold(-1, win, hop)[:, :n_frames, :]
    rms = torch.sqrt(frames.pow(2).mean(dim=-1) + 1e-12)
    energy_db = 20.0 * torch.log10(rms.clamp_min(1e-6))

    # >3 kHz HF energy via STFT
    n_fft = 1024 if sr >= 32000 else 512
    win_stft = min(n_fft, win)
    window = torch.hann_window(win_stft, device=wav.device, dtype=torch.float32)
    spec = torch.stft(
        wav_f, n_fft=n_fft, hop_length=hop, win_length=win_stft, window=window, center=True, return_complex=True
    ).abs()
    hf_bin = int(round(3000.0 * n_fft / float(sr)))
    hf_spec = spec[:, hf_bin:, :n_frames]
    if hf_spec.shape[-1] < n_frames:
        hf_spec = F.pad(hf_spec, (0, n_frames - hf_spec.shape[-1]), mode="replicate")
    scale = float(n_fft) / 2.0
    hf_rms = torch.sqrt((hf_spec / scale).pow(2).mean(dim=1) + 1e-12)
    hf_energy_db = 20.0 * torch.log10(hf_rms.clamp_min(1e-6))
    return energy_db, hf_energy_db


class HSVCPipeline:
    """Complete 48 kHz voice conversion pipeline for HSVC (Harmonic-Spectral Voice Conversion)."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        embedder_name: str = "contentvec",
        rmvpe_path: str = "rvc/models/predictors/rmvpe.pt",
        index_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.speaker_map = HSVCGenerator.from_package(model_path, device=self.device)
        self.sr = self.model.config.sample_rate
        self.hop = self.model.config.hop_in
        self.streamer = HSVCStreamer(self.model, device=self.device)
        self.rmvpe = RMVPE0Predictor(rmvpe_path, device=self.device)
        self.contentvec = load_embedding(embedder_name).to(self.device).eval()

        self.faiss_index = None
        self.faiss_feats = None
        if index_path and os.path.isfile(index_path):
            self.load_index(index_path)

    def load_index(self, index_path: str):
        """Load a FAISS ``.index`` file and cache its feature vectors for retrieval."""
        import faiss

        self.faiss_index = faiss.read_index(index_path)
        self.faiss_feats = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)

    def resolve_speaker(self, speaker: Union[int, str]) -> int:
        """Resolve an integer speaker ID, numeric string, or speaker name in ``speaker_map``."""
        if isinstance(speaker, int):
            return speaker
        if speaker.isdigit():
            return int(speaker)
        if speaker in self.speaker_map:
            return int(self.speaker_map[speaker])
        raise KeyError(f"Unknown speaker '{speaker}'. Available: {list(self.speaker_map.keys())[:20]}...")

    def _apply_index_retrieval(self, c50: np.ndarray, index_rate: float, k: int = 8) -> np.ndarray:
        """Blend 50 Hz ContentVec frames with top-k inverse-distance weighted FAISS neighbors."""
        if self.faiss_index is None or self.faiss_feats is None or index_rate <= 0.0:
            return c50
        npy = np.ascontiguousarray(c50.astype(np.float32))
        score, ix = self.faiss_index.search(npy, k=min(k, self.faiss_index.ntotal))
        weight = np.square(1.0 / np.maximum(score, 1e-6))
        weight /= np.sum(weight, axis=1, keepdims=True)
        retrieved = np.sum(self.faiss_feats[ix] * np.expand_dims(weight, axis=2), axis=1)
        return (index_rate * retrieved + (1.0 - index_rate) * npy).astype(np.float32)

    @torch.inference_mode()
    def extract_features(self, audio: np.ndarray, orig_sr: int, index_rate: float = 0.0):
        """
        Extract aligned 100 Hz ContentVec, RMVPE F0/VUV confidence, and fullband/HF energy curves.

        Args:
            audio (np.ndarray): Input mono or stereo waveform.
            orig_sr (int): Sampling rate of ``audio`` in Hz.
            index_rate (float): FAISS feature retrieval blend factor in [0, 1].

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (content [T, 768], f0 [T], vuv_conf [T], energy_db [T], hf_energy_db [T]).
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        wav_48k = (
            soxr.resample(audio.astype(np.float32), orig_sr, self.sr, quality="HQ")
            if orig_sr != self.sr
            else audio.astype(np.float32)
        )
        n_frames = len(wav_48k) // self.hop
        if n_frames == 0:
            raise ValueError("Input audio is too short (<10 ms).")
        wav_48k = wav_48k[: n_frames * self.hop]
        wav_16k = soxr.resample(wav_48k, self.sr, 16000, quality="HQ").astype(np.float32)

        audio_t = torch.from_numpy(wav_16k).to(self.device).unsqueeze(0)
        mel = self.rmvpe.mel_extractor(audio_t, center=True)
        hidden = self.rmvpe.mel2hidden(mel).squeeze(0).cpu().numpy()
        vuv_conf = np.max(hidden, axis=1).astype(np.float32)
        f0 = self.rmvpe.decode(hidden, thred=0.03).astype(np.float32)

        def _fit(arr):
            if len(arr) >= n_frames:
                return arr[:n_frames]
            return np.pad(arr, (0, n_frames - len(arr)), mode="edge")

        f0 = _fit(f0)
        vuv_conf = _fit(vuv_conf)

        c50 = self.contentvec(audio_t)["last_hidden_state"][0].float().cpu().numpy()
        if index_rate > 0.0 and self.faiss_index is not None:
            c50 = self._apply_index_retrieval(c50, index_rate=index_rate)
        c50_t = torch.from_numpy(c50).T.unsqueeze(0)
        content = F.interpolate(c50_t, size=n_frames, mode="linear", align_corners=False)[0].T.numpy()

        wav_48k_t = torch.from_numpy(wav_48k).to(self.device).unsqueeze(0)
        e_t, hf_t = compute_energy_100hz_torch(wav_48k_t, self.sr, n_frames)
        energy_db = e_t[0].cpu().numpy()
        hf_energy_db = hf_t[0].cpu().numpy()

        return content, f0, vuv_conf, energy_db, hf_energy_db

    @torch.inference_mode()
    def convert_array(
        self,
        audio: np.ndarray,
        orig_sr: int,
        speaker_id: Union[int, str] = 0,
        pitch_shift: float = 0.0,
        index_rate: float = 0.0,
        noise_scale: float = 1.0,
        streaming: bool = False,
        chunk_frames: int = 8,
    ) -> np.ndarray:
        """
        Convert an in-memory waveform to the target speaker at 48 kHz.

        Args:
            audio (np.ndarray): Input mono or stereo audio array.
            orig_sr (int): Sampling rate of ``audio`` in Hz.
            speaker_id (Union[int, str]): Target speaker ID or speaker name.
            pitch_shift (float): Pitch transposition in semitones.
            index_rate (float): FAISS index retrieval blend ratio in [0, 1].
            noise_scale (float): Aperiodic/breath noise multiplier (default: 1.0).
            streaming (bool): If True, run bounded-context phase-continuous streaming.
            chunk_frames (int): Streaming chunk size in 10 ms frames (default: 8 = 80 ms).

        Returns:
            np.ndarray: Clipped 48 kHz mono output waveform.
        """
        sid = self.resolve_speaker(speaker_id)
        content, f0, vuv_conf, energy_db, hf_energy_db = self.extract_features(
            audio, orig_sr, index_rate=index_rate
        )
        if pitch_shift != 0.0:
            f0 = np.where(f0 > 1.0, f0 * (2.0 ** (pitch_shift / 12.0)), 0.0)

        if streaming:
            return self.streamer.stream_full_features(
                content,
                f0,
                vuv_conf,
                energy_db,
                hf_energy_db,
                sid,
                chunk_frames=chunk_frames,
                noise_scale=noise_scale,
            )

        c_t = torch.from_numpy(content).unsqueeze(0).to(self.device)
        f0_t = torch.from_numpy(f0).unsqueeze(0).to(self.device)
        vc_t = torch.from_numpy(vuv_conf).unsqueeze(0).to(self.device)
        e_t = torch.from_numpy(energy_db).unsqueeze(0).to(self.device)
        hf_t = torch.from_numpy(hf_energy_db).unsqueeze(0).to(self.device)
        sid_t = torch.tensor([sid], dtype=torch.long, device=self.device)
        out = self.model(c_t, f0_t, vc_t, e_t, hf_t, sid_t, noise_scale=noise_scale)[0, 0].cpu().float().numpy()
        return np.clip(out, -1.0, 1.0)

    def convert_file(
        self,
        input_path: str,
        output_path: str,
        speaker_id: Union[int, str] = 0,
        pitch_shift: float = 0.0,
        index_rate: float = 0.0,
        noise_scale: float = 1.0,
        streaming: bool = False,
        chunk_frames: int = 8,
    ) -> str:
        wav, sr = sf.read(input_path, dtype="float32")
        out = self.convert_array(
            wav,
            sr,
            speaker_id=speaker_id,
            pitch_shift=pitch_shift,
            index_rate=index_rate,
            noise_scale=noise_scale,
            streaming=streaming,
            chunk_frames=chunk_frames,
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, out, self.sr, subtype="PCM_16")
        return output_path


def main():
    p = argparse.ArgumentParser(description="HSVC 48 kHz Voice Conversion CLI")
    p.add_argument("--model", required=True, help="Path to HSVC .pth package")
    p.add_argument("--input", required=True, help="Input WAV/FLAC path")
    p.add_argument("--output", required=True, help="Output 48 kHz WAV path")
    p.add_argument("--speaker", default="0", help="Target speaker ID or speaker name from model's speaker_map")
    p.add_argument("--pitch", type=float, default=0.0, help="Pitch shift in semitones")
    p.add_argument("--index", default="", help="Optional FAISS .index path")
    p.add_argument("--index-rate", type=float, default=0.0, help="FAISS retrieval blend ratio [0..1]")
    p.add_argument("--noise-scale", type=float, default=1.0, help="Aperiodic/breath noise scale [0..1.5]")
    p.add_argument("--streaming", action="store_true", help="Use bounded-context phase-continuous streaming")
    p.add_argument("--chunk-frames", type=int, default=8, help="Streaming chunk size in 10ms frames (8 = 80ms)")
    p.add_argument("--device", default=None, help="cuda or cpu")
    p.add_argument("--list-speakers", action="store_true", help="Print speaker_map and exit")
    args = p.parse_args()

    pipe = HSVCPipeline(args.model, device=args.device, index_path=args.index or None)
    if args.list_speakers:
        print(f"Speakers ({len(pipe.speaker_map)}):", sorted(pipe.speaker_map.items(), key=lambda kv: kv[1])[:40])
        return
    out = pipe.convert_file(
        args.input,
        args.output,
        speaker_id=args.speaker,
        pitch_shift=args.pitch,
        index_rate=args.index_rate,
        noise_scale=args.noise_scale,
        streaming=args.streaming,
        chunk_frames=args.chunk_frames,
    )
    print(f"Converted -> {out}")


if __name__ == "__main__":
    main()
