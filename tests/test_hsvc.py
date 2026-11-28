"""Comprehensive Verification & Spectral Artifact Forensic Unit Tests for HSVC."""

import json
import os
import tempfile

import librosa
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from rvc.infer.hsvc_pipeline import compute_energy_100hz_torch
from rvc.lib.algorithm.hsvc import (
    DualDomainDiscriminator,
    HardenedSpectralSpeakerCritic,
    HSVCConfig,
    HSVCGenerator,
    HSVCLoss,
    HSVCStreamer,
    UtteranceSpeakerAdversary,
    is_hsvc_checkpoint,
)


def measure_stationary_comb_db(wav: np.ndarray, sr: int = 48000, comb_hz: float = 200.0) -> float:
    """Forensic detector for stationary STFT frame-rate / ConvTranspose horizontal comb lines."""
    S = np.abs(librosa.stft(wav.astype(np.float32), n_fft=8192, hop_length=480))
    S_db = 20.0 * np.log10(np.maximum(S, 1e-7))
    mean_spec = S_db.mean(axis=1)
    hp = mean_spec - uniform_filter1d(mean_spec, size=15)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
    peaks = []
    for k in range(1, 11):
        f_t = k * comb_hz
        idx = int(np.argmin(np.abs(freqs - f_t)))
        peaks.append(float(hp[idx]))
    return float(np.mean(peaks))


def test_shapes_and_configs():
    for cfg_path in ("rvc/configs/hsvc/48000.json", "rvc/configs/hsvc/48000_base.json"):
        with open(cfg_path) as f:
            cfg = HSVCConfig.from_dict(json.load(f)["model"])
        assert cfg.use_spec_res is False, f"{cfg_path} must disable unmodulated spec_res!"
        G = HSVCGenerator(cfg).eval()
        B, T = 2, 40
        c = torch.randn(B, T, 768)
        f0 = torch.full((B, T), 260.0)
        vc = torch.ones(B, T)
        e = torch.full((B, T), -20.0)
        hf = torch.full((B, T), -30.0)
        sid = torch.tensor([0, 1])
        with torch.no_grad():
            y, z, turns = G(c, f0, vc, e, hf, sid, return_bottleneck=True)
        assert y.shape == (B, 1, T * 480)
        assert z.shape == (B, T, cfg.bottleneck_dim)
        assert torch.isfinite(y).all()
        print(f"  [OK] {cfg_path}: G={sum(p.numel() for p in G.parameters())/1e6:.2f}M, out={tuple(y.shape)}")


def test_silence_gate_and_zero_200hz_comb():
    torch.manual_seed(42)
    with open("rvc/configs/hsvc/48000_base.json") as f:
        cfg = HSVCConfig.from_dict(json.load(f)["model"])
    G = HSVCGenerator(cfg).eval()
    T = 300
    c = torch.randn(1, T, 768)
    # Continuous logarithmic chirp 140->440 Hz so no real vocal harmonic sits on a fixed 200Hz line
    f0 = torch.from_numpy(np.geomspace(140.0, 440.0, T).astype(np.float32)).unsqueeze(0)
    vc = torch.ones(1, T)
    e = torch.full((1, T), -20.0)
    hf = torch.full((1, T), -30.0)
    sid = torch.tensor([0])
    with torch.no_grad():
        y_sing = G(c, f0, vc, e, hf, sid)[0, 0].numpy()
    comb_200 = measure_stationary_comb_db(y_sing, 48000, 200.0)
    assert comb_200 < 1.5, f"200 Hz stationary STFT comb artifact detected: {comb_200:+.2f} dB!"

    # Pure silence check
    f0_s = torch.zeros(1, 50)
    vc_s = torch.zeros(1, 50)
    e_s = torch.full((1, 50), -95.0)
    hf_s = torch.full((1, 50), -95.0)
    with torch.no_grad():
        y_sil = G(c[:, :50], f0_s, vc_s, e_s, hf_s, sid)[0, 0].numpy()
    sil_db = 20.0 * np.log10(np.sqrt(np.mean(y_sil**2)) + 1e-12)
    assert sil_db < -80.0, f"Silence leak too high: {sil_db:.1f} dBFS"
    print(f"  [OK] Silence floor={sil_db:.1f} dBFS | 200 Hz comb={comb_200:+.2f} dB (clean!)")


def test_streaming_and_fp16_package_autodetect():
    with open("rvc/configs/hsvc/48000_base.json") as f:
        cfg = HSVCConfig.from_dict(json.load(f)["model"])
    G = HSVCGenerator(cfg).eval()
    T = 80
    c = np.random.randn(T, 768).astype(np.float32)
    t_sec = np.arange(T) * 0.01
    f0 = (260.0 + 25.0 * np.sin(2.0 * np.pi * 5.0 * t_sec)).astype(np.float32)
    vc = np.ones(T, dtype=np.float32)
    e = np.full(T, -20.0, dtype=np.float32)
    hf = np.full(T, -30.0, dtype=np.float32)
    st = HSVCStreamer(G, context_frames=40, lookahead_frames=2, crossfade_samples=240, device="cpu")
    wav_s = st.stream_full_features(c, f0, vc, e, hf, sid=0, chunk_frames=8)
    assert wav_s.shape == (T * 480,) and np.isfinite(wav_s).all()

    with tempfile.TemporaryDirectory() as td:
        p32 = os.path.join(td, "hsvc_fp32.pth")
        p16 = os.path.join(td, "hsvc_fp16.pth")
        G.export_package(p32, speaker_map={"demo": 0}, fp16=False)
        G.export_package(p16, speaker_map={"demo": 0}, fp16=True)
        ck16 = torch.load(p16, map_location="cpu", weights_only=True)
        assert is_hsvc_checkpoint(ck16)
        assert not is_hsvc_checkpoint({"config": [1025, 32], "weight": {}})
        G16, smap = HSVCGenerator.from_package(p16, device="cpu")
        assert smap == {"demo": 0}
        with torch.no_grad():
            torch.manual_seed(0)
            y32 = G(torch.from_numpy(c[:20]).unsqueeze(0), torch.from_numpy(f0[:20]).unsqueeze(0), torch.from_numpy(vc[:20]).unsqueeze(0), torch.from_numpy(e[:20]).unsqueeze(0), torch.from_numpy(hf[:20]).unsqueeze(0), torch.tensor([0]))
            torch.manual_seed(0)
            y16 = G16(torch.from_numpy(c[:20]).unsqueeze(0), torch.from_numpy(f0[:20]).unsqueeze(0), torch.from_numpy(vc[:20]).unsqueeze(0), torch.from_numpy(e[:20]).unsqueeze(0), torch.from_numpy(hf[:20]).unsqueeze(0), torch.tensor([0]))
        err = float((y32 - y16).abs().max())
        assert err < 1e-2, f"FP16 drift too large: {err}"
        print(f"  [OK] Streaming + FP16 package ({os.path.getsize(p16)/1e6:.1f} MB vs {os.path.getsize(p32)/1e6:.1f} MB FP32, max_err={err:.5f})")


def test_training_modules_and_torch_energy():
    e16, hf16 = compute_energy_100hz_torch(torch.randn(1, 16000) * 0.1, 16000, 100)
    e48, hf48 = compute_energy_100hz_torch(torch.randn(1, 48000) * 0.1, 48000, 100)
    assert e16.shape == (1, 100) and e48.shape == (1, 100) and hf16.shape == (1, 100) and hf48.shape == (1, 100)
    D = DualDomainDiscriminator()
    L = HSVCLoss()
    A = UtteranceSpeakerAdversary(256, 16)
    C = HardenedSpectralSpeakerCritic(16)
    y = torch.randn(2, 1, 480 * 40) * 0.1
    yh = torch.randn(2, 1, 480 * 40) * 0.1
    assert len(D(y, yh)[0]) == 8
    ls = L(y, yh, torch.full((2, 40), -20.0))
    assert all(torch.isfinite(v) for v in ls.values())
    assert A(torch.randn(2, 40, 256), torch.ones(2, 40)).shape == (2, 16)
    assert C(y, torch.ones(2, 40), apply_specaug=True)[0].shape == (2, 16)
    print("  [OK] DualDomainDiscriminator, HSVCLoss, Adversary, Critic & torch energy extractor")


def test_multi_sample_rate_adaptation():
    with open("rvc/configs/hsvc/48000_base.json") as f:
        cfg = HSVCConfig.from_dict(json.load(f)["model"])
    G = HSVCGenerator(cfg).eval()
    B, T = 1, 40
    c = torch.randn(B, T, 768)
    f0 = torch.full((B, T), 260.0)
    vc = torch.ones(B, T)
    e = torch.full((B, T), -20.0)
    hf = torch.full((B, T), -30.0)
    sid = torch.tensor([0])
    with tempfile.TemporaryDirectory() as td:
        for sr in (44100, 40000, 32000, 24000, 48000):
            G.adapt_sample_rate(sr)
            with torch.no_grad():
                y = G(c, f0, vc, e, hf, sid)
            expected_len = T * G.config.hop_in
            assert y.shape == (B, 1, expected_len) and torch.isfinite(y).all()
            loss_fn = HSVCLoss(sr=sr, hop_in=G.config.hop_in)
            ls = loss_fn(y, y, e)
            assert all(torch.isfinite(v) for v in ls.values())
            p = os.path.join(td, f"hsvc_{sr}.pth")
            G.export_package(p, speaker_map={"demo": 0})
            G_re, _ = HSVCGenerator.from_package(p, device="cpu")
            assert G_re.config.sample_rate == sr and G_re.config.hop_in == G.config.hop_in
    print("  [OK] Universal 48k -> 44.1k / 40k / 32k / 24k adaptation + export/reload")


if __name__ == "__main__":
    test_shapes_and_configs()
    test_silence_gate_and_zero_200hz_comb()
    test_streaming_and_fp16_package_autodetect()
    test_training_modules_and_torch_energy()
    test_multi_sample_rate_adaptation()
    print("ALL HSVC VERIFICATION TESTS PASSED!")
