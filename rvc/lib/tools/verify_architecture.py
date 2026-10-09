"""Hardware smoke verification, explicitly separate from perceptual evaluation."""

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rvc.configs.neural import DEFAULT_BATCH_SIZE, AcousticConfig, VocoderConfig
from rvc.infer.acoustic import Converter, LiveConverter
from rvc.realtime.streaming import FeatureStream
from rvc.train.extract.features import FeatureExtractor
from rvc.train.process.checkpoints import export_checkpoint
from rvc.train.acoustic.data import atomic_json, prepare
from rvc.train.acoustic.trainer import resolve_device, train


def verify(encoder, output="logs/v3-verification", device="auto", full_size=True):
    import soundfile as sf

    device = resolve_device(device)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    recordings = root / "recordings"
    recordings.mkdir(exist_ok=True)
    sample_rate = 44100
    t = np.arange(round(1.5 * sample_rate)) / sample_rate
    for i in range(3):
        sf.write(
            recordings / f"signal{i}.wav",
            0.08 * np.sin(2 * np.pi * (220 + 40 * i) * t),
            sample_rate,
            subtype="FLOAT",
        )
    extractor = FeatureExtractor(encoder, device=device, profile="bounded")
    source = (0.08 * np.sin(2 * np.pi * 220 * np.arange(70001) / sample_rate)).astype(
        np.float32
    )
    before = time.perf_counter()
    expected = extractor.extract(source)
    frontend = FeatureStream(extractor)
    packets = []
    for offset in range(0, len(source), 1777):
        packets.extend(frontend.push(source[offset : offset + 1777]))
    packets.extend(frontend.push(np.empty(0, np.float32), final=True))
    differences = {}
    for key in packets[0]:
        actual = np.concatenate([p[key] for p in packets])
        if actual.shape != expected[key].shape:
            raise AssertionError(f"Frontend frame count differs for {key}")
        differences[key] = float(np.abs(actual - expected[key]).max())
        np.testing.assert_allclose(actual, expected[key], rtol=1e-5, atol=1e-5)
    frontend_seconds = time.perf_counter() - before
    manifest = prepare(recordings, root / "prepared", extractor, segment_seconds=2)
    extractor.encoder.to("cpu")
    extractor.mel.to("cpu")
    extractor.device = torch.device("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    acoustic = (
        AcousticConfig()
        if full_size
        else AcousticConfig(
            condition_width=32,
            predictor_width=32,
            refiner_width=32,
            predictor_depth=2,
            refiner_depth=2,
        )
    )
    vocoder = VocoderConfig() if full_size else VocoderConfig(channels=8, depth=2)
    stages = []
    for name, kind, phase, pretrained, config in (
        ("predictor", "acoustic", "predictor", None, acoustic),
        ("flow", "acoustic", "flow", root / "predictor/last.pt", None),
        ("shortcut", "acoustic", "shortcut", root / "flow/last.pt", None),
        ("adapt", "acoustic", "adapt", root / "shortcut/last.pt", None),
        ("vocoder", "vocoder", "predictor", None, vocoder),
    ):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        before = time.perf_counter()
        result = list(
            train(
                manifest,
                root / name,
                kind=kind,
                phase=phase,
                steps=1,
                batch_size=DEFAULT_BATCH_SIZE,
                crop_frames=128,
                device=str(device),
                precision="auto",
                checkpoint_every=1,
                pretrained=pretrained,
                config=config,
            )
        )[-1]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        result.update(stage=name, elapsed_seconds=time.perf_counter() - before)
        stages.append(result)
        print(json.dumps(result), flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    export_checkpoint(root / "adapt/last.pt", root / "acoustic.pt")
    export_checkpoint(root / "vocoder/last.pt", root / "vocoder.pt")
    # Restore extraction onto the inference device and share the same pinned encoder.
    extractor.encoder.to(device)
    extractor.mel.to(device)
    extractor.device = device
    converter = Converter(
        root / "acoustic.pt",
        root / "vocoder.pt",
        encoder,
        device=str(device),
        extractor=extractor,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = time.perf_counter()
    converted = converter.convert(source, steps=4, seed=13)
    elapsed = time.perf_counter() - before
    inference_memory = (
        {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        if device.type == "cuda"
        else {}
    )
    live = LiveConverter(converter, steps=4, seed=13)
    pieces = [live.push(source[i : i + 1777]) for i in range(0, len(source), 1777)] + [
        live.flush()
    ]
    streamed = np.concatenate(pieces)
    np.testing.assert_allclose(streamed, converted, atol=1e-4, rtol=1e-4)
    assert len(converted) == len(source) and np.isfinite(converted).all()
    sf.write(
        root / "untrained_smoke_output.wav", converted, sample_rate, subtype="FLOAT"
    )
    report = {
        "purpose": "Functional synthetic-signal smoke test; no audio-quality conclusion",
        "torch": str(torch.__version__),
        "device": str(device),
        "full_size": full_size,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "acoustic_config": asdict(acoustic),
        "vocoder_config": asdict(vocoder),
        "stages": stages,
        "frontend_max_difference": differences,
        "frontend_two_pass_seconds": frontend_seconds,
        "conversion_seconds": elapsed,
        "audio_seconds": len(source) / sample_rate,
        "conversion_rtf": elapsed / (len(source) / sample_rate),
        "stream_max_difference": float(np.abs(streamed - converted).max()),
        "live_latency": live.latency,
        "inference_memory": inference_memory,
        "parameters": {
            "acoustic": sum(p.numel() for p in converter.acoustic.parameters()),
            "vocoder": sum(p.numel() for p in converter.vocoder.parameters()),
            "encoder": sum(p.numel() for p in extractor.encoder.parameters()),
        },
    }
    atomic_json(root / "report.json", report)
    return report


def verify_contracts(output):
    """CPU-only synthetic regression checks, without encoders or released weights.

    Uses a new output directory so existing reports and fixtures are preserved.
    This checks contracts and exact continuation, not perceptual quality.
    """
    from unittest.mock import patch

    import soundfile as sf
    from torch import nn

    from rvc.configs.architectures import BACKEND, FORMAT_VERSION, compatible_vocoders
    from rvc.configs.neural import FeatureConfig, MelConfig
    from rvc.lib.algorithm.acoustic.nsf_hifigan import NSFHiFiGANVocoder
    from rvc.lib.algorithm.acoustic.spectral import MelExtractor
    from rvc.lib.algorithm.acoustic.vocoder import SpectralVocoder
    from rvc.train.acoustic.data import extract_preprocessed, preprocess_audio
    from rvc.train.process.checkpoints import atomic_save, load_payload

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    mel = MelConfig(sample_rate=8192, n_fft=256, hop_length=64,
                    n_mels=16, fmin=0, fmax=4096)

    class SyntheticExtractor:
        config = FeatureConfig(encoder_id="synthetic", encoder_hash="synthetic",
                               content_dim=4, pitch_id="synthetic")
        mel_config = mel

        def extract(self, audio):
            with torch.no_grad():
                mels = MelExtractor(mel)(torch.from_numpy(audio)[None])[0].T.numpy()
            frames = len(mels)
            values = {key: np.zeros(frames, dtype=np.float32) for key in
                      ("f0", "observed_f0", "voiced", "confidence",
                       "confidence_valid", "energy")}
            values.update(mel=mels, content=np.ones((frames, 4), dtype=np.float32))
            return values

    source = root / "source" / "speaker"
    source.mkdir(parents=True)
    for index in range(3):
        t = np.arange(8192, dtype=np.float32) / 8192
        sf.write(source / f"{index}.wav", .1 * np.sin(2 * np.pi * (180 + index * 20) * t),
                 mel.sample_rate, subtype="FLOAT")
    manifest_path = extract_preprocessed(
        preprocess_audio(source.parent, root / "data", mel_config=mel, seed=11),
        SyntheticExtractor(),
    )
    config = AcousticConfig(content_dim=4, mel_dim=16, speakers=1, condition_width=16,
                            predictor_width=16, refiner_width=16, predictor_depth=2,
                            refiner_depth=2, kernel_size=3, causal=False,
                            prediction_centered=False, family="shallow-flow")
    settings = dict(config=config, batch_size=2, crop_frames=64, device="cpu",
                    precision="fp32", seed=17, checkpoint_every=1)
    vocoder_config = VocoderConfig(sample_rate=8192, hop_length=64, mel_dim=16,
                                   streams=2, n_fft=128, channels=4, depth=1)
    torch.manual_seed(17)
    vocoder = SpectralVocoder(vocoder_config)
    renderer = root / "synthetic_vocoder.pth"
    atomic_save(renderer, dict(backend=BACKEND, format_version=FORMAT_VERSION,
                              kind="vocoder", vocoder_backend="spectral", mel=asdict(mel),
                              model_config=asdict(vocoder_config), weights=vocoder.state_dict()))
    checks = {}
    for name, objectives in (
        ("mel_adversarial", dict(mel_adversarial_weight=.01)),
        ("waveform", dict(waveform_weight=.01, spectral_vocoder=str(renderer))),
    ):
        uninterrupted, resumed = root / name / "whole", root / name / "resumed"
        list(train(manifest_path, uninterrupted, steps=2, **settings, **objectives))
        list(train(manifest_path, resumed, steps=1, **settings, **objectives))
        list(train(manifest_path, resumed, steps=1, resume=resumed / "last.pt",
                   **settings, **objectives))
        left, right = load_payload(uninterrupted / "last.pt"), load_payload(resumed / "last.pt")

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return torch.equal(a, b)
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b

        for key in ("weights", "ema", "optimizer", "rank_rng", "critics", "critic_optimizer"):
            if key in left and not equal(left[key], right[key]):
                raise AssertionError(f"{name}: continuation differs in {key}")
        changed = dict(objectives)
        changed[f"{name}_weight"] = .02
        try:
            list(train(manifest_path, resumed, steps=1, resume=resumed / "last.pt",
                       **settings, **changed))
        except ValueError as error:
            if "unchanged training settings" not in str(error):
                raise
        else:
            raise AssertionError("Changed resume objective was accepted")
        checks[name + "_exact_resume"] = True

    # Exercise the wrapper boundary without allocating the full NSF generator.
    class Capture(nn.Module):
        def forward(self, mels, pitch):
            self.pitch = pitch.clone()
            return mels[:, :1]

    wrapper = NSFHiFiGANVocoder.__new__(NSFHiFiGANVocoder)
    nn.Module.__init__(wrapper)
    wrapper.generator = Capture()
    f0, voiced = torch.tensor([[220., 220., 220.]]), torch.tensor([[1., 0., 1.]])
    for policy, expected in (("continuous", f0),
                             ("legacy-zero-unvoiced", torch.tensor([[220., 0., 220.]]))):
        wrapper.f0_policy = policy
        wrapper(torch.zeros(1, 128, 3), f0, voiced)
        if not torch.equal(wrapper.generator.pitch, expected):
            raise AssertionError("NSF pitch policy mismatch")
    checks["nsf_pitch_policies"] = True
    metadata = dict(backend=BACKEND, kind="acoustic", mel=asdict(mel))
    with patch("rvc.configs.architectures.inspect_model", side_effect=[
        metadata, dict(kind="vocoder", mel=dict(asdict(mel), unknown_key=1)),
        dict(kind="vocoder", mel=asdict(mel)),
    ]):
        if compatible_vocoders("voice", [("bad", "bad"), ("good", "good")]) != [("good", "good")]:
            raise AssertionError("Unknown mel fields were accepted")
    checks["unknown_mel_fields_rejected"] = True
    report = dict(passed=True, checks=checks, device="cpu", quality_claim=False)
    atomic_json(root / "report.json", report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic CPU contract verification")
    parser.add_argument("--contracts-only", action="store_true", required=True)
    parser.add_argument("--output", required=True, help="New directory for fixtures and report")
    arguments = parser.parse_args()
    print(json.dumps(verify_contracts(arguments.output), indent=2))
