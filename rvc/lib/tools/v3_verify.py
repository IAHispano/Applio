"""Hardware smoke verification, explicitly separate from perceptual evaluation."""

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rvc.configs.v3 import DEFAULT_BATCH_SIZE, AcousticConfig, VocoderConfig
from rvc.infer.v3 import Converter, LiveConverter
from rvc.realtime.v3_streaming import FeatureStream
from rvc.train.extract.v3 import FeatureExtractor
from rvc.train.process.v3_checkpoints import export_checkpoint
from rvc.train.v3.data import atomic_json, prepare
from rvc.train.v3.trainer import resolve_device, train


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
