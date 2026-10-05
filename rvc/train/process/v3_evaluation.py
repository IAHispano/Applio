"""Held-out self-reconstruction and reference-mel vocoder diagnosis.

Evaluate identical cached conditions across explicit refinement budgets. The
ceiling render uses reference mel with the same vocoder to separate waveform
synthesis limitations from acoustic error. This diagnostic ceiling is not a
guaranteed lower bound on paired L1: smoothing can reduce error without improving
perceptual fidelity. Cached-condition RTF excludes frontend and file I/O.
"""

import csv
import time
from pathlib import Path

import torch

from rvc.configs.v3 import require_contract
from rvc.lib.algorithm.v3.acoustic import masked_mean
from rvc.lib.algorithm.v3.spectral import MelExtractor, spectral_loss
from rvc.realtime.v3_streaming import coordinate_noise
from rvc.train.process.v3_checkpoints import construct, load_payload
from rvc.train.v3.data import AcousticDataset, atomic_json, collate, condition_batch
from rvc.train.v3.trainer import resolve_device, to_device


@torch.inference_mode()
def evaluate(
    manifest,
    acoustic,
    vocoder,
    output,
    budgets=(0, 2, 4, 8, 16),
    device="auto",
    seed=1234,
    limit=0,
):
    import soundfile as sf

    device = resolve_device(device)
    data = AcousticDataset(manifest, "validation", 0)
    a, v = load_payload(acoustic), load_payload(vocoder)
    if a["kind"] != "acoustic" or v["kind"] != "vocoder":
        raise ValueError("Evaluation requires acoustic and vocoder packages")
    require_contract(a["mel"], v["mel"], "evaluation acoustic/vocoder mel")
    require_contract(a["mel"], data.manifest["contract"]["mel"], "evaluation data mel")
    require_contract(
        a["features"], data.manifest["contract"]["features"], "evaluation data features"
    )
    if a["speakers"] != data.manifest["speakers"]:
        raise ValueError(
            "Held-out reconstruction requires the package's target speaker mapping"
        )
    acoustic_model, vocoder_model = (
        construct(a, device).eval(),
        construct(v, device).eval(),
    )
    c = vocoder_model.config
    if (c.sample_rate, c.hop_length, c.mel_dim) != (
        data.config.sample_rate,
        data.config.hop_length,
        data.config.n_mels,
    ):
        raise ValueError("Vocoder dimensions disagree with the evaluation contract")
    for budget in budgets:
        if budget not in {0, 1, 2, 4, 8, 16, 32}:
            raise ValueError("Unsupported evaluation budget")
        if budget and not bool(acoustic_model.flow_trained):
            raise ValueError(
                "Evaluate a predictor checkpoint with --budgets 0; flow training is required for refinement"
            )
        if 0 < budget < 8 and not bool(acoustic_model.shortcut_trained):
            raise ValueError(
                "Few-step evaluation requires a shortcut-trained checkpoint"
            )
    mel_extractor = MelExtractor(data.config).to(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    # A limited evaluation should not silently contain only the first speaker.
    groups = {}
    for index, entry in enumerate(data.entries):
        groups.setdefault(entry["speaker"], []).append(index)
    selected = []
    for offset in range(max(len(group) for group in groups.values())):
        for group in groups.values():
            if offset < len(group):
                selected.append(group[offset])
    selected = selected[: min(limit or len(data), len(data))]
    for i, data_index in enumerate(selected):
        batch = to_device(collate([data[data_index]]), device)
        condition = condition_batch(acoustic_model, batch)
        frames = batch["mel"].shape[-1]
        noise = torch.from_numpy(
            coordinate_noise(frames, data.config.n_mels, seed=seed + i).T.copy()
        ).to(device)[None]
        prior_noise = torch.from_numpy(
            coordinate_noise(
                frames * data.config.hop_length, seed=seed + i + 1
            ).T.copy()
        ).to(device)
        real = batch["waveform"]
        for budget in ("ceiling", *budgets):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            before = time.perf_counter()
            mel = (
                batch["mel"]
                if budget == "ceiling"
                else acoustic_model.sample(condition, int(budget), noise=noise)
            )
            waveform = (
                vocoder_model(mel, batch["f0"], batch["voiced"], noise=prior_noise)
                * batch["waveform_mask"]
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - before
            row = {
                "example": i,
                "dataset_segment": data_index,
                "source": data.entries[data_index]["source"],
                "start_sample": data.entries[data_index]["start_sample"],
                "speaker": int(batch["speaker"].item()),
                "budget": budget,
                "mel_l1": float(masked_mean((mel - batch["mel"]).abs(), batch["mask"])),
                "waveform_mel_l1": float(
                    masked_mean(
                        (mel_extractor(waveform) - batch["mel"]).abs(), batch["mask"]
                    )
                ),
                "spectral_loss": float(spectral_loss(waveform, real)),
                "synthesis_seconds": elapsed,
                "audio_seconds": int(batch["waveform_length"][0])
                / data.config.sample_rate,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            }
            row["rtf"] = elapsed / row["audio_seconds"]
            rows.append(row)
            length = int(batch["waveform_length"][0])
            sf.write(
                str(output / f"example_{i:04d}_budget_{budget}.wav"),
                waveform[0, :length].cpu().numpy(),
                data.config.sample_rate,
                subtype="FLOAT",
            )
        sf.write(
            str(output / f"example_{i:04d}_reference.wav"),
            real[0, :length].cpu().numpy(),
            data.config.sample_rate,
            subtype="FLOAT",
        )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "vocoder_backend": v.get("vocoder_backend", "spectral"),
        "acoustic_step": a.get("step", 0),
        "acoustic_phase": a.get("phase", "acoustic"),
        "task": "held-out self-reconstruction",
        "scope": "Not a cross-speaker identity or blinded quality evaluation",
        "dataset_id": data.manifest["dataset_id"],
        "seed": seed,
        "budgets": list(budgets),
        "selection": "Deterministic round-robin across validation speakers",
        "selected_segments": selected,
        "device": str(device),
        "timing_scope": "Acoustic sampling and vocoder; condition encoding/frontend and file I/O excluded",
        "rows": rows,
    }
    atomic_json(output / "report.json", report)
    return report
