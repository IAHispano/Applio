"""Reproducible small-data VCTK learning experiment using the integrated services."""

import json
import shutil
import time
from pathlib import Path

import torch

from rvc.configs.v3 import DEFAULT_BATCH_SIZE


def run_experiment(
    model_name,
    predictor_steps=1000,
    vocoder_steps=2000,
    flow_steps=500,
    shortcut_steps=500,
    limit=8,
    device="cuda",
):
    from core import (
        run_v3_evaluate_script,
        run_v3_export_script,
        run_v3_infer_script,
        run_v3_train_script,
        v3_project,
    )
    from rvc.train.v3.data import atomic_json

    torch.set_num_threads(4)
    project = v3_project(model_name)
    manifest = project / "data/manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    settings = dict(
        batch_size=DEFAULT_BATCH_SIZE,
        crop_frames=128,
        precision="auto",
        device=device,
        seed=1234,
        checkpoint_every=100,
    )
    before = time.perf_counter()
    history = []

    def train_stage(stage, steps, base=None, resume=None):
        start = time.perf_counter()
        for update in run_v3_train_script(
            model_name, stage, steps=steps, base_model=base, resume=resume, **settings
        ):
            if (
                update["step"] == 1
                or update["step"] % 25 == 0
                or "validation_mel_l1" in update
            ):
                print(json.dumps(update), flush=True)
        history.append(
            dict(stage=stage, updates=steps, wall_seconds=time.perf_counter() - start)
        )
        return project / "checkpoints" / stage / "last.pt"

    if any(
        (project / "checkpoints" / stage / "last.pt").exists()
        for stage in ("predictor", "vocoder", "flow", "shortcut")
    ):
        raise ValueError(
            "Choose a new experiment model name to preserve existing checkpoints"
        )
    initial_predictor = train_stage("predictor", 1)
    baseline = project / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    shutil.copy2(initial_predictor, baseline / "predictor.pt")
    initial_vocoder = train_stage("vocoder", 1)
    shutil.copy2(initial_vocoder, baseline / "vocoder.pt")
    initial_report = run_v3_evaluate_script(
        str(manifest),
        str(initial_predictor),
        str(initial_vocoder),
        str(project / "evaluation/initial"),
        budgets=[0],
        limit=limit,
        device=device,
    )
    predictor = train_stage(
        "predictor", predictor_steps - 1, resume=str(initial_predictor)
    )
    vocoder = train_stage("vocoder", vocoder_steps - 1, resume=str(initial_vocoder))
    predictor_report = run_v3_evaluate_script(
        str(manifest),
        str(predictor),
        str(vocoder),
        str(project / "evaluation/predictor"),
        budgets=[0],
        limit=limit,
        device=device,
    )
    flow = train_stage("flow", flow_steps, base=str(predictor))
    shortcut = train_stage("shortcut", shortcut_steps, base=str(flow))
    final_report = run_v3_evaluate_script(
        str(manifest),
        str(shortcut),
        str(vocoder),
        str(project / "evaluation/final"),
        budgets=[0, 2, 4, 8, 16],
        limit=limit,
        device=device,
    )
    acoustic_export = run_v3_export_script(
        str(shortcut), str(project / (model_name + "_acoustic.pth"))
    )
    vocoder_export = run_v3_export_script(
        str(vocoder), str(project / (model_name + "_vocoder.pth"))
    )
    heldout = next(r for r in data["recordings"] if r["split"] == "validation")
    source = Path(data["source_root"]) / heldout["source"]
    conversions = []
    for speaker in (
        heldout["speaker"],
        (heldout["speaker"] + 1) % len(data["speakers"]),
    ):
        output = (
            project / "evaluation" / f"conversion_to_{data['speakers'][speaker]}.wav"
        )
        conversions.append(
            run_v3_infer_script(
                str(source),
                str(output),
                acoustic_export,
                vocoder_export,
                sid=speaker,
                refinement_steps=4,
                device=device,
            )
        )

    def aggregate(report):
        result = {}
        for budget in sorted({str(row["budget"]) for row in report["rows"]}):
            rows = [row for row in report["rows"] if str(row["budget"]) == budget]
            result[budget] = {
                key: sum(row[key] for row in rows) / len(rows)
                for key in ("mel_l1", "waveform_mel_l1", "spectral_loss", "rtf")
            }
        return result

    report = dict(
        dataset_id=data["dataset_id"],
        speakers=data["speakers"],
        recordings=len(data["recordings"]),
        segments=len(data["segments"]),
        split_recordings={
            split: sum(r["split"] == split for r in data["recordings"])
            for split in ("train", "validation")
        },
        audio_minutes=sum(e["samples"] for e in data["segments"]) / 44100 / 60,
        history=history,
        wall_seconds=time.perf_counter() - before,
        initial=aggregate(initial_report),
        predictor=aggregate(predictor_report),
        final=aggregate(final_report),
        exports=[acoustic_export, vocoder_export],
        conversions=conversions,
        limitations="Small four-speaker learning test; no MOS, speaker-similarity or superiority claim. Cached-feature timings exclude the frontend. End-to-end conversions use the real encoder.",
    )
    atomic_json(project / "experiment_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report
