"""CPU-only evaluation planning and paired summaries, separate from training.

Plans bind listening cases to original held-out recording hashes. Summaries
compare the same conditions across refinement budgets and bootstrap by speaker
so multiple clips from one voice do not pretend to be independent speakers.
Neither utility loads a model, modifies caches or schedules a training job.
"""

import json
from pathlib import Path

import numpy as np

from rvc.configs.neural import fingerprint
from rvc.train.extract.features import file_hash
from rvc.train.acoustic.data import atomic_json


def prepare_plan(audio_manifest, output, segments_per_speaker=2):
    path = Path(audio_manifest).resolve()
    if Path(output).resolve() == path:
        raise ValueError("Evaluation output must differ from the input manifest")
    data = json.loads(path.read_text(encoding="utf-8"))
    identity = {k: v for k, v in data.items() if k != "audio_id"}
    if data.get("schema") != 1 or fingerprint(identity) != data.get("audio_id"):
        raise ValueError("Invalid audio manifest identity")
    if segments_per_speaker < 1:
        raise ValueError("Select at least one held-out segment per speaker")
    train = {r["source_hash"] for r in data["recordings"] if r["split"] == "train"}
    groups = {}
    for record in data["recordings"]:
        if record["split"] == "validation":
            if record["source_hash"] in train:
                raise ValueError("Recording leakage across evaluation and training")
            groups.setdefault(record["speaker"], []).append(record)
    if not groups:
        raise ValueError("No held-out recordings")
    root = Path(data["source_root"]).resolve()
    if Path(output).resolve().is_relative_to(root):
        raise ValueError("Save evaluation plans outside the original dataset")
    # Match the shared evaluator's deterministic round-robin segment order.
    segment_groups = {}
    for index, segment in enumerate(data["segments"]):
        if segment["split"] == "validation":
            segment_groups.setdefault(segment["speaker"], []).append((index, segment))
    selected = []
    verified = set()
    for offset in range(segments_per_speaker):
        for speaker, segments in segment_groups.items():
            if offset >= len(segments):
                continue
            index, segment = segments[offset]
            source = (root / segment["source"]).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Evaluation source must stay inside the dataset")
            if segment["source_hash"] not in verified:
                if file_hash(source) != segment["source_hash"]:
                    raise ValueError(
                        "Evaluation source differs from the prepared recording"
                    )
                verified.add(segment["source_hash"])
            selected.append(
                dict(
                    source=segment["source"],
                    source_sha256=segment["source_hash"],
                    speaker_id=speaker,
                    speaker=data["speakers"][speaker],
                    segment_index=index,
                    start_sample=segment["start_sample"],
                )
            )
    plan = dict(
        schema=1,
        preprocessing_id=data["audio_id"],
        seed=1234,
        selection="Deterministic round-robin across validation speakers, matching core.py evaluate",
        evaluator_limit=len(selected),
        held_out=selected,
        speakers=len(groups),
        mel=data["mel"],
        refinement_budgets=[0, 1, 2, 4, 8, 16],
        controls="Same exported acoustic weights, frozen vocoder, source, target speaker and seed at every budget. Skip unsupported budgets.",
        conversion_cases=[
            "Unseen source voice to several target speakers (keep sources outside training)",
            "Pitch shifts -12, -5, 0, +5, +12 semitones on the same sources",
            "30-second recording, short tail, silence-only and speech separated by silence",
            "Repeat with the same seed; verify finite output, duration and sample rate",
        ],
        quality_checks=[
            "Reference transcript WER/CER when transcripts exist; ASR agreement otherwise, clearly labeled",
            "Target speaker similarity against separate target references, alongside source similarity",
            "Voiced pitch cents error and voiced/unvoiced agreement; report unsupported cases",
            "Blinded listening for intelligibility, identity, artifacts and chunk boundaries",
        ],
        integrity="Hash exported voice and frozen vocoder before/after checks. Never evaluate a checkpoint while it is being written.",
        scope="A protocol, not completed quality evidence. Reconstruction alone does not establish voice conversion quality.",
    )
    atomic_json(output, plan)
    return plan


def summarize_report(report_path, output, seed=1234):
    path = Path(report_path)
    if Path(output).resolve() == path.resolve():
        raise ValueError("Summary output must differ from the original report")
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in report["rows"] if r["budget"] != "ceiling"]
    metrics = ("mel_l1", "waveform_mel_l1", "spectral_loss", "rtf")
    groups = {}
    for row in rows:
        key = (row["source"], row["start_sample"], row["speaker"])
        budget = row["budget"]
        if not isinstance(budget, int) or budget not in {0, 1, 2, 4, 8, 16, 32}:
            raise ValueError("Unsupported report budget")
        group = groups.setdefault(budget, {})
        if key in group:
            raise ValueError("Duplicate evaluation condition within a budget")
        if any(not np.isfinite(row[m]) or row[m] < 0 for m in metrics):
            raise ValueError("Invalid evaluation metric")
        group[key] = row
    baseline = groups.get(0)
    if not baseline:
        raise ValueError("A predictor (budget 0) baseline is required")
    summary = {}
    for budget, cases in sorted(groups.items()):
        if cases.keys() != baseline.keys():
            raise ValueError(
                "Budgets must contain exactly the same evaluation conditions"
            )
        summary[str(budget)] = {}
        for metric in metrics:
            before = np.array([baseline[k][metric] for k in baseline])
            after = np.array([cases[k][metric] for k in baseline])
            change = after - before
            by_speaker = {}
            for key, delta in zip(baseline, change):
                by_speaker.setdefault(key[2], []).append(delta)
            means = np.array([np.mean(v) for v in by_speaker.values()])
            rng = np.random.default_rng(seed)
            bootstrap = means[rng.integers(len(means), size=(2000, len(means)))].mean(
                axis=1
            )
            summary[str(budget)][metric] = dict(
                predictor_mean=float(before.mean()),
                candidate_mean=float(after.mean()),
                paired_mean_change=float(change.mean()),
                improved_cases=int((change < 0).sum()),
                cases=len(cases),
                speakers=len(means),
                equal_speaker_mean_change=float(means.mean()),
                speaker_bootstrap_95=np.quantile(bootstrap, [0.025, 0.975]).tolist()
                if len(means) > 1
                else None,
            )
    best = min(
        groups, key=lambda b: summary[str(b)]["waveform_mel_l1"]["candidate_mean"]
    )
    result = dict(
        schema=1,
        report_sha256=file_hash(path),
        dataset_id=report.get("dataset_id"),
        task=report.get("task"),
        refinement=summary,
        lowest_waveform_mel_error_budget=best,
        scope="Best measured reconstruction error is not a perceptual or speaker-identity recommendation. Bootstrap describes the sampled speakers only. RTF follows the original report's timing scope and is not an end-to-end latency measurement.",
    )
    atomic_json(output, result)
    return result
