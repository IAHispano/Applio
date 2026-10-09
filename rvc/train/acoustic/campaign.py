"""Resumable full-corpus acoustic training with a frozen compatible vocoder.

This orchestrates the existing preprocess, extractor, trainer and evaluator;
it introduces no new architecture or GPU-specific training preset. Architecture
references supply dimensions/frontend semantics only, never training weights.
"""

import gc
import json
import os
import shutil
import signal
import sys
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event

from rvc.configs.neural import AcousticConfig, MelConfig, fingerprint, require_contract
from rvc.train.extract.features import file_hash
from rvc.train.process.checkpoints import load_payload
from rvc.train.process.console import ConsoleProgress
from rvc.train.acoustic.data import atomic_json, preprocess_audio


class CampaignStopped(Exception):
    """A graceful user stop; rerunning resumes from the saved checkpoint."""


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        if not self.log.closed:
            self.log.write(value)
            self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def __getattr__(self, name):
        # Model loaders/tqdm also inspect isatty, encoding and fileno.
        return getattr(self.terminal, name)


@contextmanager
def campaign_lock(project):
    """OS-owned lock is released on crashes; stale files never block resume."""
    lock = (project / "campaign.lock").open("a+b")
    try:
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(
                "This campaign is already running in another window"
            ) from error
        yield
    finally:
        lock.close()


@contextmanager
def keep_awake():
    """Prevent Windows sleep for this worker only; restore its state on exit."""
    previous = None
    if os.name == "nt":
        import ctypes

        previous = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if previous:
            ctypes.windll.kernel32.SetThreadExecutionState(previous)


def run_corpus(
    model_name,
    dataset_path,
    vocoder_path,
    architecture_from=None,
    encoder_path="rvc/models/embedders/contentvec",
    batch_size=2,
    accumulation_steps=1,
    predictor_steps=60000,
    flow_steps=None,
    shortcut_steps=None,
    device="auto",
    precision="auto",
    checkpoint_every=1000,
    seed=1234,
    check_only=False,
    crop_frames=None,
    compact_cache=False,
    compact_cache_audit=None,
    sampling_mode="segments",
    validation_limit=0,
    validation_manifest=None,
):
    """Preflight now; train only when invoked without --check-only.

    Stage targets are total optimizer updates, not additional resume budgets.
    The fixed plan rejects changed datasets/settings on resume. Ctrl+C or the
    STOP file requests a checkpoint before exit; clicking the launcher resumes.
    """
    import torch
    import soundfile as sf
    import core
    from rvc.train.acoustic.trainer import resolve_device, resolve_precision

    project = core.acoustic_project(model_name)
    root = Path(dataset_path).resolve()
    vocoder_path = Path(vocoder_path).resolve()
    if not root.is_dir() or project.resolve().is_relative_to(root):
        raise ValueError("Dataset must exist and training output must be outside it")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac"})
    if not files:
        raise ValueError("Dataset contains no WAV or FLAC recordings")
    corpus_map = root / "corpus.json"
    corpus_speakers = None
    if corpus_map.exists():
        corpus = json.loads(corpus_map.read_text(encoding="utf-8"))
        if corpus.get("version") != 1:
            raise ValueError("Unsupported combined corpus index")
        corpus_speakers = corpus["speakers"]
        available = {path.relative_to(root).as_posix(): path for path in files}
        if set(corpus_speakers) - available.keys():
            raise ValueError("Combined corpus files missing; reindex before training")
        files = [available[name] for name in sorted(corpus_speakers)]
    groups = {}
    inventory = []
    for path in files:
        relative = path.relative_to(root)
        identity = (
            corpus_speakers[relative.as_posix()]
            if corpus_speakers is not None
            else str(relative.parent)
        )
        groups.setdefault(identity, []).append(path)
        stat = path.stat()
        inventory.append((relative.as_posix(), stat.st_size, stat.st_mtime_ns))
    # One identity per directory. Nested style folders are not speaker identities.
    if corpus_speakers is None and any(len(Path(name).parts) != 1 for name in groups):
        raise ValueError(
            "Use one direct dataset folder per speaker; do not pass nested style/channel trees"
        )
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("Each speaker needs at least two recordings for validation")
    target_device = resolve_device(device)
    resolved_precision = resolve_precision(target_device, precision)
    reference = load_payload(architecture_from) if architecture_from else None
    if reference and reference["kind"] != "acoustic":
        raise ValueError("Architecture reference must be an acoustic model")
    config = (
        AcousticConfig(**reference["model_config"]) if reference else AcousticConfig()
    )
    config = replace(config, speakers=len(groups))
    joint = config.family == "shallow-flow"
    flow_steps = (0 if joint else 15000) if flow_steps is None else flow_steps
    shortcut_steps = (
        (0 if joint else 15000) if shortcut_steps is None else shortcut_steps
    )
    crop_frames = (256 if joint else 128) if crop_frames is None else crop_frames
    if crop_frames < 1:
        raise ValueError("Training crop must be positive")
    if shortcut_steps and not flow_steps:
        raise ValueError("Shortcut training requires a flow budget")
    if config.family == "shallow-flow" and (flow_steps or shortcut_steps):
        raise ValueError("Joint shallow-flow uses --flow-steps 0 --shortcut-steps 0")
    vocoder = load_payload(vocoder_path)
    if vocoder["kind"] != "vocoder" or vocoder.get(
        "vocoder_backend", "spectral"
    ) not in {"spectral", "bigvgan-v2", "wavehax", "nsf-hifigan"}:
        raise ValueError("This campaign requires a supported frozen vocoder package")
    # Scratch acoustics learn physical targets for the selected frozen renderer.
    # An architecture reference still binds its own trained spectral semantics.
    mel = MelConfig(**(reference["mel"] if reference else vocoder["mel"]))
    require_contract(asdict(mel), vocoder["mel"], "campaign acoustic/vocoder mel")
    vocoder_hash = file_hash(vocoder_path)
    del vocoder
    # Estimate from each speaker's source encoding. VCTK uses uncompressed PCM.
    # FLAC size cannot establish duration, so inspect every compressed header.
    seconds = 0.0
    for group in groups.values():
        wav = [p for p in group if p.suffix.lower() == ".wav"]
        if wav:
            sample = sf.info(wav[0])
            bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}.get(
                sample.subtype
            )
            if not bits:
                raise ValueError("Unsupported source WAV encoding for disk estimation")
            seconds += sum(p.stat().st_size for p in wav) / (
                sample.samplerate * sample.channels * bits / 8
            )
        seconds += sum(
            sf.info(p).duration for p in group if p.suffix.lower() == ".flac"
        )
    estimate = (
        int(
            seconds
            * (
                mel.sample_rate * 4
                + (config.content_dim + mel.n_mels + 5)
                * 4
                * mel.sample_rate
                / mel.hop_length
            )
        )
        + 6 * 1024**3
    )
    from rvc.train.acoustic.storage import storage_contract
    storage = storage_contract(compact_cache)
    if compact_cache_audit and not compact_cache:
        raise ValueError("A compact storage audit requires --compact-cache")
    if compact_cache:
        # Raw PCM24/FP16 upper estimate, unless a validated, contract-matched
        # corpus-specific compression audit supplies a measured margin.
        estimate = int(seconds * (mel.sample_rate * 3 +
            (config.content_dim * 2 + (mel.n_mels + 6) * 4) * mel.sample_rate / mel.hop_length)) + 12 * 1024**3
        if compact_cache_audit:
            audit_path = Path(compact_cache_audit).resolve()
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            integrity = json.loads((audit_path.parent / "integrity.json").read_text(encoding="utf-8"))
            audited_corpus_path = root / "corpus.json"
            if not audit.get("passed") or not integrity.get("passed") or not audited_corpus_path.is_file():
                raise ValueError("Compact storage needs a passed corpus-specific qualification")
            if audit["corpus_sha256"] != file_hash(audited_corpus_path) or audit["corpus_recordings"] != len(files) or audit["storage_contract"] != storage:
                raise ValueError("Compact storage audit does not match this corpus/cache")
            require_contract(audit["mel_contract"], asdict(mel), "compact audit mel")
            if not reference:
                raise ValueError("A measured compact audit requires a frontend/architecture reference")
            require_contract(audit["feature_contract"], reference["features"], "compact audit frontend")
            for name, digest in integrity["immutable_sha256"].items():
                # Check the actual storage implementation used by qualification.
                if name.replace("\\", "/").startswith("rvc/train/acoustic/") and file_hash(Path(core.current_script_directory) / name) != digest:
                    raise ValueError("Compact cache implementation changed after qualification")
            estimate = int(audit["conservative_required_bytes"])
    existing = sum(
        p.stat().st_size for p in (project / "data").rglob("*") if p.is_file()
    )
    free = shutil.disk_usage(project.parent).free
    needed = max(0, estimate - existing)
    reserved = []
    if architecture_from:
        prior_manifest = Path(architecture_from).resolve().parent / "data/manifest.json"
        if prior_manifest.exists():
            prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
            reserved = sorted(
                r["source"] for r in prior["recordings"] if r["split"] == "validation"
            )
            missing = [name for name in reserved if not (root / name).is_file()]
            if missing:
                raise ValueError(
                    "Prior validation recordings are missing from this corpus"
                )
    if validation_manifest:
        prior = json.loads(Path(validation_manifest).read_text(encoding="utf-8"))
        for record in prior["recordings"]:
            if record["split"] == "validation":
                path = (root / record["source"]).resolve()
                if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != record["source_hash"]:
                    raise ValueError("Reserved evaluation recording changed or is absent")
                reserved.append(record["source"])
    if reserved and corpus_map.exists() and corpus.get("groups"):
        # A reserved singer utterance reserves its entire indexed song-folder
        # group, retaining the corpus split semantics after overriding the seed.
        split_groups = corpus["groups"]
        units = {split_groups[Path(name).as_posix()] for name in reserved}
        reserved.extend(name for name, unit in split_groups.items() if unit in units)
    reserved = sorted(set(reserved))
    plan = dict(
        dataset_path=str(root),
        inventory_id=fingerprint(inventory),
        speakers=sorted(groups),
        recordings=len(files),
        architecture=asdict(config),
        features=reference["features"] if reference else None,
        mel=asdict(mel),
        encoder_path=str(Path(encoder_path).resolve()),
        vocoder_path=str(vocoder_path),
        vocoder_sha256=vocoder_hash,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        precision=resolved_precision,
        stages=dict(
            predictor=predictor_steps, flow=flow_steps, shortcut=shortcut_steps
        ),
        crop_frames=crop_frames,
        learning_rate=2e-4,
        checkpoint_every=checkpoint_every,
        seed=seed,
        segment_seconds=4,
        validation_fraction=0.1,
        reserved_validation_sources=reserved,
        normalize_overflow=True,
        initialization="scratch; architecture/frontend reference only",
    )
    if storage is not None:
        plan["storage"] = storage
        plan["compact_cache_audit_sha256"] = file_hash(compact_cache_audit) if compact_cache_audit else None
    if sampling_mode not in {"segments", "domain-speaker"}:
        raise ValueError("Unknown corpus sampling mode")
    if sampling_mode != "segments":
        plan["sampling_mode"] = sampling_mode
        plan["sampling_version"] = 1
    if validation_limit < 0:
        raise ValueError("Validation limit cannot be negative")
    if validation_limit:
        plan["validation_limit"] = validation_limit
        plan["validation_selection_version"] = 1
    if validation_manifest:
        plan["validation_manifest_sha256"] = file_hash(validation_manifest)
    print(
        f"Full corpus: {len(groups)} speakers, {len(files):,} recordings; no recording cap.",
        flush=True,
    )
    print(
        f"Device: {target_device}; precision: {resolved_precision}; batch: {batch_size}.",
        flush=True,
    )
    print(
        f"Disk: {free / 1024**3:.1f} GiB free; approximately {needed / 1024**3:.1f} GiB more required (conservative).",
        flush=True,
    )
    print(
        f"Frozen vocoder; stages: {plan['stages']}; reserved prior holdouts: {len(reserved)}.",
        flush=True,
    )
    if free < needed:
        raise ValueError(
            "Insufficient disk space for the full cache and checkpoint reserve"
        )
    if not Path(encoder_path).is_dir():
        raise ValueError("Local content encoder is missing; install/download it first")
    project.mkdir(parents=True, exist_ok=True)
    saved = project / "campaign_plan.json"
    if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != plan:
        raise ValueError(
            "Campaign dataset/settings changed; restore them or choose a new model name"
        )
    if check_only:
        print("Preflight passed. No preprocessing or training was started.", flush=True)
        return dict(
            speakers=len(groups),
            recordings=len(files),
            estimated_additional_bytes=needed,
            family=config.family,
            stages=plan["stages"],
            crop_frames=crop_frames,
        )
    with (
        campaign_lock(project),
        keep_awake(),
        (project / "console.log").open("a", encoding="utf-8") as log,
    ):
        with (
            redirect_stdout(Tee(sys.stdout, log)),
            redirect_stderr(Tee(sys.stderr, log)),
            ConsoleProgress(model_name) as console,
        ):
            if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != plan:
                raise ValueError("Campaign settings changed while acquiring its lock")
            atomic_json(saved, plan)
            atomic_json(project / "acoustic_config.json", plan["architecture"])
            stop = Event()
            stop_file = project / "STOP"
            stop_file.unlink(missing_ok=True)

            def request_stop(signum, frame):
                if stop.is_set():
                    raise KeyboardInterrupt
                print(
                    "\nStopping after the current operation; saving a checkpoint...",
                    flush=True,
                )
                stop.set()

            previous_handler = signal.signal(signal.SIGINT, request_stop)
            started = time.monotonic()

            def stopped():
                return stop.is_set() or stop_file.exists()

            def status(campaign_phase, **details):
                details.pop("phase", None)
                atomic_json(
                    project / "campaign_status.json",
                    dict(
                        pid=os.getpid(),
                        phase=campaign_phase,
                        updated_at=time.time(),
                        elapsed_seconds=time.monotonic() - started,
                        **details,
                    ),
                )

            def progress(phase):
                previous_operation = None
                last_status = 0.0

                def update(item):
                    nonlocal previous_operation, last_status
                    if stopped():
                        raise CampaignStopped()
                    now = time.monotonic()
                    operation = item.get("operation", phase)
                    if (
                        operation != previous_operation
                        or item["recording"] == item["total"]
                        or now - last_status >= 0.5
                    ):
                        status(phase, **item)
                        last_status = now
                        previous_operation = operation
                    console.preparation(item)

                return update

            try:
                status("preprocess")
                preprocess_audio(
                    root,
                    project / "data",
                    mel_config=mel,
                    seed=seed,
                    normalize_overflow=True,
                    validation_sources=reserved,
                    progress=progress("preprocess"),
                    compact_cache=compact_cache,
                )
                console.close()
                status("extract")
                print(
                    "Extracting features: loading the content encoder and pitch model…",
                    flush=True,
                )
                core.run_acoustic_extract_script(
                    model_name,
                    encoder_path=encoder_path,
                    device=str(target_device),
                    base_model=architecture_from,
                    progress=progress("extract"),
                )
                console.close()
                gc.collect()
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
                base = None
                exports = []
                for stage, budget in plan["stages"].items():
                    if not budget:
                        continue
                    if stopped():
                        raise CampaignStopped()
                    directory = project / "checkpoints" / stage
                    last = directory / "last.pt"
                    initial = int(load_payload(last)["step"]) if last.exists() else 0
                    if initial > budget:
                        raise ValueError("Checkpoint exceeds the planned stage target")
                    if initial < budget:
                        status(stage, step=initial, target=budget)
                        print(
                            f"Initializing {stage} and validating cached data; the first update may take time.",
                            flush=True,
                        )
                        last_training_status = 0.0
                        for item in core.run_acoustic_train_script(
                            model_name,
                            stage,
                            steps=budget - initial,
                            resume=str(last) if initial else None,
                            base_model=str(base) if not initial and base else None,
                            config=str(project / "acoustic_config.json")
                            if stage == "predictor"
                            else None,
                            batch_size=batch_size,
                            accumulation_steps=accumulation_steps,
                            precision=resolved_precision,
                            device=str(target_device),
                            seed=seed,
                            crop_frames=crop_frames,
                            learning_rate=2e-4,
                            checkpoint_every=checkpoint_every,
                            stop_requested=stopped,
                            sampling_mode=sampling_mode,
                            validation_limit=validation_limit,
                        ):
                            now = time.monotonic()
                            # Console/TensorBoard still receive every update; the
                            # UI status snapshot needs at most two writes/second.
                            if (
                                now - last_training_status >= 0.5
                                or item.get("stopped")
                                or "validation_mel_l1" in item
                                or item.get("step") == budget
                            ):
                                status(stage, target=budget, **item)
                                last_training_status = now
                            console.training(item, budget)
                            if item.get("stopped"):
                                raise CampaignStopped()
                    console.close()
                    base = directory / "best.pt"
                    export = project / f"{model_name}_{stage}.pth"
                    core.run_acoustic_export_script(str(base), str(export))
                    exports.append(str(export))
                    status("evaluation", stage=stage)
                    core.run_acoustic_evaluate_script(
                        str(project / "data/manifest.json"),
                        str(export),
                        str(vocoder_path),
                        str(project / "evaluation" / stage / "matched"),
                        budgets=[0, 1, 2, 4, 8, 16, 32]
                        if joint
                        else [0]
                        if stage == "predictor"
                        else [0, 8]
                        if stage == "flow"
                        else [0, 1, 4],
                        limit=len(groups),
                        device=str(target_device),
                        seed=seed,
                    )
                    from rvc.train.process.tensorboard import sync_evaluation_audio

                    sync_evaluation_audio(project)
                    gc.collect()
                    if target_device.type == "cuda":
                        torch.cuda.empty_cache()
                if file_hash(vocoder_path) != vocoder_hash:
                    raise ValueError("Frozen vocoder integrity check failed")
                status("complete", exports=exports, vocoder_unchanged=True)
                print(
                    f"Finished. Models, curves and listening samples: {project}",
                    flush=True,
                )
                return dict(exports=exports)
            except CampaignStopped:
                status("stopped", reason="User stop; rerun the same launcher to resume")
                print("Stopped safely. Click the same launcher to resume.", flush=True)
            except BaseException as error:
                status("failed", error=repr(error))
                raise
            finally:
                signal.signal(signal.SIGINT, previous_handler)
