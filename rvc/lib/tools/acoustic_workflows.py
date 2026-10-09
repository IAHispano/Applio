"""Shared acoustic workflows for the GUI, CLI and corpus campaigns.

Model and frontend dependencies are loaded inside each operation. Importing
this module never starts preparation, training or inference.
"""

import json
from pathlib import Path

current_script_directory = str(Path(__file__).resolve().parents[3])


def acoustic_project(model_name):
    """Keep training artifacts inside the existing logs/<model> structure."""
    from pathlib import Path

    if (
        not model_name
        or Path(model_name).name != model_name
        or model_name in {".", ".."}
    ):
        raise ValueError("Model name must be one directory name")
    return Path(current_script_directory) / "logs" / model_name


def run_acoustic_preprocess_script(
    model_name,
    dataset_path,
    validation_fraction=0.1,
    segment_seconds=4,
    seed=1234,
    speakers=(),
    recordings_per_speaker=0,
    progress=None,
    base_model=None,
    vocoder_path=None,
):
    """Preprocess V3 audio on CPU; feature models are loaded only by Extract."""
    from rvc.train.acoustic.data import atomic_json, preprocess_audio

    project = acoustic_project(model_name)
    mel = None
    if base_model or vocoder_path:
        from rvc.configs.architectures import inspect_model
        from rvc.configs.neural import MelConfig, require_contract

        metadata = inspect_model(base_model or vocoder_path)
        expected_kind = "acoustic" if base_model else "vocoder"
        if metadata.get("kind") != expected_kind:
            raise ValueError(f"Preprocessing requires a {expected_kind} package")
        mel = MelConfig(**metadata["mel"])
        if vocoder_path and base_model:
            wave = inspect_model(vocoder_path)
            if wave.get("kind") != "vocoder":
                raise ValueError("Choose a universal vocoder package")
            require_contract(metadata["mel"], wave["mel"], "preprocessing voice/vocoder mel")
    result = preprocess_audio(
        dataset_path,
        project / "data",
        validation_fraction,
        segment_seconds,
        seed,
        speaker_names=speakers,
        recordings_per_speaker=int(recordings_per_speaker),
        mel_config=mel,
        progress=progress,
    )
    atomic_json(
        project / "preparation.json",
        dict(
            model_name=model_name,
            dataset_path=str(dataset_path),
            validation_fraction=validation_fraction,
            segment_seconds=segment_seconds,
            seed=seed,
            speakers=list(speakers),
            recordings_per_speaker=int(recordings_per_speaker),
            base_model=str(base_model) if base_model else None,
            vocoder_path=str(vocoder_path) if vocoder_path else None,
        ),
    )
    return str(result)


def run_acoustic_extract_script(
    model_name,
    encoder_path="rvc/models/embedders/contentvec",
    pitch_extractor="swift",
    pitch_path=None,
    profile="bounded",
    device="auto",
    progress=None,
    base_model=None,
):
    """Extract cached content, pitch, energy and mel from preprocessed V3 audio."""
    from rvc.train.extract.features import FeatureExtractor
    from rvc.train.acoustic.data import atomic_json, extract_preprocessed
    from rvc.train.acoustic.trainer import resolve_device
    from rvc.configs.neural import MelConfig, require_contract

    project = acoustic_project(model_name)
    audio = project / "data/audio_manifest.json"
    if not audio.exists():
        raise ValueError(
            "Run Preprocess Dataset for this V3 model before Extract Features"
        )
    prepared = json.loads(audio.read_text(encoding="utf-8"))
    feature_options = dict(mel=MelConfig(**prepared["mel"]), pitch=pitch_extractor, profile=profile)
    if base_model:
        from rvc.configs.architectures import inspect_model
        from dataclasses import asdict

        metadata = inspect_model(base_model)
        if metadata.get("kind") != "acoustic":
            raise ValueError(
                "Choose a V3 pretrained voice model before extracting features"
            )
        features = metadata["features"]
        require_contract(metadata["mel"], prepared["mel"], "pretrained preprocessing mel")
        feature_options = dict(
            mel=MelConfig(**metadata["mel"]),
            layer=features["layer"],
            pitch=features["pitch_method"],
            threshold=features["pitch_threshold"],
            profile=features["profile"],
            context_seconds=features["context_seconds"],
            lookahead_seconds=features["lookahead_seconds"],
            packet_seconds=features["packet_seconds"],
        )
    extractor = FeatureExtractor(
        encoder_path,
        device=resolve_device(device),
        pitch_path=pitch_path or None,
        **feature_options,
    )
    if base_model:
        require_contract(
            asdict(extractor.config), features, "pretrained feature extraction"
        )
    result = extract_preprocessed(audio, extractor, progress=progress)
    atomic_json(
        project / "extraction.json",
        dict(
            model_name=model_name,
            encoder_path=str(encoder_path),
            pitch_extractor=extractor.config.pitch_method,
            pitch_path=str(pitch_path) if pitch_path else None,
            profile=extractor.config.profile,
            device=device,
            base_model=str(base_model) if base_model else None,
        ),
    )
    return str(result)


def run_acoustic_prepare_script(
    model_name,
    dataset_path,
    encoder_path="rvc/models/embedders/contentvec",
    pitch_extractor="swift",
    pitch_path=None,
    profile="bounded",
    device="auto",
    validation_fraction=0.1,
    segment_seconds=4,
    seed=1234,
    speakers=(),
    recordings_per_speaker=0,
    progress=None,
):
    from rvc.train.extract.features import FeatureExtractor
    from rvc.train.acoustic.data import atomic_json, prepare
    from rvc.train.acoustic.trainer import resolve_device

    project = acoustic_project(model_name)
    extractor = FeatureExtractor(
        encoder_path,
        device=resolve_device(device),
        pitch=pitch_extractor,
        pitch_path=pitch_path or None,
        profile=profile,
    )
    result = prepare(
        dataset_path,
        project / "data",
        extractor,
        validation_fraction,
        segment_seconds,
        seed,
        speaker_names=speakers,
        recordings_per_speaker=int(recordings_per_speaker),
        progress=progress,
    )
    atomic_json(
        project / "preparation.json",
        dict(
            model_name=model_name,
            dataset_path=dataset_path,
            encoder_path=encoder_path,
            pitch_extractor=pitch_extractor,
            pitch_path=pitch_path,
            profile=profile,
            device=device,
            validation_fraction=validation_fraction,
            segment_seconds=segment_seconds,
            seed=seed,
            speakers=list(speakers),
            recordings_per_speaker=int(recordings_per_speaker),
        ),
    )
    return str(result)


def run_acoustic_train_script(
    model_name,
    stage="predictor",
    manifest=None,
    output_dir=None,
    base_model=None,
    config=None,
    **kwargs,
):
    from pathlib import Path

    from rvc.configs.neural import AcousticConfig, VocoderConfig
    from rvc.train.acoustic.trainer import train

    project = acoustic_project(model_name)
    kind = "vocoder" if stage == "vocoder" else "acoustic"
    if config:
        constructor = VocoderConfig if kind == "vocoder" else AcousticConfig
        config = constructor(**json.loads(Path(config).read_text(encoding="utf-8")))
    dataset = Path(manifest) if manifest else project / "data/manifest.json"
    if not dataset.exists():
        raise ValueError("Run Preprocess Dataset and Extract Features before training")
    yield from train(
        dataset,
        output_dir or project / "checkpoints" / stage,
        kind=kind,
        phase="predictor" if kind == "vocoder" else stage,
        pretrained=base_model or None,
        config=config,
        **kwargs,
    )


def run_acoustic_train_all_script(
    model_name,
    refine=False,
    steps=10000,
    manifest=None,
    output_dir=None,
    base_model=None,
    resume=None,
    config=None,
    **kwargs,
):
    """Train and export a usable voice in one action with the shared recipe.

    Predictor and vocoder are required. Flow and shortcut are optional and run
    in dependency order. Unlike individual-stage training's additional budget,
    steps here is the target per part: restarting the pipeline resumes last.pt
    and skips completed parts. Stop never starts the next part or exports an
    unfinished pipeline. Stage checkpoints remain available for advanced use.
    """
    import os
    from pathlib import Path

    import psutil

    from rvc.train.process.checkpoints import load_payload

    if base_model or resume or config:
        raise ValueError(
            "Use an individual-stage mode for custom base weights, resume paths or architecture overrides. Complete model resumes its own saved checkpoints automatically."
        )
    if int(steps) < 1:
        raise ValueError("Training duration must be positive")
    if kwargs.get("waveform_weight") or kwargs.get("waveform_adversarial_weight"):
        raise ValueError(
            "Use an individual predictor/adaptation stage for waveform supervision"
        )
    project = acoustic_project(model_name)
    dataset = Path(manifest) if manifest else project / "data/manifest.json"
    if not dataset.exists():
        raise ValueError("Run Preprocess Dataset and Extract Features before training")
    dataset_id = json.loads(dataset.read_text(encoding="utf-8"))["dataset_id"]
    campaign_path = project / "campaign_status.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(campaign["pid"])
            live = (
                process.is_running() and process.create_time() <= campaign["updated_at"]
            )
        except (psutil.Error, KeyError):
            live = False
        if live and process.pid != os.getpid():
            raise ValueError(
                "This model already has a background training job. Follow it in Live training progress or use a different model name."
            )
    root = Path(output_dir) if output_dir else project / "checkpoints"
    stages = ["predictor", "vocoder"] + (["flow", "shortcut"] if refine else [])
    kwargs.setdefault("checkpoint_every", 1000)
    stop = kwargs.get("stop_requested")
    acoustic_stage = "predictor"
    for part, stage in enumerate(stages, 1):
        if stop and stop():
            return
        checkpoint = root / stage / "last.pt"
        initial = 0
        if checkpoint.exists():
            payload = load_payload(checkpoint)
            if payload.get("dataset_id") != dataset_id or payload.get("phase") != stage:
                raise ValueError(
                    f"Saved {stage} progress belongs to another dataset or stage. Use a new model name."
                )
            initial = int(payload["step"])
            del payload
        base = None
        if stage in {"flow", "shortcut"}:
            base = root / acoustic_stage / "best.pt"
        if initial >= int(steps):
            yield {
                "status": "skipped",
                "phase": stage,
                "step": initial,
                "part": part,
                "parts": len(stages),
            }
        else:
            stage_options = dict(kwargs)
            if stage != "predictor":
                stage_options.pop("mel_detail_weight", None)
                stage_options.pop("mel_adversarial_weight", None)
                stage_options.pop("pitch_guidance", None)
            for update in run_acoustic_train_script(
                model_name,
                stage,
                manifest=str(dataset),
                output_dir=str(root / stage),
                base_model=str(base) if base and not checkpoint.exists() else None,
                resume=str(checkpoint) if checkpoint.exists() else None,
                steps=int(steps) - initial,
                **stage_options,
            ):
                yield dict(update, phase=stage, part=part, parts=len(stages))
                if update.get("stopped"):
                    return
        if stage != "vocoder":
            acoustic_stage = stage
    if stop and stop():
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return
    acoustic = project / f"{model_name}_acoustic.pth"
    vocoder = project / f"{model_name}_vocoder.pth"
    run_acoustic_export_script(str(root / acoustic_stage / "best.pt"), str(acoustic))
    run_acoustic_export_script(str(root / "vocoder" / "best.pt"), str(vocoder))
    yield {"status": "exported", "acoustic": str(acoustic), "vocoder": str(vocoder)}


def run_acoustic_finetune_script(
    model_name, base_model, vocoder_path, steps=10000, **kwargs
):
    """Adapt only the voice model, retaining the frozen shared vocoder.

    Resume uses this project's adapter checkpoint; export merges the adapters
    into a standalone acoustic package. The selected vocoder is referenced,
    never copied, optimized or overwritten by voice fine-tuning.
    """
    from pathlib import Path
    from rvc.train.process.checkpoints import load_payload
    from rvc.configs.neural import require_contract
    from rvc.train.acoustic.data import atomic_json

    if not base_model or not vocoder_path:
        raise ValueError(
            "Choose a pretrained voice model and universal vocoder in Model Settings first"
        )
    if int(steps) < 1:
        raise ValueError("Training duration must be positive")
    base, vocoder = load_payload(base_model), load_payload(vocoder_path)
    if base["kind"] != "acoustic" or vocoder["kind"] != "vocoder":
        raise ValueError("Choose a V3 voice model and its universal vocoder")
    if base.get("adapters"):
        raise ValueError(
            "Select an exported voice model with merged adapters as the pretrained base"
        )
    require_contract(base["mel"], vocoder["mel"], "pretrained voice/vocoder")
    project = acoustic_project(model_name)
    destination = project / f"{model_name}_acoustic.pth"
    if (
        Path(base_model).resolve() == destination.resolve()
        or Path(vocoder_path).resolve() == destination.resolve()
    ):
        raise ValueError(
            "Use a different model name to preserve the pretrained weights"
        )
    manifest = project / "data/manifest.json"
    campaign = project / "campaign_status.json"
    if campaign.exists():
        import psutil

        state = json.loads(campaign.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(state["pid"])
            running = (
                process.is_running() and process.create_time() <= state["updated_at"]
            )
        except (psutil.Error, KeyError):
            running = False
        if running:
            raise ValueError(
                "This project has an active background training job. Use a new model name for fine-tuning"
            )
    if not manifest.exists():
        raise ValueError(
            "Run Preprocess Dataset and Extract Features before fine-tuning"
        )
    from rvc.train.acoustic.data import AcousticDataset

    dataset = AcousticDataset(manifest)
    require_contract(
        base["features"],
        dataset.manifest["contract"]["features"],
        "pretrained features; extract again using the selected pretrained model",
    )
    require_contract(base["mel"], dataset.manifest["contract"]["mel"], "pretrained mel")
    checkpoint = project / "checkpoints/adapt/last.pt"
    from rvc.train.extract.features import file_hash

    recipe = dict(
        base_model=str(Path(base_model).resolve()),
        vocoder=str(Path(vocoder_path).resolve()),
        dataset_id=dataset.manifest["dataset_id"],
        base_hash=file_hash(Path(base_model)),
        vocoder_hash=file_hash(Path(vocoder_path)),
    )
    settings = project / "finetuning.json"
    if checkpoint.exists():
        previous = (
            json.loads(settings.read_text(encoding="utf-8"))
            if settings.exists()
            else {}
        )
        # The vocoder is independent of adaptation. Another compatible vocoder
        # may be selected without invalidating the acoustic optimizer state.
        if any(previous.get(key) != recipe[key] for key in ("base_hash", "dataset_id")):
            raise ValueError(
                "Saved fine-tuning uses another pretrained model or dataset. Use a new model name"
            )
        initial = int(load_payload(checkpoint)["step"])
    else:
        initial = 0
    # Do not overwrite a selected pretrained checkpoint through a reused name.
    if Path(base_model).resolve().is_relative_to(
        (project / "checkpoints").resolve()
    ) or Path(vocoder_path).resolve().is_relative_to(
        (project / "checkpoints/adapt").resolve()
    ):
        raise ValueError("Use a new model name for the voice you want to fine-tune")
    atomic_json(settings, recipe)
    del base, vocoder, dataset
    kwargs["adaptation"] = "lora"
    if initial < int(steps):
        for update in run_acoustic_train_script(
            model_name,
            "adapt",
            base_model=None if checkpoint.exists() else base_model,
            resume=str(checkpoint) if checkpoint.exists() else None,
            steps=int(steps) - initial,
            **kwargs,
        ):
            yield update
            if update.get("stopped"):
                return
    if kwargs.get("stop_requested") and kwargs["stop_requested"]():
        return
    run_acoustic_export_script(
        str(project / "checkpoints/adapt/best.pt"), str(destination)
    )
    yield dict(status="exported", acoustic=str(destination), vocoder=str(vocoder_path))


def run_acoustic_infer_script(
    input_path,
    output_path,
    pth_path,
    vocoder_path,
    encoder_path="rvc/models/embedders/contentvec",
    pitch_path=None,
    sid=0,
    pitch=0,
    refinement_steps=0,
    seed=0,
    device="auto",
    ordinary_flow=False,
    batch=False,
):
    from rvc.infer.acoustic import Converter

    if not vocoder_path:
        raise ValueError("Applio v3 requires a separate --vocoder-path")
    converter = Converter(
        pth_path, vocoder_path, encoder_path, pitch_path or None, device
    )
    options = dict(
        speaker=int(sid),
        semitones=float(pitch),
        steps=int(refinement_steps),
        seed=int(seed),
        ordinary=ordinary_flow,
    )
    if not batch:
        return converter.convert_file(input_path, output_path, **options)
    return converter.convert_directory(input_path, output_path, **options)


def run_acoustic_export_script(checkpoint, output_path):
    from rvc.train.process.checkpoints import export_checkpoint

    return str(export_checkpoint(checkpoint, output_path))


def run_acoustic_evaluate_script(
    manifest, pth_path, vocoder_path, output_dir, **kwargs
):
    from rvc.train.process.evaluation import evaluate

    return evaluate(manifest, pth_path, vocoder_path, output_dir, **kwargs)
