"""Architecture-dependent controls inside the existing Training workflow.

Keep widget/service routing here; learning objectives belong in the trainer.
Both architectures use separate audio preprocessing and feature extraction. Session-local stop events avoid one browser cancelling
another session. The existing output and buttons serve both architectures.
"""

import inspect
import json
import threading
import time
from pathlib import Path

import gradio as gr

import core

_training_events = {}
_training_signature = inspect.signature(core.run_train_script)


def pretrained_choices(kind):
    from rvc.configs.architectures import discover_models, inspect_model

    return [
        (Path(path).stem, path)
        for _, path in discover_models(kind=kind)
        if inspect_model(path).get("inference_only")
    ]


def workflow_options():
    """Beginner choices live in Model Settings, before dataset preparation."""
    with gr.Column(visible=False) as settings:
        workflow = gr.Radio(
            [
                ("Fine-tune a pretrained model (LoRA)", "finetune"),
                ("Train from scratch", "scratch"),
                ("Advanced stage training", "advanced"),
            ],
            value="finetune",
            label="How would you like to train?",
            info="Fine-tuning learns your voice using a pretrained acoustic model and its frozen audio renderer.",
        )
        with gr.Row() as pretrained:
            base = gr.Dropdown(
                pretrained_choices("acoustic"),
                value=None,
                label="Pretrained voice model",
                allow_custom_value=True,
                info="Choose exported V3 pretrained weights.",
            )
            vocoder = gr.Dropdown(
                pretrained_choices("vocoder"),
                value=None,
                label="Audio renderer (vocoder)",
                allow_custom_value=True,
                info="Reused for inference; it will not be trained.",
            )
        refresh = gr.Button("Refresh pretrained models")
        status = gr.Markdown(
            "Select a pretrained voice model and vocoder to fine-tune. If you do not have pretrained weights, choose Train from scratch."
        )
        workflow.change(
            lambda mode: (
                gr.update(visible=mode == "finetune"),
                gr.update(visible=mode == "finetune"),
            ),
            workflow,
            [pretrained, refresh],
        )
        refresh.click(
            lambda: (
                gr.update(choices=pretrained_choices("acoustic")),
                gr.update(choices=pretrained_choices("vocoder")),
            ),
            outputs=[base, vocoder],
        )

        def select_vocoders(path, current):
            from rvc.configs.architectures import compatible_vocoders, preferred_vocoder

            if not path:
                return gr.update(choices=pretrained_choices("vocoder"), value=None)
            try:
                choices = compatible_vocoders(path, pretrained_choices("vocoder"))
                return gr.update(
                    choices=choices, value=preferred_vocoder(path, choices, current)
                )
            except (ValueError, OSError, KeyError) as error:
                gr.Warning(str(error))
                return gr.update(choices=[], value=None)

        def pair_status(mode, voice, wave):
            if mode != "finetune":
                return ""
            if not voice:
                return "Choose a pretrained V3 voice model to begin."
            if not wave:
                return "Choose a compatible universal vocoder. Only the voice model will be fine-tuned."
            try:
                from rvc.configs.architectures import compatible_vocoders

                if not compatible_vocoders(voice, [("selected", wave)]):
                    return "These models have different mel settings. Choose a compatible vocoder."
            except (ValueError, OSError, KeyError) as error:
                return str(error)
            return "Ready: preprocess your recordings, extract features, then Train Model. Training fits LoRA adapters and your speaker embedding; the vocoder stays frozen."

        base.change(select_vocoders, [base, vocoder], vocoder)
        for control in (workflow, base, vocoder):
            control.change(pair_status, [workflow, base, vocoder], status)
    return settings, [workflow, base, vocoder]


def preparation_options():
    with gr.Column(visible=False) as settings:
        with gr.Accordion("Dataset selection and audio settings", open=False):
            speakers = gr.Textbox(
                label="Speaker folders",
                info="Optional comma-separated subfolders, one voice per folder. Leave empty to include every speaker.",
            )
            cap = gr.Number(
                value=0,
                minimum=0,
                precision=0,
                label="Recordings per speaker",
                info="0 includes all recordings. Set a limit for a smaller initial experiment.",
            )
            with gr.Row():
                fraction = gr.Number(
                    value=0.1,
                    minimum=0.001,
                    maximum=0.999,
                    label="Validation fraction",
                    info="0.1 reserves 10% of recordings for evaluation. A recording never appears in both splits.",
                )
                segment = gr.Number(
                    value=4,
                    minimum=0.05,
                    label="Segment duration (seconds)",
                    info="Maximum saved audio segment length. Longer recordings are split after the validation split is assigned.",
                )
                seed = gr.Number(
                    value=1234,
                    precision=0,
                    label="Dataset selection seed",
                    info="Controls speaker sampling and recording splits. Keep it fixed to reproduce the same dataset.",
                )
    return settings, [speakers, cap, fraction, segment, seed]


def extraction_options():
    with gr.Column(visible=False) as settings:
        with gr.Row():
            pitch = gr.Dropdown(
                ["swift", "rmvpe"],
                value="swift",
                label="Pitch extraction algorithm",
                info="SwiftF0 is the default. RMVPE requires its checkpoint below.",
            )
            device = gr.Dropdown(
                ["auto", "cuda", "cpu"],
                value="auto",
                label="Extraction device",
                info="Auto uses CUDA when available, otherwise CPU. Extraction models are frozen.",
            )
        with gr.Accordion("Feature extraction settings", open=False):
            encoder = gr.Textbox(
                value="rvc/models/embedders/contentvec",
                label="Content encoder directory",
                info="Folder containing the encoder weights and configuration. Use the same encoder for inference.",
            )
            profile = gr.Dropdown(
                [("Bounded context", "bounded"), ("Full context", "offline")],
                value="bounded",
                label="Feature profile",
                info="Bounded context limits future audio used by feature extraction. Live conversion also requires a streaming acoustic model and renderer. Full context is for file conversion.",
            )
            pitch_path = gr.Textbox(
                label="RMVPE checkpoint",
                visible=False,
                info="Path to RMVPE weights. Required only when RMVPE is selected.",
            )
        pitch.change(
            lambda method: gr.update(visible=method == "rmvpe"), pitch, pitch_path
        )
    return settings, [encoder, pitch, pitch_path, profile, device]


def _progress_callback(progress):
    if not progress:
        return None

    def update(item):
        labels = {
            "scan": "Finding recordings",
            "hash": "Checking recordings",
            "reserve_validation": "Reserving validation",
            "preprocess": "Preprocessing audio",
            "extract": "Extracting features",
        }
        label = labels.get(item.get("operation"))
        description = f"{label}: {item['source']}" if label else item["source"]
        progress((item["recording"], item["total"]), desc=description)

    return update


def prepare_dataset(mode, legacy_args, options, progress=None):
    if mode == "classic":
        return core.run_preprocess_script(*legacy_args)
    speakers, cap, fraction, segment, seed = options[:5]
    workflow, base = options[5:] if len(options) > 5 else ("scratch", None)
    if workflow == "finetune" and not base:
        raise ValueError(
            "Choose a pretrained voice model in Model Settings before preprocessing"
        )
    path = core.run_acoustic_preprocess_script(
        legacy_args[0],
        legacy_args[1],
        validation_fraction=float(fraction),
        segment_seconds=float(segment),
        seed=int(seed),
        speakers=[s.strip() for s in (speakers or "").split(",") if s.strip()],
        recordings_per_speaker=int(cap),
        progress=_progress_callback(progress),
        base_model=base if workflow == "finetune" else None,
    )
    return f"Audio preprocessing complete. Next, run Extract Features. Saved audio manifest: {path}"


def extract_dataset(mode, legacy_args, options, progress=None):
    if mode == "classic":
        return core.run_extract_script(*legacy_args)
    encoder, pitch, pitch_path, profile, device = options[:5]
    workflow, base = options[5:] if len(options) > 5 else ("scratch", None)
    if workflow == "finetune" and not base:
        raise ValueError(
            "Choose a pretrained voice model in Model Settings before extracting features"
        )
    path = core.run_acoustic_extract_script(
        legacy_args[0],
        encoder,
        pitch,
        pitch_path or None,
        profile,
        device,
        progress=_progress_callback(progress),
        base_model=base if workflow == "finetune" else None,
    )
    return f"Feature extraction complete. Your dataset is ready for training. Manifest: {path}"


_STAGE_HELP = {
    "all": "Trains the two required parts and exports a voice model and vocoder. Saved progress is resumed automatically. Flow and shortcut are optional and are skipped.",
    "all_refiners": "Trains the predictor, vocoder, flow and shortcut in order, then exports both models. Refinement is experimental; compare the resulting audio before using it.",
    "predictor": "**1. Acoustic predictor** learns how content, pitch and the selected speaker become a mel spectrogram. Start here for a new voice model. A matching vocoder is also needed to listen to results.",
    "vocoder": "**2. Universal vocoder** learns to turn reference mel spectrograms into audio. Train it independently or use a compatible trained vocoder. It can be shared by voice models with the same mel settings.",
    "flow": "**3. Flow refiner (optional)** learns corrections to the predictor's mel spectrogram. Start from a trained predictor checkpoint. Evaluate held-out audio before choosing a refinement budget.",
    "shortcut": "**4. Shortcut refiner (optional)** learns to use fewer refinement evaluations. Start from a trained flow checkpoint. This stage does not replace predictor or vocoder training.",
    "adapt": "**Voice adaptation** fits a compatible acoustic model to your dataset's speakers. Supply an acoustic base checkpoint. LoRA trains small adapters; full adaptation updates the complete model.",
}


def stage_options():
    with gr.Column(visible=False) as settings:
        stage = gr.Dropdown(
            [
                ("Complete scratch model", "all"),
                ("Complete model with experimental refinement", "all_refiners"),
                ("1. Acoustic predictor", "predictor"),
                ("2. Universal vocoder", "vocoder"),
                ("3. Flow refiner (optional)", "flow"),
                ("4. Shortcut refiner (optional)", "shortcut"),
                ("Adapt an existing voice model", "adapt"),
            ],
            value="all",
            label="Stage or scratch recipe",
            info="Choose an individual stage or a complete pretraining recipe.",
        )
        guide = gr.Markdown(_STAGE_HELP["all"])
    stage.change(lambda value: _STAGE_HELP[value], stage, guide)
    return settings, stage


def training_options(stage=None, workflow=None):
    with gr.Column(visible=False) as settings:
        if stage is None:
            stage_settings, stage = stage_options()
            stage_settings.visible = True
        with gr.Accordion(
            "Starting weights and resume", open=False, visible=False
        ) as weights:
            base = gr.Textbox(
                label="Base or previous-stage checkpoint",
                info="Optional for predictor/vocoder scratch training. Required for flow, shortcut and adaptation. Initializes a new run from trained weights.",
            )
            resume = gr.Textbox(
                label="Resume checkpoint",
                info="Continue an interrupted run from last.pt, restoring optimizer and random states. Keep the dataset, stage, batch, crop and optimizer settings unchanged. Choose either base or resume.",
            )
        with gr.Row(visible=False) as adaptation_settings:
            adaptation = gr.Dropdown(
                [("LoRA adapters", "lora"), ("Full model", "full")],
                value="lora",
                label="Adaptation method",
                info="LoRA uses fewer trainable parameters. Full adaptation updates all acoustic weights.",
            )
            rank = gr.Number(
                value=8,
                minimum=1,
                precision=0,
                label="Adapter rank",
                info="LoRA capacity. Higher ranks add trainable parameters and memory use; ignored for full adaptation.",
            )
        with gr.Accordion("Memory and performance", open=False):
            with gr.Row():
                crop = gr.Number(
                    value=128,
                    minimum=1,
                    precision=0,
                    label="Training crop (frames)",
                    info="Audio context per example; 128 frames is about 1.49 seconds. Shorter crops use less memory but provide less context.",
                )
                accumulation = gr.Number(
                    value=1,
                    minimum=1,
                    precision=0,
                    label="Gradient accumulation",
                    info="Combine this many microbatches per update. Effective batch = batch size × accumulation per GPU.",
                )
            with gr.Row():
                precision = gr.Dropdown(
                    ["auto", "bf16", "fp16", "fp32"],
                    value="auto",
                    label="Precision",
                    info="Auto chooses supported BF16 or FP16 on CUDA, and FP32 on CPU. Keep Auto for normal training.",
                )
                device = gr.Dropdown(
                    ["auto", "cuda", "cpu"],
                    value="auto",
                    label="Training device",
                    info="Auto chooses CUDA when available. CPU training is supported but slower.",
                )
        with gr.Accordion("Optimizer and reproducibility", open=False):
            lr = gr.Number(
                value=0.0002,
                minimum=0.00000001,
                label="Learning rate",
                info="Optimizer update size. The shared default is 0.0002; change only when evaluating a training recipe.",
            )
            seed = gr.Number(
                value=1234,
                precision=0,
                label="Training seed",
                info="Controls initialization, crop sampling and training randomness. Keep fixed for comparisons and exact resume.",
            )

        def manual_settings(value, mode="advanced"):
            return (
                gr.update(visible=mode == "advanced" and value == "adapt"),
                gr.update(
                    visible=mode == "advanced" and value not in {"all", "all_refiners"}
                ),
            )

        inputs = [stage, workflow] if workflow is not None else stage
        stage.change(manual_settings, inputs, [adaptation_settings, weights])
        if workflow is not None:
            workflow.change(manual_settings, inputs, [adaptation_settings, weights])
        adaptation.change(
            lambda method: gr.update(visible=method == "lora"), adaptation, rank
        )
    return settings, [
        stage,
        base,
        resume,
        crop,
        accumulation,
        lr,
        precision,
        device,
        seed,
        adaptation,
        rank,
    ]


def train_model(mode, legacy_args, options, session_hash, progress=None):
    if mode == "classic":
        if progress is not None:
            progress(
                (0, None), desc="Training classic model — see live logs for details"
            )
        yield core.run_train_script(*legacy_args)
        return
    values = _training_signature.bind(*legacy_args).arguments
    (
        stage,
        base,
        resume,
        crop,
        accumulation,
        lr,
        precision,
        device,
        seed,
        adaptation,
        rank,
    ) = options[:11]
    # Accept callbacks from the withdrawn experimental detail control. Keep its
    # setting for exact resume, but do not offer it for new GUI training jobs.
    detail = bool(options[11]) if len(options) in {12, 15} else False
    workflow_start = 12 if len(options) in {12, 15} else 11
    workflow, pretrained, vocoder = (
        options[workflow_start:]
        if len(options) > workflow_start
        else ("advanced", None, None)
    )
    if workflow == "scratch":
        stage = "all"
    elif workflow == "finetune":
        stage = "finetune"
    event = threading.Event()
    key = (session_hash, values["model_name"])
    _training_events[key] = event
    try:
        if progress is not None:
            progress(
                0,
                desc="Preparing fine-tuning"
                if stage == "finetune"
                else "Preparing training",
            )
        final_step = None
        pipeline = stage in {"all", "all_refiners"}
        parts = (4 if stage == "all_refiners" else 2) if pipeline else 1
        stopped = False
        trainer = (
            core.run_acoustic_finetune_script
            if stage == "finetune"
            else (
                core.run_acoustic_train_all_script if pipeline else core.run_acoustic_train_script
            )
        )
        arguments = (
            dict(
                refine=stage == "all_refiners",
            )
            if pipeline
            else dict(stage=stage, base_model=base or None, resume=resume or None)
        )
        if stage == "finetune":
            arguments = dict(base_model=pretrained, vocoder_path=vocoder)
        for update in trainer(
            values["model_name"],
            **arguments,
            steps=int(values["total_epoch"]),
            batch_size=int(values["batch_size"]),
            checkpoint_every=int(values["save_every_epoch"]),
            crop_frames=int(crop),
            accumulation_steps=int(accumulation),
            learning_rate=float(lr),
            mel_detail_weight=0.5
            if detail and stage in {"predictor", "adapt", "all", "all_refiners", "finetune"}
            else 0.0,
            precision=precision,
            device=device,
            seed=int(seed),
            adaptation=adaptation,
            adapter_rank=int(rank),
            stop_requested=event.is_set,
        ):
            if update.get("status") == "exported":
                if progress is not None:
                    progress(1, desc="Training complete — models exported")
                yield f"Training complete.\nVoice model: {update['acoustic']}\nVocoder: {update['vocoder']}"
                continue
            if update.get("status") == "skipped":
                if progress is not None:
                    progress(
                        0.98 * update["part"] / parts,
                        desc=f"Part {update['part']}/{parts}: {update['phase']} already trained",
                    )
                yield f"{update['phase'].capitalize()} already reached the selected duration. Continuing…"
                continue
            if final_step is None:
                final_step = update["step"] + int(values["total_epoch"]) - 1
            target = (
                int(values["total_epoch"])
                if pipeline or stage == "finetune"
                else final_step
            )
            phase = (
                "Fine-tuning"
                if stage == "finetune"
                else (update["phase"] if pipeline else stage)
            )
            heading = f"Part {update['part']}/{update['parts']} — " if pipeline else ""
            if progress is not None:
                fraction = min(1, update["step"] / target)
                overall = (
                    ((update["part"] - 1) + fraction) / parts if pipeline else fraction
                )
                # Reserve the final 2% for exporting complete scratch/LoRA models.
                exporting = pipeline or stage == "finetune"
                description = f"{heading}{phase.capitalize()} — {update['step']:,}/{target:,} updates"
                if update.get("stopped"):
                    description = "Stopped — progress saved"
                elif (
                    fraction == 1
                    and exporting
                    and (not pipeline or update["part"] == parts)
                ):
                    description = "Exporting trained model"
                progress((0.98 if exporting else 1) * overall, desc=description)
            text = f"{heading}{phase.capitalize()} | Update: {update['step']}/{target} | Loss: {update['loss']:.4f}"
            if "validation_mel_l1" in update:
                text += f"\nValidation mel error: {update['validation_mel_l1']:.4f} (lower is better)."
            if not update.get("optimizer_updated", True):
                text += "\nOptimizer update skipped because gradients were nonfinite."
            if update.get("stopped"):
                stopped = True
                text += (
                    "\nStopped safely. Click Start Training again to continue from saved progress."
                    if pipeline or stage == "finetune"
                    else "\nStopped safely. Resume from this stage's last.pt checkpoint."
                )
            yield text
        if (
            progress is not None
            and not stopped
            and not event.is_set()
            and not (pipeline or stage == "finetune")
        ):
            progress(1, desc="Training complete")
    finally:
        _training_events.pop(key, None)


def stop_training(mode, model_name, request: gr.Request):
    if mode == "classic":
        from tabs.settings.sections.restart import stop_train

        return stop_train(model_name)
    event = next(
        (
            event
            for (session, _), event in _training_events.items()
            if session == request.session_hash
        ),
        None,
    )
    if event:
        event.set()
        gr.Info(
            "Training will stop after the current update and save a resume checkpoint."
        )
    else:
        gr.Info("There is no active training job for this model in this session.")


def _log_tail(path, lines=40, maximum_bytes=32768):
    """Bound monitor reads even when a long training job produces large logs."""
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - maximum_bytes))
            data = stream.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-lines:])
    except FileNotFoundError:
        return ""


def monitored_runs():
    root = Path(core.logs_path)
    runs = []
    for project in root.iterdir():
        if not project.is_dir() or project.name == "_archive":
            continue
        files = list((project / "checkpoints").glob("*/metrics.jsonl"))
        if (project / "train.log").exists():
            files.append(project / "train.log")
        if (project / "campaign_status.json").exists():
            files.append(project / "campaign_status.json")
        if files:
            runs.append((max(path.stat().st_mtime for path in files), project.name))
    return [name for _, name in sorted(runs, reverse=True)]


def training_progress(name, stage="auto"):
    """Read existing job artifacts; watching never starts or interrupts training."""
    if not name:
        return "Select a training run to monitor.", ""
    project = core.acoustic_project(name)
    files = list((project / "checkpoints").glob("*/metrics.jsonl"))
    if stage != "auto":
        files = [path for path in files if path.parent.name == stage]
    if not files:
        classic = _log_tail(project / "train.log")
        campaign_path = project / "campaign_status.json"
        if campaign_path.exists():
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            phase = campaign.get("phase", "preparation")
            summary = f"**{name}** — **{phase}**"
            if campaign.get("total"):
                summary += f" · {campaign['recording']:,} / {campaign['total']:,}"
            return summary, _log_tail(
                project / "console.log"
            ) or "Preparing the dataset…"
        return (
            f"**{name}** — waiting for training logs.",
            classic or "No updates have been logged yet.",
        )
    path = max(files, key=lambda value: value.stat().st_mtime)
    raw = _log_tail(path)
    records = []
    for line in raw.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        return f"**{name}** — waiting for a complete log entry.", raw
    latest = records[-1]
    phase = path.parent.name
    age = max(0, int(time.time() - path.stat().st_mtime))
    state = "Receiving updates" if age < 30 else "No recent updates"
    campaign_path = project / "campaign_status.json"
    if campaign_path.exists():
        import psutil

        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(campaign["pid"])
            live = (
                process.is_running() and process.create_time() <= campaign["updated_at"]
            )
        except (psutil.Error, KeyError):
            live = False
        state = "Running" if live else "Process stopped"
        active_phase = campaign.get("phase")
        if active_phase in {"complete", "stopped", "failed"}:
            state = {"complete": "Completed", "stopped": "Stopped", "failed": "Failed"}[
                active_phase
            ]
        if stage == "auto" and active_phase in {"preprocess", "extract", "evaluation"}:
            summary = f"**{name} — {state}**\n\nStage: **{active_phase}**"
            if campaign.get("total"):
                summary += f" · {campaign['recording']:,} / {campaign['total']:,}"
            return summary, _log_tail(project / "console.log") or raw
        if (
            stage == "auto"
            and active_phase in {"predictor", "vocoder", "flow", "shortcut", "adapt"}
            and phase != active_phase
        ):
            consoles = list(Path(core.logs_path).glob(f"{name}*console.log"))
            if (project / "console.log").exists():
                consoles.append(project / "console.log")
            console = (
                _log_tail(max(consoles, key=lambda value: value.stat().st_mtime))
                if consoles
                else ""
            )
            return (
                f"**{name} — {state}**\n\nStage: **{active_phase}** · Preparing training; optimizer updates have not been logged yet.",
                console or "Preparing dataset and stage statistics…",
            )
    step = latest["step"]
    target = None
    plan = project / "campaign_plan.json"
    if plan.exists():
        target = (
            json.loads(plan.read_text(encoding="utf-8")).get("stages", {}).get(phase)
        )
    progress = (
        f"{step:,} / {target:,} updates ({100 * step / target:.1f}%)"
        if target
        else f"{step:,} updates"
    )
    summary = f"**{name} — {state}**\n\nStage: **{phase}** · {progress} · Last log update: **{age}s ago**"
    loss = latest.get("loss")
    if loss is not None:
        summary += f"\n\nTraining loss: **{loss:.4f}**"
    best_path = path.parent / "best.json"
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        if "validation_mel_l1" in best:
            summary += f" · Best validation mel error: **{best['validation_mel_l1']:.4f}** at update {best['step']:,}"
    errors = sorted(
        Path(core.logs_path).glob(f"{name}*error.log"),
        key=lambda value: value.stat().st_mtime,
        reverse=True,
    )
    if errors:
        error = _log_tail(errors[0], lines=10, maximum_bytes=4096)
        if error:
            raw += "\n\nLatest console error output:\n" + error
    return summary, raw


def monitor_options():
    """Follow CLI/background and GUI runs in the shared Gradio Training tab."""
    with gr.Accordion("Live training progress", open=True):
        gr.Markdown(
            "Watch training even when it was started in the background or from the CLI. This panel refreshes every 2 seconds. Use the TensorBoard tab for curves; watching a run does not start another training job."
        )
        runs = monitored_runs()
        with gr.Row():
            run = gr.Dropdown(
                runs,
                value=runs[0] if runs else None,
                label="Training run to monitor",
                info="Most recently updated run is selected by default.",
            )
            stage = gr.Dropdown(
                ["auto", "predictor", "vocoder", "flow", "shortcut", "adapt"],
                value="auto",
                label="Stage to monitor",
                info="Auto follows the stage whose log is updating most recently.",
            )
        with gr.Row():
            follow = gr.Checkbox(value=True, label="Refresh automatically")
            refresh = gr.Button("Refresh runs")
        summary = gr.Markdown("Waiting for the first refresh…")
        log = gr.Textbox(
            label="Realtime training logs",
            lines=10,
            max_lines=14,
            interactive=False,
            info="Recent raw log entries, newest at the bottom. Full logs remain saved in the run's directory.",
        )
        timer = gr.Timer(2, active=True)
        timer.tick(training_progress, [run, stage], [summary, log], queue=False)
        follow.change(
            lambda enabled: gr.update(active=enabled), follow, timer, queue=False
        )
        run.change(training_progress, [run, stage], [summary, log], queue=False)
        stage.change(training_progress, [run, stage], [summary, log], queue=False)
        refresh.click(
            lambda: gr.update(choices=monitored_runs()), outputs=run, queue=False
        ).then(training_progress, [run, stage], [summary, log], queue=False)
    return summary, log


def export_options(model_name):
    with gr.Column(visible=False) as settings:
        gr.Markdown(
            "Export trained weights for inference. Export the acoustic voice model and matching vocoder separately; inference needs both packages. Keep last.pt if you want to resume training later."
        )
        checkpoint = gr.Textbox(
            label="Training checkpoint",
            info="Select this stage's best.pt for the lowest validation error, or last.pt for the latest saved update.",
        )
        destination = gr.Textbox(
            label="Output package",
            info="Optional .pth path; the default is in this model's logs directory. Export uses EMA weights and merges any LoRA adapters.",
        )
        export = gr.Button("Export Model")
        result = gr.File(label="Exported model", interactive=False)

        def export_callback(name, checkpoint, destination):
            from rvc.train.process.checkpoints import load_payload

            kind = load_payload(checkpoint)["kind"]
            output = destination or str(core.acoustic_project(name) / f"{name}_{kind}.pth")
            return core.run_acoustic_export_script(checkpoint, output)

        export.click(
            export_callback,
            [model_name, checkpoint, destination],
            result,
            concurrency_id="model_gpu",
            concurrency_limit=1,
        )
        with gr.Accordion("Evaluate held-out recordings", open=False):
            acoustic = gr.Textbox(label="Acoustic package")
            vocoder = gr.Textbox(label="Universal vocoder package")
            budgets = gr.CheckboxGroup(
                [0, 2, 4, 8, 16], value=[0], label="Refinement steps"
            )
            limit = gr.Number(
                value=8, minimum=1, precision=0, label="Validation segments"
            )
            device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
            run = gr.Button("Evaluate")
            report = gr.Textbox(label="Evaluation report", lines=8)

            def evaluate_callback(name, acoustic, vocoder, budgets, limit, device):
                project = core.acoustic_project(name)
                result = core.run_acoustic_evaluate_script(
                    str(project / "data/manifest.json"),
                    acoustic,
                    vocoder,
                    str(project / "evaluation"),
                    budgets=[int(b) for b in budgets],
                    limit=int(limit),
                    device=device,
                )
                return json.dumps(
                    {k: v for k, v in result.items() if k != "rows"}, indent=2
                )

            run.click(
                evaluate_callback,
                [model_name, acoustic, vocoder, budgets, limit, device],
                report,
                concurrency_id="model_gpu",
                concurrency_limit=1,
            )
    return settings
