"""Architecture-dependent controls inside the existing Training workflow.

Keep widget/service routing here; learning objectives belong in the trainer.
Classic preparation/extraction remain separate, while V3 preparation caches
both PCM and features. Session-local stop events avoid one browser cancelling
another session. The existing output and buttons serve both architectures.
"""

import inspect
import json
import threading

import gradio as gr

import core

_training_events = {}
_training_signature = inspect.signature(core.run_train_script)


def preparation_options():
    with gr.Column(visible=False) as settings:
        with gr.Accordion("Advanced Settings", open=False):
            encoder = gr.Textbox(
                value="rvc/models/embedders/contentvec",
                label="Content encoder directory",
            )
            with gr.Row():
                pitch = gr.Dropdown(
                    ["swift", "rmvpe"],
                    value="swift",
                    label="Pitch extraction algorithm",
                )
                profile = gr.Dropdown(
                    ["bounded", "offline"], value="bounded", label="Frontend profile"
                )
                device = gr.Dropdown(
                    ["auto", "cuda", "cpu"], value="auto", label="Device"
                )
            pitch_path = gr.Textbox(label="RMVPE checkpoint (when selected)")
            speakers = gr.Textbox(
                label="Speaker folders",
                info="Optional comma-separated folders. Leave empty to use all speakers.",
            )
            cap = gr.Number(
                value=0,
                minimum=0,
                precision=0,
                label="Recordings per speaker",
                info="0 uses all recordings.",
            )
            fraction = gr.Number(
                value=0.1, minimum=0.001, maximum=0.999, label="Validation fraction"
            )
            segment = gr.Number(
                value=4, minimum=0.05, label="Segment duration (seconds)"
            )
        gr.Markdown(
            "Preparation extracts audio and features together, keeping validation recordings separate before segmentation."
        )
    return settings, [
        encoder,
        pitch,
        pitch_path,
        profile,
        device,
        speakers,
        cap,
        fraction,
        segment,
    ]


def prepare_dataset(mode, legacy_args, options, progress=None):
    if mode == "classic":
        return core.run_preprocess_script(*legacy_args)
    encoder, pitch, pitch_path, profile, device, speakers, cap, fraction, segment = (
        options
    )
    return core.run_v3_prepare_script(
        legacy_args[0],
        legacy_args[1],
        encoder,
        pitch,
        pitch_path or None,
        profile,
        device,
        fraction,
        segment,
        speakers=[s.strip() for s in speakers.split(",") if s.strip()],
        recordings_per_speaker=int(cap),
        progress=(
            lambda item: progress(
                (item["recording"], item["total"]), desc=item["source"]
            )
        )
        if progress
        else None,
    )


def training_options():
    with gr.Column(visible=False) as settings:
        stage = gr.Dropdown(
            ["predictor", "vocoder", "flow", "shortcut", "adapt"],
            value="predictor",
            label="Training stage",
            info="Train predictor and vocoder, then flow and shortcut. Adaptation requires compatible base weights.",
        )
        with gr.Accordion("Advanced Settings", open=False):
            base = gr.Textbox(label="Base or previous-stage checkpoint")
            resume = gr.Textbox(
                label="Resume checkpoint",
                info="Choose either exact resume or a base checkpoint.",
            )
            with gr.Row():
                crop = gr.Number(value=128, minimum=1, precision=0, label="Crop frames")
                accumulation = gr.Number(
                    value=1, minimum=1, precision=0, label="Gradient accumulation"
                )
                lr = gr.Number(value=0.0002, label="Learning rate")
            with gr.Row():
                precision = gr.Dropdown(
                    ["bf16", "fp16", "fp32"], value="bf16", label="Precision"
                )
                device = gr.Dropdown(
                    ["auto", "cuda", "cpu"], value="auto", label="Device"
                )
                seed = gr.Number(value=1234, precision=0, label="Training seed")
            config = gr.Textbox(
                label="Model configuration JSON",
                info="Optional configuration for training from scratch.",
            )
            adaptation = gr.Dropdown(
                ["lora", "full"], value="lora", label="Adaptation method"
            )
            rank = gr.Number(value=8, minimum=1, precision=0, label="Adapter rank")
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
        config,
        adaptation,
        rank,
    ]


def train_model(mode, legacy_args, options, session_hash):
    if mode == "classic":
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
        config,
        adaptation,
        rank,
    ) = options
    event = threading.Event()
    key = (session_hash, values["model_name"])
    _training_events[key] = event
    try:
        for update in core.run_v3_train_script(
            values["model_name"],
            stage,
            base_model=base or None,
            resume=resume or None,
            config=config or None,
            steps=int(values["total_epoch"]),
            batch_size=int(values["batch_size"]),
            checkpoint_every=int(values["save_every_epoch"]),
            crop_frames=int(crop),
            accumulation_steps=int(accumulation),
            learning_rate=float(lr),
            precision=precision,
            device=device,
            seed=int(seed),
            adaptation=adaptation,
            adapter_rank=int(rank),
            stop_requested=event.is_set,
        ):
            yield json.dumps(update, indent=2)
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


def export_options(model_name):
    with gr.Column(visible=False) as settings:
        checkpoint = gr.Textbox(label="Training checkpoint")
        destination = gr.Textbox(
            label="Output package",
            info="Optional .pth path; the default is in this model's logs directory.",
        )
        export = gr.Button("Export Model")
        result = gr.File(label="Exported model", interactive=False)

        def export_callback(name, checkpoint, destination):
            from rvc.train.process.v3_checkpoints import load_payload

            kind = load_payload(checkpoint)["kind"]
            output = destination or str(core.v3_project(name) / f"{name}_{kind}.pth")
            return core.run_v3_export_script(checkpoint, output)

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
                project = core.v3_project(name)
                result = core.run_v3_evaluate_script(
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
