"""Metadata-driven V3 controls inside the shared Inference interface.

Voice selection chooses which advanced settings are visible. Conversion routes
to the classic callback or V3 acoustic/vocoder pair while retaining common audio,
pitch, target-speaker and output controls. Model math stays outside UI callbacks.
"""

import inspect
from pathlib import Path

import gradio as gr

import core
from rvc.configs.architectures import discover_models, inspect_model, model_architecture

_single_signature = inspect.signature(core.run_infer_script)
_batch_signature = inspect.signature(core.run_batch_infer_script)


def inference_options(model_file=None):
    with gr.Column(visible=False) as settings:
        vocoder = gr.Dropdown(
            discover_models(kind="vocoder"),
            value=None,
            allow_custom_value=True,
            label="Universal vocoder",
            info="Choose a v3 vocoder matching the voice model's mel contract.",
        )
        gr.Button("Refresh vocoders").click(
            lambda: gr.update(choices=discover_models(kind="vocoder")), outputs=vocoder
        )
        with gr.Row():
            budget = gr.Dropdown(
                [0, 1, 2, 4, 8, 16, 32],
                value=0,
                label="Refinement steps",
                info="0 uses the predictor. Evaluate trained refiners before choosing a higher budget.",
            )
            device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
        encoder = gr.Textbox(
            value="rvc/models/embedders/contentvec", label="Content encoder directory"
        )
        pitch_path = gr.Textbox(
            label="RMVPE checkpoint",
            info="Required only for models prepared with RMVPE.",
        )
        seed = gr.Number(value=0, minimum=0, precision=0, label="Sampling seed")
        ordinary = gr.Checkbox(value=False, label="Ordinary flow reference sampler")
    if model_file is not None:
        # Loading a predictor-only voice clears unsupported settings left over
        # from another model without changing the shared interface's layout.
        for trigger in (model_file.change, ordinary.change):
            trigger(
                refinement_settings,
                inputs=[model_file, ordinary, budget],
                outputs=budget,
            )
    return settings, [vocoder, encoder, pitch_path, budget, seed, device, ordinary]


def model_settings(path):
    modern = bool(path) and model_architecture(path) == "v3"
    metadata = inspect_model(path) if path else {}
    if modern and metadata.get("kind") != "acoustic":
        raise gr.Error(
            "Please select an acoustic voice model. Choose its vocoder in Advanced Settings."
        )
    if modern:
        speakers = [(name, index) for index, name in enumerate(metadata["speakers"])]
    else:
        speakers = list(range(int(metadata.get("speakers_id") or 1)))
    return (
        gr.update(visible=not modern),
        gr.update(visible=modern),
        gr.update(visible=not modern),
        gr.update(visible=modern),
        gr.update(visible=not modern, **({"value": ""} if modern else {})),
        gr.update(choices=speakers, value=0),
        gr.update(choices=speakers, value=0),
    )


def refinement_settings(path, ordinary=False, current=0):
    """Offer only budgets supported by an exported acoustic model's capabilities."""
    choices = [0, 1, 2, 4, 8, 16, 32]
    if path and model_architecture(path) == "v3":
        capabilities = inspect_model(path).get("capabilities")
        if isinstance(capabilities, dict):
            if not capabilities.get("ordinary_flow"):
                choices = [0]
            elif not capabilities.get("shortcuts") and not ordinary:
                choices = [0, 8, 16, 32]
    value = int(current or 0)
    return gr.update(choices=choices, value=value if value in choices else 0)


def route_conversion(legacy_args, options, batch=False):
    legacy = core.run_batch_infer_script if batch else core.run_infer_script
    signature = _batch_signature if batch else _single_signature
    values = signature.bind(*legacy_args).arguments
    if model_architecture(values["pth_path"]) == "classic":
        return legacy(*legacy_args)
    vocoder, encoder, pitch_path, budget, seed, device, ordinary = options
    source = values["input_folder" if batch else "input_path"]
    destination = values["output_folder" if batch else "output_path"]
    if not batch:
        destination = str(Path(destination).with_suffix(".wav"))
    result = core.run_v3_infer_script(
        source,
        destination,
        values["pth_path"],
        vocoder,
        encoder,
        pitch_path or None,
        sid=values.get("sid", 0),
        pitch=values["pitch"],
        refinement_steps=int(budget),
        seed=int(seed),
        device=device,
        ordinary_flow=ordinary,
        batch=batch,
    )
    return (
        f"Converted {len(result)} audio files."
        if batch
        else ("Conversion completed.", result)
    )
