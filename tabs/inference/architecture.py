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
            label="Audio renderer (vocoder)",
            info="Choose a renderer compatible with the selected acoustic voice model.",
        )
        refresh = gr.Button("Refresh vocoders")
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
            visible=False,
        )
        seed = gr.Number(value=0, minimum=0, precision=0, label="Sampling seed")
        ordinary = gr.Checkbox(value=False, label="Ordinary flow reference sampler")
    if model_file is not None:

        def select_vocoders(path, current):
            from rvc.configs.architectures import compatible_vocoders, preferred_vocoder

            if not path or model_architecture(path) != "v3":
                return gr.update(value=None)
            choices = compatible_vocoders(path, discover_models(kind="vocoder"))
            return gr.update(
                choices=choices, value=preferred_vocoder(path, choices, current)
            )

        model_file.change(select_vocoders, [model_file, vocoder], vocoder)
        refresh.click(select_vocoders, [model_file, vocoder], vocoder)

        def runtime_settings(path):
            metadata = (
                inspect_model(path) if path and model_architecture(path) == "v3" else {}
            )
            return (
                gr.update(
                    visible=metadata.get("features", {}).get("pitch_method") == "rmvpe"
                ),
                gr.update(
                    visible=metadata.get("capabilities", {}).get("ordinary_flow", False)
                    and metadata.get("model_config", {}).get("family") != "shallow-flow"
                ),
            )

        model_file.change(runtime_settings, model_file, [pitch_path, ordinary])
        # Loading a predictor-only voice clears unsupported settings left over
        # from another model without changing the shared interface's layout.
        model_file.change(
            lambda path, ordinary, current: refinement_settings(
                path, ordinary, current, loading=True
            ),
            inputs=[model_file, ordinary, budget],
            outputs=budget,
        )
        ordinary.change(
            refinement_settings, inputs=[model_file, ordinary, budget], outputs=budget
        )
    else:
        refresh.click(
            lambda: gr.update(choices=discover_models(kind="vocoder")), outputs=vocoder
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


def refinement_settings(path, ordinary=False, current=0, loading=False):
    """Offer integration budgets supported by the exported model.

    Joint shallow flow learns its velocity in the predictor stage and supports
    few-step integration directly. Staged residual flow requires trained
    shortcuts or an explicit ordinary-flow selection for small budgets.
    """
    choices = [0, 1, 2, 4, 8, 16, 32]
    if path and model_architecture(path) == "v3":
        metadata = inspect_model(path)
        capabilities = metadata.get("capabilities")
        if metadata.get("model_config", {}).get("family") == "shallow-flow":
            choices = (
                choices if (capabilities or {}).get("ordinary_flow") else [0]
            )
            value = 8 if loading and 8 in choices else int(current or 0)
            return gr.update(
                choices=choices,
                value=value if value in choices else choices[0],
                label="Generative flow steps",
                info="Start with 8 steps and compare quality and speed. Lower budgets need fewer model evaluations. 0 uses the coarse prediction without flow refinement.",
            )
        if isinstance(capabilities, dict):
            if not capabilities.get("ordinary_flow"):
                choices = [0]
            elif not capabilities.get("shortcuts") and not ordinary:
                choices = [0, 8, 16, 32]
    value = int(current or 0)
    return gr.update(
        choices=choices,
        value=value if value in choices else 0,
        label="Refinement steps",
        info="0 uses the predictor. Evaluate trained refiners before choosing a higher budget.",
    )


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
    try:
        result = core.run_acoustic_infer_script(
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
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error
    return (
        f"Converted {len(result)} audio files."
        if batch
        else ("Conversion completed.", result)
    )
