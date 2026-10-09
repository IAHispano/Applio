"""Advanced research commands, kept separate from the everyday CLI."""

import copy
import json

import click

from rvc.configs.neural import DEFAULT_BATCH_SIZE
from rvc.lib.tools.acoustic_workflows import (
    run_acoustic_evaluate_script,
    run_acoustic_export_script,
)


@click.group(invoke_without_command=True)
@click.pass_context
def research(ctx):
    """Advanced corpus, renderer, evaluation and experimental runtime tools."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@research.command("train-corpus")
@click.option("--model-name", required=True)
@click.option(
    "--dataset-path", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "--vocoder-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--architecture-from",
    type=click.Path(exists=True, dir_okay=False),
    help="Reuse architecture/frontend semantics only; initialize new acoustic weights from scratch.",
)
@click.option(
    "--encoder-path",
    default="rvc/models/embedders/contentvec",
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--batch-size", default=DEFAULT_BATCH_SIZE, type=click.IntRange(min=1))
@click.option("--accumulation-steps", default=1, type=click.IntRange(min=1))
@click.option("--predictor-steps", default=60000, type=click.IntRange(min=1))
@click.option("--flow-steps", default=None, type=click.IntRange(min=0), help="Residual flow updates; defaults to 15000 for residual models and zero for joint shallow flow.")
@click.option("--shortcut-steps", default=None, type=click.IntRange(min=0), help="Residual shortcut updates; defaults to 15000 for residual models and zero for joint shallow flow.")
@click.option("--crop-frames", default=None, type=click.IntRange(min=1), help="Training context; defaults to 128 residual frames or 256 joint shallow-flow frames.")
@click.option("--device", default="auto")
@click.option(
    "--precision", default="auto", type=click.Choice(["auto", "bf16", "fp16", "fp32"])
)
@click.option("--checkpoint-every", default=1000, type=click.IntRange(min=1))
@click.option("--compact-cache", is_flag=True, help="Store derived audio as scaled PCM24 FLAC and content as FP16; originals remain unchanged.")
@click.option("--compact-cache-audit", type=click.Path(exists=True, dir_okay=False), help="Passed contract-matched corpus storage qualification for a measured disk estimate.")
@click.option("--sampling-mode", default="segments", type=click.Choice(["segments", "domain-speaker"]), help="Segment-uniform sampling or duration-tempered domains with balanced voices.")
@click.option("--validation-limit", default=0, type=click.IntRange(min=0), help="Fixed voice-covered checkpoint validation panel; zero validates every reserved segment.")
@click.option("--validation-manifest", type=click.Path(exists=True, dir_okay=False), help="Preserve prior held-out recordings and their corpus split groups in the new validation split.")
@click.option("--seed", default=1234, type=int)
@click.option(
    "--check-only",
    is_flag=True,
    help="Check inputs, disk space and resume compatibility without training.",
)
def train_corpus(**kwargs):
    """Preprocess, extract and train every corpus speaker with a frozen vocoder."""
    from rvc.train.acoustic.campaign import run_corpus

    try:
        run_corpus(**kwargs)
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error


@research.command("prepare-pitch-views")
@click.option("--manifest", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-manifest", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--per-speaker", default=16, type=click.IntRange(min=4), show_default=True
)
@click.option("--seed", default=5678, type=int, show_default=True)
def prepare_pitch_views(manifest, output_manifest, per_speaker, seed):
    """Prepare optional V3 training pitch counterexamples; keep natural validation."""
    from rvc.train.acoustic.augmentation import prepare_pitch_views as prepare

    try:
        result = prepare(
            manifest,
            output_manifest,
            per_speaker,
            seed,
            progress=lambda item: (
                click.echo(f"Speaker {item['speaker']}: {item['views']} views prepared")
                if item["views"] % 16 == 0
                else None
            ),
        )
        click.echo(result)
    except (ValueError, ImportError, OSError) as error:
        raise click.ClickException(str(error)) from error


@research.command("export-model")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--output-path", required=True, type=click.Path())
def export_model(checkpoint, output_path):
    """Export v3 EMA weights, merging adaptation adapters for inference."""
    click.echo(run_acoustic_export_script(checkpoint, output_path))


@research.command("import-vocoder")
@click.option("--output-path", required=True, type=click.Path())
@click.option("--checkpoint", type=click.Path(exists=True), default=None)
@click.option("--config", "configuration", type=click.Path(exists=True), default=None)
@click.option("--backend", type=click.Choice(["bigvgan-v2", "nsf-hifigan"]), default="bigvgan-v2")
def import_vocoder(output_path, checkpoint, configuration, backend):
    """Package compatible frozen vocoder weights for file inference/fine-tuning.

    Without local checkpoint/config paths, download the pinned official
    44.1-kHz BigVGAN model. NSF-HiFiGAN requires a local converted OpenVPI export
    with its own mel contract and weight license. Both backends are file-only.
    """
    from rvc.train.process.pretrained import import_bigvgan, import_nsf_hifigan

    try:
        if backend == "nsf-hifigan":
            if not checkpoint or configuration:
                raise ValueError("NSF-HiFiGAN requires --checkpoint and uses its serialized configuration")
            click.echo(import_nsf_hifigan(output_path, checkpoint))
        else:
            click.echo(import_bigvgan(output_path, checkpoint, configuration))
    except (ValueError, OSError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error


@research.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--pth-path", "--model-path", required=True, type=click.Path(exists=True))
@click.option("--vocoder-path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option(
    "--budget",
    "budgets",
    multiple=True,
    type=click.Choice(["0", "1", "2", "4", "8", "16", "32"]),
    default=["0", "2", "4", "8", "16"],
)
@click.option("--device", default="auto")
@click.option("--seed", type=int, default=1234)
@click.option("--limit", type=click.IntRange(min=0), default=0)
@click.option("--oracle-start", is_flag=True, help="Research diagnostic: shallow flow initialized with reference mel unavailable during conversion.")
def evaluate(budgets, **kwargs):
    """Render recording-disjoint validation audio, budget curves and vocoder ceiling."""
    result = run_acoustic_evaluate_script(budgets=[int(b) for b in budgets], **kwargs)
    click.echo(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


@research.command("prepare-evaluation")
@click.option("--audio-manifest", required=True, type=click.Path(exists=True))
@click.option("--output-path", required=True, type=click.Path())
@click.option("--segments-per-speaker", type=click.IntRange(min=1), default=2)
def prepare_evaluation(audio_manifest, output_path, segments_per_speaker):
    """Save a fixed held-out listening and conversion protocol without loading models."""
    from rvc.train.process.reports import prepare_plan

    try:
        plan = prepare_plan(audio_manifest, output_path, segments_per_speaker)
        click.echo(
            f"Prepared {len(plan['held_out'])} segments across {plan['speakers']} speakers. Use evaluate --limit {plan['evaluator_limit']} --seed {plan['seed']} after training."
        )
    except (ValueError, OSError, KeyError) as error:
        raise click.ClickException(str(error)) from error


@research.command("summarize-evaluation")
@click.option("--report-path", required=True, type=click.Path(exists=True))
@click.option("--output-path", required=True, type=click.Path())
def summarize_evaluation(report_path, output_path):
    """Compare refinement budgets on identical cases with speaker-level uncertainty."""
    from rvc.train.process.reports import summarize_report

    try:
        result = summarize_report(report_path, output_path)
        click.echo(
            f"Saved paired summary. Lowest waveform mel error: budget {result['lowest_waveform_mel_error_budget']}. Listening and identity checks are still required."
        )
    except (ValueError, OSError, KeyError) as error:
        raise click.ClickException(str(error)) from error


@research.command("download-encoder")
@click.option(
    "--output-dir", default="rvc/models/embedders/contentvec", type=click.Path()
)
@click.option("--repo", default="IAHispano/Applio")
@click.option("--revision", default="70ed563897504c756ec94067c12c902c4fd42025")
@click.option("--prefix", default="Resources/embedders/contentvec")
def download_encoder(output_dir, repo, revision, prefix):
    """Download pinned content encoder weights without remote Python code."""
    import re
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise click.ClickException("Pin --revision to a full 40-character commit SHA")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "pytorch_model.bin"):
        source = hf_hub_download(repo, f"{prefix}/{name}", revision=revision)
        shutil.copy2(source, destination / name)
    click.echo(str(destination))


@research.command("verify-architecture")
@click.option(
    "--encoder-path",
    default="rvc/models/embedders/contentvec",
    type=click.Path(exists=True),
)
@click.option("--output-dir", default="logs/v3-verification")
@click.option("--device", default="auto")
@click.option("--small", is_flag=True)
def verify_architecture(encoder_path, output_dir, device, small):
    """Measure graph, streaming and GPU memory on a synthetic engineering fixture."""
    from rvc.lib.tools.verify_architecture import verify

    click.echo(
        json.dumps(
            verify(encoder_path, output_dir, device, full_size=not small), indent=2
        )
    )


def _acoustic_runtime_options(func):
    for option in reversed(
        [
            click.option(
                "--pth-path",
                "--model-path",
                required=True,
                type=click.Path(exists=True),
            ),
            click.option("--vocoder-path", required=True, type=click.Path(exists=True)),
            click.option(
                "--encoder-path",
                default="rvc/models/embedders/contentvec",
                type=click.Path(exists=True),
            ),
            click.option("--pitch-path", type=click.Path(exists=True)),
            click.option("--device", default="auto"),
        ]
    ):
        func = option(func)
    return func


@research.command("serve")
@_acoustic_runtime_options
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=click.IntRange(1, 65535), default=7862)
def serve_acoustic(
    pth_path, vocoder_path, encoder_path, pitch_path, device, host, port
):
    """Serve the V3 WebSocket API; the microphone UI lives in Applio's Realtime tab."""
    import uvicorn

    from rvc.infer.acoustic import Converter
    from rvc.realtime.transport import create_app

    converter = Converter(pth_path, vocoder_path, encoder_path, pitch_path, device)
    uvicorn.run(create_app(converter), host=host, port=port)


@research.command("realtime")
@_acoustic_runtime_options
@click.option("--input-device", type=int)
@click.option("--output-device", type=int)
@click.option("--sample-rate", type=click.IntRange(min=8000), default=48000)
@click.option("--block-size", type=click.IntRange(min=1), default=1024)
@click.option("--sid", type=click.IntRange(min=0), default=0)
@click.option("--pitch", type=click.FloatRange(-24, 24), default=0)
@click.option(
    "--refinement-steps",
    type=click.Choice(["0", "1", "2", "4", "8", "16", "32"]),
    default="0",
)
@click.option("--seed", type=click.IntRange(min=0), default=0)
def realtime_acoustic(
    pth_path,
    vocoder_path,
    encoder_path,
    pitch_path,
    device,
    sid,
    pitch,
    refinement_steps,
    seed,
    **kwargs,
):
    """Run v3 conversion on explicitly selected native audio devices."""
    from rvc.infer.acoustic import Converter
    from rvc.realtime.transport import NativeSession

    converter = Converter(pth_path, vocoder_path, encoder_path, pitch_path, device)
    NativeSession(
        converter,
        speaker=sid,
        semitones=pitch,
        steps=int(refinement_steps),
        seed=seed,
        **kwargs,
    ).run()


@research.command("test-vctk")
@click.option("--model-name", required=True)
@click.option("--predictor-steps", type=click.IntRange(min=2), default=1000)
@click.option("--vocoder-steps", type=click.IntRange(min=2), default=2000)
@click.option("--flow-steps", type=click.IntRange(min=1), default=500)
@click.option("--shortcut-steps", type=click.IntRange(min=1), default=500)
@click.option("--limit", type=click.IntRange(min=1), default=8)
@click.option("--device", default="cuda")
def test_vctk(**kwargs):
    """Run staged learning and held-out audio tests on an already prepared VCTK subset."""
    from rvc.lib.tools.corpus_experiment import run_experiment

    run_experiment(**kwargs)


def register_research(cli):
    """Expose one group while retaining old script invocations as hidden aliases."""
    cli.add_command(research)
    legacy_names = {"serve": "serve-v3", "realtime": "realtime-v3"}
    for name, command in research.commands.items():
        alias = copy.copy(command)
        alias.hidden = True
        cli.add_command(alias, name=legacy_names.get(name, name))
