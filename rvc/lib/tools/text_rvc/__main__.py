"""Train and synthesize speech directly from text with an RVC voice model.

Run with python -m rvc.lib.tools.text_rvc --help. Standalone synthesis
uses CPU by default. Accelerated training requires the Rust job scheduler.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile

from rvc.lib.tools.text_rvc.data import digest, save_json


def _torch_save(path, payload):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "applio-text-rvc-data-v1":
        raise ValueError("Unsupported prepared dataset format")
    records = data["records"]
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate utterance IDs")
    if not all(row["split"] in ("train", "validation") for row in records):
        raise ValueError("Invalid dataset split")
    if not records or not all(
        re.fullmatch(r"(?:p\d{3}_\d+|[a-f0-9]{24})", row["id"])
        and isinstance(row["frames"], int)
        and 4 <= row["frames"] <= 3000
        for row in records
    ):
        raise ValueError("Invalid transcribed speech record contract")
    from rvc.lib.tools.text_rvc.data import normalize_text

    training_text = {
        normalize_text(row["text"]) for row in records if row["split"] == "train"
    }
    if training_text & {
        normalize_text(row["text"]) for row in records if row["split"] == "validation"
    }:
        raise ValueError("Training and validation transcripts overlap")
    return path, data


def _example(root, row, dimensions):
    import torch

    path = (root / row["features"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Feature path escapes the prepared dataset")
    value = torch.load(path, map_location="cpu", weights_only=True)
    features, f0 = value["features"], value["f0"]
    if (
        features.ndim != 2
        or features.shape != (row["frames"], dimensions)
        or f0.shape != (len(features),)
    ):
        raise ValueError("Feature/pitch dimensions do not match the dataset contract")
    if (
        not torch.isfinite(features).all()
        or not torch.isfinite(f0).all()
        or (f0 < 0).any()
    ):
        raise ValueError("Invalid acoustic targets")
    return {
        **value,
        "text": row["text"],
        **({"durations": row["durations"]} if "durations" in row else {}),
        **({"tokens": row["tokens"]} if "tokens" in row else {}),
    }


def _load(path):
    import torch
    from rvc.lib.tools.text_rvc.model import TextFrontend, DurationFrontend

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "applio-text-rvc-frontend-v1":
        raise ValueError("Unsupported frontend checkpoint")
    config = dict(payload["config"])
    architecture = config.pop("architecture", "autoregressive")
    if architecture not in ("autoregressive", "duration"):
        raise ValueError("Unknown frontend architecture")
    model = (DurationFrontend if architecture == "duration" else TextFrontend)(
        **config
    ).cpu()
    model.load_state_dict(payload["weights"], strict=True)
    if getattr(model, "phonetic_head", None) is not None:
        model.phonetic_head.requires_grad_(False)
    model.duration_scale = float(payload.get("duration_scale", 1.0))
    if not math.isfinite(model.duration_scale) or not 0.25 <= model.duration_scale <= 4:
        raise ValueError("Invalid duration calibration")
    if payload.get("units"):
        raise ValueError("This frontend checkpoint variant is not supported")
    model.eval()
    return model, payload


def _validation(model, records, root, mean, scale):
    import torch
    from rvc.lib.tools.text_rvc.model import batch, loss

    model.eval()
    values = []
    with torch.inference_mode():
        for row in records:
            args = batch(
                [_example(root, row, model.feature_dim)], mean, scale, model.reduction
            )
            device = next(model.parameters()).device
            args = tuple(value.to(device) for value in args)
            total, metrics = loss(model, *args)
            values.append({"loss": float(total), **metrics})
    return {key: sum(row[key] for row in values) / len(values) for key in values[0]}


def train(args):
    import torch
    from rvc.lib.tools.text_rvc.model import (
        TextFrontend,
        DurationFrontend,
        batch,
        loss,
    )

    manifest, data = _manifest(args.manifest)
    train_rows = [row for row in data["records"] if row["split"] == "train"]
    validation = [row for row in data["records"] if row["split"] == "validation"]
    if not train_rows or not validation:
        raise ValueError("Separate training and validation utterances are required")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new training output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dimensions = data["feature_space"]["feature_dim"]
    count = 0
    sums, squares = torch.zeros(dimensions, dtype=torch.float64), torch.zeros(
        dimensions, dtype=torch.float64
    )
    # Compute normalization on training frames only, without retaining the corpus.
    for row in train_rows:
        features = _example(manifest.parent, row, dimensions)["features"].double()
        count += len(features)
        sums += features.sum(0)
        squares += features.square().sum(0)
    mean = (sums / count).float()
    scale = (squares / count - (sums / count).square()).clamp_min(1e-6).sqrt().float()
    aligned = all("durations" in row for row in data["records"])
    vocabulary = data.get("alignment", {}).get("phoneme_vocabulary")
    model = (
        DurationFrontend(
            dimensions,
            hidden=args.hidden,
            layers=args.layers,
            phoneme_vocabulary=vocabulary,
            language=data.get("alignment", {}).get("language", "en"),
        )
        if aligned
        else TextFrontend(dimensions, hidden=args.hidden, layers=args.layers)
    )
    if args.initialize_from:
        initialization_hash = digest(args.initialize_from)
        source_model, source = _load(args.initialize_from)
        if digest(args.initialize_from) != initialization_hash:
            raise ValueError("Initialization checkpoint changed while loading")
        if (
            source["feature_space"] != data["feature_space"]
            or isinstance(source_model, DurationFrontend) != aligned
        ):
            raise ValueError("Initialization feature space or architecture differs")
        if getattr(source_model, "phoneme_vocabulary", None) != vocabulary or getattr(
            source_model, "language", "en"
        ) != getattr(model, "language", "en"):
            if not args.reset_tokenizer:
                raise ValueError(
                    "Initialization text tokenizer differs; use --reset-tokenizer for acoustic transfer"
                )
            transferred = {
                key: value
                for key, value in source["weights"].items()
                if not key.startswith(("embedding.", "phonetic_head."))
            }
            missing, unexpected = model.load_state_dict(transferred, strict=False)
            if missing != ["embedding.weight"] or unexpected:
                raise ValueError("Acoustic transfer architecture differs")
        else:
            model = source_model
        if set(source["train_ids"]) & {row["id"] for row in validation}:
            raise ValueError("Initialization checkpoint has seen validation utterances")
        mean, scale = source["mean"], source["scale"]
    if args.phonetic_loss:
        from rvc.lib.tools.text_rvc.model import fit_phonetic_head

        if not aligned:
            raise ValueError("Phonetic supervision requires aligned training data")
        fit_phonetic_head(
            model,
            [_example(manifest.parent, row, dimensions) for row in train_rows],
            mean,
            scale,
        )
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    baseline = _validation(model, validation, manifest.parent, mean, scale)
    history = [{"step": 0, "validation": baseline}]
    best = float("inf")
    train_ids = [row["id"] for row in train_rows]
    validation_ids = [row["id"] for row in validation]
    for step in range(1, args.steps + 1):
        model.train()
        rows = rng.sample(train_rows, min(args.batch_size, len(train_rows)))
        examples = [_example(manifest.parent, row, dimensions) for row in rows]
        batch_args = batch(examples, mean, scale, model.reduction)
        batch_args = tuple(value.to(args.device) for value in batch_args)
        optimizer.zero_grad(set_to_none=True)
        total, metrics = loss(model, *batch_args)
        if not torch.isfinite(total):
            raise ValueError("Training produced a non-finite loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            print(
                json.dumps(
                    {"step": step, "train_loss": float(total.detach()), **metrics}
                ),
                flush=True,
            )
        if step % args.validate_every == 0 or step == args.steps:
            metrics = _validation(model, validation, manifest.parent, mean, scale)
            history.append({"step": step, "validation": metrics})
            payload = {
                "format": "applio-text-rvc-frontend-v1",
                "config": model.config,
                "weights": model.state_dict(),
                "mean": mean,
                "scale": scale,
                "feature_space": data["feature_space"],
                "train_ids": train_ids,
                "validation_ids": validation_ids,
                "seed": args.seed,
                "step": step,
                "validation": metrics,
                "duration_scale": model.duration_scale,
                "initialized_from_sha256": (
                    initialization_hash if args.initialize_from else None
                ),
                "reset_tokenizer": args.reset_tokenizer,
            }
            _torch_save(output / "last.pt", payload)
            if metrics["loss"] < best:
                best = metrics["loss"]
                _torch_save(output / "best.pt", payload)
            save_json(
                output / "training.json",
                {
                    "manifest_sha256": digest(manifest),
                    "corpus": data["corpus"],
                    "speaker": data["speaker"],
                    "history": history,
                },
            )
            print(json.dumps(history[-1]), flush=True)


def evaluate(args):
    import torch

    model, payload = _load(args.frontend)
    manifest, data = _manifest(args.manifest)
    if payload["feature_space"] != data["feature_space"]:
        raise ValueError("Feature space differs from the frontend training contract")
    rows_by_id = {row["id"]: row for row in data["records"]}
    rows = [rows_by_id[identity] for identity in payload["validation_ids"]]
    if set(payload["train_ids"]) & {row["id"] for row in rows}:
        raise ValueError("Validation leakage detected")
    mean, scale = payload["mean"], payload["scale"]
    results = []
    with torch.inference_mode():
        for row in rows[: args.limit]:
            truth = _example(manifest.parent, row, model.feature_dim)
            tokens = torch.tensor([model.tokenize(row["text"])])
            generated = model.generate(tokens)
            target = (truth["features"] - mean) / scale
            # Length-normalized error is diagnostic, not a perceptual speech score.
            aligned = torch.nn.functional.interpolate(
                generated[:, :-2].T.unsqueeze(0),
                size=len(target),
                mode="linear",
                align_corners=False,
            )[0].T
            voiced = truth["f0"] > 0
            generated_pitch = torch.nn.functional.interpolate(
                generated[:, -2:].T.unsqueeze(0),
                size=len(target),
                mode="linear",
                align_corners=False,
            )[0].T
            voiced_prediction = generated_pitch[:, -1] >= 0.5
            reference_pitch = (truth["f0"].clamp_min(1).log() - 5.3) / 0.5
            cents = (
                (generated_pitch[:, 0] - reference_pitch).abs()
                * 0.5
                * 1200
                / math.log(2)
            )[voiced & voiced_prediction]
            result = {
                "id": row["id"],
                "feature_mse": float((aligned - target).square().mean()),
                "training_mean_mse": float(target.square().mean()),
                "duration_ratio": len(generated) / len(target),
                "voicing_accuracy": float((voiced_prediction == voiced).float().mean()),
                "voiced_pitch_mae_cents": float(cents.mean()) if len(cents) else None,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    save_json(
        args.output,
        {
            "frontend_sha256": digest(args.frontend),
            "teacher_forced": _validation(model, rows, manifest.parent, mean, scale),
            "free_running": results,
            "summary": {
                "feature_mse": sum(row["feature_mse"] for row in results)
                / len(results),
                "training_mean_mse": sum(row["training_mean_mse"] for row in results)
                / len(results),
                "duration_relative_mae": sum(
                    abs(row["duration_ratio"] - 1) for row in results
                )
                / len(results),
                "voicing_accuracy": sum(row["voicing_accuracy"] for row in results)
                / len(results),
            },
            "perceptual_quality_verified": False,
        },
    )


def predict(frontend, text, max_frames=2000, rate=0, loaded=None):
    import torch
    from rvc.lib.tools.text_rvc.model import denormalize

    model, payload = loaded or _load(frontend)
    tokens = torch.tensor([model.tokenize(text)])
    original_scale = model.duration_scale
    speed = 1 + rate / 100 if rate >= 0 else 1 / (1 - rate / 100)
    model.duration_scale /= speed
    try:
        values = model.generate(tokens, max_frames=max_frames)
    finally:
        model.duration_scale = original_scale
    features, f0, coarse = denormalize(values, payload["mean"], payload["scale"])
    return {
        "features": features,
        "f0": f0,
        "coarse_f0": coarse,
        "feature_space": payload["feature_space"],
        "text": text,
    }


def fit(args):
    from huggingface_hub import snapshot_download
    from rvc.lib.tools.text_rvc.data import prepare, validate_speech
    from rvc.lib.tools.text_rvc.alignment import align
    import argparse

    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new training output directory")
    validate_speech(args.vctk_root, args.speaker, args.language)
    version = "v2"
    if args.rvc:
        import torch

        voice = torch.load(args.rvc, map_location="cpu", weights_only=True)
        if (
            "weight" not in voice
            or voice.get("embedder_model", "contentvec") != "contentvec"
        ):
            raise ValueError("An exported ContentVec RVC voice model is required")
        version = voice.get("version", "v1")
        if version not in ("v1", "v2"):
            raise ValueError("Unsupported RVC feature version")
    if (
        not (Path(args.embedder) / "pytorch_model.bin").is_file()
        or not Path(args.rmvpe).is_file()
    ):
        raise FileNotFoundError(
            "ContentVec and RMVPE assets are required; run Install / Repair first"
        )
    recognizer = Path(args.recognizer).resolve()
    multilingual = args.language != "en"
    if multilingual:
        recognizer = recognizer.with_name("phonemes-multilingual")
    snapshot_download(
        (
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            if multilingual
            else "facebook/wav2vec2-lv-60-espeak-cv-ft"
        ),
        revision=(
            "2c733782da5604684829819a5eb744c193fe9398"
            if multilingual
            else "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
        ),
        local_dir=str(recognizer),
        allow_patterns=["*.json", "pytorch_model.bin", "README.md", "LICENSE*"],
    )
    prepare(
        args.vctk_root,
        root / "prepared",
        args.embedder,
        args.rmvpe,
        speaker=args.speaker,
        version=version,
        language=args.language,
    )
    manifest = root / "prepared" / "phonemes.json"
    align(
        root / "prepared" / "manifest.json",
        manifest,
        recognizer,
        phonemes=True,
        language=args.language,
    )
    training = argparse.Namespace(
        **{**vars(args), "manifest": str(manifest), "output": str(root / "training")}
    )
    train(training)
    calibrate(
        argparse.Namespace(
            frontend=root / "training" / "best.pt",
            manifest=manifest,
            output=root / "frontend.pt",
        )
    )


def calibrate(args):
    import torch
    from rvc.lib.tools.text_rvc.model import DurationFrontend

    model, payload = _load(args.frontend)
    manifest, data = _manifest(args.manifest)
    if (
        not isinstance(model, DurationFrontend)
        or payload["feature_space"] != data["feature_space"]
    ):
        raise ValueError("Calibration requires a compatible duration frontend")
    if Path(args.output).exists():
        raise FileExistsError("Use a new calibrated checkpoint path")
    rows = {row["id"]: row for row in data["records"]}
    predicted, actual = 0.0, 0
    with torch.inference_mode():
        for identity in payload["train_ids"]:
            row = rows[identity]
            if row["split"] != "train":
                raise ValueError(
                    "Duration calibration must use training utterances only"
                )
            tokens = torch.tensor([model.tokenize(row["text"])])
            _, durations = model.encode(tokens)
            predicted += float(
                durations.clamp(0, math.log(301)).expm1()[tokens != model.eos].sum()
            )
            actual += row["frames"]
    ratio = actual / max(predicted, 1e-8)
    if not 0.25 <= ratio <= 4:
        raise ValueError("Duration predictions are too inaccurate to calibrate")
    _torch_save(
        args.output,
        {
            **payload,
            "duration_scale": ratio,
            "calibration_manifest_sha256": digest(manifest),
        },
    )
    print(
        json.dumps(
            {"duration_scale": ratio, "training_utterances": len(payload["train_ids"])}
        )
    )


def decode_rvc(acoustic, checkpoint, sid=0, device="cpu"):
    import torch
    from rvc.lib.algorithm.synthesizers import Synthesizer

    voice = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not all(key in voice for key in ("weight", "config")):
        raise ValueError(
            "An exported RVC voice checkpoint is required, not training weights"
        )
    weights = voice["weight"]
    config = list(voice["config"])
    dimensions = 768 if voice.get("version", "v1") == "v2" else 256
    if (
        dimensions != acoustic["features"].shape[-1]
        or acoustic["feature_space"]["frame_hz"] != 100
    ):
        raise ValueError("RVC checkpoint and frontend feature spaces are incompatible")
    if voice.get("embedder_model", "contentvec") != "contentvec":
        raise ValueError(
            "Direct text synthesis supports ContentVec RVC checkpoints only"
        )
    speakers = weights["emb_g.weight"].shape[0]
    if not 0 <= sid < speakers:
        raise ValueError("Speaker ID is outside the checkpoint speaker table")
    config[-3] = speakers
    model = Synthesizer(
        *config,
        use_f0=bool(voice.get("f0", 1)),
        text_enc_hidden_dim=dimensions,
        vocoder=voice.get("vocoder", "HiFi-GAN")
    )
    del model.enc_q
    model.load_state_dict(weights, strict=True)
    model.to(device).float().eval().requires_grad_(False)
    features = acoustic["features"].unsqueeze(0).to(device)
    pitch = acoustic["coarse_f0"].unsqueeze(0).to(device) if model.use_f0 else None
    f0 = acoustic["f0"].unsqueeze(0).to(device) if model.use_f0 else None
    with torch.inference_mode():
        output = (
            model.infer(
                features,
                torch.tensor([features.size(1)], device=device),
                pitch,
                f0,
                torch.tensor([sid], device=device),
            )[0][0, 0]
            .float()
            .cpu()
        )
    if not torch.isfinite(output).all() or not len(output):
        raise ValueError("RVC decoding produced invalid audio")
    output = output * min(1.0, 0.95 / max(float(output.abs().max()), 1e-8))
    return output.numpy(), int(config[-1])


def synthesize(args):
    import numpy as np
    import soundfile as sf
    from rvc.lib.tools.text_rvc.data import text_chunks

    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".wav", delete=False
    ) as handle:
        temporary = Path(handle.name)
    stream = None
    frames = 0
    loaded = _load(args.frontend)
    speed = 1 + args.rate / 100 if args.rate >= 0 else 1 / (1 - args.rate / 100)
    try:
        for text in text_chunks(
            args.text, max_bytes=max(32, min(120, int(120 * speed)))
        ):
            acoustic = predict(
                args.frontend, text, args.max_frames, args.rate, loaded=loaded
            )
            audio, sample_rate = decode_rvc(
                acoustic, args.rvc, args.sid, device=args.device
            )
            if stream is None:
                stream = sf.SoundFile(
                    temporary,
                    "w",
                    samplerate=sample_rate,
                    channels=1,
                    subtype="PCM_16",
                    format="WAV",
                )
            else:
                if stream.samplerate != sample_rate:
                    raise ValueError("Voice sample rate changed during synthesis")
                stream.write(np.zeros(round(sample_rate * 0.08), dtype=np.float32))
            stream.write(audio)
            frames += len(acoustic["features"])
        stream.close()
        temporary.replace(destination)
    finally:
        if stream is not None:
            stream.close()
        temporary.unlink(missing_ok=True)
    save_json(
        str(destination) + ".json",
        {
            "frontend_sha256": digest(args.frontend),
            "language": getattr(loaded[0], "language", "en"),
            "rvc_sha256": digest(args.rvc),
            "sid": args.sid,
            "frames": frames,
            "sample_rate": sample_rate,
        },
    )


def main():
    from rvc.lib.tools.text_rvc.alignment import LANGUAGES

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Workflow: download-vctk, prepare, align, train, calibrate, assess, synthesize. Use the managed engine Python. Phoneme mode requires piper-tts; alignment and assessment require local recognizer model folders.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate-data")
    validation.add_argument("--dataset", required=True)
    validation.add_argument("--speaker", default="p225")
    validation.add_argument("--language", choices=LANGUAGES, default="en")
    download = commands.add_parser("download-vctk")
    download.add_argument("--output", required=True)
    download.add_argument("--speaker", default="p225")
    download.add_argument("--limit", type=int, default=64)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--vctk-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--embedder", required=True)
    prepare.add_argument("--rmvpe", required=True)
    prepare.add_argument("--speaker", default="p225")
    prepare.add_argument("--mic", choices=("mic1", "mic2"), default="mic1")
    prepare.add_argument("--version", choices=("v1", "v2"), default="v2")
    prepare.add_argument("--language", choices=LANGUAGES, default="en")
    alignment = commands.add_parser("align")
    alignment.add_argument("--manifest", required=True)
    alignment.add_argument("--output", required=True)
    alignment.add_argument("--recognizer", required=True)
    alignment.add_argument("--phonemes", action="store_true")
    alignment.add_argument("--language", choices=LANGUAGES, default="en")
    training = commands.add_parser("train")
    training.add_argument("--manifest", required=True)
    fitting = commands.add_parser(
        "fit", help="Prepare transcribed speech and train a text frontend"
    )
    for command in (training, fitting):
        command.add_argument("--output", required=True)
        command.add_argument("--steps", type=int, default=1000)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--hidden", type=int, default=128)
        command.add_argument("--layers", type=int, default=2)
        command.add_argument("--learning-rate", type=float, default=0.0003)
        command.add_argument("--validate-every", type=int, default=50)
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--initialize-from")
        command.add_argument("--reset-tokenizer", action="store_true")
        command.add_argument("--phonetic-loss", action="store_true")
    fitting.add_argument("--dataset", "--vctk-root", dest="vctk_root", required=True)
    fitting.add_argument("--rvc", default="")
    fitting.add_argument("--language", choices=LANGUAGES, default="en")
    fitting.add_argument("--speaker", default="p225")
    fitting.add_argument("--embedder", required=True)
    fitting.add_argument("--rmvpe", required=True)
    fitting.add_argument("--recognizer", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--frontend", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--limit", type=int, default=8)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--frontend", required=True)
    calibration.add_argument("--manifest", required=True)
    calibration.add_argument("--output", required=True)
    assessment = commands.add_parser("assess")
    assessment.add_argument("--frontend", required=True)
    assessment.add_argument("--manifest", required=True)
    assessment.add_argument("--rvc", required=True)
    assessment.add_argument("--recognizer", required=True)
    assessment.add_argument("--output", required=True)
    assessment.add_argument("--sid", type=int, default=0)
    assessment.add_argument("--limit", type=int, default=8)
    for command in ("predict", "synthesize"):
        item = commands.add_parser(command)
        item.add_argument("--frontend", required=True)
        text_input = item.add_mutually_exclusive_group(required=True)
        text_input.add_argument("--text")
        text_input.add_argument("--text-file")
        item.add_argument("--output", required=True)
        item.add_argument("--max-frames", type=int, default=2000)
        item.add_argument(
            "--rate", type=int, choices=range(-100, 101), default=0, metavar="-100..100"
        )
        if command == "synthesize":
            item.add_argument("--rvc", required=True)
            item.add_argument("--sid", type=int, default=0)
    args = parser.parse_args()
    if getattr(args, "text_file", None):
        args.text = Path(args.text_file).read_text(encoding="utf-8-sig")
    if not 1 <= args.threads <= 16:
        parser.error("threads must be between 1 and 16")
    if args.device != "cpu":
        import uuid

        try:
            uuid.UUID(os.environ.get("APPLIO_TEXT_RVC_GPU_LEASE", ""))
        except ValueError:
            parser.error("Accelerated text-to-RVC work requires managed job scheduling")
        if args.command not in ("train", "fit", "synthesize"):
            parser.error("Acceleration requires a scheduled training or synthesis job")
    if args.command == "download-vctk":
        from rvc.lib.tools.text_rvc.data import download_vctk

        download_vctk(args.output, args.speaker, args.limit)
        return
    if args.command == "validate-data":
        from rvc.lib.tools.text_rvc.data import validate_speech

        validate_speech(args.dataset, args.speaker, args.language)
        return
    import torch

    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(
        0,
        str(Path(os.environ.get("APPLIO_BACKEND_ROOT") or root)),
    )
    if args.device == "auto":
        from rvc.configs.config import Config

        args.device = Config().device
    if args.command == "prepare":
        from rvc.lib.tools.text_rvc.data import prepare as prepare_data

        prepare_data(
            args.vctk_root,
            args.output,
            args.embedder,
            args.rmvpe,
            args.speaker,
            args.mic,
            args.version,
            language=args.language,
        )
    elif args.command == "align":
        from rvc.lib.tools.text_rvc.alignment import align

        align(args.manifest, args.output, args.recognizer, args.phonemes, args.language)
    elif args.command in ("train", "fit"):
        if min(args.steps, args.batch_size, args.validate_every, args.layers) < 1:
            parser.error("training counts must be positive")
        if args.reset_tokenizer and not args.initialize_from:
            parser.error("reset-tokenizer requires initialize-from")
        (fit if args.command == "fit" else train)(args)
    elif args.command == "evaluate":
        if args.limit < 1:
            parser.error("evaluation limit must be positive")
        evaluate(args)
    elif args.command == "assess":
        from rvc.lib.tools.text_rvc.assessment import assess

        if not 1 <= args.limit <= 100:
            parser.error("assessment limit must be between 1 and 100")
        assess(args)
    elif args.command == "calibrate":
        calibrate(args)
    else:
        if not 4 <= args.max_frames <= 3000:
            parser.error("max-frames must be between 4 and 3000")
        if args.command == "predict":
            acoustic = predict(args.frontend, args.text, args.max_frames, args.rate)
            _torch_save(args.output, acoustic)
        else:
            synthesize(args)


if __name__ == "__main__":
    main()
