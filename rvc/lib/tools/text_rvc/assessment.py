"""Held-out intelligibility checks with an independent frozen English recognizer."""

import math
from pathlib import Path
import re

from rvc.lib.tools.text_rvc.data import digest, save_json


def error_rate(reference, hypothesis):
    """Word edit distance; insertions can legitimately produce rates above one."""
    reference = re.findall(r"[a-z']+", reference.lower())
    hypothesis = re.findall(r"[a-z']+", hypothesis.lower())
    if not reference:
        raise ValueError("A nonempty reference transcript is required")
    previous = list(range(len(hypothesis) + 1))
    for index, word in enumerate(reference, 1):
        current = [index]
        for other, candidate in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other] + 1,
                    previous[other - 1] + (word != candidate),
                )
            )
        previous = current
    return previous[-1], len(reference)


def assess(args):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from rvc.lib.tools.text_rvc.__main__ import _example, _load, _manifest, decode_rvc
    from rvc.lib.tools.text_rvc.model import denormalize

    model, payload = _load(args.frontend)
    torch.manual_seed(7)
    manifest, data = _manifest(args.manifest)
    if payload["feature_space"] != data["feature_space"]:
        raise ValueError("Incompatible evaluation feature space")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new assessment output directory")
    output.mkdir(parents=True, exist_ok=True)
    processor = Wav2Vec2Processor.from_pretrained(
        args.recognizer, local_files_only=True
    )
    recognizer = (
        Wav2Vec2ForCTC.from_pretrained(args.recognizer, local_files_only=True)
        .cpu()
        .eval()
    )
    recognizer.requires_grad_(False)

    def transcribe(audio, sr):
        divisor = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // divisor, sr // divisor).astype(np.float32)
        with torch.inference_mode():
            logits = recognizer(
                processor(audio, sampling_rate=16000, return_tensors="pt").input_values
            ).logits
        return processor.batch_decode(logits.argmax(-1))[0]

    selected = set(payload["validation_ids"])
    if selected & set(payload["train_ids"]):
        raise ValueError("Validation leakage detected")
    rows = [row for row in data["records"] if row["id"] in selected][: args.limit]
    if not rows:
        raise ValueError("No held-out utterances are available")
    results = []
    for row in rows:
        truth = _example(manifest.parent, row, model.feature_dim)
        values = model.generate(torch.tensor([model.tokenize(row["text"])]))
        features, f0, coarse = denormalize(values, payload["mean"], payload["scale"])
        acoustic = dict(
            features=features,
            f0=f0,
            coarse_f0=coarse,
            feature_space=data["feature_space"],
        )
        audio, sr = decode_rvc(acoustic, args.rvc, args.sid)
        sf.write(output / f"{row['id']}-generated.wav", audio, sr, subtype="PCM_16")
        generated_transcript = transcribe(audio, sr)
        # Decode reference features through the same voice to isolate frontend errors.
        f0 = truth["f0"]
        target = torch.cat(
            (
                (truth["features"] - payload["mean"]) / payload["scale"],
                torch.where(f0 > 0, (f0.clamp_min(1).log() - 5.3) / 0.5, 0)[:, None],
                (f0 > 0).float()[:, None],
            ),
            -1,
        )
        features, f0, coarse = denormalize(target, payload["mean"], payload["scale"])
        audio, sr = decode_rvc(
            dict(
                features=features,
                f0=f0,
                coarse_f0=coarse,
                feature_space=data["feature_space"],
            ),
            args.rvc,
            args.sid,
        )
        sf.write(
            output / f"{row['id']}-reference-decoded.wav", audio, sr, subtype="PCM_16"
        )
        decoded_transcript = transcribe(audio, sr)
        audio, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
        original_transcript = transcribe(audio.mean(1), sr)
        result = {
            "id": row["id"],
            "reference": row["text"],
            "generated": generated_transcript,
            "reference_decoded": decoded_transcript,
            "original": original_transcript,
        }
        for key in ("generated", "reference_decoded", "original"):
            errors, words = error_rate(row["text"], result[key])
            result[key + "_errors"] = errors
            result[key + "_wer"] = errors / words
        result["words"] = words
        results.append(result)
        save_json(
            output / "assessment.json",
            {
                "frontend_sha256": digest(args.frontend),
                "rvc_sha256": digest(args.rvc),
                "recognizer_sha256": digest(
                    Path(args.recognizer) / "pytorch_model.bin"
                ),
                "results": results,
                "summary": {
                    key
                    + "_wer": sum(row[key + "_errors"] for row in results)
                    / sum(row["words"] for row in results)
                    for key in ("generated", "reference_decoded", "original")
                },
                "human_listening_verified": False,
            },
        )
        print(result, flush=True)
