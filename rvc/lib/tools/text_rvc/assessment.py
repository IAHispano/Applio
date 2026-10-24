"""Held-out intelligibility checks with an independent frozen recognizer."""

import math
from pathlib import Path
import unicodedata

from rvc.lib.tools.text_rvc.data import (
    digest,
    has_seen_recordings,
    quiet_phrase_edges,
    save_json,
)


def load_seeded_decoder(checkpoint, sid):
    import torch
    from rvc.lib.tools.text_rvc.__main__ import load_rvc

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        decoder = load_rvc(checkpoint, sid)
        return decoder, torch.random.get_rng_state()


def decode_seeded(acoustic, checkpoint, sid, loaded=None):
    import torch
    from rvc.lib.tools.text_rvc.__main__ import decode_rvc

    # Decoder noise must not depend on a previous candidate's predicted length.
    # Keep paired reference audio identical across assessments of different models.
    with torch.random.fork_rng(devices=[]):
        if loaded is not None:
            decoder, state = loaded
            # Model construction also consumes noise. Reuse its post-load RNG
            # state so cached and fresh decoders produce identical waveforms.
            torch.random.set_rng_state(state)
            return decode_rvc(acoustic, checkpoint, sid, loaded=decoder)
        torch.manual_seed(7)
        return decode_rvc(acoustic, checkpoint, sid)


def error_rate(reference, hypothesis, language="en"):
    """Word edit distance; insertions can legitimately produce rates above one."""

    def words(text):
        if language == "tr":
            text = text.replace("I", "ı").replace("İ", "i")
        text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
        return "".join(
            char
            if unicodedata.category(char)[0] in ("L", "M", "N") or char == "'"
            else " "
            for char in text
        ).split()

    reference = words(reference)
    hypothesis = words(hypothesis)
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
    from rvc.lib.tools.text_rvc.__main__ import _example, _load, _manifest
    from rvc.lib.tools.text_rvc.model import denormalize

    model, payload = _load(args.frontend)
    language = getattr(model, "language", "en")
    recognizer_language = getattr(args, "recognizer_language", "en")
    cleanup_mode = getattr(args, "cleanup_mode", "off")
    cleanup_strength = getattr(args, "cleanup_strength", 12)
    if language.split("-")[0] != recognizer_language.split("-")[0]:
        raise ValueError("The independent recognizer language must match the TTS model")
    torch.manual_seed(7)
    manifest, data = _manifest(args.manifest)
    if not args.external_validation and payload.get("manifest_sha256") not in (
        None,
        digest(manifest),
    ):
        raise ValueError(
            "Checkpoint validation requires its original manifest; use external-validation for a separate corpus"
        )
    if payload["feature_space"] != data["feature_space"]:
        raise ValueError("Incompatible evaluation feature space")
    selected = (
        {row["id"] for row in data["records"] if row["split"] == "validation"}
        if args.external_validation
        else set(payload["validation_ids"])
    )
    if selected & set(payload.get("seen_ids", payload["train_ids"])):
        raise ValueError("Validation leakage detected")
    rows = [row for row in data["records"] if row["id"] in selected][: args.limit]
    if not rows:
        raise ValueError("No held-out utterances are available")
    if has_seen_recordings(payload, rows):
        raise ValueError("Validation recordings were used during training")
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

    decoder = load_seeded_decoder(args.rvc, args.sid)
    results = []
    for row in rows:
        truth = _example(manifest.parent, row, model.feature_dim)
        if args.oracle_durations:
            if "durations" not in row or "tokens" not in row:
                raise ValueError(
                    "Oracle-duration diagnostics require an aligned phoneme manifest"
                )
            with torch.inference_mode():
                values, _ = model(
                    torch.tensor([row["tokens"]]),
                    durations=torch.tensor([row["durations"]]),
                )
                values = values[0].clone()
                values[:, -1] = values[:, -1].sigmoid()
        else:
            values = model.generate(torch.tensor([model.tokenize(row["text"])]))
        features, f0, coarse = denormalize(values, payload["mean"], payload["scale"])
        acoustic = dict(
            features=features,
            f0=f0,
            coarse_f0=coarse,
            feature_space=data["feature_space"],
        )
        audio, sr = decode_seeded(acoustic, args.rvc, args.sid, loaded=decoder)
        if cleanup_mode != "off":
            from rvc.lib.tools.audio_cleanup import cleanup_audio

            audio = quiet_phrase_edges(audio, sr)
            audio = cleanup_audio(audio, sr, cleanup_mode, cleanup_strength)
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
        audio, sr = decode_seeded(
            dict(
                features=features,
                f0=f0,
                coarse_f0=coarse,
                feature_space=data["feature_space"],
            ),
            args.rvc,
            args.sid,
            loaded=decoder,
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
            errors, words = error_rate(row["text"], result[key], language)
            result[key + "_errors"] = errors
            result[key + "_wer"] = errors / words
        result["words"] = words
        results.append(result)
        save_json(
            output / "assessment.json",
            {
                "frontend_sha256": digest(args.frontend),
                "language": language,
                "recognizer_language": recognizer_language,
                "scoring": "unicode-words-numbers-v2",
                "external_validation": args.external_validation,
                "decoder_rng": "isolated-per-decode-v1",
                "decoder_seed": 7,
                "audio_processing": (
                    "quiet-phrase-edges-v2" if cleanup_mode != "off" else "none"
                ),
                "cleanup_mode": cleanup_mode,
                "cleanup_strength": cleanup_strength,
                "uses_reference_timing": args.oracle_durations,
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
