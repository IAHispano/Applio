"""Language-aware CTC timing supervision used only during preparation."""

import json
import math
from pathlib import Path
import re

from rvc.lib.tools.text_rvc.data import digest, normalize_text, save_json, tokenize

_phonemizer = None
LANGUAGES = ("en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "tr", "hi", "ar")


def phoneme_labels(text, vocabulary, language="en"):
    if language not in LANGUAGES:
        raise ValueError("Unsupported speech language")
    global _phonemizer
    if _phonemizer is None:
        from rvc.lib.tools.tts_providers import ensure_piper_runtime

        ensure_piper_runtime()
        from piper.phonemize_espeak import EspeakPhonemizer

        _phonemizer = EspeakPhonemizer()
    spoken = "".join(
        "".join(sentence)
        for sentence in _phonemizer.phonemize(language, normalize_text(text, language))
    )
    spoken = re.sub(r"[ˈˌ.,!?;:͡]", "", spoken)
    phones = sorted(
        (phone for phone in vocabulary if not phone.startswith("<")),
        key=len,
        reverse=True,
    )
    labels = []
    for word in spoken.split():
        while word:
            phone = next((phone for phone in phones if word.startswith(phone)), None)
            if phone is None:
                if word.startswith("ː"):
                    # Some eSpeak diphthong length variants have no CTC label.
                    word = word[1:]
                    continue
                raise ValueError(f"The phoneme recognizer does not support {word[0]!r}")
            labels.append(vocabulary[phone])
            word = word[len(phone) :]
    if not labels or len(labels) > 512:
        raise ValueError("Invalid phoneme sequence length")
    return labels


def alignment_text(text):
    text = normalize_text(text)
    if re.search(r"[^a-z\s'.,!?;:\-\"()]", text):
        raise ValueError("The English aligner requires spelled-out English text")
    result = " ".join(re.sub(r"[^a-z' ]", " ", text).split())
    if not result:
        raise ValueError("The transcript contains no English letters")
    return result


def ctc_durations(log_probs, labels, frames, blank=0):
    """Viterbi alignment with legal repeated-letter transitions and full coverage."""
    import torch

    if not labels or frames < len(labels) or len(log_probs) < len(labels):
        raise ValueError("Insufficient audio frames for the transcript")
    states = [blank]
    for label in labels:
        states.extend((label, blank))
    states = torch.tensor(states)
    score = torch.full((len(states),), -float("inf"))
    score[:2] = log_probs[0, states[:2]]
    back = torch.zeros(len(log_probs), len(states), dtype=torch.long)
    can_skip = (states != blank) & (states != torch.roll(states, 2))
    can_skip[:2] = False
    for time in range(1, len(log_probs)):
        one = torch.cat((score.new_full((1,), -float("inf")), score[:-1]))
        two = torch.cat((score.new_full((2,), -float("inf")), score[:-2]))
        two[~can_skip] = -float("inf")
        best, choices = torch.stack((score, one, two)).max(0)
        score = best + log_probs[time, states]
        back[time] = choices
    state = len(states) - 2 + int(score[-1] > score[-2])
    if not torch.isfinite(score[state]):
        raise ValueError("Transcript cannot be aligned to this recording")
    path = []
    for time in range(len(log_probs) - 1, -1, -1):
        path.append(max(0, min(len(labels) - 1, (state - 1) // 2)))
        state -= int(back[time, state])
    path.reverse()
    indices = (
        torch.arange(frames).mul(len(path) / frames).long().clamp_max(len(path) - 1)
    )
    return torch.bincount(torch.tensor(path)[indices], minlength=len(labels))


def align(manifest, output, recognizer, phonemes=False, language="en"):
    if language not in LANGUAGES or not phonemes and language != "en":
        raise ValueError("This speech language requires multilingual phoneme alignment")
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new aligned manifest path")
    # Relative feature paths must retain their original base directory.
    if output.parent != manifest.parent:
        raise ValueError("Store the aligned manifest alongside the original manifest")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("format") != "applio-text-rvc-data-v1":
        raise ValueError("Unsupported prepared dataset")
    processor = (
        Wav2Vec2FeatureExtractor if phonemes else Wav2Vec2Processor
    ).from_pretrained(recognizer, local_files_only=True)
    model = (
        Wav2Vec2ForCTC.from_pretrained(recognizer, local_files_only=True).cpu().eval()
    )
    model.requires_grad_(False)
    vocab = (
        json.loads((Path(recognizer) / "vocab.json").read_text(encoding="utf-8"))
        if phonemes
        else processor.tokenizer.get_vocab()
    )
    if phonemes and (
        max(vocab.values()) >= 510 or model.config.vocab_size != len(vocab)
    ):
        raise ValueError("Unsupported phoneme vocabulary")
    provenance = {
        "method": (
            "wav2vec2-phoneme-viterbi-v1"
            if phonemes
            else "english-wav2vec2-ctc-viterbi-v1"
        ),
        "phoneme_vocabulary": vocab if phonemes else None,
        "language": language,
        "recognizer_sha256": digest(Path(recognizer) / "pytorch_model.bin"),
        "source_manifest_sha256": digest(manifest),
    }
    partial = output.with_suffix(".partial.json")
    resumed = (
        json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else None
    )
    if resumed and resumed["alignment"] != provenance:
        raise ValueError("Interrupted alignment belongs to different inputs")
    rows = resumed["records"] if resumed else []
    completed = {row["id"] for row in rows}
    for row in data["records"]:
        if row["id"] in completed:
            if digest(row["audio"]) != row["audio_sha256"]:
                raise ValueError("Previously aligned reference audio changed")
            continue
        text = normalize_text(row["text"]) if phonemes else alignment_text(row["text"])
        labels = (
            phoneme_labels(text, vocab, language)
            if phonemes
            else [
                vocab[character.upper() if character != " " else "|"]
                for character in text
            ]
        )
        if digest(row["audio"]) != row["audio_sha256"]:
            raise ValueError("Reference audio changed since preparation")
        audio, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
        divisor = math.gcd(sr, 16000)
        audio = resample_poly(audio.mean(1), 16000 // divisor, sr // divisor).astype(
            np.float32
        )
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values).logits[0]
            durations = ctc_durations(
                logits.log_softmax(-1), labels, row["frames"], model.config.pad_token_id
            )
        # EOS has no acoustic duration; boundary characters include edge silence.
        durations = durations.tolist() + [0]
        tokens = [label + 1 for label in labels] + [511] if phonemes else tokenize(text)
        if len(durations) != len(tokens) or sum(durations) != row["frames"]:
            raise ValueError("Alignment does not cover all acoustic frames")
        rows.append(
            {
                **row,
                "text": text,
                "durations": durations,
                **({"tokens": tokens} if phonemes else {}),
            }
        )
        save_json(partial, {"alignment": provenance, "records": rows})
        print(f"Aligned {row['id']}: {len(labels)} tokens", flush=True)
    save_json(
        output,
        {
            **data,
            "records": rows,
            "alignment": provenance,
        },
    )
    partial.unlink(missing_ok=True)
