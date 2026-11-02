"""Transcribed speech preparation with holdouts and explicit feature provenance."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import urlencode
from urllib.request import urlopen

CORPUS = "https://doi.org/10.7488/ds/2645"
MIRROR = "sanchit-gandhi/vctk"
TRANSCRIBER = "openai/whisper-base"
TRANSCRIBER_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
AUDIO_EXTENSIONS = frozenset((".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"))
SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？", "؟", "।", "॥")
CLOSING_MARKS = "\"'”’»)]}"
NONTERMINAL_ABBREVIATIONS = frozenset(
    (
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "dra.",
        "prof.",
        "sr.",
        "sra.",
        "srta.",
        "jr.",
        "hr.",
        "fr.",
        "nr.",
        "e.g.",
        "i.e.",
        "vs.",
    )
)


def sentence_ending(text):
    ending = text.rstrip().rstrip(CLOSING_MARKS)
    if not ending.endswith(SENTENCE_ENDINGS):
        return False
    word = ending.split()[-1]
    # Keep titles, initials and dotted acronyms with the phrase that follows.
    return not (
        word.casefold() in NONTERMINAL_ABBREVIATIONS
        or re.fullmatch(r"(?:[^\W\d_]\.)+", word)
    )


def normalize_text(text, language="en"):
    # Read-speech corpora and pasted prose commonly mark emphasis with Markdown.
    # Keep unpaired stars and arithmetic such as 2*3 as literal text.
    text = re.sub(r"(?<!\w)(\*{1,2})(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", text)
    if language == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    text = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    if not text or len(text.encode("utf-8")) > 512:
        raise ValueError("Text must contain between 1 and 512 UTF-8 bytes")
    return text


def tokenize(text):
    # Zero is padding, 1..256 are bytes, 257 marks the end of the text.
    return [byte + 1 for byte in normalize_text(text).encode("utf-8")] + [257]


def text_chunks(text, max_bytes=120):
    words = unicodedata.normalize("NFKC", text).split()
    if not words:
        raise ValueError("Text is required")
    current = []
    for word in words:
        if len(word.encode("utf-8")) > max_bytes:
            raise ValueError("A word exceeds the speech model's text limit")
        if current and len(" ".join([*current, word]).encode("utf-8")) > max_bytes:
            yield " ".join(current)
            current = []
        current.append(word)
        if sentence_ending(word):
            yield " ".join(current)
            current = []
    if current:
        yield " ".join(current)


def boundary_pause(text, speed=1.0):
    ending = text.rstrip().rstrip(CLOSING_MARKS)
    if sentence_ending(text):
        seconds = 0.22
    elif ending.endswith((";", ":", "؛")):
        seconds = 0.12
    elif ending.endswith((",", "،", "，")):
        seconds = 0.08
    else:
        # Length limits can split a sentence mid-phrase; avoid adding a full pause.
        seconds = 0.015
    return seconds / speed


def quiet_phrase_edges(audio, sample_rate, speed=1.0):
    """Suppress low-level phrase-edge noise while preserving phrase timing."""
    import numpy as np

    hop = max(1, round(sample_rate * 0.02))
    if len(audio) < hop:
        return audio
    energy = np.array(
        [np.sqrt(np.mean(audio[i : i + hop] ** 2)) for i in range(0, len(audio), hop)]
    )
    # A relative ceiling protects quiet voices. Only sustained, low-energy
    # phrase edges qualify. Retain context around speech and leave internal gaps alone.
    threshold = min(10 ** (-45 / 20), float(energy.max()) * 10 ** (-35 / 20))
    active = np.flatnonzero(energy > threshold)
    if not len(active):
        return audio
    first = int(active[0]) * hop
    last = min(len(audio), (int(active[-1]) + 1) * hop)
    minimum = round(sample_rate * 0.12 / speed)
    retain = round(sample_rate * 0.05 / speed)
    attack = max(1, round(sample_rate * 0.02 / speed))
    release = max(1, round(sample_rate * 0.08 / speed))
    result = audio.copy()
    if first >= minimum:
        end = max(0, first - retain)
        start = max(0, end - attack)
        result[:start] = 0
        phase = np.linspace(0, np.pi, end - start, dtype=np.float32)
        result[start:end] *= (1 - np.cos(phase)) * 0.5
    if len(audio) - last >= minimum:
        start = min(len(audio), last + retain)
        end = min(len(audio), start + release)
        phase = np.linspace(0, np.pi, end - start, dtype=np.float32)
        result[start:end] *= (1 + np.cos(phase)) * 0.5
        result[end:] = 0
    return result


def has_seen_recordings(payload, rows):
    """Match both prepared audio and its identity before conversion."""
    seen = set(payload.get("seen_audio_sha256", [])) | set(
        payload.get("seen_source_audio_sha256", [])
    )
    return any(
        row["audio_sha256"] in seen or row.get("source_audio_sha256") in seen
        for row in rows
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(
            (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def vctk_records(root, speaker="p225", mic="mic1"):
    if not re.fullmatch(r"p\d{3}", speaker) or mic not in ("mic1", "mic2"):
        raise ValueError("Invalid VCTK speaker or microphone")
    root = Path(root).resolve()
    provenance = root / "source.json"
    selected = None
    if provenance.is_file():
        source = json.loads(provenance.read_text(encoding="utf-8"))
        if source.get("microphone") != mic:
            raise ValueError("Requested microphone differs from the downloaded subset")
        selected = {row["id"]: row["sha256"] for row in source["utterances"]}
    records = []
    for transcript in sorted((root / "txt" / speaker).glob("*.txt")):
        if selected is not None and transcript.stem not in selected:
            continue
        candidates = [
            root / "wav48_silence_trimmed" / speaker / f"{transcript.stem}_{mic}{ext}"
            for ext in (".flac", ".wav")
        ]
        audio = next((path for path in candidates if path.is_file()), None)
        if audio is not None:
            if selected is not None and digest(audio) != selected[transcript.stem]:
                raise ValueError(
                    "Downloaded audio changed after the subset was recorded"
                )
            records.append(
                {
                    "id": transcript.stem,
                    "speaker": speaker,
                    "audio": str(audio),
                    "text": normalize_text(transcript.read_text(encoding="utf-8-sig")),
                }
            )
    if len(records) < 4:
        raise ValueError(
            "At least four matched VCTK recordings and transcripts are required"
        )
    return records


def speech_records(root, speaker="p225", mic="mic1", language="en"):
    root = Path(root).resolve()
    if (root / "txt" / speaker).is_dir():
        return vctk_records(root, speaker, mic)
    origins = {}
    provenance = root / "source.json"
    if provenance.is_file():
        source = json.loads(provenance.read_text(encoding="utf-8"))
        if (
            source.get("transcriber_revision")
            and "sources" in source
            and "clips" in source
        ):
            originals = {row["relative"]: row for row in source["sources"]}
            counts = Counter(row["source"] for row in source["clips"])
            for clip in source["clips"]:
                original = originals[clip["source"]]
                # Keep a whole recording's identity across resampling/encoding.
                # Long-recording chunks need distinct identities for their splits.
                identity = original["audio_sha256"]
                whole = (
                    counts[clip["source"]] == 1
                    and clip["start_sample"] == 0
                    and clip["source_samples"]
                    == clip.get("source_total_samples", clip["source_samples"])
                )
                if not whole:
                    identity = hashlib.sha256(
                        f"{identity}:{clip['start_sample']}:{clip['source_samples']}:{clip['source_sample_rate']}".encode()
                    ).hexdigest()
                origins[clip["audio"]] = identity
    records = []
    for audio in sorted(root.rglob("*")):
        if not audio.is_file() or audio.suffix.lower() not in (
            ".wav",
            ".flac",
            ".ogg",
            ".aiff",
        ):
            continue
        transcript = audio.with_suffix(".txt")
        if not transcript.is_file():
            raise ValueError(
                f"Missing transcript: {transcript}. Add a UTF-8 .txt file beside each recording."
            )
        relative = audio.relative_to(root).as_posix()
        records.append(
            {
                "id": hashlib.sha256(relative.encode()).hexdigest()[:24],
                "speaker": "voice",
                "audio": str(audio),
                "text": normalize_text(
                    transcript.read_text(encoding="utf-8-sig"), language
                ),
                **(
                    {"source_audio_sha256": origins[relative]}
                    if relative in origins
                    else {}
                ),
            }
        )
    if len(records) < 4:
        raise ValueError(
            "At least four recordings with matching transcripts are required"
        )
    return records


def validate_speech(root, speaker="p225", language="en"):
    import soundfile as sf

    records = speech_records(root, speaker, language=language)
    for row in records:
        info = sf.info(row["audio"])
        if not 0.25 <= info.duration <= 12:
            raise ValueError(
                f"{row['audio']}: TTS requires transcribed clips between 0.25 and 12 seconds; split longer recordings first"
            )
    partition(records)
    return records


def audio_sources(root, speaker="p225", language="en"):
    """Inventory originals without requiring transcripts or modifying user files."""
    import soundfile as sf

    root = Path(root).resolve()
    if (root / "txt" / speaker).is_dir():
        records = vctk_records(root, speaker)
    else:
        records = []
        for audio in sorted(root.rglob("*")):
            if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            transcript = audio.with_suffix(".txt")
            records.append(
                {
                    "audio": str(audio),
                    "text": (
                        transcript.read_text(encoding="utf-8-sig").strip()
                        if transcript.is_file()
                        else ""
                    ),
                }
            )
    if not records:
        raise ValueError("No supported audio files found")
    sources = []
    for row in records:
        info = sf.info(row["audio"])
        if not math.isfinite(info.duration) or info.duration < 0.25:
            continue
        sources.append(
            {
                "audio": row["audio"],
                "relative": Path(row["audio"]).relative_to(root).as_posix(),
                "audio_sha256": digest(row["audio"]),
                "text": row["text"],
            }
        )
    if not sources:
        raise ValueError("No supported audio files found")
    return sources


def _transcriber(path, language):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    path = Path(path)
    snapshot_download(
        TRANSCRIBER,
        revision=TRANSCRIBER_REVISION,
        local_dir=str(path),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "merges.txt",
            "README.md",
            "LICENSE*",
        ],
    )
    processor = WhisperProcessor.from_pretrained(str(path), local_files_only=True)
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, use_safetensors=True
        )
        .cpu()
        .float()
        .eval()
        .requires_grad_(False)
    )
    language = "en" if language == "en-us" else language
    no_speech = processor.tokenizer.convert_tokens_to_ids("<|nospeech|>")

    def transcribe(audio):
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            encoded = model.get_encoder()(inputs.input_features, return_dict=True)
            start = torch.tensor([[model.config.decoder_start_token_id]])
            logits = model(encoder_outputs=encoded, decoder_input_ids=start).logits
            if float(logits[0, 0].softmax(-1)[no_speech]) >= 0.6:
                return ""
            tokens = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                encoder_outputs=encoded,
                language=language,
                task="transcribe",
                return_timestamps=False,
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
            )
        return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    return transcribe


def automatic_transcripts(root, output, transcriber, speaker="p225", language="en"):
    """Create short paired clips in job storage; retain supplied short-clip text."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Use a new training output directory")
    sources = audio_sources(root, speaker, language)
    provenance = output / "source.json"
    if provenance.exists():
        cached = json.loads(provenance.read_text(encoding="utf-8"))
        if (
            cached["sources"] != sources
            or cached["language"] != language
            or cached["transcriber_revision"] != TRANSCRIBER_REVISION
        ):
            raise ValueError("TTS recordings or transcripts changed after preparation")
        for clip in cached["clips"]:
            for key in ("audio", "transcript"):
                path = (output / clip[key]).resolve()
                if (
                    not path.is_relative_to(output)
                    or digest(path) != clip[key + "_sha256"]
                ):
                    raise ValueError(
                        "TTS recordings or transcripts changed after preparation"
                    )
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new training output directory")
    output.mkdir(parents=True, exist_ok=True)
    recognize, clips = None, []
    for source in sources:
        with sf.SoundFile(source["audio"]) as stream:
            rate, total = stream.samplerate, len(stream)
            start = 0
            while start < total:
                stream.seek(start)
                audio = stream.read(
                    min(total - start, 12 * rate), dtype="float32", always_2d=True
                ).mean(axis=1)
                if total - start > 12 * rate:
                    # Prefer a quiet boundary rather than cutting a word at a fixed timestamp.
                    hop = max(1, rate // 50)
                    choices = range(6 * rate, 10 * rate, hop)
                    end = min(
                        choices,
                        key=lambda i: float(np.square(audio[i : i + hop]).mean()),
                    )
                    audio = audio[: end + hop // 2]
                length = len(audio)
                if not np.isfinite(audio).all():
                    raise ValueError("Invalid transcribed speech record contract")
                divisor = math.gcd(rate, 16000)
                samples = resample_poly(
                    audio, 16000 // divisor, rate // divisor
                ).astype(np.float32)
                text, origin = source["text"], "provided"
                if not text or total > 12 * rate:
                    origin = "automatic"
                    if float(np.square(samples).mean()) < 1e-8:
                        start += length
                        continue
                    if recognize is None:
                        recognize = _transcriber(transcriber, language)
                    text = recognize(samples)
                text = normalize_text(text, language) if text.strip() else ""
                if text:
                    identity = hashlib.sha256(source["relative"].encode()).hexdigest()[
                        :24
                    ]
                    name = f"{identity}-{start:012d}"
                    audio_path, text_path = (
                        output / f"{name}.wav",
                        output / f"{name}.txt",
                    )
                    sf.write(audio_path, samples, 16000, subtype="PCM_16")
                    text_path.write_text(text + "\n", encoding="utf-8")
                    clips.append(
                        {
                            "audio": audio_path.name,
                            "transcript": text_path.name,
                            "audio_sha256": digest(audio_path),
                            "transcript_sha256": digest(text_path),
                            "source": source["relative"],
                            "start_sample": start,
                            "source_sample_rate": rate,
                            "source_samples": length,
                            "source_total_samples": total,
                            "transcript_origin": origin,
                        }
                    )
                    print(
                        json.dumps({"tts_transcribed_clip": name, "origin": origin}),
                        flush=True,
                    )
                start += length
    if audio_sources(root, speaker, language) != sources:
        raise ValueError("TTS recordings or transcripts changed after preparation")
    save_json(
        provenance,
        {
            "sources": sources,
            "clips": clips,
            "language": language,
            "transcriber": TRANSCRIBER,
            "transcriber_revision": TRANSCRIBER_REVISION,
        },
    )
    validate_speech(output, language=language)
    return output


def partition(records, validation_fraction=0.2):
    if len(records) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("A nonempty train/validation split is required")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError(
            "Duplicate utterance IDs would leak between train and validation"
        )
    ordered = sorted(
        records, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()
    )
    count = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * validation_fraction)))
    held_out = {row["id"] for row in ordered[:count]}
    held_text = {
        normalize_text(row["text"])
        for row in records
        if row["id"] in held_out and "text" in row
    }
    result = [
        {
            **row,
            "split": (
                "validation"
                if row["id"] in held_out
                or ("text" in row and normalize_text(row["text"]) in held_text)
                else "train"
            ),
        }
        for row in records
    ]
    if not any(row["split"] == "train" for row in result):
        raise ValueError(
            "Recordings need distinct transcripts for a train/validation split"
        )
    return result


def download_vctk(output, speaker="p225", limit=64):
    """Download a bounded mic1 subset from the public VCTK mirror."""
    if not re.fullmatch(r"p\d{3}", speaker) or not 4 <= limit <= 400:
        raise ValueError("Use a VCTK speaker ID and a limit between 4 and 400")
    output = Path(output).resolve()
    previous = output / "source.json"
    previous_source = (
        json.loads(previous.read_text(encoding="utf-8")) if previous.is_file() else None
    )
    previous_revision = previous_source["revision"] if previous_source else None
    previous_count = len(previous_source["utterances"]) if previous_source else 0
    selected = []
    revisions = set()
    # Metadata only is scanned; audio is downloaded solely for the chosen speaker.
    for offset in range(0, 5000, 100):
        query = urlencode(
            {
                "dataset": MIRROR,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        with urlopen(
            f"https://datasets-server.huggingface.co/rows?{query}", timeout=30
        ) as response:
            rows = json.load(response)["rows"]
        for item in rows:
            row = item["row"]
            if row["speaker_id"] != speaker:
                continue
            utterance = f"{speaker}_{row['text_id']}"
            if not re.fullmatch(r"p\d{3}_\d+", utterance):
                raise ValueError("Unexpected corpus utterance ID")
            if Path(row["file"]).name == f"{utterance}_mic2.flac":
                continue
            url = row["audio"][0]["src"]
            revision = re.search(r"/--/([a-f0-9]{40})/--/", url)
            if not revision:
                raise ValueError("Dataset revision is missing from the audio URL")
            revisions.add(revision[1])
            if len(revisions) != 1 or (
                previous_revision is not None and previous_revision != revision[1]
            ):
                raise ValueError("Dataset changed during download")
            if Path(row["file"]).name != f"{utterance}_mic1.flac":
                raise ValueError("Unexpected microphone in the VCTK mirror")
            audio = (
                output / "wav48_silence_trimmed" / speaker / f"{utterance}_mic1.flac"
            )
            text = output / "txt" / speaker / f"{utterance}.txt"
            audio.parent.mkdir(parents=True, exist_ok=True)
            text.parent.mkdir(parents=True, exist_ok=True)
            if not audio.is_file():
                with urlopen(url, timeout=30) as response:
                    content = response.read(10 * 1024 * 1024 + 1)
                if len(content) > 10 * 1024 * 1024 or content[:4] != b"fLaC":
                    raise ValueError("Invalid or oversized VCTK audio response")
                temporary = audio.with_suffix(".part")
                temporary.write_bytes(content)
                temporary.replace(audio)
            text.write_text(row["text"], encoding="utf-8")
            selected.append({"id": utterance, "sha256": digest(audio)})
            # Preserve a verified, usable subset if the mirror interrupts a larger run.
            if len(selected) >= max(4, previous_count) or len(selected) == limit:
                save_json(
                    output / "source.json",
                    {
                        "corpus": CORPUS,
                        "mirror": MIRROR,
                        "revision": next(iter(revisions)),
                        "microphone": "mic1",
                        "utterances": selected,
                    },
                )
            print(f"VCTK {speaker}: {len(selected)}/{limit}", flush=True)
            if len(selected) == limit:
                return
        if not rows:
            break
    raise ValueError(
        "The bounded metadata scan did not find enough recordings; use a local VCTK corpus"
    )


def prepare(
    root,
    output,
    embedder,
    rmvpe,
    speaker="p225",
    mic="mic1",
    version="v2",
    max_seconds=12,
    language="en",
    device="cpu",
):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torch.nn.functional as functional
    from rvc.lib.utils import HubertModelWithFinalProj
    from rvc.lib.predictors.f0 import RMVPE

    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a new output directory for each prepared dataset")
    embedding_file = Path(embedder).resolve() / "pytorch_model.bin"
    feature_space = {
        "version": version,
        "embedder_sha256": digest(embedding_file),
        "embedder_config_sha256": digest(Path(embedder) / "config.json"),
        "frame_hz": 100,
        "feature_dim": 768 if version == "v2" else 256,
        "pitch": "rmvpe",
        "rmvpe_sha256": digest(rmvpe),
    }
    model = (
        HubertModelWithFinalProj.from_pretrained(
            str(Path(embedder).resolve()), local_files_only=True
        )
        .to(device)
        .float()
        .eval()
    )
    model.requires_grad_(False)
    predictor = RMVPE(
        device=device,
        model_name=str(Path(rmvpe).resolve()),
        high_register={"enabled": False},
    )
    (output / "features").mkdir(parents=True, exist_ok=True)
    recordings = speech_records(root, speaker, mic, language)
    source_recordings = [
        {"id": row["id"], "text": row["text"], "audio_sha256": digest(row["audio"])}
        for row in recordings
    ]
    source_hashes = {row["id"]: row["audio_sha256"] for row in source_recordings}
    rows = []
    for row in recordings:
        audio, sample_rate = sf.read(row["audio"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not 0.25 <= len(audio) / sample_rate <= max_seconds:
            continue
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(
            np.float32
        )
        peak = float(np.abs(audio).max())
        if not np.isfinite(audio).all() or peak < 1e-5:
            continue
        if peak > 0.95:
            audio *= 0.95 / peak
        with torch.inference_mode():
            features = model(torch.from_numpy(audio).unsqueeze(0).to(device))[
                "last_hidden_state"
            ]
            if version == "v1":
                features = model.final_proj(features)
            features = functional.interpolate(
                features.transpose(1, 2), scale_factor=2
            ).transpose(1, 2)[0]
        f0 = predictor.get_f0(audio)
        frames = min(len(features), len(f0), len(audio) // 160)
        if frames < 4:
            continue
        path = output / "features" / f"{row['id']}.pt"
        torch.save(
            {
                "features": features[:frames].cpu().contiguous(),
                "f0": torch.tensor(f0[:frames], dtype=torch.float32),
            },
            path,
        )
        if digest(row["audio"]) != source_hashes[row["id"]]:
            raise ValueError("TTS recordings or transcripts changed after preparation")
        rows.append(
            {
                **row,
                "features": str(path.relative_to(output)),
                "frames": frames,
                "audio_sha256": source_hashes[row["id"]],
            }
        )
        print(f"Prepared {row['id']}: {frames} frames", flush=True)
    if len(rows) < 4:
        raise ValueError("Fewer than four usable utterances remain")
    save_json(
        output / "manifest.json",
        {
            "format": "applio-text-rvc-data-v1",
            "corpus": CORPUS if (Path(root) / "txt" / speaker).is_dir() else "user",
            "speaker": (
                speaker if (Path(root) / "txt" / speaker).is_dir() else Path(root).name
            ),
            "microphone": mic,
            "feature_space": feature_space,
            "source_recordings": source_recordings,
            "records": partition(rows),
        },
    )
