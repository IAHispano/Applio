"""Index a combined corpus without copying audio or merging speaker identities.

Run with ``python -m rvc.lib.tools.corpus --root assets/datasets/combined``.
The recording index preserves original paths and metadata. Preparation consumes
only the completed speaker map; interrupted indexing never publishes that map.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def speaker_identity(relative):
    """Use real corpus identities rather than style, song or session folders."""
    parts = relative.parts
    dataset = parts[0]
    if dataset in {"vctk", "ears"}:
        if len(parts) < 3 or not re.fullmatch(r"p\d+", parts[1]):
            raise ValueError(f"Unrecognized {dataset} speaker: {relative}")
        speaker = parts[1]
    elif dataset == "m4singer":
        speaker = parts[1].split("#", 1)[0]
        if not re.fullmatch(r"(?:Alto|Bass|Soprano|Tenor)-\d+", speaker):
            raise ValueError(f"Unrecognized singing identity: {relative}")
    elif dataset == "expresso":
        match = re.match(r"(ex\d+)_", relative.stem)
        if not match:
            raise ValueError(f"Ambiguous Expresso identity: {relative}")
        speaker = match[1]
        if speaker not in parts:
            raise ValueError(f"Expresso filename/folder disagreement: {relative}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return f"{dataset}/{speaker}"


def index_corpus(root):
    root = Path(root).resolve()
    paths = sorted(root.rglob("*.wav"))
    if not paths:
        raise ValueError("No recordings found")
    rows_path = root / "recordings.jsonl"
    temporary = rows_path.with_suffix(".jsonl.tmp")
    speakers, hashes, failures = {}, {}, []
    counts, seconds = Counter(), Counter()
    with temporary.open("w", encoding="utf-8") as output:
        for path in tqdm(paths, desc="Indexing corpus", unit="files"):
            relative = path.relative_to(root)
            try:
                identity = speaker_identity(relative)
                info = sf.info(path)
                if info.frames <= 0 or info.samplerate <= 0:
                    raise ValueError("Empty or invalid audio")
                digest = hashlib.sha256()
                with path.open("rb") as audio:
                    for block in iter(lambda: audio.read(1024 * 1024), b""):
                        digest.update(block)
                checksum = digest.hexdigest()
                duplicate = hashes.get(checksum)
                row = dict(
                    path=relative.as_posix(),
                    speaker=identity,
                    dataset=relative.parts[0],
                    frames=info.frames,
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    duration=info.duration,
                    bytes=path.stat().st_size,
                    sha256=checksum,
                    duplicate_of=duplicate,
                )
                output.write(json.dumps(row) + "\n")
                if duplicate:
                    # Duplicate files are preserved but selected only once.
                    if speakers[duplicate] != identity:
                        raise ValueError("Duplicate audio has conflicting identities")
                    continue
                hashes[checksum] = relative.as_posix()
                speakers[relative.as_posix()] = identity
                counts[relative.parts[0]] += 1
                seconds[relative.parts[0]] += info.duration
            except (ValueError, RuntimeError, OSError) as error:
                failures.append(dict(path=relative.as_posix(), error=str(error)))
    temporary.replace(rows_path)
    report = dict(
        files=len(paths),
        selected=len(speakers),
        speaker_count=len(set(speakers.values())),
        counts=dict(counts),
        hours={k: v / 3600 for k, v in seconds.items()},
        failures=failures,
    )
    (root / "inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        raise ValueError(f"{len(failures)} corpus failures; inspect inventory.json")
    map_path = root / "corpus.json"
    map_temp = map_path.with_suffix(".json.tmp")
    map_temp.write_text(
        json.dumps(
            dict(
                version=1,
                speakers=speakers,
                groups={
                    name: "/".join(Path(name).parts[:2])
                    if name.startswith("m4singer/")
                    else name
                    for name in speakers
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    map_temp.replace(map_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    index_corpus(parser.parse_args().root)
