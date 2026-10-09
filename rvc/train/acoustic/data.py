"""Recording-disjoint preparation, cache identities and frame-aligned crops.

Whole recordings are assigned to train/validation before segmentation. Source
hashes prevent byte-identical recordings crossing the split. Feature/waveform
hashes protect cache reuse; dataset identity also binds speaker IDs and segment
offsets for exact resume. Dataset tensors use content [T,C], mel [M,T] and
frame controls [T]; collate adds batch axes and masks for padded frames/samples.
"""

import json
import os
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rvc.configs.neural import MelConfig, fingerprint
from rvc.train.extract.features import file_hash, read_audio
from rvc.train.acoustic.storage import (
    read_segment, storage_contract, stored_features, validate_storage, waveform_scale,
)


def atomic_json(path, value):
    """Publish complete JSON, tolerating short Windows reader/scanner locks.

    Windows readers may briefly prevent replacing the destination. Keep the old
    valid snapshot while retrying; never fall back to truncating a file in place.
    A private temporary file also prevents concurrent writers sharing one .tmp.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(payload)
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError as error:
                if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 7:
                    raise
                time.sleep(min(0.025 * 2**attempt, 0.2))
    finally:
        temporary.unlink(missing_ok=True)


def dataset_identity(manifest):
    """Bind exact resume to recordings, segmentation, split and speaker mapping."""
    return fingerprint(
        {
            "contract": manifest["cache_id"],
            "speakers": manifest["speakers"],
            "segments": [
                {
                    k: e[k]
                    for k in (
                        "source_hash",
                        "speaker",
                        "split",
                        "start_sample",
                        "samples",
                        "features_sha256",
                        "waveform_sha256",
                    )
                } | ({"waveform_scale": e["waveform_scale"]} if "waveform_scale" in e else {})
                for e in manifest["segments"]
            ],
        }
    )


def _select_recordings(
    root,
    validation_fraction,
    seed,
    speaker_names,
    recordings_per_speaker,
    progress=None,
):
    """Share deterministic speaker selection, deduplication and recording splits."""
    if progress:
        progress(
            dict(
                recording=0,
                total=1,
                source="Finding audio recordings",
                operation="scan",
            )
        )
    files = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif"}
    )
    if not files:
        raise ValueError("No supported audio recordings found")
    input_recordings = len(files)
    corpus_path = root / "corpus.json"
    corpus_speakers = None
    corpus_groups = None
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        if corpus.get("version") != 1:
            raise ValueError("Unsupported combined corpus index")
        corpus_speakers = corpus["speakers"]
        corpus_groups = corpus.get("groups")
        if corpus_groups is not None and set(corpus_groups) != set(corpus_speakers):
            raise ValueError("Corpus split groups disagree with recordings")
        available = {p.relative_to(root).as_posix(): p for p in files}
        missing = set(corpus_speakers) - available.keys()
        if missing:
            raise ValueError(f"Combined corpus recordings missing: {len(missing)}")
        files = [available[name] for name in sorted(corpus_speakers)]
    elif all(
        (root / name).is_dir() for name in ("vctk", "ears", "m4singer", "expresso")
    ):
        raise ValueError(
            "Index the combined corpus before preparation: python -m rvc.lib.tools.corpus --root DATASET"
        )

    def recording_speaker(path):
        if corpus_speakers is not None:
            return corpus_speakers[path.relative_to(root).as_posix()]
        return str(path.relative_to(root).parent)

    groups = {}
    for path in files:
        groups.setdefault(recording_speaker(path), []).append(path)
    if speaker_names:
        missing = set(speaker_names) - groups.keys()
        if missing:
            raise ValueError(f"Unknown dataset speakers: {sorted(missing)}")
        groups = {name: groups[name] for name in sorted(set(speaker_names))}
    if recordings_per_speaker < 0:
        raise ValueError("Recordings per speaker must be nonnegative")
    selection_rng = random.Random(seed)
    if recordings_per_speaker:
        for name, group in sorted(groups.items()):
            selection_rng.shuffle(group)
            groups[name] = group[:recordings_per_speaker]
    files = sorted(path for group in groups.values() for path in group)
    # Content hashes group duplicate recordings, independently of filenames.
    speakers = sorted({recording_speaker(p) for p in files})
    speaker_map = {name: i for i, name in enumerate(speakers)}
    records, seen = [], {}
    for index, path in enumerate(files):
        digest = file_hash(path)
        if progress:
            progress(
                dict(
                    recording=index + 1,
                    total=len(files),
                    source=str(path.relative_to(root)),
                    operation="hash",
                )
            )
        if digest in seen:
            if seen[digest] != speaker_map[recording_speaker(path)]:
                raise ValueError("Identical recording assigned to different speakers")
            continue
        seen[digest] = speaker_map[recording_speaker(path)]
        records.append(
            {
                "source": str(path.relative_to(root)),
                "source_hash": digest,
                "speaker": speaker_map[recording_speaker(path)],
            }
        )
    # Group once without changing record order or RNG calls: the split remains
    # identical while selection scales linearly with the recording count.
    split_groups = {speaker: [] for speaker in speaker_map.values()}
    for record in records:
        split_groups[record["speaker"]].append(record)
    rng = random.Random(seed)
    for group in split_groups.values():
        if corpus_groups is not None:
            units = {}
            for record in group:
                units.setdefault(
                    corpus_groups[Path(record["source"]).as_posix()], []
                ).append(record)
            units = list(units.values())
            rng.shuffle(units)
            n_val = (
                min(len(units) - 1, max(1, round(len(units) * validation_fraction)))
                if len(units) > 1
                else 0
            )
            for i, unit in enumerate(units):
                for record in unit:
                    record["split"] = "validation" if i < n_val else "train"
            continue
        rng.shuffle(group)
        n_val = (
            min(len(group) - 1, max(1, round(len(group) * validation_fraction)))
            if len(group) > 1
            else 0
        )
        for i, record in enumerate(group):
            record["split"] = "validation" if i < n_val else "train"
    if not any(r["split"] == "validation" for r in records):
        raise ValueError(
            "Need at least two distinct recordings for one speaker to create recording-disjoint validation"
        )
    return speakers, records, input_recordings


def prepare(
    input_dir,
    output_dir,
    extractor,
    validation_fraction=0.1,
    segment_seconds=4,
    seed=1234,
    speaker_names=None,
    recordings_per_speaker=0,
    progress=None,
):
    """Extract and cache a deterministic whole-recording split outside the source tree.

    Recordings are resampled/read without modifying originals. Segment offsets are
    retained in the manifest so held-out renders can be matched to source audio.
    The cache ID binds extraction semantics; dataset ID additionally binds the
    selected recordings, split, speaker vocabulary and concrete cache hashes.
    """

    import soundfile as sf

    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not root.is_dir() or root == output or output.is_relative_to(root):
        raise ValueError("Dataset input must exist and output must be outside its tree")
    if not 0 < validation_fraction < 1 or segment_seconds <= 0:
        raise ValueError("Invalid split or segment duration")
    speakers, records, input_recordings = _select_recordings(
        root, validation_fraction, seed, speaker_names, recordings_per_speaker, progress
    )
    contract = {
        "mel": asdict(extractor.mel_config),
        "features": asdict(extractor.config),
        "implementation": 1,
    }
    cache_id = fingerprint(contract)
    cache = output / "cache" / cache_id
    cache.mkdir(parents=True, exist_ok=True)
    segment_samples = max(
        extractor.mel_config.hop_length,
        int(segment_seconds * extractor.mel_config.sample_rate),
    )
    segment_samples -= segment_samples % extractor.mel_config.hop_length
    entries = []
    for index, record in enumerate(records):
        audio = read_audio(root / record["source"], extractor.mel_config.sample_rate)
        if np.max(np.abs(audio)) > 1.01:
            raise ValueError(
                f"Recording exceeds full scale: {record['source']}; normalize explicitly before preparation"
            )
        for start in range(0, len(audio), segment_samples):
            segment = audio[start : start + segment_samples]
            key = fingerprint(
                {
                    "record": record["source_hash"],
                    "start": start,
                    "samples": len(segment),
                    "contract": cache_id,
                }
            )
            feature_file, waveform_file = cache / (key + ".npz"), cache / (key + ".wav")
            metadata_file = cache / (key + ".json")
            valid_cache = False
            if (
                feature_file.exists()
                and waveform_file.exists()
                and metadata_file.exists()
            ):
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                valid_cache = metadata.get("features_sha256") == file_hash(
                    feature_file
                ) and metadata.get("waveform_sha256") == file_hash(waveform_file)
            if not valid_cache:
                features = extractor.extract(segment)
                temporary = feature_file.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **features)
                temporary.replace(feature_file)
                sf.write(
                    str(waveform_file),
                    segment,
                    extractor.mel_config.sample_rate,
                    subtype="FLOAT",
                )
                atomic_json(
                    metadata_file,
                    {
                        "features_sha256": file_hash(feature_file),
                        "waveform_sha256": file_hash(waveform_file),
                    },
                )
            entries.append(
                {
                    **record,
                    "start_sample": start,
                    "samples": len(segment),
                    "features": str(feature_file.relative_to(output)),
                    "waveform": str(waveform_file.relative_to(output)),
                    "features_sha256": file_hash(feature_file),
                    "waveform_sha256": file_hash(waveform_file),
                }
            )
        if progress:
            progress(
                {
                    "recording": index + 1,
                    "total": len(records),
                    "source": record["source"],
                }
            )
    manifest = {
        "schema": 1,
        "contract": contract,
        "cache_id": cache_id,
        "source_root": str(root),
        "seed": seed,
        "selection": {
            "speaker_names": list(speaker_names or []),
            "recordings_per_speaker": recordings_per_speaker,
            "input_recordings": input_recordings,
        },
        "speakers": speakers,
        "recordings": records,
        "segments": entries,
    }
    manifest["dataset_id"] = dataset_identity(manifest)
    atomic_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def preprocess_audio(
    input_dir,
    output_dir,
    validation_fraction=0.1,
    segment_seconds=4,
    seed=1234,
    speaker_names=None,
    recordings_per_speaker=0,
    mel_config=None,
    progress=None,
    normalize_overflow=False,
    validation_sources=(),
    recording_splits=None,
    compact_cache=False,
):
    """Save resampled audio and a recording-disjoint split without loading models.

    The audio manifest binds selection, offsets and waveform hashes. Extraction
    reads these immutable segments later; changing frontend settings never needs
    to resample the originals again. Sources remain untouched. Research controls
    can supply a complete relative-path recording_splits map to reuse an exact
    external train/validation split instead of adding automatic validation.
    """
    import soundfile as sf

    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not root.is_dir() or root == output or output.is_relative_to(root):
        raise ValueError("Dataset input must exist and output must be outside its tree")
    if not 0 < validation_fraction < 1 or segment_seconds <= 0:
        raise ValueError("Invalid split or segment duration")
    mel = mel_config or MelConfig()
    speakers, records, input_recordings = _select_recordings(
        root, validation_fraction, seed, speaker_names, recordings_per_speaker, progress
    )
    if recording_splits is not None:
        explicit = {
            Path(name).as_posix(): split for name, split in recording_splits.items()
        }
        selected = {Path(record["source"]).as_posix() for record in records}
        if set(explicit) != selected or set(explicit.values()) - {
            "train",
            "validation",
        }:
            raise ValueError(
                "Explicit recording splits must cover exactly the selected sources"
            )
        for record in records:
            record["split"] = explicit[Path(record["source"]).as_posix()]
        if not any(record["split"] == "validation" for record in records):
            raise ValueError(
                "Explicit recording splits must include validation recordings"
            )
    reserved = set(validation_sources)
    reserved_hashes = set()
    for index, name in enumerate(sorted(reserved)):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Reserved validation source must exist inside the dataset")
        reserved_hashes.add(file_hash(path))
        if progress:
            progress(
                dict(
                    recording=index + 1,
                    total=len(reserved),
                    source=name,
                    operation="reserve_validation",
                )
            )
    for record in records:
        if record["source_hash"] in reserved_hashes:
            if recording_splits is not None and record["split"] != "validation":
                raise ValueError(
                    "Reserved validation conflicts with the explicit recording split"
                )
            record["split"] = "validation"
    if any(
        not any(r["speaker"] == i and r["split"] == "train" for r in records)
        for i in range(len(speakers))
    ):
        raise ValueError(
            "Reserved validation leaves a speaker without training recordings"
        )
    storage = storage_contract(compact_cache)
    audio_key = asdict(mel) if storage is None else dict(mel=asdict(mel), storage=storage)
    audio_cache = output / "audio" / fingerprint(audio_key)
    audio_cache.mkdir(parents=True, exist_ok=True)
    segment_samples = max(mel.hop_length, int(segment_seconds * mel.sample_rate))
    segment_samples -= segment_samples % mel.hop_length
    entries = []
    for index, record in enumerate(records):
        audio = read_audio(root / record["source"], mel.sample_rate)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.01:
            if not normalize_overflow:
                raise ValueError(
                    f"Recording exceeds full scale: {record['source']}; normalize explicitly before preprocessing"
                )
            # Scale only derived segments; source hashes still identify original recordings.
            record["normalization_gain"] = 0.98 / peak
            audio = audio * record["normalization_gain"]
        for start in range(0, len(audio), segment_samples):
            segment = audio[start : start + segment_samples]
            key = fingerprint(
                dict(
                    record=record["source_hash"],
                    start=start,
                    samples=len(segment),
                    mel=asdict(mel),
                )
            )
            if storage is not None:
                key = fingerprint(dict(segment=key, storage=storage))
            waveform = audio_cache / (key + (".flac" if storage is not None else ".wav"))
            scale = waveform_scale(segment) if storage is not None else 1.0
            metadata = waveform.with_suffix(".json")
            valid = waveform.exists() and metadata.exists()
            waveform_hash = file_hash(waveform) if valid else None
            if valid:
                valid = (
                    json.loads(metadata.read_text(encoding="utf-8"))["waveform_sha256"]
                    == waveform_hash
                )
            if not valid:
                temporary = waveform.with_name(waveform.stem + ".tmp" + waveform.suffix)
                sf.write(str(temporary), segment / scale if storage is not None else segment,
                         mel.sample_rate, subtype="PCM_24" if storage is not None else "FLOAT")
                temporary.replace(waveform)
                waveform_hash = file_hash(waveform)
                atomic_json(metadata, {"waveform_sha256": waveform_hash})
            entries.append(
                {
                    **record,
                    "start_sample": start,
                    "samples": len(segment),
                    "waveform": str(waveform.relative_to(output)),
                    "waveform_sha256": waveform_hash,
                }
            )
            if storage is not None:
                entries[-1]["waveform_scale"] = scale
        if progress:
            progress(
                dict(
                    recording=index + 1,
                    total=len(records),
                    source=record["source"],
                    operation="preprocess",
                )
            )
    manifest = dict(
        schema=1,
        mel=asdict(mel),
        source_root=str(root),
        seed=seed,
        selection=dict(
            speaker_names=list(speaker_names or []),
            recordings_per_speaker=recordings_per_speaker,
            input_recordings=input_recordings,
            normalize_overflow=normalize_overflow,
            reserved_validation_sources=sorted(reserved),
        ),
        speakers=speakers,
        recordings=records,
        segments=entries,
    )
    if recording_splits is not None:
        manifest["selection"]["recording_splits"] = explicit
    if storage is not None:
        manifest["storage"] = storage
    manifest["audio_id"] = fingerprint(manifest)
    atomic_json(output / "audio_manifest.json", manifest)
    return output / "audio_manifest.json"


def extract_preprocessed(audio_manifest, extractor, progress=None):
    """Cache frontend features from verified audio; publish a train-ready manifest.

    Audio paths/hashes, segment lengths and split identity are checked before
    use. Feature hashes bind cache reuse to the exact frontend contract. Legacy
    combined manifests remain readable by the dataset and training code.
    """
    from rvc.configs.neural import require_contract

    path = Path(audio_manifest)
    audio = json.loads(path.read_text(encoding="utf-8"))
    identity = {key: value for key, value in audio.items() if key != "audio_id"}
    if audio.get("schema") != 1 or fingerprint(identity) != audio.get("audio_id"):
        raise ValueError("Preprocessed audio manifest is invalid; run Preprocess again")
    require_contract(asdict(extractor.mel_config), audio["mel"], "preprocessed mel")
    output = path.parent.resolve()
    contract = dict(
        mel=audio["mel"], features=asdict(extractor.config), implementation=1
    )
    storage = audio.get("storage")
    validate_storage(storage)
    if storage is not None:
        contract["storage"] = storage
    cache_id = fingerprint(contract)
    cache = output / "cache" / cache_id
    cache.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, entry in enumerate(audio["segments"]):
        if ("waveform_scale" in entry) != (storage is not None):
            raise ValueError("Waveform scale disagrees with cache storage contract")
        waveform = (output / entry["waveform"]).resolve()
        if (
            not waveform.is_relative_to(output)
            or file_hash(waveform) != entry["waveform_sha256"]
        ):
            raise ValueError(
                f"Preprocessed audio changed: {entry['source']}; run Preprocess again"
            )
        key = fingerprint(
            dict(
                record=entry["source_hash"],
                start=entry["start_sample"],
                samples=entry["samples"],
                contract=cache_id,
            )
        )
        feature_file = cache / (key + ".npz")
        metadata = feature_file.with_suffix(".json")
        valid = feature_file.exists() and metadata.exists()
        feature_hash = file_hash(feature_file) if valid else None
        if valid:
            hashes = json.loads(metadata.read_text(encoding="utf-8"))
            valid = (
                hashes.get("features_sha256") == feature_hash
                and hashes.get("waveform_sha256") == entry["waveform_sha256"]
            )
        if not valid:
            segment = read_segment(waveform, entry, extractor.mel_config.sample_rate)
            temporary = feature_file.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **stored_features(extractor.extract(segment), storage))
            temporary.replace(feature_file)
            feature_hash = file_hash(feature_file)
            atomic_json(
                metadata,
                dict(
                    features_sha256=feature_hash,
                    waveform_sha256=entry["waveform_sha256"],
                ),
            )
        entries.append(
            {
                **entry,
                "features": str(feature_file.relative_to(output)),
                "features_sha256": feature_hash,
            }
        )
        if progress:
            progress(
                dict(
                    recording=index + 1,
                    total=len(audio["segments"]),
                    source=entry["source"],
                    operation="extract",
                )
            )
    manifest = {
        key: audio[key]
        for key in (
            "schema",
            "source_root",
            "seed",
            "selection",
            "speakers",
            "recordings",
        )
    }
    manifest.update(
        contract=contract,
        cache_id=cache_id,
        segments=entries,
        preprocessing_id=audio["audio_id"],
    )
    manifest["dataset_id"] = dataset_identity(manifest)
    atomic_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def _link_cached_waveform(source, output, relative):
    """Keep immutable shared audio storage inside the destination cache boundary."""
    waveform = (source / relative).resolve()
    if not waveform.is_relative_to(source.resolve()):
        raise ValueError("Source waveform must remain inside its cache directory")
    destination = output / "waveforms" / (file_hash(waveform) + waveform.suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not os.path.samefile(waveform, destination):
        # Distinct recordings can yield byte-identical segments (for example
        # silence). Preserve a hardlink to each immutable source rather than
        # treating a content-hash name collision as a changed cache.
        destination = destination.with_name(
            destination.stem
            + "_"
            + fingerprint(Path(relative).as_posix())[:16]
            + waveform.suffix
        )
    if destination.exists():
        if not os.path.samefile(waveform, destination):
            raise ValueError("Retargeted waveform must share immutable source storage")
    else:
        os.link(waveform, destination)
    return str(destination.relative_to(output))


def retarget_preprocessed_audio(source_manifest, output_dir, mel_config):
    """Reuse a verified recording split for a different spectral contract.

    This changes only mel semantics before feature extraction. Sample rate and
    hop must remain fixed. Audio hardlinks preserve storage and are immutable;
    no original recording is resampled again or reassigned to another split.
    """
    source, output = Path(source_manifest).resolve(), Path(output_dir).resolve()
    if source.parent == output:
        raise ValueError("Retargeted audio needs a separate output directory")
    prepared = json.loads(source.read_text(encoding="utf-8"))
    identity = {key: value for key, value in prepared.items() if key != "audio_id"}
    if prepared.get("schema") != 1 or fingerprint(identity) != prepared.get("audio_id"):
        raise ValueError("Source audio manifest identity is invalid")
    old = MelConfig(**prepared["mel"])
    if (old.sample_rate, old.hop_length) != (
        mel_config.sample_rate,
        mel_config.hop_length,
    ):
        raise ValueError("Retargeting requires unchanged sample rate and hop")
    prepared["mel"] = asdict(mel_config)
    for entry in prepared["segments"]:
        waveform = (source.parent / entry["waveform"]).resolve()
        if (
            not waveform.is_relative_to(source.parent)
            or file_hash(waveform) != entry["waveform_sha256"]
        ):
            raise ValueError("Source preprocessed waveform changed")
        entry["waveform"] = _link_cached_waveform(
            source.parent, output, entry["waveform"]
        )
    prepared.pop("audio_id")
    prepared["audio_id"] = fingerprint(prepared)
    path = output / "audio_manifest.json"
    atomic_json(path, prepared)
    return path


def retarget_mel_manifest(source_manifest, output_dir, mel_config, progress=None):
    """Recompute mel targets while reusing verified content, pitch and audio.

    Sample rate/hop must stay unchanged, so the feature clock and segmentation
    remain valid. A new immutable manifest/cache is published; originals are
    untouched. Waveforms are hardlinked, not copied, to avoid duplicating a
    corpus merely because a frozen renderer uses different spectral semantics.
    This is an explicit new experiment, not compatible optimizer continuation.
    """
    import copy
    from rvc.lib.algorithm.acoustic.spectral import MelExtractor

    source, output = Path(source_manifest).resolve(), Path(output_dir).resolve()
    if source.parent == output:
        raise ValueError("Retargeted data must use a separate output directory")
    original = json.loads(source.read_text(encoding="utf-8"))
    # Shared dataset validation checks identity, split leakage and contracts.
    AcousticDataset(source, "train", 0, load_waveform=False)
    old = MelConfig(**original["contract"]["mel"])
    if (old.sample_rate, old.hop_length) != (
        mel_config.sample_rate,
        mel_config.hop_length,
    ):
        raise ValueError(
            "Retargeting requires unchanged sample rate and hop; extract again otherwise"
        )
    contract = copy.deepcopy(original["contract"])
    contract["mel"] = asdict(mel_config)
    cache_id = fingerprint(contract)
    cache = output / "cache" / cache_id
    cache.mkdir(parents=True, exist_ok=True)
    extractor = MelExtractor(mel_config)

    entries = []
    for index, entry in enumerate(original["segments"]):
        feature_source, waveform = (
            source.parent / entry["features"],
            source.parent / entry["waveform"],
        )
        if (
            file_hash(feature_source) != entry["features_sha256"]
            or file_hash(waveform) != entry["waveform_sha256"]
        ):
            raise ValueError("Source feature/waveform cache changed")
        with np.load(feature_source, allow_pickle=False) as loaded:
            features = {
                key: np.asarray(loaded[key], dtype=np.float32) for key in loaded.files
            }
        audio = read_segment(waveform, entry, mel_config.sample_rate)
        frames = mel_config.frames(len(audio))
        if any(
            len(value) != frames or not np.isfinite(value).all()
            for value in features.values()
        ):
            raise ValueError("Source features are nonfinite or misaligned")
        with torch.inference_mode():
            features["mel"] = extractor(torch.from_numpy(audio)[None])[0].T.numpy()
        key = fingerprint(
            dict(source_features=entry["features_sha256"], contract=cache_id)
        )
        destination = cache / (key + ".npz")
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **stored_features(features, contract.get("storage")))
        temporary.replace(destination)
        entries.append(
            dict(
                entry,
                waveform=_link_cached_waveform(
                    source.parent, output, entry["waveform"]
                ),
                features=str(destination.relative_to(output)),
                features_sha256=file_hash(destination),
            )
        )
        if progress:
            progress(
                dict(
                    recording=index + 1,
                    total=len(original["segments"]),
                    operation="retarget_mel",
                )
            )
    manifest = copy.deepcopy(original)
    manifest.update(contract=contract, cache_id=cache_id, segments=entries)
    manifest["mel_retargeting"] = dict(
        source_manifest_sha256=file_hash(source),
        source_dataset_id=original["dataset_id"],
        waveform_storage="Hardlinked immutable source cache",
    )
    audio_source = source.parent / "audio_manifest.json"
    if audio_source.exists():
        prepared = json.loads(audio_source.read_text(encoding="utf-8"))
        audio_identity = {
            key: value for key, value in prepared.items() if key != "audio_id"
        }
        if (
            fingerprint(audio_identity) != prepared["audio_id"]
            or original.get("preprocessing_id") != prepared["audio_id"]
        ):
            raise ValueError("Source audio manifest identity is invalid")
        prepared["mel"] = asdict(mel_config)
        for entry in prepared["segments"]:
            entry["waveform"] = _link_cached_waveform(
                source.parent, output, entry["waveform"]
            )
        prepared.pop("audio_id")
        prepared["audio_id"] = fingerprint(prepared)
        atomic_json(output / "audio_manifest.json", prepared)
        manifest["preprocessing_id"] = prepared["audio_id"]
    manifest["dataset_id"] = dataset_identity(manifest)
    path = output / "manifest.json"
    atomic_json(path, manifest)
    return path


class AcousticDataset(Dataset):
    """Serve verified cached segments or frame crops for acoustic and vocoder stages."""

    def __init__(self, manifest, split="train", crop_frames=128, *, load_waveform=True):
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != 1:
            raise ValueError("Unsupported dataset schema")
        audio_path = self.path.parent / "audio_manifest.json"
        if audio_path.exists():
            prepared = json.loads(audio_path.read_text(encoding="utf-8"))
            if self.manifest.get("preprocessing_id") != prepared.get("audio_id"):
                raise ValueError(
                    "Audio preprocessing changed; run Extract Features before training"
                )
        self.config = MelConfig(**self.manifest["contract"]["mel"])
        validate_storage(self.manifest["contract"].get("storage"))
        self.entries = [e for e in self.manifest["segments"] if e["split"] == split]
        if not self.entries:
            raise ValueError(f"Dataset has no {split} examples")
        if fingerprint(self.manifest["contract"]) != self.manifest["cache_id"]:
            raise ValueError("Manifest contract fingerprint is invalid")
        if self.manifest.get("dataset_id") != dataset_identity(self.manifest):
            raise ValueError("Manifest recording/split identity is invalid")
        dataset_root = self.path.parent.resolve()
        for entry in self.manifest["segments"]:
            if ("waveform_scale" in entry) != (self.manifest["contract"].get("storage") is not None):
                raise ValueError("Waveform scale disagrees with cache storage contract")
            for key in ("features", "waveform"):
                if (
                    not (self.path.parent / entry[key])
                    .resolve()
                    .is_relative_to(dataset_root)
                ):
                    raise ValueError(
                        "Cache paths must remain within the dataset directory"
                    )
            if not 0 <= entry["speaker"] < len(self.manifest["speakers"]):
                raise ValueError("Invalid cached speaker ID")
        train_hashes = {
            e["source_hash"] for e in self.manifest["segments"] if e["split"] == "train"
        }
        val_hashes = {
            e["source_hash"]
            for e in self.manifest["segments"]
            if e["split"] == "validation"
        }
        if train_hashes & val_hashes:
            raise ValueError("Recording leakage across train/validation splits")
        # Acoustic objectives use features only; keep waveform integrity checks
        # but avoid decoding/transferring audio unless a vocoder needs it.
        self.load_waveform = load_waveform
        self.crop_frames = crop_frames
        self.split = split
        self.verified_entries = set()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        if index not in self.verified_entries:
            for key in ("features", "waveform"):
                if file_hash(self.path.parent / entry[key]) != entry[key + "_sha256"]:
                    raise ValueError(f"Cached {key} content differs from the manifest")
            self.verified_entries.add(index)
        with np.load(self.path.parent / entry["features"], allow_pickle=False) as data:
            features = {
                key: np.asarray(data[key], dtype=np.float32) for key in data.files
            }
        frames = len(features["mel"])
        if any(
            len(value) != frames or not np.isfinite(value).all()
            for value in features.values()
        ):
            raise ValueError(f"Corrupt or misaligned features: {entry['features']}")
        if frames != self.config.frames(entry["samples"]) or features["mel"].shape != (
            frames,
            self.config.n_mels,
        ):
            raise ValueError("Cached mel shape disagrees with frame/sample contract")
        if features["content"].shape != (
            frames,
            self.manifest["contract"]["features"]["content_dim"],
        ):
            raise ValueError("Cached content width disagrees with encoder contract")
        if any(
            features[k].shape != (frames,)
            for k in (
                "f0",
                "observed_f0",
                "voiced",
                "confidence",
                "confidence_valid",
                "energy",
            )
        ):
            raise ValueError("Cached pitch/energy controls must be scalar per frame")
        if np.any(features["f0"] < 0) or np.any(
            (features["voiced"] != 0) & (features["voiced"] != 1)
        ):
            raise ValueError("Invalid pitch/voicing controls")
        start = (
            int(torch.randint(max(frames - self.crop_frames + 1, 1), ()).item())
            if self.crop_frames and self.split == "train"
            else 0
        )
        end = min(frames, start + self.crop_frames) if self.crop_frames else frames
        result = {
            key: torch.from_numpy(value[start:end].copy())
            for key, value in features.items()
        }
        result["speaker"] = torch.tensor(entry["speaker"], dtype=torch.long)
        result["length"] = end - start
        if not self.load_waveform:
            return result
        waveform = read_segment(self.path.parent / entry["waveform"], entry, self.config.sample_rate)
        begin, count = (
            start * self.config.hop_length,
            (end - start) * self.config.hop_length,
        )
        waveform = waveform[begin : begin + count]
        result["waveform"] = torch.from_numpy(
            np.pad(waveform, (0, count - len(waveform)))
        )
        result["waveform_length"] = len(waveform)
        return result


def collate(examples):
    frames, samples = (
        max(e["length"] for e in examples),
        max((len(e.get("waveform", ())) for e in examples), default=0),
    )
    result = {}
    for key in examples[0]:
        if key in {"length", "waveform_length"}:
            result[key] = torch.tensor([e[key] for e in examples])
        elif key == "speaker":
            result[key] = torch.stack([e[key] for e in examples])
        else:
            target = samples if key == "waveform" else frames
            values = []
            for example in examples:
                x = example[key]
                padding = torch.zeros((target - len(x),) + x.shape[1:], dtype=x.dtype)
                values.append(torch.cat([x, padding]))
            result[key] = torch.stack(values)
    result["mask"] = (torch.arange(frames)[None] < result["length"][:, None]).float()[
        :, None
    ]
    result["mel"] = result["mel"].transpose(1, 2)
    if "waveform" in result:
        result["waveform_mask"] = (
            torch.arange(samples)[None] < result["waveform_length"][:, None]
        ).float()
    return result


def condition_batch(model, batch):
    return model.condition(
        batch["content"],
        batch["f0"],
        batch["voiced"],
        batch["energy"],
        batch["speaker"],
        batch["confidence"],
        batch["confidence_valid"],
        batch["mask"],
    )
