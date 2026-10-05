"""Recording-disjoint preparation, cache identities and frame-aligned crops.

Whole recordings are assigned to train/validation before segmentation. Source
hashes prevent byte-identical recordings crossing the split. Feature/waveform
hashes protect cache reuse; dataset identity also binds speaker IDs and segment
offsets for exact resume. Dataset tensors use content [T,C], mel [M,T] and
frame controls [T]; collate adds batch axes and masks for padded frames/samples.
"""

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rvc.configs.v3 import MelConfig, fingerprint
from rvc.train.extract.v3 import file_hash, read_audio


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
                }
                for e in manifest["segments"]
            ],
        }
    )


def _select_recordings(
    root, validation_fraction, seed, speaker_names, recordings_per_speaker
):
    """Share deterministic speaker selection, deduplication and recording splits."""
    files = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif"}
    )
    if not files:
        raise ValueError("No supported audio recordings found")
    input_recordings = len(files)
    groups = {}
    for path in files:
        groups.setdefault(str(path.relative_to(root).parent), []).append(path)
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
    speakers = sorted({str(p.relative_to(root).parent) for p in files})
    speaker_map = {name: i for i, name in enumerate(speakers)}
    records, seen = [], {}
    for path in files:
        digest = file_hash(path)
        if digest in seen:
            if seen[digest] != speaker_map[str(path.relative_to(root).parent)]:
                raise ValueError("Identical recording assigned to different speakers")
            continue
        seen[digest] = speaker_map[str(path.relative_to(root).parent)]
        records.append(
            {
                "source": str(path.relative_to(root)),
                "source_hash": digest,
                "speaker": speaker_map[str(path.relative_to(root).parent)],
            }
        )
    rng = random.Random(seed)
    for speaker in speaker_map.values():
        group = [r for r in records if r["speaker"] == speaker]
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
        root, validation_fraction, seed, speaker_names, recordings_per_speaker
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
):
    """Save resampled audio and a recording-disjoint split without loading models.

    The audio manifest binds selection, offsets and waveform hashes. Extraction
    reads these immutable segments later; changing frontend settings never needs
    to resample the originals again. Sources remain untouched.
    """
    import soundfile as sf

    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not root.is_dir() or root == output or output.is_relative_to(root):
        raise ValueError("Dataset input must exist and output must be outside its tree")
    if not 0 < validation_fraction < 1 or segment_seconds <= 0:
        raise ValueError("Invalid split or segment duration")
    mel = mel_config or MelConfig()
    speakers, records, input_recordings = _select_recordings(
        root, validation_fraction, seed, speaker_names, recordings_per_speaker
    )
    audio_cache = output / "audio" / fingerprint(asdict(mel))
    audio_cache.mkdir(parents=True, exist_ok=True)
    segment_samples = max(mel.hop_length, int(segment_seconds * mel.sample_rate))
    segment_samples -= segment_samples % mel.hop_length
    entries = []
    for index, record in enumerate(records):
        audio = read_audio(root / record["source"], mel.sample_rate)
        if np.max(np.abs(audio)) > 1.01:
            raise ValueError(
                f"Recording exceeds full scale: {record['source']}; normalize explicitly before preprocessing"
            )
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
            waveform = audio_cache / (key + ".wav")
            metadata = waveform.with_suffix(".json")
            valid = waveform.exists() and metadata.exists()
            if valid:
                valid = json.loads(metadata.read_text())[
                    "waveform_sha256"
                ] == file_hash(waveform)
            if not valid:
                temporary = waveform.with_suffix(".tmp.wav")
                sf.write(str(temporary), segment, mel.sample_rate, subtype="FLOAT")
                temporary.replace(waveform)
                atomic_json(metadata, {"waveform_sha256": file_hash(waveform)})
            entries.append(
                {
                    **record,
                    "start_sample": start,
                    "samples": len(segment),
                    "waveform": str(waveform.relative_to(output)),
                    "waveform_sha256": file_hash(waveform),
                }
            )
        if progress:
            progress(
                dict(recording=index + 1, total=len(records), source=record["source"])
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
        ),
        speakers=speakers,
        recordings=records,
        segments=entries,
    )
    manifest["audio_id"] = fingerprint(manifest)
    atomic_json(output / "audio_manifest.json", manifest)
    return output / "audio_manifest.json"


def extract_preprocessed(audio_manifest, extractor, progress=None):
    """Cache frontend features from verified audio; publish a train-ready manifest.

    Audio paths/hashes, segment lengths and split identity are checked before
    use. Feature hashes bind cache reuse to the exact frontend contract. Legacy
    combined manifests remain readable by the dataset and training code.
    """
    import soundfile as sf
    from rvc.configs.v3 import require_contract

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
    cache_id = fingerprint(contract)
    cache = output / "cache" / cache_id
    cache.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, entry in enumerate(audio["segments"]):
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
        if valid:
            hashes = json.loads(metadata.read_text())
            valid = (
                hashes.get("features_sha256") == file_hash(feature_file)
                and hashes.get("waveform_sha256") == entry["waveform_sha256"]
            )
        if not valid:
            segment, rate = sf.read(str(waveform), dtype="float32")
            if (
                rate != extractor.mel_config.sample_rate
                or segment.ndim != 1
                or len(segment) != entry["samples"]
                or not np.isfinite(segment).all()
            ):
                raise ValueError(
                    "Preprocessed audio dimensions or sample rate are invalid"
                )
            temporary = feature_file.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **extractor.extract(segment))
            temporary.replace(feature_file)
            atomic_json(
                metadata,
                dict(
                    features_sha256=file_hash(feature_file),
                    waveform_sha256=entry["waveform_sha256"],
                ),
            )
        entries.append(
            {
                **entry,
                "features": str(feature_file.relative_to(output)),
                "features_sha256": file_hash(feature_file),
            }
        )
        if progress:
            progress(
                dict(
                    recording=index + 1,
                    total=len(audio["segments"]),
                    source=entry["source"],
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


class AcousticDataset(Dataset):
    """Serve verified cached segments or frame crops for acoustic and vocoder stages."""

    def __init__(self, manifest, split="train", crop_frames=128):
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
        self.entries = [e for e in self.manifest["segments"] if e["split"] == split]
        if not self.entries:
            raise ValueError(f"Dataset has no {split} examples")
        if fingerprint(self.manifest["contract"]) != self.manifest["cache_id"]:
            raise ValueError("Manifest contract fingerprint is invalid")
        if self.manifest.get("dataset_id") != dataset_identity(self.manifest):
            raise ValueError("Manifest recording/split identity is invalid")
        for entry in self.manifest["segments"]:
            for key in ("features", "waveform"):
                if (
                    not (self.path.parent / entry[key])
                    .resolve()
                    .is_relative_to(self.path.parent.resolve())
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
        waveform = read_audio(
            self.path.parent / entry["waveform"], self.config.sample_rate
        )
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
        max(len(e["waveform"]) for e in examples),
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
