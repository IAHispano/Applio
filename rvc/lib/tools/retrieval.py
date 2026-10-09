"""Optional target-speaker content retrieval for acoustic file conversion.

Indexes contain only training content features. They do not replace pitch,
voicing, energy, speaker conditioning, acoustic weights or the renderer.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from rvc.configs.neural import FeatureConfig, require_contract


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(manifest, output, speaker=None, max_vectors=50000, seed=1234):
    """Build a bounded, deterministic FAISS index from one voice's train split."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock")
    try:
        owner = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError("This index already has a build lock; do not start duplicate builds") from error
    try:
        with owner:
            return _build_index(manifest, output, speaker, max_vectors, seed)
    finally:
        lock.unlink(missing_ok=True)


def _build_index(manifest, output, speaker, max_vectors, seed):
    import faiss

    faiss.omp_set_num_threads(2)

    if not isinstance(max_vectors, int) or not 1 <= max_vectors <= 200000:
        raise ValueError("Index vector cap must be between 1 and 200000")
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    metadata_file = output.with_suffix(output.suffix + ".json")
    if output.exists() or metadata_file.exists():
        raise ValueError("Index output already exists; choose a new output path")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    features = asdict(FeatureConfig(**data["contract"]["features"]))
    speakers = data["speakers"]
    if speaker is None and len(speakers) == 1:
        speaker = speakers[0]
    if speaker not in speakers:
        raise ValueError("Choose one target speaker name from the dataset manifest")
    sid = speakers.index(speaker)
    heldout = {e["source_hash"] for e in data["segments"] if e["split"] != "train"}
    entries = [e for e in data["segments"] if e["split"] == "train" and e["speaker"] == sid]
    if not entries or any(e["source_hash"] in heldout for e in entries):
        raise ValueError("Index needs recording-disjoint training features for this voice")
    # Priority reservoir sampling: uniform over frames, bounded memory, fixed seed.
    rng = np.random.default_rng(seed)
    vectors = np.empty((0, features["content_dim"]), dtype=np.float32)
    priorities = np.empty(0)
    total = 0
    seen_files = set()
    for entry in entries:
        file = (manifest.parent / entry["features"]).resolve()
        if not file.is_relative_to(manifest.parent):
            raise ValueError("Cached features must stay inside the dataset directory")
        if file in seen_files:
            continue
        seen_files.add(file)
        if _sha(file) != entry["features_sha256"]:
            raise ValueError("Cached content features failed their integrity check")
        with np.load(file, allow_pickle=False) as arrays:
            content = np.asarray(arrays["content"], dtype=np.float32)
        if content.ndim != 2 or content.shape[1] != features["content_dim"] or not np.isfinite(content).all():
            raise ValueError("Invalid cached content feature shape or values")
        total += len(content)
        vectors = np.concatenate((vectors, content))
        priorities = np.concatenate((priorities, rng.random(len(content))))
        if len(vectors) > max_vectors:
            keep = np.argpartition(priorities, max_vectors - 1)[:max_vectors]
            vectors, priorities = vectors[keep], priorities[keep]
    if not len(vectors):
        raise ValueError("No content frames available for this target")
    # Canonical insertion order makes repeated builds reproducible.
    vectors = np.ascontiguousarray(vectors[np.argsort(priorities)], dtype=np.float32)
    index = faiss.IndexFlatL2(features["content_dim"])
    index.add(vectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        faiss.write_index(index, str(temporary))
        metadata = dict(format="applio-acoustic-content-index", version=1,
                        features=features, speaker=speaker, dataset_id=data["dataset_id"],
                        split="train", vectors=len(vectors), available_frames=total,
                        max_vectors=max_vectors, seed=seed, algorithm="FlatL2",
                        source_recordings=len({e["source_hash"] for e in entries}),
                        index_sha256=_sha(temporary))
        metadata_temporary = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
        metadata_temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata_temporary.replace(metadata_file)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return str(output)


class ContentIndex:
    """Validated CPU nearest-neighbor retrieval; classic indexes are incompatible."""

    def __init__(self, path, features):
        import faiss

        faiss.omp_set_num_threads(2)

        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if (metadata.get("format"), metadata.get("version"), metadata.get("split"),
                metadata.get("algorithm")) != ("applio-acoustic-content-index", 1, "train", "FlatL2"):
            raise ValueError("Choose a supported acoustic content index")
        features = asdict(FeatureConfig(**features))
        require_contract(metadata["features"], features, "retrieval frontend")
        if _sha(path) != metadata["index_sha256"]:
            raise ValueError("Content index integrity check failed")
        self.index = faiss.read_index(str(path))
        if not isinstance(self.index, faiss.IndexFlatL2) or self.index.d != features["content_dim"]:
            raise ValueError("Content index dimensions or search metric disagree")
        if self.index.ntotal != metadata["vectors"] or self.index.ntotal < 1:
            raise ValueError("Content index vector count disagrees")
        self.metadata = metadata

    def require_speaker(self, speaker):
        if self.metadata["speaker"] != speaker:
            raise ValueError("Retrieval index belongs to another target speaker")

    def blend(self, content, voiced, rate, speaker, neighbors=8):
        """Blend a distance-weighted neighbor mean into voiced content frames only."""
        self.require_speaker(speaker)
        if not np.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Retrieval ratio must be finite and between zero and one")
        if rate == 0:
            return content
        content = np.asarray(content, dtype=np.float32)
        voiced = np.asarray(voiced)
        if (content.ndim != 2 or content.shape[1] != self.index.d or
                voiced.shape != (len(content),) or not np.isfinite(content).all() or
                not np.isfinite(voiced).all() or np.any((voiced < 0) | (voiced > 1))):
            raise ValueError("Retrieval needs finite, aligned content and voicing")
        if not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError("Neighbor count must be positive")
        output = content.copy()
        selected = np.flatnonzero(voiced >= .5)
        k = min(neighbors, self.index.ntotal)
        for offset in range(0, len(selected), 512):
            frames = selected[offset:offset + 512]
            distances, ids = self.index.search(np.ascontiguousarray(content[frames]), k)
            if np.any(ids < 0) or not np.isfinite(distances).all():
                raise ValueError("Content index returned invalid neighbors")
            exact = distances <= 1e-8
            weights = 1 / np.maximum(distances.astype(np.float64), 1e-8)
            weights = np.where(exact.any(axis=1, keepdims=True), exact, weights)
            weights /= weights.sum(axis=1, keepdims=True)
            retrieved = self.index.reconstruct_batch(ids.reshape(-1)).reshape(len(frames), k, self.index.d)
            mean = np.sum(retrieved * weights[..., None], axis=1).astype(np.float32)
            output[frames] = (1 - rate) * content[frames] + rate * mean
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite retrieved content")
        return output
