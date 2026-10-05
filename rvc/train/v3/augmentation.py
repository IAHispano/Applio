"""Training-only pitch counterexamples; inference still uses its frozen vocoder.

WORLD keeps a recording's spectral envelope and aperiodicity while changing its
F0 contour. Content features stay from the original recording: the acoustic
model must therefore follow the supplied F0 instead of recovering source pitch
from content. Synthetic views never enter validation, and original files and
the original feature manifest are not overwritten. These views are a research
intervention, not a substitute for held-out natural speech evaluation.
"""

import copy
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rvc.lib.algorithm.v3.spectral import MelExtractor
from rvc.train.extract.v3 import file_hash, read_audio
from rvc.train.v3.data import AcousticDataset, atomic_json, dataset_identity


def prepare_pitch_views(
    manifest, output_manifest, per_speaker=16, seed=5678, progress=None
):
    """Create balanced octave/half-octave views alongside verified cached data.

    Both manifests must share a cache root so the normal containment checks
    remain in force. Training includes equally weighted original and synthetic
    populations by repeating the smaller population's entries, not its files.
    Exact dataset identity binds every view's content hashes and ordered entries.
    The random seed, synthesis version and transformations are recorded in the
    manifest. WORLD is imported only for preparation, not model loading.
    """
    import pyworld

    manifest, output_manifest = Path(manifest), Path(output_manifest)
    if output_manifest.resolve() == manifest.resolve():
        raise ValueError("Augmentation must not overwrite the original manifest")
    if output_manifest.parent.resolve() != manifest.parent.resolve():
        raise ValueError("Augmented manifest must use the original cache root")
    if output_manifest.name.lower() in {"manifest.json", "audio_manifest.json"}:
        raise ValueError("Choose a separate research manifest, not pipeline metadata")
    if output_manifest.exists():
        raise ValueError("Output manifest already exists; choose a new research name")
    if per_speaker < 4 or per_speaker % 4:
        raise ValueError("Views per speaker must be a positive multiple of four")
    dataset = AcousticDataset(manifest, "train", 0)
    original = dataset.manifest
    if original.get("augmentation"):
        raise ValueError(
            "Prepare pitch views from the original natural-speech manifest"
        )
    folder = manifest.parent / (output_manifest.stem + "_views")
    if not folder.resolve().is_relative_to(manifest.parent.resolve()):
        raise ValueError("Pitch-view cache must stay within the dataset directory")
    folder.mkdir(exist_ok=True)
    mel = MelExtractor(dataset.config)
    rng = np.random.default_rng(seed)
    groups = {}
    for index, entry in enumerate(dataset.entries):
        groups.setdefault(entry["speaker"], []).append(index)
    views = []
    for speaker, indices in sorted(groups.items()):
        rng.shuffle(indices)
        selected = 0
        for index in indices:
            entry = dataset.entries[index]
            # Avoid tails and silence: no pitch counterexample is possible there.
            if entry["samples"] < 2 * dataset.config.sample_rate:
                continue
            item = dataset[index]
            if float(item["voiced"].mean()) < 0.2:
                continue
            shift = [-12, -6, 6, 12][selected % 4]
            factor = 2 ** (shift / 12)
            waveform = read_audio(
                manifest.parent / entry["waveform"], dataset.config.sample_rate
            ).astype(np.float64)
            period = 5.0
            times = (
                np.arange(
                    int(len(waveform) / dataset.config.sample_rate * 1000 / period) + 1
                )
                * period
                / 1000
            )
            frame_times = np.asarray(dataset.config.timestamps(item["length"]))
            nearest = np.abs(times[:, None] - frame_times[None]).argmin(axis=1)
            f0 = np.ascontiguousarray(
                np.where(
                    item["voiced"].numpy()[nearest] > 0, item["f0"].numpy()[nearest], 0
                ),
                dtype=np.float64,
            )
            envelope = pyworld.cheaptrick(
                waveform, f0, times, dataset.config.sample_rate
            )
            aperiodicity = pyworld.d4c(waveform, f0, times, dataset.config.sample_rate)
            synthetic = pyworld.synthesize(
                f0 * factor,
                envelope,
                aperiodicity,
                dataset.config.sample_rate,
                frame_period=period,
            )
            synthetic = np.pad(synthetic, (0, max(0, len(waveform) - len(synthetic))))[
                : len(waveform)
            ]
            source_rms = float(np.sqrt(np.mean(waveform**2)))
            output_rms = float(np.sqrt(np.mean(synthetic**2)))
            gain = min(
                source_rms / max(output_rms, 1e-12),
                0.98 / max(float(np.abs(synthetic).max()), 1e-12),
            )
            synthetic = (synthetic * gain).astype(np.float32)
            if len(synthetic) != len(waveform) or not np.isfinite(synthetic).all():
                raise ValueError(
                    "Pitch synthesis produced invalid audio; originals are unchanged"
                )
            wave_path = folder / f"speaker_{speaker:03d}_view_{selected:03d}.wav"
            feature_path = wave_path.with_suffix(".npz")
            sf.write(wave_path, synthetic, dataset.config.sample_rate, subtype="FLOAT")
            with np.load(
                manifest.parent / entry["features"], allow_pickle=False
            ) as archive:
                features = {key: archive[key].copy() for key in archive.files}
            features["f0"] *= factor
            features["observed_f0"] *= factor
            with torch.inference_mode():
                features["mel"] = mel(torch.from_numpy(synthetic)[None])[0].numpy().T
            np.savez_compressed(feature_path, **features)
            view = dict(entry)
            view.update(
                features=str(feature_path.relative_to(manifest.parent)),
                waveform=str(wave_path.relative_to(manifest.parent)),
                features_sha256=file_hash(feature_path),
                waveform_sha256=file_hash(wave_path),
                pitch_view=dict(
                    semitones=shift,
                    gain=gain,
                    source_features_sha256=entry["features_sha256"],
                ),
            )
            views.append(view)
            selected += 1
            if progress:
                progress(dict(speaker=speaker, views=len(views), shift=shift))
            if selected == per_speaker:
                break
        if selected < per_speaker:
            raise ValueError(
                f"Not enough voiced training recordings for speaker {speaker}"
            )
    augmented = copy.deepcopy(original)
    # Use no physical duplicates. Ordered repeated entries make sampling balanced.
    originals = [e for e in original["segments"] if e["split"] == "train"]
    validation = [e for e in original["segments"] if e["split"] == "validation"]
    synthetic_entries = [views[i % len(views)] for i in range(len(originals))]
    augmented["segments"] = originals + synthetic_entries + validation
    augmented["augmentation"] = dict(
        method="world-pitch-counterexamples",
        version=str(pyworld.__version__),
        seed=seed,
        unique_views=len(views),
        synthetic_training_entries=len(synthetic_entries),
        original_manifest_sha256=file_hash(manifest),
        shifts=[-12, -6, 6, 12],
        validation="Original recording-disjoint validation only",
    )
    augmented["dataset_id"] = dataset_identity(augmented)
    atomic_json(output_manifest, augmented)
    return str(output_manifest)
