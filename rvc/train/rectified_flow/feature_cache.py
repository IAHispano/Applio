import hashlib
import json
import os
import random

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from rvc.lib.algorithm.rectified_flow.features import to_mel_rate
from rvc.train.rectified_flow.data_utils import FlowAudioLoader, FlowItem

CACHE_VERSION = 1
# Lengths this many frames apart sort as equal, so the batches change by epoch.
LENGTH_GRID = 8
CURVES = ("f0", "energy", "breathiness", "tension")


def augmentation_plan(count, flow_config, seed):
    """
    Get the [clip, key shift, speed] of every item of the cache: each clip as
    it is, then the offline augmentation of DiffSinger. `key_shift_scale`
    times the clips are shifted, and `time_stretch_scale` times the result is
    stretched, split between plain clips, copies of shifted ones and shifted
    ones stretched in place.

    Args:
        count (int): Number of clips.
        flow_config (dict): The `flow` section of the config.
        seed (int): Seed of the draw.
    """
    rng = random.Random(seed)
    clips = range(count)
    plan = [[clip, 0.0, 1.0] for clip in clips]
    shift_range = float(flow_config.get("key_shift_range", 0.0))
    shift_scale = 0.0
    if shift_range > 0:
        shift_scale = float(flow_config.get("key_shift_scale", 0.0))
    shifted = [
        [clip, rng.uniform(-shift_range, shift_range), 1.0]
        for clip in rng.choices(clips, k=int(shift_scale * count))
    ]
    low, high = flow_config.get("time_stretch_range", (1.0, 1.0))
    stretch_scale = 0.0
    if low < high:
        stretch_scale = float(flow_config.get("time_stretch_scale", 0.0))
    stretched = []
    if stretch_scale > 0:

        def speed():
            return low * (high / low) ** rng.random()

        plain = int(stretch_scale / (1 + shift_scale) * count)
        copies = int(shift_scale * stretch_scale / (1 + shift_scale) * count)
        in_place = int(shift_scale * stretch_scale / (1 + stretch_scale) * count)
        stretched = [[clip, 0.0, speed()] for clip in rng.choices(clips, k=plain)]
        if shifted:
            stretched += [
                [clip, shift, speed()]
                for clip, shift, _ in rng.choices(shifted, k=copies)
            ]
            for item in rng.sample(shifted, k=min(in_place, len(shifted))):
                item[2] = speed()
    return plan + shifted + stretched


def cache_folder(experiment_dir, config, entries):
    """
    Get the folder of the cache of these clips under this config. Any change
    to what the features depend on gives another folder.

    Args:
        experiment_dir (str): The directory of the experiment.
        config (dict): The rectified flow config.
        entries (list): Rows of the filelist.
    """
    flow_config = config["flow"]
    clips = "\n".join(f"{entry[0]}|{entry[4]}" for entry in entries)
    recipe = {
        "version": CACHE_VERSION,
        "data": config["data"],
        "tension": bool(flow_config["model"].get("tension", False)),
        "augmentation": [
            flow_config.get(key)
            for key in (
                "key_shift_range",
                "key_shift_scale",
                "time_stretch_range",
                "time_stretch_scale",
            )
        ],
        "seed": flow_config["seed"],
        "clips": hashlib.sha256(clips.encode()).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode())
    return os.path.join(experiment_dir, "flow_cache", fingerprint.hexdigest()[:16])


def item_path(folder, number):
    return os.path.join(folder, f"{number // 1000:05d}", f"{number:08d}.npz")


class FeatureWriter(torch.utils.data.Dataset):
    """
    Dataset that writes the items of a clip on being read and returns the
    number and the frames of each.

    Args:
        dataset (FlowAudioLoader): Dataset of the clips.
        by_clip (dict): The (number, key shift, speed) of the items of each clip.
        folder (str): Folder of the cache.
    """

    def __init__(self, dataset, by_clip, folder):
        self.dataset = dataset
        self.by_clip = by_clip
        self.clips = sorted(by_clip)
        self.folder = folder

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, position):
        clip = self.clips[position]
        source, written = None, []
        for number, key_shift, speed in self.by_clip[clip]:
            path = item_path(self.folder, number)
            if os.path.exists(path):
                with np.load(path) as saved:
                    written.append((number, int(saved["f0"].shape[0])))
                continue
            if source is None:
                source = self.dataset.get_source(clip)
            hop = int(round(self.dataset.hop_length * speed))
            item = self.dataset.get_features(*source, key_shift, hop)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Under another name until whole, so an interrupted run leaves no half file
            with open(path + ".tmp", "wb") as f:
                np.savez(
                    f,
                    mel=item.mel.numpy().astype(np.float16),
                    **{curve: getattr(item, curve).numpy() for curve in CURVES},
                )
            os.replace(path + ".tmp", path)
            written.append((number, item.mel.shape[-1]))
        return written


def build_cache(experiment_dir, config, entries, num_workers):
    """
    Compute and write what is missing of the feature cache of the clips. An
    interrupted build continues from the items already written.

    Args:
        experiment_dir (str): The directory of the experiment.
        config (dict): The rectified flow config.
        entries (list): Rows of the filelist.
        num_workers (int): Number of workers that compute the features.
    """
    flow_config = config["flow"]
    folder = cache_folder(experiment_dir, config, entries)
    index_path = os.path.join(folder, "index.json")
    if os.path.exists(index_path):
        return
    plan = augmentation_plan(len(entries), flow_config, flow_config["seed"])
    by_clip = {}
    for number, (clip, key_shift, speed) in enumerate(plan):
        by_clip.setdefault(clip, []).append((number, key_shift, speed))
    os.makedirs(folder, exist_ok=True)
    print(
        f"Writing the features of {len(plan)} items ({len(entries)} clips and their augmented copies) to '{folder}', this is done once."
    )
    dataset = FlowAudioLoader(entries, config, augment=False)
    loader = torch.utils.data.DataLoader(
        FeatureWriter(dataset, by_clip, folder),
        batch_size=None,
        num_workers=num_workers,
    )
    frames = [0] * len(plan)
    with tqdm(total=len(plan), leave=False) as pbar:
        for written in loader:
            for number, count in written:
                frames[number] = int(count)
            pbar.update(len(written))
    hop = dataset.hop_length
    items = [
        [clip, frames[number], key_shift, int(round(hop * speed)) / hop]
        for number, (clip, key_shift, speed) in enumerate(plan)
    ]
    with open(index_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"items": items}, f)
    os.replace(index_path + ".tmp", index_path)


def load_cache(experiment_dir, config, entries):
    """
    Get the folder of the feature cache and its items, each one
    [clip, frames, key shift, speed].

    Args:
        experiment_dir (str): The directory of the experiment.
        config (dict): The rectified flow config.
        entries (list): Rows of the filelist.
    """
    folder = cache_folder(experiment_dir, config, entries)
    with open(os.path.join(folder, "index.json"), "r", encoding="utf-8") as f:
        return folder, json.load(f)["items"]


class CachedFlowLoader(torch.utils.data.Dataset):
    """
    Dataset that loads the items of the feature cache, cut at random to
    `max_frames` mel frames. The content is read from the clips as it is and
    not kept twice.

    Args:
        dataset (FlowAudioLoader): Dataset of the clips the cache was made from.
        folder (str): Folder of the cache.
        items (list): The [clip, frames, key shift, speed] of each item.
        max_frames (int): Number of mel frames an item is cut to.
    """

    def __init__(self, dataset, folder, items, max_frames):
        self.dataset = dataset
        self.folder = folder
        self.items = items
        self.max_frames = int(max_frames)

    def __len__(self):
        return len(self.items)

    def get_lengths(self):
        """
        Get the number of mel frames each item gives.
        """
        frames = np.array([item[1] for item in self.items])
        return np.minimum(frames, self.max_frames)

    def get_clips(self):
        """
        Get the clip each item was made from.
        """
        return np.array([item[0] for item in self.items])

    def __getitem__(self, number):
        clip, frames, key_shift, speed = self.items[number]
        length = min(frames, self.max_frames)
        start = random.randint(0, frames - length)
        stop = start + length
        with np.load(item_path(self.folder, number)) as saved:
            mel = torch.from_numpy(saved["mel"][:, start:stop].astype(np.float32))
            curves = {
                curve: torch.from_numpy(saved[curve][start:stop]) for curve in CURVES
            }
        dataset = self.dataset
        hop = int(round(dataset.hop_length * speed))
        content = dataset.get_content(dataset.entries[clip][1])
        content = to_mel_rate(content, frames, dataset.sample_rate, hop)
        return FlowItem(
            mel=mel,
            content=content[start:stop],
            key_shift=key_shift,
            speed=speed,
            sid=int(dataset.entries[clip][4]),
            **curves,
        )


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler of items of similar length, each batch within `max_frames`
    padded mel frames and `max_items` items. The batches are formed anew every
    epoch, always the same number of them, and dealt between the GPUs. An
    epoch reads every clip once, as it is or as one of its augmented copies.

    Args:
        lengths (np.ndarray): Number of mel frames of each item.
        clips (np.ndarray): The clip each item was made from.
        max_frames (int): Number of padded mel frames of a batch.
        max_items (int): Number of items of a batch.
        seed (int, optional): Seed of the draw. Defaults to 1234.
        rank (int, optional): Rank of the current process. Defaults to 0.
        n_gpus (int, optional): Number of GPUs. Defaults to 1.
    """

    def __init__(
        self, lengths, clips, max_frames, max_items, seed=1234, rank=0, n_gpus=1
    ):
        self.lengths = np.asarray(lengths)
        self.clips = np.asarray(clips)
        self.max_frames = int(max_frames)
        self.max_items = int(max_items)
        self.seed = int(seed)
        self.rank = int(rank)
        self.n_gpus = int(n_gpus)
        self.epoch = 0
        self.formed = None
        self.count = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def form_batches(self, rng):
        order = rng.permutation(len(self.lengths))
        # One item of each clip, the first of them in the shuffled order
        order = order[np.unique(self.clips[order], return_index=True)[1]]
        order = rng.permutation(order)
        grid = self.lengths[order] // LENGTH_GRID
        order = order[np.argsort(-grid, kind="stable")]
        batches, batch, longest = [], [], 0
        for index in order.tolist():
            frames = int(self.lengths[index])
            full = len(batch) == self.max_items
            if batch and (
                full or (len(batch) + 1) * max(longest, frames) > self.max_frames
            ):
                batches.append(batch)
                batch, longest = [], 0
            batch.append(index)
            longest = max(longest, frames)
        if batch:
            batches.append(batch)
        return [batches[index] for index in rng.permutation(len(batches)).tolist()]

    def get_batches(self):
        if self.formed is not None and self.formed[0] == self.epoch:
            return self.formed[1]
        if self.count is None:
            self.count = len(self.form_batches(np.random.default_rng(self.seed)))
        # The same on every rank, which then takes its own share
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = self.form_batches(rng)
        # Every epoch has the same number of steps: the batches over it are
        # left out and the ones missing are repeated
        missing = max(self.count - len(batches), 0)
        repeated = rng.choice(len(batches), missing, replace=False).tolist()
        batches = (batches + [batches[index] for index in repeated])[: self.count]
        each =len(batches) // self.n_gpus
        if each == 0:
            raise ValueError(
                f"{len(batches)} batches is fewer than one for each of {self.n_gpus} GPUs."
            )
        batches = batches[self.rank : each * self.n_gpus : self.n_gpus]
        self.formed = (self.epoch, batches)
        return batches

    def __len__(self):
        return len(self.get_batches())

    def __iter__(self):
        return iter(self.get_batches())
