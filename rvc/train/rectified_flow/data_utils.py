import os
import random
from typing import NamedTuple

import numpy as np
import torch
import torch.utils.data

from rvc.lib.algorithm.rectified_flow import Conditioning
from rvc.lib.algorithm.rectified_flow.features import (
    FEATURE_RATE,
    TENSION_SMOOTH_SECONDS,
    LogMel,
    aperiodicity,
    curve_to_mel_rate,
    f0_to_mel_rate,
    frame_energy,
    mel_frames,
    smooth_curve,
    tension,
    to_mel_rate,
    upsample_content,
)
from rvc.train.utils import load_wav_to_torch


def slice_number(path):
    """
    Get the recording and the slice number of a preprocessed slice, named
    `{sid}_{idx0}_{idx1}`, or None for any other name.

    Args:
        path (str): Path of the audio file.
    """
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    return "_".join(parts[:2]), int(parts[2])


def split_holdout(entries, count, seed=1234):
    """
    Split the filelist into the training clips and the clips held out for the
    validation loss, which are never silent files. The clips are drawn from
    the speakers in turn, and the slices that overlap a held out one are left
    out of both lists.

    Args:
        entries (list): Rows of the filelist.
        count (int): Number of clips to hold out: at least one per speaker, at most a tenth of the dataset.
        seed (int, optional): Seed of the draw. Defaults to 1234.
    """
    candidates = [
        i for i, entry in enumerate(entries) if "mute" not in os.path.basename(entry[0])
    ]
    by_speaker = {}
    for index in sorted(candidates, key=lambda i: entries[i][0]):
        by_speaker.setdefault(entries[index][4], []).append(index)
    rng = random.Random(seed)
    pools = [by_speaker[sid] for sid in sorted(by_speaker)]
    for pool in pools:
        rng.shuffle(pool)
    rng.shuffle(pools)

    target = 0
    if count > 0:
        target = min(max(count, len(pools)), len(candidates) // 10)
    held = set()
    while len(held) < target:
        for pool in pools:
            if pool and len(held) < target:
                held.add(pool.pop())

    # Preprocess cuts a recording into slices that overlap the next one
    slices = {slice_number(entry[0]): i for i, entry in enumerate(entries)}
    overlapping = set()
    for index in held:
        key = slice_number(entries[index][0])
        if key is not None:
            recording, number = key
            for step in (-1, 1):
                overlapping.add(slices.get((recording, number + step), index))
    train = [
        entry
        for i, entry in enumerate(entries)
        if i not in held and i not in overlapping
    ]
    return train, [entries[i] for i in sorted(held)]


class FlowItem(NamedTuple):
    """
    One training crop, without the batch dimension.

    Args:
        mel (torch.Tensor): Mel spectrogram, shape (n_mels, frames).
        content (torch.Tensor): Content features, shape (frames, channels).
        f0 (torch.Tensor): Pitch in Hz, shape (frames,).
        energy (torch.Tensor): Loudness curve, shape (frames,).
        breathiness (torch.Tensor): Aperiodicity curve, shape (frames,).
        tension (torch.Tensor): Tension curve, shape (frames,).
        key_shift (float): Shift of pitch and formants in semitones.
        speed (float): Time stretch.
        sid (int): Speaker id.
    """

    mel: torch.Tensor
    content: torch.Tensor
    f0: torch.Tensor
    energy: torch.Tensor
    breathiness: torch.Tensor
    tension: torch.Tensor
    key_shift: float
    speed: float
    sid: int


class FlowAudioLoader(torch.utils.data.Dataset):
    """
    Dataset that loads the mel, content, pitch, loudness, breathiness and
    tension of each clip, cropped to `segment_frames` mel frames.

    Args:
        entries (list): Rows of the filelist: audio, content, pitch, voiced pitch and speaker id.
        config (dict): The rectified flow config.
        augment (bool, optional): Crop at random and draw the key shift and time stretch. Defaults to True.
    """

    def __init__(self, entries, config, augment=True):
        self.entries = entries
        self.data = config["data"]
        self.hop_length = int(self.data["hop_length"])
        self.sample_rate = int(self.data["sample_rate"])
        self.mel = LogMel.from_config(self.data)
        flow = config["flow"]
        self.segment_frames = int(flow["segment_frames"])
        self.use_tension = bool(flow["model"].get("tension", False))
        self.key_shift_range = float(flow.get("key_shift_range", 0.0))
        self.key_shift_prob = float(flow.get("key_shift_prob", 0.0))
        self.stretch_range = tuple(flow.get("time_stretch_range", (1.0, 1.0)))
        self.stretch_prob = float(flow.get("time_stretch_prob", 0.0))
        self.augment = augment

    def get_audio(self, filename):
        audio, sample_rate = load_wav_to_torch(filename)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{sample_rate} SR doesn't match target {self.sample_rate} SR"
            )
        return audio.mean(-1) if audio.dim() == 2 else audio

    def get_content(self, filename):
        content = torch.FloatTensor(np.load(filename, allow_pickle=False))
        return upsample_content(content, self.data["content_interpolation"])

    def get_curves(self, audio, f0, frames, hop):
        """
        Get the loudness, breathiness and tension curves at the mel frames.

        Args:
            audio (torch.Tensor): Audio, shape (samples,).
            f0 (torch.Tensor): Pitch of the audio at FEATURE_RATE, shape (time,).
            frames (int): Number of mel frames.
            hop (int): Hop size of the mel.
        """
        audio, f0 = audio.unsqueeze(0), f0.unsqueeze(0)
        feature_frames = audio.shape[-1] // (self.sample_rate // FEATURE_RATE)
        curves = [
            smooth_curve(frame_energy(audio, self.sample_rate, feature_frames)),
            smooth_curve(aperiodicity(audio, self.sample_rate, f0, feature_frames)),
        ]
        if self.use_tension:
            curves.append(
                smooth_curve(
                    tension(audio, self.sample_rate, f0, feature_frames),
                    TENSION_SMOOTH_SECONDS,
                )
            )
        curves = [
            curve_to_mel_rate(curve, frames, self.sample_rate, hop)[0]
            for curve in curves
        ]
        if not self.use_tension:
            curves.append(torch.zeros(frames))
        return curves

    def get_source(self, index):
        """
        Get what `get_features` takes of a clip: its audio, its pitch and
        content at FEATURE_RATE, and its speaker id.

        Args:
            index (int): Index of the clip.
        """
        audiopath, content_path, _, pitchf_path, sid = self.entries[index]
        audio = self.get_audio(audiopath)
        pitchf = torch.FloatTensor(np.load(pitchf_path, allow_pickle=False))
        return audio, pitchf, self.get_content(content_path), int(sid)

    def get_features(self, audio, pitchf, content, sid, key_shift=0.0, hop=None):
        """
        Get the whole clip as a FlowItem. A shift moves pitch and formants; a
        longer hop reads the clip faster.

        Args:
            audio (torch.Tensor): Audio, shape (samples,).
            pitchf (torch.Tensor): Pitch of the audio at FEATURE_RATE, shape (time,).
            content (torch.Tensor): Content features at FEATURE_RATE, shape (time, channels).
            sid (int): Speaker id.
            key_shift (float, optional): Shift of pitch and formants in semitones. Defaults to 0.0.
            hop (int, optional): Hop size the clip is read with, the one of the mel when None.
        """
        hop = hop or self.hop_length
        frames = min(
            audio.shape[0] // hop,
            mel_frames(pitchf.shape[0], self.sample_rate, hop),
            mel_frames(content.shape[0], self.sample_rate, hop),
        )
        audio = audio[: frames * hop]
        content = to_mel_rate(content, frames, self.sample_rate, hop)
        f0 = f0_to_mel_rate(pitchf, frames, self.sample_rate, hop)
        f0 = f0 * 2.0 ** (key_shift / 12.0)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0), key_shift, hop)[0, :, :frames]
        energy, breathiness, strain = self.get_curves(audio, pitchf, frames, hop)
        return FlowItem(
            mel=mel,
            content=content,
            f0=f0,
            energy=energy,
            breathiness=breathiness,
            tension=strain,
            key_shift=key_shift,
            speed=hop / self.hop_length,
            sid=sid,
        )

    def __getitem__(self, index):
        key_shift, hop = 0.0, self.hop_length
        if self.augment and random.random() < self.key_shift_prob:
            key_shift = random.uniform(-self.key_shift_range, self.key_shift_range)
        if self.augment and random.random() < self.stretch_prob:
            low, high = self.stretch_range
            hop = int(round(self.hop_length * low * (high / low) ** random.random()))
        item = self.get_features(*self.get_source(index), key_shift, hop)

        frames = item.mel.shape[-1]
        length = min(frames, self.segment_frames)
        start = random.randint(0, frames - length) if self.augment else 0
        stop = start + length
        return item._replace(
            mel=item.mel[:, start:stop],
            content=item.content[start:stop],
            f0=item.f0[start:stop],
            energy=item.energy[start:stop],
            breathiness=item.breathiness[start:stop],
            tension=item.tension[start:stop],
        )

    def __len__(self):
        return len(self.entries)

    def get_reference(self, embedder_name, max_seconds=10.0):
        """
        Get the mel and the flow inputs of the clip the validation audio is
        generated from: the reference set of the embedder as speaker 0, else
        the first dataset clip of at least two seconds.

        Args:
            embedder_name (str): Name of the embedder the dataset was extracted with.
            max_seconds (float, optional): Length a dataset clip is cut to. Defaults to 10.0.
        """
        reference_dir = os.path.join("logs", "reference")
        feats_path = os.path.join(reference_dir, embedder_name, "feats.npy")
        if os.path.isfile(feats_path):
            from rvc.lib.utils import load_audio

            print("Using", embedder_name, "reference set for validation")
            audio = load_audio(
                os.path.join(reference_dir, "reference.wav"), self.sample_rate
            )
            return self.get_reference_item(
                torch.FloatTensor(audio),
                self.get_content(feats_path),
                torch.FloatTensor(
                    np.load(os.path.join(reference_dir, "pitch_fine.npy"))
                ),
                0,
            )

        print("No custom reference found, using a default audio sample for validation")
        for index in sorted(range(len(self.entries)), key=lambda i: self.entries[i]):
            clip = self.get_clip(index, max_seconds)
            if clip is not None:
                return clip
        return None

    def get_clip(self, index, max_seconds=10.0):
        """
        Get the mel and the flow inputs of a dataset clip, or None for a silent
        file or a clip under two seconds.

        Args:
            index (int): Index of the clip.
            max_seconds (float, optional): Length the clip is cut to. Defaults to 10.0.
        """
        audiopath, content_path, _, pitchf_path, sid = self.entries[index]
        if "mute" in os.path.basename(audiopath):
            return None
        audio = self.get_audio(audiopath)
        if audio.shape[0] < 2 * self.sample_rate:
            return None
        return self.get_reference_item(
            audio[: int(max_seconds * self.sample_rate)],
            self.get_content(content_path),
            torch.FloatTensor(np.load(pitchf_path, allow_pickle=False)),
            int(sid),
        )

    def get_speaker_clips(self, count, max_seconds=10.0):
        """
        Get the mel and the flow inputs of up to `count` clips, of speakers
        spread over the dataset. A speaker gives a second clip only once every
        one of them has given one.

        Args:
            count (int): Number of clips.
            max_seconds (float, optional): Length a clip is cut to. Defaults to 10.0.
        """
        by_speaker = {}
        for index in sorted(range(len(self.entries)), key=lambda i: self.entries[i]):
            by_speaker.setdefault(int(self.entries[index][4]), []).append(index)
        speakers = sorted(by_speaker)
        speakers = speakers[:: max(1, len(speakers) // max(1, count))]
        clips = []
        for turn in range(max(map(len, by_speaker.values()), default=0)):
            for speaker in speakers:
                if len(clips) == count:
                    return clips
                if turn < len(by_speaker[speaker]):
                    clip = self.get_clip(by_speaker[speaker][turn], max_seconds)
                    if clip is not None:
                        clips.append(clip)
        return clips

    def get_reference_item(self, audio, content, pitchf, sid):
        frames = min(
            audio.shape[0] // self.hop_length,
            mel_frames(
                min(pitchf.shape[0], content.shape[0]),
                self.sample_rate,
                self.hop_length,
            ),
        )
        audio = audio[: frames * self.hop_length]
        content = to_mel_rate(content, frames, self.sample_rate, self.hop_length)
        f0 = f0_to_mel_rate(pitchf, frames, self.sample_rate, self.hop_length)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0))[:, :, :frames]
        energy, breathiness, strain = self.get_curves(
            audio, pitchf, frames, self.hop_length
        )
        inputs = Conditioning(
            content=content.unsqueeze(0),
            f0=f0.unsqueeze(0),
            energy=energy.unsqueeze(0),
            speaker=torch.LongTensor([sid]),
            mask=torch.ones(1, 1, frames),
            breathiness=breathiness.unsqueeze(0),
            tension=strain.unsqueeze(0),
        )
        return mel, inputs


class FlowAudioCollate:
    """
    Collate function that pads every item to `frames` mel frames and returns
    the mel and the flow inputs of the batch.

    Args:
        frames (int, optional): Number of mel frames of the batch, the longest item when None.
    """

    def __init__(self, frames=None):
        self.frames = frames

    def __call__(self, batch):
        size = len(batch)
        frames = self.frames or max(item.mel.shape[-1] for item in batch)
        mel = torch.zeros(size, batch[0].mel.shape[0], frames)
        # Padding reads as silence: no pitch, lowest energy, fully aperiodic
        inputs = Conditioning(
            content=torch.zeros(size, frames, batch[0].content.shape[-1]),
            f0=torch.zeros(size, frames),
            energy=torch.full((size, frames), -1.0),
            speaker=torch.LongTensor([item.sid for item in batch]),
            mask=torch.zeros(size, 1, frames),
            breathiness=torch.ones(size, frames),
            key_shift=torch.FloatTensor([item.key_shift for item in batch]),
            speed=torch.FloatTensor([item.speed for item in batch]),
            tension=torch.zeros(size, frames),
        )
        for i, item in enumerate(batch):
            length = item.mel.shape[-1]
            mel[i, :, :length] = item.mel
            inputs.content[i, :length] = item.content
            inputs.f0[i, :length] = item.f0
            inputs.energy[i, :length] = item.energy
            inputs.breathiness[i, :length] = item.breathiness
            inputs.tension[i, :length] = item.tension
            inputs.mask[i, :, :length] = 1.0
        return mel, inputs
