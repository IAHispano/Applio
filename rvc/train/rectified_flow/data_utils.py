import os
import random

import numpy as np
import torch
import torch.utils.data

from rvc.lib.algorithm.rectified_flow_features import (
    FEATURE_RATE,
    TENSION_SMOOTH_SECONDS,
    LogMel,
    aperiodicity,
    f0_to_mel_rate,
    frame_energy,
    mel_frames,
    smooth_curve,
    tension,
    to_mel_rate,
    upsample_content,
)
from rvc.train.utils import load_wav_to_torch


def split_holdout(entries, count, seed=1234):
    """
    Split the filelist into the training clips and the clips held out for the
    validation loss, which are never silent files.

    Args:
        entries (list): Rows of the filelist.
        count (int): Number of clips to hold out, at most a tenth of the dataset.
        seed (int, optional): Seed of the draw. Defaults to 1234.
    """
    candidates = [
        i for i, entry in enumerate(entries) if "mute" not in os.path.basename(entry[0])
    ]
    held = set(
        random.Random(seed).sample(candidates, min(count, len(candidates) // 10))
    )
    train = [entry for i, entry in enumerate(entries) if i not in held]
    return train, [entries[i] for i in sorted(held)]


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
            to_mel_rate(curve.unsqueeze(-1), frames, self.sample_rate, hop)[0, :, 0]
            for curve in curves
        ]
        if not self.use_tension:
            curves.append(torch.zeros(frames))
        return curves

    def __getitem__(self, index):
        audiopath, content_path, _, pitchf_path, sid = self.entries[index]
        audio = self.get_audio(audiopath)
        content = self.get_content(content_path)
        pitchf = torch.FloatTensor(np.load(pitchf_path, allow_pickle=False))

        # A shift moves pitch and formants; a longer hop reads the clip faster.
        key_shift, hop = 0.0, self.hop_length
        if self.augment and random.random() < self.key_shift_prob:
            key_shift = random.uniform(-self.key_shift_range, self.key_shift_range)
        if self.augment and random.random() < self.stretch_prob:
            low, high = self.stretch_range
            hop = int(round(self.hop_length * low * (high / low) ** random.random()))
        speed = hop / self.hop_length

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

        length = min(frames, self.segment_frames)
        start = random.randint(0, frames - length) if self.augment else 0
        stop = start + length
        return (
            mel[:, start:stop],
            content[start:stop],
            f0[start:stop],
            energy[start:stop],
            breathiness[start:stop],
            key_shift,
            speed,
            int(sid),
            strain[start:stop],
        )

    def __len__(self):
        return len(self.entries)

    def get_reference(self, embedder_name, max_seconds=10.0):
        """
        Get the clip the validation audio is generated from: the reference set
        of the embedder as speaker 0, else the first dataset clip of at least
        two seconds.

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
        for audiopath, content_path, _, pitchf_path, sid in sorted(self.entries):
            if "mute" in os.path.basename(audiopath):
                continue
            audio = self.get_audio(audiopath)
            if audio.shape[0] < 2 * self.sample_rate:
                continue
            return self.get_reference_item(
                audio[: int(max_seconds * self.sample_rate)],
                self.get_content(content_path),
                torch.FloatTensor(np.load(pitchf_path, allow_pickle=False)),
                int(sid),
            )
        return None

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
        return (
            mel,
            content.unsqueeze(0),
            f0.unsqueeze(0),
            energy.unsqueeze(0),
            breathiness.unsqueeze(0),
            torch.LongTensor([sid]),
            strain.unsqueeze(0),
        )


class FlowAudioCollate:
    """
    Collate function that pads every item to `frames` mel frames and returns
    the frame mask.

    Args:
        frames (int): Number of mel frames of the batch.
    """

    def __init__(self, frames):
        self.frames = frames

    def __call__(self, batch):
        size, frames = len(batch), self.frames
        mel = torch.zeros(size, batch[0][0].shape[0], frames)
        content = torch.zeros(size, frames, batch[0][1].shape[-1])
        f0 = torch.zeros(size, frames)
        energy = torch.full((size, frames), -1.0)
        breathiness = torch.ones(size, frames)
        key_shift = torch.zeros(size)
        speed = torch.ones(size)
        sid = torch.zeros(size, dtype=torch.long)
        mask = torch.zeros(size, 1, frames)
        tension = torch.zeros(size, frames)
        for i, row in enumerate(batch):
            length = row[0].shape[-1]
            mel[i, :, :length] = row[0]
            content[i, :length] = row[1]
            f0[i, :length] = row[2]
            energy[i, :length] = row[3]
            breathiness[i, :length] = row[4]
            key_shift[i] = row[5]
            speed[i] = row[6]
            sid[i] = row[7]
            mask[i, :, :length] = 1.0
            tension[i, :length] = row[8]
        return (
            mel,
            content,
            f0,
            energy,
            breathiness,
            key_shift,
            speed,
            sid,
            mask,
            tension,
        )
