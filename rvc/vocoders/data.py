"""The run's folder, its dataset and its checkpoints."""

import glob
import json
import math
import os
import random
import re
import shutil

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from rvc.vocoders.mel import LogMel
from rvc.vocoders.paths import LOGS_DIR

#: Frame rate of the pitch files.
F0_RATE = 100
#: What a run keeps of its training checkpoints (``G_<step>.pth`` and
#: ``D_<step>.pth``): the latest, all of them, or none, when only the exports
#: are written and a stopped run cannot resume. Exports are always kept.
CHECKPOINT_MODES = ("latest", "all", "none")


def run_dir(model_name: str) -> str:
    """``logs/<model_name>/vocoder``: config, checkpoints, exports, events and previews."""
    return os.path.join(LOGS_DIR, model_name, "vocoder")


def load_run_config(model_name: str, default: str) -> dict:
    """The run's ``config.json``, copied from ``default`` on first use."""
    path = os.path.join(run_dir(model_name), "config.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(default, path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_filelist(path: str, root: str | None = None):
    """Rows of a filelist as [audio, f0], relative paths resolved against
    ``root`` (the working directory when None).

    A row is ``audio|f0``, or an RVC ``filelist.txt`` row
    (``audio|features|f0|f0_voiced|speaker``), whose fourth field is the pitch
    in Hz. The pitch is a ``.npy`` at ``F0_RATE`` frames per second, 0 where
    unvoiced.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    root = root or os.getcwd()
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split("|")
            if len(fields) < 2:
                continue
            audio, f0 = fields[0], fields[3] if len(fields) >= 4 else fields[1]
            rows.append([os.path.normpath(os.path.join(root, item)) for item in (audio, f0)])
    return rows


def split_holdout(entries, count: int, seed: int = 1234):
    """``(train, holdout)``: ``count`` non-mute clips drawn with ``seed``, kept
    out of training for the validation losses."""
    candidates = [i for i, entry in enumerate(entries) if "mute" not in os.path.basename(entry[0])]
    held = set(random.Random(seed).sample(candidates, min(count, len(candidates) // 10)))
    train = [entry for i, entry in enumerate(entries) if i not in held]
    return train, [entries[i] for i in sorted(held)]


def mel_frames(f0_frames: int, sample_rate: int, hop: int) -> int:
    """Mel frames covered by ``f0_frames`` frames at ``F0_RATE``."""
    return int(f0_frames * sample_rate / (hop * F0_RATE))


def f0_to_mel_rate(f0: torch.Tensor, frames: int, sample_rate: int, hop: int) -> torch.Tensor:
    """``[..., time]`` pitch at ``F0_RATE`` -> ``frames`` mel frames.

    Interpolated only between two voiced frames; elsewhere the nearer frame, so
    no frame gets a pitch halfway to zero.
    """
    length = f0.shape[-1]
    position = torch.arange(frames, device=f0.device, dtype=torch.float64)
    position = (position * hop * F0_RATE / sample_rate).clamp(max=length - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=length - 1)
    weight = (position - left).float()
    a, b = f0[..., left], f0[..., right]
    nearest = torch.where(weight < 0.5, a, b)
    return torch.where((a > 0) & (b > 0), a * (1 - weight) + b * weight, nearest)


def latest_checkpoint(directory: str, prefix: str):
    """Newest ``<prefix>_<step>.pth`` in ``directory``, or None."""
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.pth$")
    found = []
    for path in glob.glob(os.path.join(directory, f"{prefix}_*.pth")):
        match = pattern.search(os.path.basename(path))
        if match:
            found.append((int(match.group(1)), path))
    return max(found)[1] if found else None


def remove_older(directory: str, prefix: str, keep: str | None = None) -> None:
    """Delete ``<prefix>_*.pth`` in ``directory`` but ``keep``; all without it."""
    for path in glob.glob(os.path.join(directory, f"{prefix}_*.pth")):
        if keep is None or os.path.abspath(path) != os.path.abspath(keep):
            os.remove(path)


def pretrained_weights(path: str) -> dict:
    """Weights from a training checkpoint (its EMA when present) or an export."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    ema = checkpoint.get("ema")
    return ema["shadow"] if ema else checkpoint["model"]


class VocoderDataset(Dataset):
    """Fixed-length (mel, f0, audio) crops of ``segment_frames`` mel frames.

    The mel is taken over the whole clip before cropping, as at inference.
    Without ``augment`` the crop is the middle of the clip instead of random.
    """

    def __init__(self, entries, config: dict, segment_frames: int, augment: bool = True):
        self.entries = entries
        self.data = config["data"]
        self.segment_frames = int(segment_frames)
        self.hop = int(self.data["hop_length"])
        self.sample_rate = int(self.data["sample_rate"])
        self.mel = LogMel.from_config(self.data)
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def _audio(self, path):
        data, sample_rate = sf.read(path, dtype="float32")
        audio = torch.from_numpy(data)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{path} is {sample_rate} Hz; this config trains at {self.sample_rate} Hz."
            )
        return audio.mean(-1) if audio.dim() == 2 else audio

    def _f0(self, path):
        return torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))

    def __getitem__(self, index):
        audio_path, f0_path = self.entries[index]
        audio, f0 = self._audio(audio_path), self._f0(f0_path)
        frames = min(audio.shape[0] // self.hop, mel_frames(f0.shape[0], self.sample_rate, self.hop))
        f0 = f0_to_mel_rate(f0, frames, self.sample_rate, self.hop)
        if frames < self.segment_frames:
            audio = F.pad(audio, (0, self.segment_frames * self.hop - audio.shape[0]))
            f0 = F.pad(f0, (0, self.segment_frames - f0.shape[0]))
            frames = self.segment_frames
        audio = audio[: frames * self.hop]
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0))[0, :, :frames]
        if self.augment:
            start = random.randint(0, frames - self.segment_frames)
        else:
            start = (frames - self.segment_frames) // 2
        stop = start + self.segment_frames
        return mel[:, start:stop], f0[start:stop], audio[start * self.hop : stop * self.hop]

    def reference(self, max_seconds: float = 10.0):
        """The preview clip, (mel, f0, audio, path) with a leading batch axis,
        or None: the first non-mute clip of at least two seconds, in path
        order, cut to ``max_seconds``."""
        for audio_path, f0_path in sorted(self.entries):
            if "mute" in os.path.basename(audio_path):
                continue
            audio = self._audio(audio_path)
            if audio.shape[0] < 2 * self.sample_rate:
                continue
            f0 = self._f0(f0_path)
            frames = min(
                audio.shape[0] // self.hop,
                mel_frames(f0.shape[0], self.sample_rate, self.hop),
                int(max_seconds * self.sample_rate) // self.hop,
            )
            audio = audio[: frames * self.hop]
            f0 = f0_to_mel_rate(f0, frames, self.sample_rate, self.hop)
            with torch.no_grad():
                mel = self.mel(audio.unsqueeze(0))[:, :, :frames]
            return mel, f0.unsqueeze(0), audio.unsqueeze(0), audio_path
        return None


def collate_vocoder(batch):
    mel, f0, audio = zip(*batch)
    return torch.stack(mel), torch.stack(f0), torch.stack(audio).unsqueeze(1)


def amp_setup(precision: str, device: torch.device, tag: str):
    """Autocast dtype and GradScaler for the run's precision.

    FP16 needs the scaler; BF16 has FP32's exponent range and does not. A
    precision the GPU cannot run falls back to FP32.
    """
    from rvc.vocoders.terminal import warning

    precision = str(precision).lower()
    if device.type != "cuda" or precision == "fp32":
        return None, None
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        warning("BF16 is not supported on this GPU; training in FP32.", tag=tag)
        return None, None
    if precision == "fp16":
        scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10, growth_interval=2000)
        return torch.float16, scaler
    return torch.bfloat16, None


def precision_label(amp_dtype) -> str:
    tf32 = torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    if amp_dtype is None:
        return "FP32 (TF32 matmul/conv)" if tf32 else "FP32"
    if amp_dtype == torch.float16:
        return "FP16 autocast + GradScaler"
    return "BF16 autocast"


def volume_augment(mel, audio, probability: float):
    """SingingVocoders' volume augmentation: with ``probability`` per clip, a
    gain of e^-3 up to e^3 that keeps the peak under 1, on the audio and, as a
    shift, on its log mel."""
    peak = audio.abs().amax(dim=(1, 2)) + 1e-5
    high = torch.clamp(torch.log(1 / peak), max=3.0)
    shift = -3.0 + (high + 3.0) * torch.rand_like(high)
    shift = torch.where(torch.rand_like(high) < probability, shift, torch.zeros_like(shift))
    mel = torch.clamp(mel + shift.view(-1, 1, 1), min=math.log(1e-5))
    return mel, audio * shift.exp().view(-1, 1, 1)
