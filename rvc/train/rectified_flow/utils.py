import datetime
import math
import sys
from time import time as ttime

import torch
import torch.distributed as dist

from rvc.lib.algorithm.rectified_flow.features import normalize_mel

# Batches the statistics of the mel bins are measured over
MEL_STATS_BATCHES = 200


class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"time={current_time} | training_speed={elapsed_time_str}"


def learning_rate(base, step, warmup, total, final_ratio, anchor=(0, 0.0)):
    """
    Linear warmup, then cosine decay to `final_ratio` of the base learning rate.

    Args:
        base (float): The base learning rate.
        step (int): The current step.
        warmup (int): Number of warmup steps.
        total (int): Number of steps of the whole training.
        final_ratio (float): Share of the base learning rate reached at the end.
        anchor (tuple, optional): Step and progress along the cosine a resumed training continues from, so a changed total stretches what is left of it.
    """
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    start, done = max(anchor[0], warmup), anchor[1]
    left = min(1.0, max(0.0, (step - start) / max(1, total - start)))
    progress = done + (1.0 - done) * left
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (final_ratio + (1.0 - final_ratio) * cosine)


def cosine_progress(scale, final_ratio):
    """
    Returns how far along the cosine decay a learning rate is, from 0 to 1.

    Args:
        scale (float): The learning rate as a share of the base learning rate.
        final_ratio (float): Share of the base learning rate reached at the end.
    """
    if final_ratio >= 1.0:
        return 0.0
    height = (scale - final_ratio) / (1.0 - final_ratio)
    return math.acos(min(1.0, max(-1.0, 2.0 * height - 1.0))) / math.pi


@torch.no_grad()
def get_mel_stats(train_loader, data_config, device, n_gpus):
    """
    Returns the mean and the spread of each bin of the normalised mel over the
    first batches of every rank.

    Args:
        train_loader (DataLoader): Dataloader of the training set.
        data_config (dict): The `data` section of the config.
        device (torch.device): The device of the model.
        n_gpus (int): The total number of GPUs available for training.
    """
    total = torch.zeros(data_config["n_mels"], device=device, dtype=torch.float64)
    squares = torch.zeros_like(total)
    frames = torch.zeros((), device=device, dtype=torch.float64)
    for batch_idx, (mel, inputs) in enumerate(train_loader):
        if batch_idx >= MEL_STATS_BATCHES:
            break
        mask = inputs.mask.to(device).double()
        mel = normalize_mel(mel.to(device), data_config).double() * mask
        total += mel.sum((0, 2))
        squares += mel.square().sum((0, 2))
        frames += mask.sum()
    if n_gpus > 1:
        for value in (total, squares, frames):
            dist.all_reduce(value)
    mean = total / frames
    std = (squares / frames - mean.square()).clamp_min(0.0).sqrt()
    return mean.float(), std.float()


def freeze_voice(net_flow):
    """
    Freezes what maps time and speaker into the network, so a single speaker
    fine-tune keeps the speaker space of the pretrained model.

    Args:
        net_flow (RectifiedFlow): The flow model.
    """
    modules = [net_flow.backbone.time_mlp, *net_flow.speaker_layers()]
    for module in filter(None, modules):
        module.requires_grad_(False)


def load_pretrain(pretrain, embedder_name):
    """
    Loads the weights of a pretrained flow model and the time scale it was
    trained with, which is None when the file does not record its config.

    Args:
        pretrain (str): Path to the pre-trained flow model.
        embedder_name (str): Name of the embedder the dataset was extracted with.
    """
    try:
        pretrain_ckpt = torch.load(pretrain, map_location="cpu", weights_only=True)
        pretrain_weights = (
            pretrain_ckpt["ema"]["shadow"]
            if pretrain_ckpt.get("ema")
            else pretrain_ckpt["model"]
        )
    except Exception as e:
        print(f"The pretrain model could not be loaded: {e}")
        sys.exit(1)

    pretrain_embedder = pretrain_ckpt.get("embedder_model")
    if pretrain_embedder and pretrain_embedder.replace(
        "-", "_"
    ) != embedder_name.replace("-", "_"):
        print(
            f"The pretrain model was trained on {pretrain_embedder} features and this dataset was extracted with {embedder_name}."
        )
        sys.exit(1)

    time_scale = None
    pretrain_model = pretrain_ckpt.get("config", {}).get("flow", {}).get("model")
    if pretrain_model is not None:
        time_scale = float(
            (pretrain_model.get("backbone_args") or {}).get("time_scale", 1000.0)
        )
    return pretrain_weights, time_scale


def nonfinite_names(mel, inputs, net_flow):
    """
    Names what holds a non-finite value among a batch and the model weights.

    Args:
        mel (torch.Tensor): Mel of the batch.
        inputs (Conditioning): The inputs of the flow.
        net_flow (RectifiedFlow): The flow model.
    """
    names = [
        name
        for name, value in (("mel", mel), *zip(inputs._fields, inputs))
        if value is not None and not torch.isfinite(value).all()
    ]
    if any(not torch.isfinite(param).all() for param in net_flow.parameters()):
        names.append("model weights")
    return ", ".join(names) or "none, the loss or a gradient overflowed"
