import os
import sys

os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
import datetime
import glob
import json
import math
from random import randint
from time import time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda
from rvc.lib.algorithm.rectified_flow import build_flow, match_inputs, resize_speakers
from rvc.lib.algorithm.rectified_flow_features import denormalize_mel, normalize_mel
from rvc.lib.algorithm.vocoders import load_vocoder
from rvc.train.rectified_flow.data_utils import (
    FlowAudioCollate,
    FlowAudioLoader,
    split_holdout,
)
from rvc.train.rectified_flow.ema import WeightEMA
from rvc.train.rectified_flow.feature_cache import (
    BucketBatchSampler,
    CachedFlowLoader,
    build_cache,
    load_cache,
)
from rvc.train.rectified_flow.muon import MuonAdamW
from rvc.train.utils import (
    latest_checkpoint_path,
    load_filepaths_and_text,
    load_wav_to_torch,
    plot_spectrogram_to_numpy,
    summarize,
)

# Parse command line arguments
model_name = sys.argv[1]
save_every_epoch = int(sys.argv[2])
total_epoch = int(sys.argv[3])
pretrain = sys.argv[4]
vocoder_path = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])


def _strtobool(val):
    return val.lower() in ("yes", "true", "t", "y", "1")


save_only_latest = _strtobool(sys.argv[8])
save_every_weights = _strtobool(sys.argv[9])
cleanup = _strtobool(sys.argv[10])
feature_cache = _strtobool(sys.argv[11])

# Sampling steps of the validation audio
preview_steps = 16
# Held out clips of different speakers rendered beside the reference clip
preview_clips = 3
# Where in the trained time range the validation loss is taken
eval_fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
# FP16: the weight gradients overflow with the scale around 2^20, so the
# GradScaler is not left to grow until a step is skipped
max_grad_scale = 2.0**16
# Steps skipped in a row over a non-finite gradient before the training stops
max_skipped_in_a_row = 10
# Batches the statistics of the mel bins are measured over
mel_stats_batches = 200

current_dir = os.getcwd()

try:
    with open(
        os.path.join(current_dir, "assets", "config.json"),
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)
        precision = config["precision"]
        if (
            precision == "bf16"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            train_dtype = torch.bfloat16
        elif precision == "fp16" and torch.cuda.is_available():
            train_dtype = torch.float16
        else:
            train_dtype = torch.float32
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    train_dtype = torch.float32

experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
model_info_path = os.path.join(experiment_dir, "model_info.json")
training_files = os.path.join(experiment_dir, "filelist.txt")

try:
    with open(config_save_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config.pop("process_pids", None)
except FileNotFoundError:
    print(
        f"Config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
    )
    sys.exit(1)

if "flow" not in config:
    print(
        "This model was not extracted for Rectified Flow. Preprocess and extract it again with the 44100 sampling rate."
    )
    sys.exit(1)

# Without the feature cache the dataset is augmented as it is read
if not feature_cache:
    config["flow"]["feature_cache"] = False

torch.backends.cudnn.deterministic = False
if os.name == "nt":  # Windows
    torch.backends.cudnn.benchmark = True

# TF32 settings, should improve performance in some cases
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception as e:
    print(f"Torch tf32: {e}")

global_step = 0
skipped_steps = 0
skipped_in_a_row = 0
logged_real_mel = False

import logging

logging.getLogger("torch").setLevel(logging.ERROR)


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
        if batch_idx >= mel_stats_batches:
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


def optimizer_step(loss, net_flow, optim, scaler, grad_clip):
    """
    Runs the backward pass, clips the gradient and steps the optimizer.
    Returns the norm of the gradient and whether the step was skipped over a
    non-finite gradient.

    Args:
        loss (torch.Tensor): The loss to optimise.
        net_flow (RectifiedFlow): The flow model.
        optim (torch.optim.Optimizer): The optimizer of the flow model.
        scaler (torch.amp.GradScaler): The gradient scaler for FP16 training.
        grad_clip (float): Maximum norm of the gradient.
    """
    optim.zero_grad()
    if train_dtype != torch.float16:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net_flow.parameters(), grad_clip)
        # a non-finite gradient would reach every weight
        if not torch.isfinite(grad_norm):
            return grad_norm, True
        optim.step()
        return grad_norm, False

    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    grad_norm = torch.nn.utils.clip_grad_norm_(net_flow.parameters(), grad_clip)
    scaler.step(optim)
    scale = scaler.get_scale()
    scaler.update()
    skipped = scaler.get_scale() < scale
    if not skipped and scaler.get_scale() > max_grad_scale:
        scaler.update(max_grad_scale)
    return grad_norm, skipped


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


def main():
    """
    Main function to start the training process.
    """
    global gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    # Check sample rate
    wavs = glob.glob(os.path.join(experiment_dir, "sliced_audios", "*.wav"))
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != config["data"]["sample_rate"]:
            print(
                f"Error: Rectified Flow sample rate ({config['data']['sample_rate']} Hz) does not match dataset audio sample rate ({sr} Hz)."
            )
            os._exit(1)
    else:
        print("No wav file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        gpus = [0]
        n_gpus = 1
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        print("Training with CPU, this will take a long time.")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []
        pid_data = {"process_pids": []}
        with open(config_save_path, "r", encoding="utf-8") as pid_file:
            try:
                existing_data = json.load(pid_file)
                pid_data.update(existing_data)
            except json.JSONDecodeError:
                pass
        pid_data["process_pids"] = []
        for rank, device_id in enumerate(gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    rank,
                    n_gpus,
                    experiment_dir,
                    pretrain,
                    vocoder_path,
                    total_epoch,
                    save_every_weights,
                    config,
                    device,
                    device_id,
                ),
            )
            children.append(subproc)
            subproc.start()
            pid_data["process_pids"].append(subproc.pid)
        with open(config_save_path, "w") as pid_file:
            json.dump(pid_data, pid_file, indent=4)

        for i in range(n_gpus):
            children[i].join()

    if cleanup:
        print("Removing files from the prior training attempt...")

        # Clean up unnecessary files
        for root, dirs, files in os.walk(
            os.path.join(now_dir, "logs", model_name), topdown=False
        ):
            for name in files:
                file_path = os.path.join(root, name)
                file_name, file_extension = os.path.splitext(name)
                if file_extension == ".0" or (
                    file_name.startswith("F_") and file_extension == ".pth"
                ):
                    os.remove(file_path)
            for name in dirs:
                if name == "eval":
                    folder_path = os.path.join(root, name)
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                    os.rmdir(folder_path)

        print("Cleanup done!")

    # The features are written once, before the training processes start
    if config["flow"].get("feature_cache", False):
        entries = [
            row for row in load_filepaths_and_text(training_files) if len(row) >= 5
        ]
        entries, _ = split_holdout(entries, config["flow"]["holdout_clips"])
        build_cache(
            experiment_dir,
            config,
            entries,
            min(config["flow"]["num_workers"], os.cpu_count() or 1),
        )

    start()


def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrain,
    vocoder_path,
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
    device_id,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrain (str): Path to the pre-trained flow model.
        vocoder_path (str): Path to the vocoder that renders the validation audio.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): Whether to save the model weights at every saved epoch.
        config (dict): The rectified flow config.
        device (torch.device): The device to use for training (CPU or GPU).
        device_id (int): The index of the GPU of the current process.
    """
    global global_step, skipped_steps

    if rank == 0:
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))
    else:
        writer_eval = None

    if n_gpus > 1:
        dist.init_process_group(
            backend="gloo" if sys.platform == "win32" else "nccl",
            init_method="env://",
            world_size=n_gpus,
            rank=rank,
        )

    flow_config = config["flow"]
    model_config = flow_config["model"]

    # Every rank draws its own noise and augmentations
    torch.manual_seed(flow_config["seed"] + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device("cuda", device_id)

    # Create datasets and dataloaders
    entries = [row for row in load_filepaths_and_text(training_files) if len(row) >= 5]
    n_speakers = max(int(row[4]) for row in entries) + 1
    entries, holdout_entries = split_holdout(entries, flow_config["holdout_clips"])

    train_dataset = FlowAudioLoader(entries, config)
    segment_frames = flow_config["segment_frames"]
    collate_fn = FlowAudioCollate(segment_frames)
    num_workers = min(flow_config["num_workers"], os.cpu_count() or 1)

    # With the feature cache the training reads the cached items, and with
    # bucket batches too, in batches of whole clips of similar length
    use_cache = flow_config.get("feature_cache", False)
    use_buckets = use_cache and flow_config.get("bucket_batches", False)
    train_items = train_dataset
    if use_cache:
        max_frames = segment_frames
        if use_buckets:
            max_frames = flow_config.get("bucket_max_frames", segment_frames)
        train_items = CachedFlowLoader(
            train_dataset, *load_cache(experiment_dir, config, entries), max_frames
        )

    # Batches of whole clips change shape, which the benchmark would search anew
    if use_buckets:
        torch.backends.cudnn.benchmark = False
    elif flow_config.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if use_buckets:
        train_loader = DataLoader(
            train_items,
            batch_sampler=BucketBatchSampler(
                train_items.get_lengths(),
                batch_size * segment_frames,
                flow_config.get("bucket_max_items", 64),
                flow_config["seed"],
                rank,
                n_gpus,
            ),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=FlowAudioCollate(),
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        if n_gpus > 1:
            train_sampler = DistributedSampler(
                train_items,
                num_replicas=n_gpus,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            train_sampler = None
        train_loader = DataLoader(
            train_items,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    eval_loader = None
    if rank == 0 and holdout_entries:
        eval_loader = DataLoader(
            FlowAudioLoader(holdout_entries, config, augment=False),
            batch_size=min(batch_size, len(holdout_entries)),
            num_workers=min(2, num_workers),
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    # Validations
    if len(train_loader) < 1:
        print(
            "Not enough data present in the training set. Perhaps you forgot to slice the audio files in preprocess?"
        )
        os._exit(2333333)

    # defaults
    embedder_name = "contentvec"

    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
    except Exception as e:
        print(f"Could not load model info file: {e}. Using defaults.")
    print(f"Dataset has {n_speakers} speakers.")

    checkpoint = None
    checkpoint_path = latest_checkpoint_path(experiment_dir, "F_*.pth")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    backbone_args = model_config.setdefault("backbone_args", {})

    finetune = pretrain not in ("", "None")
    pretrain_weights = None
    if finetune:
        if rank == 0:
            print(f"Loaded pretrained (Flow) '{pretrain}'")
        pretrain_weights, time_scale = load_pretrain(pretrain, embedder_name)
        # The time scale is not in the weights, the model has to be built with it
        if time_scale is not None:
            backbone_args["time_scale"] = time_scale

    # Initialize model and optimizer
    net_flow = build_flow(config, n_speakers).to(device)

    speaker_dropout = flow_config["speaker_dropout"]
    tension_dropout = 0.0
    if model_config.get("tension", False):
        tension_dropout = flow_config["tension_dropout"]
    if finetune and n_speakers == 1 and flow_config["finetune_freeze_voice"]:
        if rank == 0:
            print("Single speaker fine-tune, freezing the time and speaker layers.")
        freeze_voice(net_flow)
        speaker_dropout = 0.0

    if finetune:
        base_lr = flow_config["finetune_learning_rate"]
        ema_decay = flow_config["finetune_ema_decay"]
        warmup = 0
    else:
        base_lr = flow_config["learning_rate"]
        ema_decay = flow_config["ema_decay"]
        warmup = flow_config["warmup_steps"]

    if flow_config["optimizer"] == "muon":
        if rank == 0:
            print("Using Muon + AdamW optimizer")
        optim = MuonAdamW(
            net_flow,
            base_lr,
            muon_weight_decay=flow_config["weight_decay"],
            betas=flow_config["betas"],
        )
    else:
        if rank == 0:
            print("Using AdamW optimizer")
        optim = torch.optim.AdamW(
            net_flow.parameters(),
            base_lr,
            betas=flow_config["betas"],
            weight_decay=flow_config["weight_decay"],
        )

    use_scaler = device.type == "cuda" and train_dtype == torch.float16
    scaler = torch.amp.GradScaler(
        init_scale=2.0**10, growth_interval=2000, enabled=use_scaler
    )
    ema = WeightEMA(net_flow, ema_decay)

    if rank == 0 and train_dtype == torch.bfloat16:
        print("Using BFloat16 for training.")
    elif rank == 0 and train_dtype == torch.float16:
        print("Using Float16 for training.")

    # Load checkpoint if available
    epoch_str = 1
    global_step = 0
    resumed = checkpoint is not None
    if checkpoint is not None:
        print("Starting training...")
        net_flow.load_state_dict(match_inputs(checkpoint["model"], net_flow))
        optim.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint.get("ema"), net_flow)
        if len(checkpoint.get("scaler", {})) > 0:
            scaler.load_state_dict(checkpoint["scaler"])
        epoch_str = checkpoint["iteration"] + 1
        global_step = checkpoint["step"]
        skipped_steps = checkpoint.get("skipped_steps", 0)
        print(
            f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['iteration']})"
        )
        del checkpoint
    elif finetune:
        try:
            print(
                f"Overriding the pretrain speaker embedding size to {n_speakers} speakers."
            )
            pretrain_weights = resize_speakers(pretrain_weights, n_speakers)
            net_flow.load_state_dict(match_inputs(pretrain_weights, net_flow))
            ema.reseed(net_flow)
        except Exception as e:
            print(
                "The parameters of the pretrain model such as the embedder or architecture do not match the selected model."
            )
            print(e)
            sys.exit(1)
    pretrain_weights = None

    # A resumed training and a fine-tune keep the mel statistics of their weights
    if flow_config.get("mel_bin_norm", False) and not resumed and not finetune:
        if rank == 0:
            print("Measuring the statistics of the mel bins...")
        net_flow.set_mel_stats(
            *get_mel_stats(train_loader, config["data"], device, n_gpus)
        )
        ema.reseed(net_flow)

    # A resumed training continues the cosine from where its learning rate was
    lr_anchor = (0, 0.0)
    if resumed and global_step >= warmup:
        scale = optim.param_groups[0]["lr"] / base_lr
        lr_anchor = (
            global_step,
            cosine_progress(scale, flow_config["lr_final_ratio"]),
        )

    # Wrap model with DDP for multi-gpu processing
    net_flow_ddp = net_flow
    if n_gpus > 1 and device.type == "cuda":
        net_flow_ddp = DDP(net_flow, device_ids=[device_id])

    # collect the vocoder and the validation clips for tensorboard evaluation
    vocoder = None
    references = []
    if rank == 0:
        if os.path.isfile(vocoder_path):
            try:
                vocoder = load_vocoder(vocoder_path, config["data"]).to(device)
                print(f"Loaded vocoder '{vocoder_path}'")
            except Exception as e:
                print(f"Could not load the vocoder: {e}")
        if vocoder is None:
            print("No vocoder loaded, validation will only show the mel spectrogram.")
        # name, mel, flow inputs and whether the clip was held out
        reference = train_dataset.get_reference(embedder_name)
        if reference is not None:
            references.append(("", *reference, False))
        if eval_loader is not None:
            holdout_clips = eval_loader.dataset.get_speaker_clips(preview_clips)
            for index, clip in enumerate(holdout_clips):
                references.append((f"holdout_{index}_", *clip, True))
        references = [
            (name, ref_mel.to(device), ref_inputs.to(device), held_out)
            for name, ref_mel, ref_inputs, held_out in references
        ]

    hps = {
        "config": config,
        "base_lr": base_lr,
        "warmup": warmup,
        "total_steps": custom_total_epoch * len(train_loader),
        "final_ratio": flow_config["lr_final_ratio"],
        "lr_anchor": lr_anchor,
        "grad_clip": flow_config["grad_clip"],
        "speaker_dropout": speaker_dropout,
        "tension_dropout": tension_dropout,
        "content_blur": flow_config.get("content_blur_prob", 0.0),
        "aux_weight": flow_config["aux_mel_weight"],
        "eval_interval": flow_config["eval_interval"],
        "n_speakers": n_speakers,
        "embedder_name": embedder_name,
        "vocoder_path": vocoder_path if vocoder is not None else "",
    }

    for epoch in range(epoch_str, total_epoch + 1):
        train_and_evaluate(
            rank,
            epoch,
            hps,
            [net_flow, net_flow_ddp, ema],
            optim,
            [train_loader, eval_loader],
            writer_eval,
            custom_save_every_weights,
            custom_total_epoch,
            device,
            references,
            vocoder,
            scaler,
        )


def evaluate(hps, net_flow, ema, eval_loader, writer, device, use_amp):
    """
    Logs the flow loss of the held out clips through the averaged weights, from
    the same noise and at the same times on every call.

    Args:
        hps (dict): Hyperparameters.
        net_flow (RectifiedFlow): The flow model.
        ema (WeightEMA): Average of the flow weights.
        eval_loader (DataLoader): Dataloader of the held out clips.
        writer (SummaryWriter): The TensorBoard writer.
        device (torch.device): The device of the model.
        use_amp (bool): Whether to use automatic mixed precision.
    """
    data_config = hps["config"]["data"]
    totals = torch.zeros(len(eval_fractions), device=device)
    aux_total = 0.0
    items = 0
    with ema.applied(net_flow):
        net_flow.eval()
        for batch_idx, (mel, inputs) in enumerate(eval_loader):
            inputs = inputs.to(device)
            mel = normalize_mel(mel.to(device), data_config) * inputs.mask
            generator = torch.Generator(device=device).manual_seed(batch_idx)
            noise = torch.randn(mel.shape, device=device, generator=generator)
            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                losses, loss_aux = net_flow.validation_losses(
                    mel, inputs, noise, eval_fractions
                )
            totals += losses.float() * mel.shape[0]
            if loss_aux is not None:
                aux_total += loss_aux.item() * mel.shape[0]
            items += mel.shape[0]
        net_flow.train()

    totals /= max(1, items)
    scalar_dict = {"loss/val/flow": totals.mean()}
    for fraction, value in zip(eval_fractions, totals):
        scalar_dict[f"loss/val/flow_t{fraction:g}"] = value
    if net_flow.aux is not None:
        scalar_dict["loss/val/aux_mel"] = aux_total / max(1, items)
    summarize(writer=writer, global_step=global_step, scalars=scalar_dict)


def generate_validation(hps, net_flow, ema, references, vocoder, writer, device):
    """
    Logs the mel and the audio of the validation clips sampled through the
    averaged weights. Each clip is also rendered from the mel of the aux
    decoder alone and, when it was held out, from the flow started at its real
    mel on the same noise, which tells the error of the aux decoder from the
    error of the flow. The real mel through the vocoder is logged once.

    Args:
        hps (dict): Hyperparameters.
        net_flow (RectifiedFlow): The flow model.
        ema (WeightEMA): Average of the flow weights.
        references (list): Name, mel, flow inputs and whether it was held out, for each validation clip.
        vocoder (torch.nn.Module): The vocoder that renders the validation audio.
        writer (SummaryWriter): The TensorBoard writer.
        device (torch.device): The device of the model.
    """
    global logged_real_mel

    data_config = hps["config"]["data"]
    image_dict = {}
    audio_dict = {}
    with ema.applied(net_flow), torch.no_grad():
        net_flow.eval()
        for name, ref_mel, inputs, held_out in references:
            real_mel = normalize_mel(ref_mel, data_config)
            noise = torch.randn_like(real_mel)
            mels = {"": net_flow.sample(inputs, steps=preview_steps, noise=noise)}
            if net_flow.aux is not None:
                mels["_aux"] = net_flow.aux_mel(inputs)
            if net_flow.starts_from_aux and held_out:
                mels["_from_real_mel"] = net_flow.sample(
                    inputs, steps=preview_steps, noise=noise, start_mel=real_mel
                )
            if not logged_real_mel:
                mels["_real_mel"] = real_mel

            image_dict[f"slice/{name}mel_org"] = plot_spectrogram_to_numpy(
                ref_mel[0].data.cpu().numpy()
            )
            image_dict[f"slice/{name}mel_gen"] = plot_spectrogram_to_numpy(
                denormalize_mel(mels[""], data_config)[0].float().data.cpu().numpy()
            )
            if vocoder is not None:
                for kind, mel in mels.items():
                    o = vocoder(mel.float(), inputs.f0)
                    audio_dict[f"gen/{name}audio{kind}_{global_step:07d}"] = o[0, :, :]
        net_flow.train()
    logged_real_mel = True

    summarize(
        writer=writer,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sample_rate=data_config["sample_rate"],
    )


def save_model(path, hps, ema, epoch):
    """
    Saves the averaged weights as a model for inference.

    Args:
        path (str): Path of the model file.
        hps (dict): Hyperparameters.
        ema (WeightEMA): Average of the flow weights.
        epoch (int): Current epoch number.
    """
    torch.save(
        {
            "kind": "rectified_flow",
            "config": hps["config"],
            "model": ema.cpu_state_dict(),
            "speaker_count": hps["n_speakers"],
            "speakers_id": hps["n_speakers"],
            "embedder_model": hps["embedder_name"],
            "vocoder": hps["vocoder_path"],
            "epoch": epoch,
            "step": global_step,
        },
        path,
    )
    print(f"Saved model '{path}' (epoch {epoch} and step {global_step})")


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optim,
    loaders,
    writer,
    custom_save_every_weights,
    custom_total_epoch,
    device,
    references,
    vocoder,
    scaler,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        hps (dict): Hyperparameters.
        nets (list): The flow model, its DDP wrapper and the average of its weights [net_flow, net_flow_ddp, ema].
        optim (torch.optim.Optimizer): The optimizer of the flow model.
        loaders (list): List of dataloaders [train_loader, eval_loader].
        writer (SummaryWriter): The TensorBoard writer.
        custom_save_every_weights (bool): Whether to save the model weights at every saved epoch.
        custom_total_epoch (int): The total number of epochs for training.
        device (torch.device): The device to use for training.
        references (list): Name, mel, flow inputs and whether it was held out, for each clip the validation audio is generated from.
        vocoder (torch.nn.Module): The vocoder that renders the validation audio.
        scaler (torch.amp.GradScaler): The gradient scaler for FP16 training.
    """
    global global_step, skipped_steps, skipped_in_a_row

    net_flow, net_flow_ddp, ema = nets
    train_loader, eval_loader = loaders
    data_config = hps["config"]["data"]

    if isinstance(train_loader.batch_sampler, BucketBatchSampler):
        train_loader.batch_sampler.set_epoch(epoch)
    elif isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    net_flow.train()

    use_amp = device.type == "cuda" and (
        train_dtype == torch.bfloat16 or train_dtype == torch.float16
    )

    epoch_recorder = EpochRecorder()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, (mel, inputs) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            mel = normalize_mel(mel, data_config) * inputs.mask

            lr = learning_rate(
                hps["base_lr"],
                global_step,
                hps["warmup"],
                hps["total_steps"],
                hps["final_ratio"],
                hps["lr_anchor"],
            )
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Forward pass
                loss_flow, loss_aux = net_flow_ddp(
                    mel,
                    inputs,
                    speaker_dropout=hps["speaker_dropout"],
                    tension_dropout=hps["tension_dropout"],
                    content_blur=hps["content_blur"],
                )
                loss_all = loss_flow
                if loss_aux is not None:
                    loss_all = loss_all + hps["aux_weight"] * loss_aux

            # Backward and update
            grad_norm, skipped = optimizer_step(
                loss_all, net_flow, optim, scaler, hps["grad_clip"]
            )
            if skipped:
                skipped_steps += 1
                skipped_in_a_row += 1
                if rank == 0:
                    print(
                        f"Step {global_step + 1} skipped over a non-finite gradient. Non-finite values: {nonfinite_names(mel, inputs, net_flow)}."
                    )
                if skipped_in_a_row >= max_skipped_in_a_row:
                    print(
                        f"{skipped_in_a_row} steps in a row had a non-finite gradient, stopping the training."
                    )
                    os._exit(1)
            else:
                skipped_in_a_row = 0
                ema.update()

            global_step += 1

            if rank == 0 and global_step % 50 == 0:
                scalar_dict = {
                    "loss/flow": loss_flow,
                    "loss/total": loss_all,
                    "learning_rate": lr,
                    "grad/norm": grad_norm,
                    "grad/skipped_steps": skipped_steps,
                }
                if train_dtype == torch.float16:
                    scalar_dict["amp/scale"] = scaler.get_scale()
                if loss_aux is not None:
                    scalar_dict["loss/aux_mel"] = loss_aux
                summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

            if (
                eval_loader is not None
                and hps["eval_interval"]
                and global_step % hps["eval_interval"] == 0
            ):
                evaluate(hps, net_flow, ema, eval_loader, writer, device, use_amp)

            pbar.update(1)
        # end of batch train
    # end of tqdm
    with torch.no_grad():
        torch.cuda.empty_cache()

    # Logging and checkpointing
    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()} | loss_flow={round(loss_flow.item(), 4)}"
        if skipped_steps > 0:
            record = record + f" | skipped_steps={skipped_steps}"
        print(record)

        done = epoch >= custom_total_epoch
        model_add = []

        if (epoch % save_every_epoch == 0 or done) and any(
            not torch.isfinite(param).all() for param in net_flow.parameters()
        ):
            print(
                "The model has non-finite weights. Nothing was saved, so the last checkpoint stands."
            )
            os._exit(1)

        if epoch % save_every_epoch == 0:
            # Validation samples through the averaged weights
            if references:
                generate_validation(
                    hps, net_flow, ema, references, vocoder, writer, device
                )

            # Save checkpoint
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"
            checkpoint_path = os.path.join(experiment_dir, "F_" + checkpoint_suffix)
            torch.save(
                {
                    "model": net_flow.state_dict(),
                    "ema": ema.state_dict(),
                    "iteration": epoch,
                    "step": global_step,
                    "skipped_steps": skipped_steps,
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved model '{checkpoint_path}' (epoch {epoch})")

            if custom_save_every_weights:
                model_add.append(
                    os.path.join(
                        experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                    )
                )

        # Check completion
        if done:
            print(
                f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_flow.item(), 4)} loss flow."
            )
            # Final model
            model_add.append(
                os.path.join(experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth")
            )

        for m in model_add:
            if not os.path.exists(m):
                save_model(m, hps, ema, epoch)

        if done:
            # Clean-up process IDs from config.json
            pid_file_path = os.path.join(experiment_dir, "config.json")
            with open(pid_file_path, "r", encoding="utf-8") as pid_file:
                pid_data = json.load(pid_file)
            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)
                json.dump(pid_data, pid_file, indent=4)
            os._exit(2333333)

        with torch.no_grad():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
