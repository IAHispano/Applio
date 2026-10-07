import copy
import os
import sys

os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
import glob
import json
from random import randint

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
from rvc.lib.algorithm.rectified_flow.features import normalize_mel
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
from rvc.train.rectified_flow.utils import (
    EpochRecorder,
    cosine_progress,
    freeze_voice,
    get_mel_stats,
    learning_rate,
    load_pretrain,
    nonfinite_names,
)
from rvc.train.rectified_flow.validation import evaluate, generate_validation
from rvc.train.utils import (
    latest_checkpoint_path,
    load_filepaths_and_text,
    load_wav_to_torch,
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
shortcut = _strtobool(sys.argv[12])

# FP16: the weight gradients overflow with the scale around 2^20, so the
# GradScaler is not left to grow until a step is skipped
max_grad_scale = 2.0**16
# Steps skipped in a row over a non-finite gradient before the training stops
max_skipped_in_a_row = 10
# How much of the dataset is shifted and stretched: drawn per item, or made
# ahead for the feature cache. A fine-tune has its own, under `finetune_`
finetune_augmentation = (
    "key_shift_prob",
    "time_stretch_prob",
    "key_shift_scale",
    "time_stretch_scale",
)

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

# A fine-tune shifts and stretches less of the dataset than a pretrain
if pretrain not in ("", "None"):
    for key in finetune_augmentation:
        if "finetune_" + key in config["flow"]:
            config["flow"][key] = config["flow"]["finetune_" + key]

# Whether the flow is a shortcut model is decided by the training, not the config
config["flow"]["model"]["shortcut"] = shortcut

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

    # A pretrain does not fine-tune to more than one speaker
    if pretrain not in ("", "None"):
        rows = [row for row in load_filepaths_and_text(training_files) if len(row) >= 5]
        n_speakers = max(int(row[4]) for row in rows) + 1
        if n_speakers > 1:
            print(
                f"Error: Rectified Flow can not fine-tune a pretrained model on {n_speakers} speakers. Use a dataset of one speaker, or train without a pretrained model."
            )
            os._exit(1)

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
        # Sorted, the filelist is shuffled anew by every extraction
        entries = sorted(
            row for row in load_filepaths_and_text(training_files) if len(row) >= 5
        )
        entries, _ = split_holdout(entries, config["flow"]["holdout_clips"])
        build_cache(
            experiment_dir,
            config,
            entries,
            min(config["flow"]["num_workers"], os.cpu_count() or 1),
        )

    start()


def get_loaders(config, entries, holdout_entries, experiment_dir, rank, n_gpus):
    """
    Returns the training dataset, its dataloader and the dataloader of the
    held out clips, which is None without any or off the first rank.

    Args:
        config (dict): The rectified flow config.
        entries (list): Rows of the filelist of the training clips.
        holdout_entries (list): Rows of the filelist of the held out clips.
        experiment_dir (str): The directory of the experiment.
        rank (int): The rank of the current process.
        n_gpus (int): The total number of GPUs available for training.
    """
    flow_config = config["flow"]
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
                batch_size,
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
    return train_dataset, train_loader, eval_loader


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
    entries = sorted(
        row for row in load_filepaths_and_text(training_files) if len(row) >= 5
    )
    n_speakers = max(int(row[4]) for row in entries) + 1
    entries, holdout_entries = split_holdout(entries, flow_config["holdout_clips"])

    train_dataset, train_loader, eval_loader = get_loaders(
        config, entries, holdout_entries, experiment_dir, rank, n_gpus
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
    if rank == 0 and isinstance(train_loader.batch_sampler, BucketBatchSampler):
        segment_frames = flow_config["segment_frames"]
        print(
            f"Batches of up to {batch_size} whole clips and {batch_size * segment_frames} mel frames: {len(train_loader.dataset)} items in {len(train_loader)} steps per epoch."
        )

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
    elif rank == 0 and checkpoint is None:
        print("No pretrained (Flow), training from scratch.")

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

    total_steps = custom_total_epoch * len(train_loader)
    if finetune:
        base_lr = flow_config["finetune_learning_rate"]
        ema_decay = flow_config["finetune_ema_decay"]
        # The optimizer starts cold on trained weights and the step of Muon
        # does not shrink with the gradient; a short fine-tune is not all warmup
        warmup = min(flow_config.get("finetune_warmup_steps", 0), total_steps // 4)
    else:
        base_lr = flow_config["learning_rate"]
        ema_decay = flow_config["ema_decay"]
        warmup = flow_config["warmup_steps"]

    optim = MuonAdamW(
        net_flow,
        base_lr,
        muon_weight_decay=flow_config["weight_decay"],
        betas=flow_config["betas"],
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

    # A shortcut model learns its jumps from a copy of itself that lags it,
    # which starts at the averaged weights
    teacher = None
    if net_flow.shortcut_levels:
        with ema.applied(net_flow):
            teacher = copy.deepcopy(net_flow).requires_grad_(False).eval()
        if rank == 0:
            print("Training a shortcut flow.")

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
        elif eval_loader is not None:
            # Without a reference clip, one held out clip is rendered instead
            for index, clip in enumerate(eval_loader.dataset.get_speaker_clips(1)):
                references.append((f"holdout_{index}_", *clip, True))
        references = [
            (name, ref_mel.to(device), ref_inputs.to(device), held_out)
            for name, ref_mel, ref_inputs, held_out in references
        ]

    hps = {
        "config": config,
        "base_lr": base_lr,
        "warmup": warmup,
        "total_steps": total_steps,
        "final_ratio": flow_config["lr_final_ratio"],
        "lr_anchor": lr_anchor,
        "grad_clip": flow_config["grad_clip"],
        "speaker_dropout": speaker_dropout,
        "tension_dropout": tension_dropout,
        "shortcut_share": flow_config.get("shortcut_share", 0.125),
        "shortcut_ema_decay": flow_config.get("shortcut_ema_decay", 0.999),
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
            [net_flow, net_flow_ddp, ema, teacher],
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
        nets (list): The flow model, its DDP wrapper, the average of its weights and the lagging copy of a shortcut model [net_flow, net_flow_ddp, ema, teacher].
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
    global global_step, skipped_steps, skipped_in_a_row, logged_real_mel

    net_flow, net_flow_ddp, ema, teacher = nets
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
                    teacher=teacher,
                    shortcut_share=hps["shortcut_share"],
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
                if teacher is not None:
                    with torch.no_grad():
                        torch._foreach_lerp_(
                            list(teacher.parameters()),
                            list(net_flow.parameters()),
                            1.0 - hps["shortcut_ema_decay"],
                        )

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
                evaluate(
                    hps,
                    net_flow,
                    ema,
                    eval_loader,
                    writer,
                    device,
                    global_step,
                    train_dtype if use_amp else None,
                )

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
                # the real mel through the vocoder is logged once
                generate_validation(
                    hps,
                    net_flow,
                    ema,
                    references,
                    vocoder,
                    writer,
                    global_step,
                    not logged_real_mel,
                )
                logged_real_mel = True

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
