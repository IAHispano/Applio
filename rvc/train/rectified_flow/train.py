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
mean_flow = _strtobool(sys.argv[11])

# Sampling steps of the validation audio
preview_steps = 16
# Where in the trained time range the validation loss is taken
eval_fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
# FP16: the weight gradients overflow with the scale around 2^20, so the
# GradScaler is not left to grow until a step is skipped
max_grad_scale = 2.0**16

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


def learning_rate(base, step, warmup, total, final_ratio):
    """
    Linear warmup, then cosine decay to `final_ratio` of the base learning rate.

    Args:
        base (float): The base learning rate.
        step (int): The current step.
        warmup (int): Number of warmup steps.
        total (int): Number of steps of the whole training.
        final_ratio (float): Share of the base learning rate reached at the end.
    """
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (final_ratio + (1.0 - final_ratio) * cosine)


def freeze_voice(net_flow):
    """
    Freezes what maps time and speaker into the network, so a single speaker
    fine-tune keeps the speaker space of the pretrained model.

    Args:
        net_flow (RectifiedFlow): The flow model.
    """
    modules = [
        net_flow.backbone.time_mlp,
        net_flow.encoder.speaker_proj,
        net_flow.backbone.voice,
    ]
    modules += [layer.modulation for layer in net_flow.backbone.layers]
    for module in filter(None, modules):
        module.requires_grad_(False)


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
    collate_fn = FlowAudioCollate(flow_config["segment_frames"])
    if n_gpus > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=n_gpus, rank=rank, shuffle=True, drop_last=True
        )
    else:
        train_sampler = None
    num_workers = min(flow_config["num_workers"], os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
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

    # A resumed training keeps the mean flow choice it was started with
    use_mean_flow = mean_flow
    if checkpoint is not None:
        use_mean_flow = any(
            key.startswith("backbone.span_mlp.") for key in checkpoint["model"]
        )
        if use_mean_flow != mean_flow and rank == 0:
            print(
                f"This model was started {'with' if use_mean_flow else 'without'} Mean Flow and resumes that way. Enable Fresh Training to change it."
            )
    model_config["mean_flow"] = use_mean_flow
    backbone_args = model_config.setdefault("backbone_args", {})

    finetune = pretrain not in ("", "None")
    pretrain_ckpt = None
    if finetune:
        if rank == 0:
            print(f"Loaded pretrained (Flow) '{pretrain}'")
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

        # The time scale is not in the weights, the model has to be built with it
        pretrain_model = pretrain_ckpt.get("config", {}).get("flow", {}).get("model")
        if pretrain_model is not None:
            backbone_args["time_scale"] = float(
                (pretrain_model.get("backbone_args") or {}).get("time_scale", 1000.0)
            )
        del pretrain_ckpt

    if rank == 0 and use_mean_flow and backbone_args.get("time_scale", 1000.0) > 10:
        print(
            "Mean Flow with a time scale over 10 has diverged. Set flow.model.backbone_args.time_scale to 1 in the config.json of the model for a training from scratch."
        )

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
        mean_warmup = 0
    else:
        base_lr = flow_config["learning_rate"]
        ema_decay = flow_config["ema_decay"]
        warmup = flow_config["warmup_steps"]
        mean_warmup = flow_config["mean_flow_warmup_steps"]

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
    if checkpoint is not None:
        print("Starting training...")
        net_flow.load_state_dict(checkpoint["model"])
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

    # Wrap model with DDP for multi-gpu processing
    net_flow_ddp = net_flow
    if n_gpus > 1 and device.type == "cuda":
        net_flow_ddp = DDP(net_flow, device_ids=[device_id])

    # collect the vocoder and the reference audio for tensorboard evaluation
    vocoder = None
    reference = None
    if rank == 0:
        if os.path.isfile(vocoder_path):
            try:
                vocoder = load_vocoder(vocoder_path, config["data"]).to(device)
                print(f"Loaded vocoder '{vocoder_path}'")
            except Exception as e:
                print(f"Could not load the vocoder: {e}")
        if vocoder is None:
            print("No vocoder loaded, validation will only show the mel spectrogram.")
        reference = train_dataset.get_reference(embedder_name)
        if reference is not None:
            reference = [tensor.to(device) for tensor in reference]

    hps = {
        "config": config,
        "base_lr": base_lr,
        "warmup": warmup,
        "total_steps": custom_total_epoch * len(train_loader),
        "final_ratio": flow_config["lr_final_ratio"],
        "grad_clip": flow_config["grad_clip"],
        "speaker_dropout": speaker_dropout,
        "tension_dropout": tension_dropout,
        "aux_weight": flow_config["aux_mel_weight"],
        "mean_ratio": flow_config["mean_flow_ratio"] if use_mean_flow else 0.0,
        "mean_warmup": mean_warmup,
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
            reference,
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
        for batch_idx, info in enumerate(eval_loader):
            info = [tensor.to(device) for tensor in info]
            (
                mel,
                content,
                pitchf,
                energy,
                breathiness,
                key_shift,
                speed,
                sid,
                mask,
                tension,
            ) = info
            mel = normalize_mel(mel, data_config) * mask
            generator = torch.Generator(device=device).manual_seed(batch_idx)
            noise = torch.randn(mel.shape, device=device, generator=generator)
            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                losses, loss_aux = net_flow.validation_losses(
                    mel,
                    content,
                    pitchf,
                    energy,
                    sid,
                    mask,
                    breathiness,
                    key_shift,
                    speed,
                    noise,
                    eval_fractions,
                    tension,
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
    reference,
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
        reference (list): The clip the validation audio is generated from.
        vocoder (torch.nn.Module): The vocoder that renders the validation audio.
        scaler (torch.amp.GradScaler): The gradient scaler for FP16 training.
    """
    global global_step, skipped_steps

    net_flow, net_flow_ddp, ema = nets
    train_loader, eval_loader = loaders
    data_config = hps["config"]["data"]

    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    net_flow.train()

    use_amp = device.type == "cuda" and (
        train_dtype == torch.bfloat16 or train_dtype == torch.float16
    )

    epoch_recorder = EpochRecorder()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in enumerate(train_loader):
            info = [tensor.to(device, non_blocking=True) for tensor in info]
            (
                mel,
                content,
                pitchf,
                energy,
                breathiness,
                key_shift,
                speed,
                sid,
                mask,
                tension,
            ) = info
            mel = normalize_mel(mel, data_config) * mask

            lr = learning_rate(
                hps["base_lr"],
                global_step,
                hps["warmup"],
                hps["total_steps"],
                hps["final_ratio"],
            )
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            # The bootstrapped target comes in once there is a field to differentiate
            mean_bootstrap = 1.0
            if hps["mean_warmup"]:
                mean_bootstrap = min(1.0, global_step / hps["mean_warmup"])

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Forward pass
                loss_flow, loss_aux, loss_mean = net_flow_ddp(
                    mel,
                    content,
                    pitchf,
                    energy,
                    sid,
                    mask,
                    speaker_dropout=hps["speaker_dropout"],
                    breathiness=breathiness,
                    key_shift=key_shift,
                    speed=speed,
                    tension=tension,
                    mean_ratio=hps["mean_ratio"],
                    tension_dropout=hps["tension_dropout"],
                    mean_bootstrap=mean_bootstrap,
                )
                loss_all = loss_flow
                if loss_mean is not None:
                    loss_all = (1.0 - hps["mean_ratio"]) * loss_flow + hps[
                        "mean_ratio"
                    ] * loss_mean[0]
                    # the flow loss without the Mean Flow weighting
                    loss_flow = loss_mean[1]
                if loss_aux is not None:
                    loss_all = loss_all + hps["aux_weight"] * loss_aux

            # Backward and update
            optim.zero_grad()
            if train_dtype == torch.float16:
                scaler.scale(loss_all).backward()
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net_flow.parameters(), hps["grad_clip"]
                )
                scaler.step(optim)
                scale = scaler.get_scale()
                scaler.update()
                if scaler.get_scale() < scale:
                    # non-finite gradients, the optimizer step was skipped
                    skipped_steps += 1
                elif scaler.get_scale() > max_grad_scale:
                    scaler.update(max_grad_scale)
            else:
                loss_all.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net_flow.parameters(), hps["grad_clip"]
                )
                optim.step()
            ema.update()

            global_step += 1

            if rank == 0 and global_step % 50 == 0:
                scalar_dict = {
                    "loss/flow": loss_flow,
                    "loss/total": loss_all,
                    "learning_rate": lr,
                    "grad/norm": grad_norm,
                }
                if train_dtype == torch.float16:
                    scalar_dict["amp/scale"] = scaler.get_scale()
                    scalar_dict["amp/skipped_steps"] = skipped_steps
                if loss_aux is not None:
                    scalar_dict["loss/aux_mel"] = loss_aux
                if loss_mean is not None:
                    scalar_dict["loss/mean_flow"] = loss_mean[2]
                    # over 1 the Mean Flow target is feeding on itself
                    scalar_dict["loss/mean_flow_bootstrap"] = loss_mean[3]
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

        if epoch % save_every_epoch == 0:
            # Validation samples through the averaged weights
            if reference is not None:
                ref_mel, content, pitchf, energy, breathiness, sid, tension = reference
                mask = torch.ones(1, 1, pitchf.shape[1], device=device)
                one_step_mel = None
                with ema.applied(net_flow):
                    net_flow.eval()
                    gen_mel = net_flow.sample(
                        content,
                        pitchf,
                        energy,
                        sid,
                        mask,
                        steps=preview_steps,
                        breathiness=breathiness,
                        tension=tension,
                    )
                    if hps["mean_ratio"] > 0:
                        one_step_mel = net_flow.sample(
                            content,
                            pitchf,
                            energy,
                            sid,
                            mask,
                            steps=1,
                            method="mean",
                            breathiness=breathiness,
                            tension=tension,
                        )
                    net_flow.train()

                image_dict = {
                    "slice/mel_org": plot_spectrogram_to_numpy(
                        ref_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": plot_spectrogram_to_numpy(
                        denormalize_mel(gen_mel, data_config)[0]
                        .float()
                        .data.cpu()
                        .numpy()
                    ),
                }
                audio_dict = {}
                if vocoder is not None:
                    with torch.no_grad():
                        o = vocoder(gen_mel.float(), pitchf)
                        audio_dict[f"gen/audio_{global_step:07d}"] = o[0, :, :]
                        if one_step_mel is not None:
                            o = vocoder(one_step_mel.float(), pitchf)
                            audio_dict[f"gen/audio_1_step_{global_step:07d}"] = o[
                                0, :, :
                            ]
                summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    audios=audio_dict,
                    audio_sample_rate=data_config["sample_rate"],
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
                    m,
                )
                print(f"Saved model '{m}' (epoch {epoch} and step {global_step})")

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
