import os
import sys

os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
import datetime
import glob
import json
from collections import deque
from distutils.util import strtobool
from random import randint, shuffle
from time import time as ttime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import (
    MultiScaleMelSpectrogramLoss,
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from utils import (
    HParams,
    latest_checkpoint_path,
    load_checkpoint,
    load_wav_to_torch,
    plot_spectrogram_to_numpy,
    save_checkpoint,
    summarize,
)

# Zluda hijack
import rvc.lib.zluda
from rvc.lib.algorithm import commons
from rvc.train.process.extract_model import extract_model

# Parse command line arguments
model_name = sys.argv[1]
save_every_epoch = int(sys.argv[2])
total_epoch = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])
sample_rate = int(sys.argv[8])
save_only_latest = strtobool(sys.argv[9])
save_every_weights = strtobool(sys.argv[10])
cache_data_in_gpu = strtobool(sys.argv[11])
overtraining_detector = strtobool(sys.argv[12])
overtraining_threshold = int(sys.argv[13])
cleanup = strtobool(sys.argv[14])
vocoder = sys.argv[15]
checkpointing = strtobool(sys.argv[16])
# experimental settings
randomized = True
d_lr_coeff = 1.0
g_lr_coeff = 1.0
d_step_per_g_step = 1
multiscale_mel_loss = False
bf16_adamw = False

current_dir = os.getcwd()

try:
    with open(os.path.join(current_dir, "assets", "config.json"), "r") as f:
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
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")

try:
    with open(config_save_path, "r") as f:
        config = json.load(f)
    config = HParams(**config)
except FileNotFoundError:
    print(
        f"Config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
    )
    sys.exit(1)

config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

global_step = 0
last_loss_gen_all = 0
overtrain_save_epoch = 0
loss_gen_history = []
smoothed_loss_gen_history = []
loss_disc_history = []
smoothed_loss_disc_history = []
lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
training_file_path = os.path.join(experiment_dir, "training_data.json")

avg_losses = {
    "grad_d_50": deque(maxlen=50),
    "grad_g_50": deque(maxlen=50),
    "disc_loss_50": deque(maxlen=50),
    "adv_loss_50": deque(maxlen=50),
    "fm_loss_50": deque(maxlen=50),
    "kl_loss_50": deque(maxlen=50),
    "mel_loss_50": deque(maxlen=50),
    "gen_loss_50": deque(maxlen=50),
}

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


def main():
    """
    Main function to start the training process.
    """
    global training_file_path, last_loss_gen_all, smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history, overtrain_save_epoch, gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    # Check sample rate
    wavs = glob.glob(
        os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*.wav")
    )
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != config.data.sample_rate:
            print(
                f"Error: Pretrained model sample rate ({config.data.sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz)."
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
        with open(config_save_path, "r") as pid_file:
            try:
                existing_data = json.load(pid_file)
                pid_data.update(existing_data)
            except json.JSONDecodeError:
                pass
        with open(config_save_path, "w") as pid_file:
            for rank, device_id in enumerate(gpus):
                subproc = mp.Process(
                    target=run,
                    args=(
                        rank,
                        n_gpus,
                        experiment_dir,
                        pretrainG,
                        pretrainD,
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
            json.dump(pid_data, pid_file, indent=4)

        for i in range(n_gpus):
            children[i].join()

    def load_from_json(file_path):
        """
        Load data from a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                return (
                    data.get("loss_disc_history", []),
                    data.get("smoothed_loss_disc_history", []),
                    data.get("loss_gen_history", []),
                    data.get("smoothed_loss_gen_history", []),
                )
        return [], [], [], []

    def continue_overtrain_detector(training_file_path):
        """
        Continues the overtrain detector by loading the training history from a JSON file.

        Args:
            training_file_path (str): The file path of the JSON file containing the training history.
        """
        if overtraining_detector:
            if os.path.exists(training_file_path):
                (
                    loss_disc_history,
                    smoothed_loss_disc_history,
                    loss_gen_history,
                    smoothed_loss_gen_history,
                ) = load_from_json(training_file_path)

    if cleanup:
        print("Removing files from the prior training attempt...")

        # Clean up unnecessary files
        for root, dirs, files in os.walk(
            os.path.join(now_dir, "logs", model_name), topdown=False
        ):
            for name in files:
                file_path = os.path.join(root, name)
                file_name, file_extension = os.path.splitext(name)
                if (
                    file_extension == ".0"
                    or (file_name.startswith("D_") and file_extension == ".pth")
                    or (file_name.startswith("G_") and file_extension == ".pth")
                    or (file_name.startswith("added") and file_extension == ".index")
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

    continue_overtrain_detector(training_file_path)
    start()


def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
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
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): The interval (in epochs) at which to save model weights.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, smoothed_value_gen, smoothed_value_disc

    smoothed_value_gen = 0
    smoothed_value_disc = 0

    if rank == 0:
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))
    else:
        writer_eval = None

    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # Create datasets and dataloaders
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # Validations
    if len(train_loader) < 3:
        print(
            "Not enough data present in the training set. Perhaps you forgot to slice the audio files in preprocess?"
        )
        os._exit(2333333)

    # defaults
    embedder_name = "contentvec"
    spk_dim = config.model.spk_embed_dim  # 109 default speakers

    try:
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
            spk_dim = model_info["speakers_id"]
    except Exception as e:
        print(f"Could not load model info file: {e}. Using defaults.")

    # Try to load speaker dim from latest checkpoint or pretrainG
    try:
        last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
        chk_path = (
            last_g if last_g else (pretrainG if pretrainG not in ("", "None") else None)
        )

        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Using default number of speakers.")

    # update config before the model init
    print(f"Initializing the generator with {spk_dim} speakers.")
    config.model.spk_embed_dim = spk_dim

    # Initialize models and optimizers
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
    from rvc.lib.algorithm.synthesizers import Synthesizer

    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=True,
        sr=config.data.sample_rate,
        vocoder=vocoder,
        checkpointing=checkpointing,
        randomized=randomized,
    )

    net_d = MultiPeriodDiscriminator(
        config.model.use_spectral_norm, checkpointing=checkpointing
    )

    if torch.cuda.is_available():
        net_g = net_g.cuda(device_id)
        net_d = net_d.cuda(device_id)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    if bf16_adamw == True and train_dtype == torch.bfloat16:
        print("Using BFload16 AdamW optimizer")
        from rvc.train.anyprecision_optimizer import AnyPrecisionAdamW

        optimizer = AnyPrecisionAdamW
    else:
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW

    optim_g = optimizer(
        net_g.parameters(),
        config.train.learning_rate * g_lr_coeff,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    optim_d = optimizer(
        net_d.parameters(),
        config.train.learning_rate * d_lr_coeff,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    if multiscale_mel_loss:
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=config.data.sample_rate)
        print("Using Multi-Scale Mel loss function")
    else:
        fn_mel_loss = torch.nn.L1Loss()
        print("Using Single-Scale Mel loss function")

    # Wrap models with DDP for multi-gpu processing
    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id])
        net_d = DDP(net_d, device_ids=[device_id])

    if rank == 0 and train_dtype == torch.bfloat16:
        print("Using BFloat16 for training.")
    elif rank == 0 and train_dtype == torch.float16:
        print("Using Float16 for training.")

    # Load checkpoint if available
    try:
        print("Starting training...")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"), net_g, optim_g
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

    except Exception as e:
        epoch_str = 1
        global_step = 0

        if pretrainG not in ("", "None"):
            if rank == 0:
                print(f"Loaded pretrained (G) '{pretrainG}'")
            try:
                ckpt = torch.load(pretrainG, map_location="cpu", weights_only=True)[
                    "model"
                ]
                if hasattr(net_g, "module"):
                    net_g.module.load_state_dict(ckpt)
                else:
                    net_g.load_state_dict(ckpt)
                del ckpt
            except Exception as e:
                print(
                    "The parameters of the pretrain model such as the sample rate or architecture do not match the selected model."
                )
                print(e)
                sys.exit(1)

        if pretrainD not in ("", "None"):
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            try:
                ckpt = torch.load(pretrainD, map_location="cpu", weights_only=True)[
                    "model"
                ]
                if hasattr(net_d, "module"):
                    net_d.module.load_state_dict(ckpt)
                else:
                    net_d.load_state_dict(ckpt)
                del ckpt
            except Exception as e:
                print(
                    "The parameters of the pretrain model such as the sample rate or architecture do not match the selected model."
                )
                print(e)
                sys.exit(1)

    # Initialize schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )

    use_scaler = device.type == "cuda" and train_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    cache = []
    # collect the reference audio for tensorboard evaluation
    if os.path.isfile(os.path.join("logs", "reference", embedder_name, "feats.npy")):
        print("Using", embedder_name, "reference set for validation")
        phone = np.load(os.path.join("logs", "reference", embedder_name, "feats.npy"))
        # expanding x2 to match pitch size
        phone = np.repeat(phone, 2, axis=0)
        phone_lengths = torch.LongTensor([phone.shape[0]]).to(device)
        phone = torch.FloatTensor(phone).unsqueeze(0).to(device)
        pitch = np.load(os.path.join("logs", "reference", "pitch_coarse.npy"))
        # removed last frame to match features
        pitch = torch.LongTensor(pitch[:-1]).unsqueeze(0).to(device)
        pitchf = np.load(os.path.join("logs", "reference", "pitch_fine.npy"))
        # removed last frame to match features
        pitchf = torch.FloatTensor(pitchf[:-1]).unsqueeze(0).to(device)
        sid = torch.LongTensor([0]).to(device)
        reference = (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            sid,
        )
    else:
        print("No custom reference found, using a default audio sample for validation")
        info = next(iter(train_loader))
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
        reference = (
            phone.to(device),
            phone_lengths.to(device),
            pitch.to(device),
            pitchf.to(device),
            sid.to(device),
        )

    for epoch in range(epoch_str, total_epoch + 1):
        train_and_evaluate(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            [train_loader, None],
            [writer_eval],
            cache,
            custom_save_every_weights,
            custom_total_epoch,
            device,
            device_id,
            reference,
            fn_mel_loss,
            scaler,
        )

        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    loaders,
    writers,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    device,
    device_id,
    reference,
    fn_mel_loss,
    scaler,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        hps (Namespace): Hyperparameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, optim_d].
        loaders (list): List of dataloaders [train_loader, eval_loader].
        writers (list): List of TensorBoard writers [writer_eval].
        cache (list): List to cache data in GPU memory.
        use_cpu (bool): Whether to use CPU for training.
    """
    global global_step, lowest_value, loss_disc, consecutive_increases_gen, consecutive_increases_disc, smoothed_value_gen, smoothed_value_disc

    if epoch == 1:
        lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
        consecutive_increases_gen = 0
        consecutive_increases_disc = 0

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    use_amp = device.type == "cuda" and (
        train_dtype == torch.bfloat16 or train_dtype == torch.float16
    )

    # Data caching
    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                # phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
                cache.append((batch_idx, info))
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            if device.type == "cuda" and not cache_data_in_gpu:
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            # else iterator is going thru a cached list with a device already assigned

            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Forward pass
                model_output = net_g(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                )
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
                    model_output
                )
                # slice of the original waveform to match a generate slice
                if randomized:
                    wave = commons.slice_segments(
                        wave,
                        ids_slice * config.data.hop_length,
                        config.train.segment_size,
                        dim=3,
                    )
            for _ in range(d_step_per_g_step):  # default x1
                with torch.amp.autocast(
                    device_type="cuda", enabled=use_amp, dtype=train_dtype
                ):
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                # Discriminator backward and update
                optim_d.zero_grad()
                if train_dtype == torch.float16:
                    scaler.scale(loss_disc).backward()
                    scaler.unscale_(optim_d)
                    grad_norm_d = commons.grad_norm(net_d.parameters())
                    scaler.step(optim_d)
                else:
                    loss_disc.backward()
                    grad_norm_d = commons.grad_norm(net_d.parameters())
                    optim_d.step()

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Generator backward and update
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

            if multiscale_mel_loss:
                loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
            else:
                wave_mel = mel_spectrogram_torch(
                    wave.float().squeeze(1),
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.hop_length,
                    config.data.win_length,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.hop_length,
                    config.data.win_length,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                loss_mel = fn_mel_loss(wave_mel, y_hat_mel) * config.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            if loss_gen_all < lowest_value["value"]:
                lowest_value = {
                    "step": global_step,
                    "value": loss_gen_all,
                    "epoch": epoch,
                }
            optim_g.zero_grad()
            if train_dtype == torch.float16:
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.grad_norm(net_g.parameters())
                scaler.step(optim_g)
                scaler.update()
            else:
                loss_gen_all.backward()
                grad_norm_g = commons.grad_norm(net_g.parameters())
                optim_g.step()

            global_step += 1

            # queue for rolling losses over 50 steps
            avg_losses["grad_d_50"].append(grad_norm_d)
            avg_losses["grad_g_50"].append(grad_norm_g)
            avg_losses["disc_loss_50"].append(loss_disc.detach())
            avg_losses["adv_loss_50"].append(loss_gen.detach())
            avg_losses["fm_loss_50"].append(loss_fm.detach())
            avg_losses["kl_loss_50"].append(loss_kl.detach())
            avg_losses["mel_loss_50"].append(loss_mel.detach())
            avg_losses["gen_loss_50"].append(loss_gen_all.detach())

            if rank == 0 and global_step % 50 == 0:
                # logging rolling averages
                scalar_dict = {
                    "grad_avg_50/norm_d": sum(avg_losses["grad_d_50"])
                    / len(avg_losses["grad_d_50"]),
                    "grad_avg_50/norm_g": sum(avg_losses["grad_g_50"])
                    / len(avg_losses["grad_g_50"]),
                    "loss_avg_50/d/adv": torch.mean(
                        torch.stack(list(avg_losses["disc_loss_50"]))
                    ),
                    "loss_avg_50/g/adv": torch.mean(
                        torch.stack(list(avg_losses["adv_loss_50"]))
                    ),
                    "loss_avg_50/g/fm": torch.mean(
                        torch.stack(list(avg_losses["fm_loss_50"]))
                    ),
                    "loss_avg_50/g/kl": torch.mean(
                        torch.stack(list(avg_losses["kl_loss_50"]))
                    ),
                    "loss_avg_50/g/mel": torch.mean(
                        torch.stack(list(avg_losses["mel_loss_50"]))
                    ),
                    "loss_avg_50/g/total": torch.mean(
                        torch.stack(list(avg_losses["gen_loss_50"]))
                    ),
                }
                summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                )

            pbar.update(1)
        # end of batch train
    # end of tqdm
    with torch.no_grad():
        torch.cuda.empty_cache()

    # Logging and checkpointing
    if rank == 0:
        # used for tensorboard chart - all/mel
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )
        # used for tensorboard chart - slice/mel_org
        if randomized:
            y_mel = commons.slice_segments(
                mel,
                ids_slice,
                config.train.segment_size // config.data.hop_length,
                dim=3,
            )
        else:
            y_mel = mel
        # used for tensorboard chart - slice/mel_gen
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.hop_length,
            config.data.win_length,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )

        lr = optim_g.param_groups[0]["lr"]

        scalar_dict = {
            "loss/g/total": loss_gen_all,
            "loss/d/adv": loss_disc,
            "learning_rate": lr,
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "loss/g/adv": loss_gen,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
        }

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }

        if epoch % save_every_epoch == 0:
            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                with torch.no_grad():
                    if hasattr(net_g, "module"):
                        o, *_ = net_g.module.infer(*reference)
                    else:
                        o, *_ = net_g.infer(*reference)
            audio_dict = {f"gen/audio_{global_step:07d}": o[0, :, :]}
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
            )

    # Save checkpoint
    model_add = []
    model_del = []
    done = False

    if rank == 0:
        overtrain_info = ""
        # Check overtraining
        if overtraining_detector and rank == 0 and epoch > 1:
            # Add the current loss to the history
            current_loss_disc = float(loss_disc)
            loss_disc_history.append(current_loss_disc)
            # Update smoothed loss history with loss_disc
            smoothed_value_disc = update_exponential_moving_average(
                smoothed_loss_disc_history, current_loss_disc
            )
            # Check overtraining with smoothed loss_disc
            is_overtraining_disc = check_overtraining(
                smoothed_loss_disc_history, overtraining_threshold * 2
            )
            if is_overtraining_disc:
                consecutive_increases_disc += 1
            else:
                consecutive_increases_disc = 0
            # Add the current loss_gen to the history
            current_loss_gen = float(lowest_value["value"])
            loss_gen_history.append(current_loss_gen)
            # Update the smoothed loss_gen history
            smoothed_value_gen = update_exponential_moving_average(
                smoothed_loss_gen_history, current_loss_gen
            )
            # Check for overtraining with the smoothed loss_gen
            is_overtraining_gen = check_overtraining(
                smoothed_loss_gen_history, overtraining_threshold, 0.01
            )
            if is_overtraining_gen:
                consecutive_increases_gen += 1
            else:
                consecutive_increases_gen = 0
            overtrain_info = f"Smoothed loss_g {smoothed_value_gen:.3f} and loss_d {smoothed_value_disc:.3f}"
            # Save the data in the JSON file if the epoch is divisible by save_every_epoch
            if epoch % save_every_epoch == 0:
                save_to_json(
                    training_file_path,
                    loss_disc_history,
                    smoothed_loss_disc_history,
                    loss_gen_history,
                    smoothed_loss_gen_history,
                )

            if (
                is_overtraining_gen
                and consecutive_increases_gen == overtraining_threshold
                or is_overtraining_disc
                and consecutive_increases_disc == overtraining_threshold * 2
            ):
                print(
                    f"Overtraining detected at epoch {epoch} with smoothed loss_g {smoothed_value_gen:.3f} and loss_d {smoothed_value_disc:.3f}"
                )
                done = True
            else:
                print(
                    f"New best epoch {epoch} with smoothed loss_g {smoothed_value_gen:.3f} and loss_d {smoothed_value_disc:.3f}"
                )
                old_model_files = glob.glob(
                    os.path.join(experiment_dir, f"{model_name}_*e_*s_best_epoch.pth")
                )
                for file in old_model_files:
                    model_del.append(file)
                model_add.append(
                    os.path.join(
                        experiment_dir,
                        f"{model_name}_{epoch}e_{global_step}s_best_epoch.pth",
                    )
                )

        # Print training progress
        lowest_value_rounded = float(lowest_value["value"])
        lowest_value_rounded = round(lowest_value_rounded, 3)

        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        if epoch > 1:
            record = (
                record
                + f" | lowest_value={lowest_value_rounded} (epoch {lowest_value['epoch']} and step {lowest_value['step']})"
            )

        if overtraining_detector:
            remaining_epochs_gen = overtraining_threshold - consecutive_increases_gen
            remaining_epochs_disc = (
                overtraining_threshold * 2 - consecutive_increases_disc
            )
            record = (
                record
                + f" | Number of epochs remaining for overtraining: g/total: {remaining_epochs_gen} d/total: {remaining_epochs_disc} | smoothed_loss_gen={smoothed_value_gen:.3f} | smoothed_loss_disc={smoothed_value_disc:.3f}"
            )
        print(record)

        # Save weights every N epochs
        if epoch % save_every_epoch == 0:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"
            save_checkpoint(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
            )
            save_checkpoint(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "D_" + checkpoint_suffix),
            )
            if custom_save_every_weights:
                model_add.append(
                    os.path.join(
                        experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                    )
                )

        # Check completion
        if epoch >= custom_total_epoch:
            lowest_value_rounded = float(lowest_value["value"])
            lowest_value_rounded = round(lowest_value_rounded, 3)
            print(
                f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_all.item(), 3)} loss gen."
            )
            print(
                f"Lowest generator loss: {lowest_value_rounded} at epoch {lowest_value['epoch']}, step {lowest_value['step']}"
            )
            # Final model
            model_add.append(
                os.path.join(
                    experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                )
            )
            done = True

        # Clean-up old best epochs
        for m in model_del:
            os.remove(m)

        if model_add:
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=config.data.sample_rate,
                        name=model_name,
                        model_path=m,
                        epoch=epoch,
                        step=global_step,
                        hps=hps,
                        overtrain_info=overtrain_info,
                        vocoder=vocoder,
                    )

        if done:
            # Clean-up process IDs from config.json
            pid_file_path = os.path.join(experiment_dir, "config.json")
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)
                json.dump(pid_data, pid_file, indent=4)
            os._exit(2333333)

        with torch.no_grad():
            torch.cuda.empty_cache()


def check_overtraining(smoothed_loss_history, threshold, epsilon=0.004):
    """
    Checks for overtraining based on the smoothed loss history.

    Args:
        smoothed_loss_history (list): List of smoothed losses for each epoch.
        threshold (int): Number of consecutive epochs with insignificant changes or increases to consider overtraining.
        epsilon (float): The maximum change considered insignificant.
    """
    if len(smoothed_loss_history) < threshold + 1:
        return False

    for i in range(-threshold, -1):
        if smoothed_loss_history[i + 1] > smoothed_loss_history[i]:
            return True
        if abs(smoothed_loss_history[i + 1] - smoothed_loss_history[i]) >= epsilon:
            return False
    return True


def update_exponential_moving_average(
    smoothed_loss_history, new_value, smoothing=0.987
):
    """
    Updates the exponential moving average with a new value.

    Args:
        smoothed_loss_history (list): List of smoothed values.
        new_value (float): New value to be added.
        smoothing (float): Smoothing factor.
    """
    if smoothed_loss_history:
        smoothed_value = (
            smoothing * smoothed_loss_history[-1] + (1 - smoothing) * new_value
        )
    else:
        smoothed_value = new_value
    smoothed_loss_history.append(smoothed_value)
    return smoothed_value


def save_to_json(
    file_path,
    loss_disc_history,
    smoothed_loss_disc_history,
    loss_gen_history,
    smoothed_loss_gen_history,
):
    """
    Save the training history to a JSON file.
    """
    data = {
        "loss_disc_history": loss_disc_history,
        "smoothed_loss_disc_history": smoothed_loss_disc_history,
        "loss_gen_history": loss_gen_history,
        "smoothed_loss_gen_history": smoothed_loss_gen_history,
    }
    with open(file_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
