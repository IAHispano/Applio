"""The training loop of the mel vocoders: NSF-BigVGAN and NSF-HiFiGAN, each by
its own recipe's spectral losses, against the v3 discriminator.

``train.py`` builds the run's spec from the command line; this takes it as a
JSON file.
"""

import json
import math
import os
import random
from collections import defaultdict, deque
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rvc.vocoders.balance import FamilyReducer, head_accuracies
from rvc.vocoders.build import (
    ARCHITECTURES,
    build_discriminator,
    build_generator,
    export_payload,
    generator_hparams,
    pretrained_generator,
    shipped_config,
    training_state,
)
from rvc.vocoders.data import (
    VocoderDataset,
    amp_setup,
    collate_vocoder,
    latest_checkpoint,
    load_run_config,
    precision_label,
    pretrained_weights,
    read_filelist,
    remove_older,
    run_dir,
    split_holdout,
    volume_augment,
)
from rvc.vocoders.diagnostics import branch_separation, clip_or_sample_grad_norm
from rvc.vocoders.distributed import Ranks, launch, parse_gpus
from rvc.vocoders.ema import WeightEMA
from rvc.vocoders.losses import (
    MultiResolutionSTFTLoss,
    MultiScaleSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    loud_crop,
    r1_penalty,
)
from rvc.vocoders.mel import LogMel
from rvc.vocoders.previews import RectifiedPreviews
from rvc.vocoders.progress import EpochRecorder, emit_machine_progress
from rvc.vocoders.schedules import prepare_schedulers
from rvc.vocoders.setup import apply_precision_policy, loader_workers, normalize_san_weights
from rvc.vocoders.stop import finish_stop, install_stop_handlers, uninterruptible_save
from rvc.vocoders.terminal import (
    info,
    print_model_summary,
    print_settings_panel,
    progress_task,
    success,
)

TAG = "[VOCODER]"
METRICS_INTERVAL = 8
ACCURACY_SAMPLE_EVERY = 4
SAN_DIRECTION_WEIGHT = 0.25
#: Ceiling of the FP16 GradScaler's scale: left to grow, it doubles until a
#: gradient overflows and a step is skipped.
MAX_GRAD_SCALE = 2.0**16


def generator_gradient_metrics(net_g):
    """Gradient norms of the generator: the decoder, and ``conv_post`` kept
    apart as ``output``."""
    groups = {"decoder": [], "output": []}
    for name, parameter in net_g.named_parameters():
        if parameter.requires_grad:
            groups["output" if name.startswith("conv_post.") else "decoder"].append(parameter)
    metrics = {}
    for group, parameters in groups.items():
        if not parameters:
            continue
        parameter_norm = torch.sqrt(sum(p.detach().float().square().sum() for p in parameters))
        terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
        gradient_norm = torch.sqrt(sum(terms)) if terms else parameter_norm.new_zeros(())
        metrics[f"grad_norm_{group}"] = gradient_norm
        metrics[f"grad_to_param_{group}"] = gradient_norm / parameter_norm.clamp_min(1e-8)
    return metrics


class PlateauScale:
    """A factor on the scheduled LR, cut by ``factor`` whenever the monitored
    mean fails to beat its best by ``threshold`` (relative) for ``patience``
    evaluations in a row; never below ``min_scale``."""

    def __init__(self, factor: float, patience: int, threshold: float, min_scale: float):
        self.factor = factor
        self.patience = max(1, patience)
        self.threshold = threshold
        self.min_scale = min_scale
        self.scale = 1.0
        self.best = math.inf
        self.bad = 0

    def update(self, value: float) -> bool:
        """Record one evaluation; returns whether the scale was cut."""
        if value < self.best * (1.0 - self.threshold):
            self.best, self.bad = value, 0
            return False
        self.bad += 1
        if self.bad < self.patience or self.scale <= self.min_scale:
            return False
        self.scale = max(self.min_scale, self.scale * self.factor)
        self.bad = 0
        return True

    def state_dict(self) -> dict:
        return {"scale": self.scale, "best": self.best, "bad": self.bad}

    def load_state_dict(self, state: dict) -> None:
        self.scale, self.best, self.bad = state["scale"], state["best"], state["bad"]


class SpectralLoss:
    """The recipe's reconstruction loss, at its weights: L1 on the log mel,
    read up to Nyquist, plus the multi-resolution STFT loss when the recipe
    sets ``c_stft``."""

    def __init__(self, config: dict, device):
        data, settings = config["data"], config["vocoder"]
        sample_rate = data["sample_rate"]
        self.mel = LogMel(
            sample_rate, data["n_fft"], data["win_length"], data["hop_length"],
            settings.get("loss_n_mels", data["n_mels"]), settings.get("loss_fmin", data["mel_fmin"]),
            sample_rate / 2,
        ).to(device)
        self.weight = settings["c_mel"]
        self.label = f"mel x{self.weight:g}"
        self.stft = None
        # The last call's terms, for the log, when there are two.
        self.terms = {}
        if settings.get("c_stft"):
            self.stft = MultiResolutionSTFTLoss(
                settings["loss_fft_sizes"], settings["loss_hop_sizes"], settings["loss_win_lengths"]
            ).to(device)
            self.stft_weight = settings["c_stft"]
            self.label += f" + MR-STFT x{self.stft_weight:g}"

    def __call__(self, y, y_hat):
        loss_mel = F.l1_loss(self.mel(y_hat.squeeze(1)), self.mel(y.squeeze(1))) * self.weight
        if self.stft is None:
            return loss_mel
        sc_loss, mag_loss = self.stft(y_hat.squeeze(1), y.squeeze(1))
        loss_stft = (sc_loss + mag_loss) * self.stft_weight
        self.terms = {"loss_mel": loss_mel.detach(), "loss_stft": loss_stft.detach()}
        return loss_stft + loss_mel


def main(spec_path: str) -> None:
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    install_stop_handlers()
    launch(train, spec_path, parse_gpus(spec.get("gpu", "0")))


def train(ranks: Ranks, spec_path: str) -> None:
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    install_stop_handlers()
    ranks.setup()
    main_rank = ranks.main

    name = spec["model_name"]
    architecture = spec["architecture"]
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}, not {architecture!r}.")
    config = load_run_config(name, spec.get("config") or shipped_config(architecture))
    settings = config["vocoder"]
    if settings.get("architecture", architecture) != architecture:
        raise ValueError(
            f"{name}'s config is a {settings['architecture']} recipe, not {architecture}. "
            "Use another model name."
        )
    data = config["data"]
    sample_rate = data["sample_rate"]
    hop = data["hop_length"]
    out_dir = run_dir(name)
    os.makedirs(out_dir, exist_ok=True)

    device = ranks.device
    seed = settings.get("seed")
    if seed is not None:
        # Offset per rank, as ``Ranks.setup`` does; rank 0 keeps the seed.
        random.seed(int(seed) + ranks.rank)
        torch.manual_seed(int(seed) + ranks.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    entries, holdout_entries = split_holdout(
        read_filelist(spec["filelist"], spec.get("data_root")), int(settings.get("holdout_clips", 0))
    )
    dataset = VocoderDataset(entries, config, settings["segment_size"] // hop)
    # Loaded once: fixed middle crops, the same on every rank and every call.
    holdout = list(
        DataLoader(
            VocoderDataset(
                holdout_entries, config,
                int(settings.get("eval_segment_size", 131072)) // hop, augment=False,
            ),
            batch_size=min(int(spec["batch_size"]), len(holdout_entries)),
            collate_fn=collate_vocoder,
        )
    ) if holdout_entries else []
    eval_interval = int(settings.get("eval_interval", 0)) if holdout else 0
    # Per GPU.
    batch_size = int(spec["batch_size"])
    if len(dataset) // ranks.world < batch_size:
        raise ValueError(
            f"{len(dataset)} clips is fewer than one batch of {batch_size} on each of {ranks.world} GPU(s)."
        )
    workers = loader_workers(4)
    sampler = ranks.sampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_vocoder,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    hparams = generator_hparams(architecture, config)
    net_g = build_generator(architecture, hparams).to(device)
    net_d = build_discriminator(config).to(device)
    learning_rate = float(settings["learning_rate"])
    if spec.get("pretrained_g"):
        learning_rate = float(settings.get("finetune_learning_rate", learning_rate))
    adam = {
        "betas": tuple(settings["betas"]),
        "eps": float(settings.get("eps", 1e-9)),
        "weight_decay": float(settings.get("weight_decay", 0.01)),
    }
    optim_g = torch.optim.AdamW(net_g.parameters(), learning_rate, **adam)
    optim_d = torch.optim.AdamW(net_d.parameters(), learning_rate, **adam)
    amp_dtype, scaler = amp_setup(spec.get("precision", "fp32"), device, TAG)

    # Reduce on plateau of the spectral loss, the only non-adversarial one: the
    # training mean over ``plateau_interval`` steps, or with ``plateau_metric``
    # "val", the held-out one at each evaluation.
    plateau_metric = str(settings.get("plateau_metric", "train"))
    if plateau_metric not in ("train", "val"):
        raise ValueError(f"plateau_metric must be 'train' or 'val', not {plateau_metric!r}.")
    if plateau_metric == "val" and not eval_interval:
        raise ValueError("plateau_metric 'val' needs holdout_clips and eval_interval.")
    plateau_interval = eval_interval if plateau_metric == "val" else int(settings.get("plateau_interval", 0))
    plateau = PlateauScale(
        float(settings.get("plateau_factor", 0.5)),
        int(settings.get("plateau_patience", 3)),
        float(settings.get("plateau_threshold", 0.005)),
        float(settings.get("plateau_min_scale", 0.1)),
    )

    epoch, step, skipped = 1, 0, 0
    resume_g = None if spec.get("fresh") else latest_checkpoint(out_dir, "G")
    resume_d = None if spec.get("fresh") else latest_checkpoint(out_dir, "D")
    resumed = bool(resume_g and resume_d)
    ema = WeightEMA(net_g, settings["ema_decay"])
    starting_point = "scratch"
    if resumed:
        state_g = torch.load(resume_g, map_location="cpu", weights_only=True)
        state_d = torch.load(resume_d, map_location="cpu", weights_only=True)
        if state_g.get("architecture", architecture) != architecture:
            raise ValueError(
                f"{resume_g} is a {state_g['architecture']} checkpoint, not {architecture}. "
                "Use --fresh or another model name."
            )
        net_g.load_state_dict(state_g["model"])
        net_d.load_state_dict(state_d["model"])
        optim_g.load_state_dict(state_g["optimizer"])
        optim_d.load_state_dict(state_d["optimizer"])
        if "ema" in state_g:
            ema.load_state_dict(state_g["ema"], net_g)
        if scaler is not None and state_g.get("scaler"):
            scaler.load_state_dict(state_g["scaler"])
        epoch, step = state_g["epoch"] + 1, state_g["step"]
        skipped = int(state_g.get("amp_skipped_steps", 0))
        # A best from the other metric is on another scale; that one restarts.
        if state_g.get("plateau") and state_g["plateau"].get("metric", "train") == plateau_metric:
            plateau.load_state_dict(state_g["plateau"])
        starting_point = f"resumed from {os.path.basename(resume_g)}"
    else:
        if spec.get("pretrained_g"):
            weights = pretrained_generator(spec["pretrained_g"])
            net_g.load_state_dict(training_state(weights, net_g.state_dict()))
            ema.reseed(net_g)
            starting_point = f"G from {spec['pretrained_g']}"
        if spec.get("pretrained_d"):
            net_d.load_state_dict(pretrained_weights(spec["pretrained_d"]))
            starting_point += f", D from {spec['pretrained_d']}"

    # "exp decay epoch" or "exp decay step"; the step variant spreads
    # ``lr_decay`` over the epoch's steps.
    lr_scheduler = str(settings.get("lr_scheduler", "exp decay epoch"))
    lr_final_ratio = settings.get("lr_final_ratio")
    scheduler_g, scheduler_d = prepare_schedulers(
        optim_g, optim_d, lr_scheduler, lr_scheduler, float(settings["lr_decay"]),
        int(spec["total_epochs"]), epoch, step, loader,
        fresh_start=not resumed,
        lr_final_ratio=None if lr_final_ratio is None else float(lr_final_ratio),
    )
    step_schedulers = lr_scheduler == "exp decay step"
    # Linear ramps from the first step: the LR of both optimizers, and the
    # generator's adversarial weight.
    lr_warmup = int(settings.get("warmup_steps", 0))
    adv_warmup = int(settings.get("adv_warmup_steps", 0))

    def ramp(length: int) -> float:
        return 1.0 if length <= 0 else min(1.0, (step + 1) / length)

    apply_precision_policy(net_g, amp_dtype)
    # The generator update reads D's frozen weights in two passes, real then
    # fake: cached, each weight norm is built once. Not under spectral norm,
    # whose power iteration advances per build.
    cache_d_weights = not getattr(net_d, "use_spectral_norm", False)
    # Only the two training forwards go through DDP. The generator update's
    # D pass and R1 use ``net_d`` itself: with D frozen no gradient hook
    # fires, and an armed allreduce would never complete.
    train_g = ranks.wrap(net_g)
    train_d = ranks.wrap(
        net_d, find_unused_parameters=bool(getattr(net_d, "uses_branchwise_r1", False))
    )

    def autocast():
        return torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None)

    spectral_loss = SpectralLoss(config, device)
    stft_distance = MultiScaleSTFTLoss().to(device)
    san = bool(getattr(net_d, "supports_san", False))
    branch_weights = tuple(net_d.branch_weights) if getattr(net_d, "uses_branch_weights", False) else None
    # ``average_heads``: the GAN losses, and R1 with them, as a mean over the
    # discriminator's heads instead of their sum.
    heads = len(net_d.discriminators) if settings.get("average_heads", False) else 1
    c_fm = float(settings["c_fm"])
    r1_gamma = float(getattr(net_d, "r1_gamma", 0.0))
    r1_branches = len(net_d.discriminators) if r1_gamma > 0 else 0
    r1_every = max(1, math.ceil(int(getattr(net_d, "r1_interval", 16)) / r1_branches)) if r1_branches else 1
    r1_period = r1_every * max(1, r1_branches)
    # R1's squared input gradient can underflow in FP16, so it runs in FP32 there.
    r1_dtype = amp_dtype if amp_dtype == torch.bfloat16 else None

    writer = SummaryWriter(os.path.join(out_dir, "eval")) if main_rank else None
    previews = RectifiedPreviews(out_dir, config, step, device) if main_rank else None
    reference = dataset.reference() if main_rank else None
    total_epochs = int(spec["total_epochs"])
    save_every = max(1, int(spec["save_every"]))
    volume_prob = float(settings.get("volume_aug_prob", 0.0))
    grad_clip = float(settings.get("grad_clip") or "inf")

    if main_rank:
        print_model_summary(
            [("Generator", net_g), ("Discriminator", net_d)],
            title=f"{architecture} {sample_rate} Hz",
        )
        print_settings_panel(
            [
                ("Model", name),
                ("Architecture", architecture),
                ("Clips", f"{len(dataset)} ({len(loader)} steps per epoch)"
                 + (f", {len(holdout_entries)} held out" if holdout_entries else "")),
                ("Batch size", batch_size),
                ("Volume augmentation", f"{volume_prob:g}" if volume_prob > 0 else "off"),
                ("Epochs", f"{epoch} -> {total_epochs}, saving every {save_every}"),
                ("Starting point", starting_point),
                ("Precision", precision_label(amp_dtype)),
                ("Seed", "random" if seed is None else int(seed)),
                ("Device", (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
                           + (f" x {ranks.world} GPUs" if ranks.world > 1 else "")),
                ("Optimizer", f"AdamW, lr {learning_rate:g}, {lr_scheduler} {settings['lr_decay']}"
                 + (f", final ratio {lr_final_ratio}" if lr_final_ratio is not None else "")
                 + (f", warmup {lr_warmup} steps" if lr_warmup > 0 else "")
                 + (f", {plateau_metric} plateau x{plateau.factor:g} after {plateau.patience} x"
                    f" {plateau_interval} steps (at x{plateau.scale:g})" if plateau_interval > 0 else "")),
                ("Losses", f"{spectral_loss.label}, FM x{c_fm:g}, adversarial"
                 + (f" ramped over {adv_warmup} steps" if adv_warmup > 0 else "")),
                ("Discriminator", f"v3, R1 gamma {r1_gamma:g}"
                 + (f", UnivHD x{net_d.univhd_weight:g}" if getattr(net_d, "use_univhd", False) else "")
                 + (", SAN" if san else "")
                 + (f", mean over {heads} heads" if heads > 1 else "")),
            ],
            title="Vocoder training",
        )

    def save(current_epoch: int):
        keep = spec.get("checkpoints", "latest")
        saved = []
        with uninterruptible_save("Vocoder checkpoint"):
            if keep == "none":
                # An older run's checkpoints would otherwise be resumed from,
                # silently undoing everything trained since.
                remove_older(out_dir, "G")
                remove_older(out_dir, "D")
            else:
                g_path = os.path.join(out_dir, f"G_{step}.pth")
                d_path = os.path.join(out_dir, f"D_{step}.pth")
                torch.save(
                    {"model": net_g.state_dict(), "optimizer": optim_g.state_dict(),
                     "ema": ema.state_dict(), "epoch": current_epoch, "step": step,
                     "architecture": architecture,
                     "scaler": scaler.state_dict() if scaler is not None else None,
                     "amp_skipped_steps": skipped, "plateau": {**plateau.state_dict(), "metric": plateau_metric}},
                    g_path,
                )
                torch.save(
                    {"model": net_d.state_dict(), "optimizer": optim_d.state_dict(),
                     "epoch": current_epoch, "step": step},
                    d_path,
                )
                if keep == "latest":
                    remove_older(out_dir, "G", g_path)
                    remove_older(out_dir, "D", d_path)
                saved.append(os.path.basename(g_path))
            export = os.path.join(out_dir, f"{name}_vocoder_{current_epoch}e_{step}s.pth")
            torch.save(
                {**export_payload(architecture, config, hparams, ema.cpu_state_dict()),
                 "epoch": current_epoch, "step": step},
                export,
            )
        saved.append(os.path.basename(export))
        success(f"Saved {' and '.join(saved)}.", tag=TAG)

    def optimizer_step(optimizer):
        """Step at the warmed-up, plateau-scaled LR, then put the scheduler's
        back: the exponential schedulers scale whatever LR the group holds."""
        scheduled = [group["lr"] for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] *= ramp(lr_warmup) * plateau.scale
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
        for group, lr in zip(optimizer.param_groups, scheduled):
            group["lr"] = lr

    def backward(loss, optimizer, parameters):
        """Backward and step; returns the gradient norm on sampled steps."""
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        norm = clip_or_sample_grad_norm(parameters, grad_clip, step, METRICS_INTERVAL)
        optimizer_step(optimizer)
        return norm

    rolling = int(settings.get("rolling_loss_steps", 50))
    preview_interval = int(settings.get("preview_interval", 500))
    labels = tuple(getattr(net_d, "branch_labels", ()))
    families = FamilyReducer(labels, branch_weights, device)
    caches = defaultdict(lambda: deque(maxlen=rolling))
    disc_cache, adv_cache, accuracy_cache = (deque(maxlen=rolling) for _ in range(3))
    r1_cache, skip_cache = deque(maxlen=rolling), deque(maxlen=rolling)

    def log_rolling():
        scalars = {
            "learning_rate/lr_d": optim_d.param_groups[0]["lr"] * plateau.scale,
            "learning_rate/lr_g": optim_g.param_groups[0]["lr"] * plateau.scale,
        }
        if plateau_interval > 0:
            scalars["learning_rate/plateau_scale"] = plateau.scale
        if lr_warmup > 0 or adv_warmup > 0:
            scalars["learning_rate/warmup"] = ramp(lr_warmup)
            scalars["diag/adv_weight"] = ramp(adv_warmup)
        if accuracy_cache:
            for family, value in zip(families.names, torch.stack(list(accuracy_cache)).mean(0).tolist()):
                scalars[f"balance_accuracy_{rolling}/{family}"] = value
        if r1_cache:
            per_branch = defaultdict(list)
            for branch, value in zip((b for b, _ in r1_cache), torch.stack([v for _, v in r1_cache]).tolist()):
                per_branch[branch].append(value)
            total = 0.0
            for branch, values in sorted(per_branch.items()):
                mean = sum(values) / len(values)
                total += mean
                scalars[f"r1_grad_sq/{labels[branch] if branch < len(labels) else branch}"] = mean
            scalars["r1/penalty"] = 0.5 * r1_gamma * total / heads
        if scaler is not None:
            scalars["AMP/grad_scaler_scale"] = scaler.get_scale()
            scalars["AMP/skipped_steps_total"] = skipped
            if skip_cache:
                scalars[f"AMP/skip_rate_{rolling}"] = sum(skip_cache) / len(skip_cache)
        for key, queue in caches.items():
            if queue:
                category = "loss" if key.startswith("loss_") else "grad"
                scalars[f"{category}_avg_{rolling}/{key}_{rolling}"] = torch.stack(list(queue)).mean().item()
        for prefix, cache in ((f"disc_sep_{rolling}", disc_cache), (f"adv_sep_{rolling}", adv_cache)):
            if cache:
                means = torch.stack(list(cache)).mean(0).tolist()
                for label, value in zip(labels, means):
                    scalars[f"{prefix}/{label}"] = value
        for key, value in scalars.items():
            writer.add_scalar(key, value, step)

    def render_preview():
        if reference is None:
            return
        ref_mel, ref_f0, ref_audio, ref_path = reference
        with ema.applied(net_g), torch.no_grad():
            generated = net_g(ref_mel.to(device), ref_f0.to(device))
        previews.log(epoch, step, ref_path, generated, ref_audio.to(device))

    @torch.no_grad()
    def evaluate():
        """Held-out spectral loss (at the scale of ``loss_spectral``) and
        MR-STFT distance through the EMA weights, with the source drawn from
        the same seeds on every call. Every rank must call it."""
        totals = torch.zeros(2, device=device)
        items = 0
        with ema.applied(net_g):
            for index, (mel, f0, y) in enumerate(holdout):
                y = y.to(device)
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(index)
                    y_hat = net_g(mel.to(device), f0.to(device)).float()
                totals[0] += spectral_loss(y, y_hat) * y.shape[0]
                totals[1] += stft_distance(y_hat, y) * y.shape[0]
                items += y.shape[0]
        return ranks.mean(totals / max(1, items)).tolist()

    def plateau_update(value: float):
        if plateau.update(value) and main_rank:
            info(f"{plateau_metric.capitalize()} spectral loss plateaued at {plateau.best:.3f}; "
                 f"LR scale now x{plateau.scale:g}.", tag=TAG)
        if main_rank:
            writer.add_scalar("learning_rate/plateau_metric", value, step)

    # Held off until both warmups end: the adversarial ramp raises the spectral
    # loss by itself.
    plateau_start = max(lr_warmup, adv_warmup)
    plateau_sum = torch.zeros((), device=device)
    plateau_count = 0

    recorder = EpochRecorder()
    net_g.train()
    net_d.train()
    while epoch <= total_epochs:
        metrics = ""
        epoch_sums = defaultdict(lambda: torch.zeros((), device=device))
        epoch_steps = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        with progress_task(
            len(loader), f"Epoch {epoch}/{total_epochs}", training=True, disable=not main_rank
        ) as (progress, task):
            for batch_index, (mel, f0, y) in enumerate(loader):
                mel = mel.to(device, non_blocking=True)
                f0 = f0.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if volume_prob > 0:
                    mel, y = volume_augment(mel, y, volume_prob)

                with autocast():
                    y_hat = train_g(mel, f0)
                    y_d_r, y_d_g, _, _ = train_d(y, y_hat.detach(), san_training=san, combine_inputs=True)
                    loss_d, loss_d_real, loss_d_fake = (
                        loss / heads for loss in discriminator_loss(
                            y_d_r, y_d_g, san_direction_weight=SAN_DIRECTION_WEIGHT,
                            branch_weights=branch_weights,
                        )
                    )
                loss_d_step = loss_d
                if r1_branches and step % r1_every == 0:
                    branch = (step // r1_every) % r1_branches
                    count = max(1, int(round(y.shape[0] * float(getattr(net_d, "r1_batch_fraction", 0.5)))))
                    real = loud_crop(y, count, int(getattr(net_d, "r1_segment", 0)))
                    r1 = r1_penalty(net_d, real, branch, dtype=r1_dtype)
                    r1_cache.append((branch, r1.detach()))
                    loss_d_step = loss_d + 0.5 * r1_gamma * r1_period * r1 / heads
                if families.names and step % ACCURACY_SAMPLE_EVERY == 0:
                    accuracy_cache.append(families(head_accuracies(y_d_r, y_d_g)))
                optim_d.zero_grad(set_to_none=True)
                grad_norm_d = backward(loss_d_step, optim_d, net_d.parameters())
                normalize_san_weights(net_d)
                disc_cache.append(branch_separation(y_d_r, y_d_g))
                del y_d_r, y_d_g

                # The generator update reads D and never trains it: frozen, its
                # backward skips D's weight gradients and still reaches y_hat.
                net_d.requires_grad_(False)
                with autocast(), (torch.nn.utils.parametrize.cached() if cache_d_weights else nullcontext()):
                    _, y_d_g, fmap_r, fmap_g = net_d(y, y_hat, no_grad_real=True)
                    loss_mel = spectral_loss(y, y_hat)
                    loss_fm = feature_loss(fmap_r, fmap_g, branch_weights=branch_weights) * c_fm / heads
                    loss_adv, branch_adv = generator_loss(
                        y_d_g, san_direction_weight=SAN_DIRECTION_WEIGHT, use_softplus=san,
                        branch_weights=branch_weights, per_branch=True,
                    )
                    loss_adv = loss_adv / heads
                    loss_g = loss_mel + loss_fm + loss_adv * ramp(adv_warmup)
                del fmap_r, fmap_g
                optim_g.zero_grad(set_to_none=True)
                if scaler is None:
                    loss_g.backward()
                else:
                    scaler.scale(loss_g).backward()
                    scaler.unscale_(optim_g)
                module_metrics = generator_gradient_metrics(net_g) if step % METRICS_INTERVAL == 0 else {}
                grad_norm_g = clip_or_sample_grad_norm(net_g.parameters(), grad_clip, step, METRICS_INTERVAL)
                optimizer_step(optim_g)
                net_d.requires_grad_(True)
                if scaler is not None:
                    scale = scaler.get_scale()
                    scaler.update()
                    overflowed = scaler.get_scale() < scale
                    skipped += int(overflowed)
                    skip_cache.append(float(overflowed))
                    if not overflowed and scaler.get_scale() > MAX_GRAD_SCALE:
                        scaler.update(MAX_GRAD_SCALE)
                ema.update(net_g)
                if step_schedulers:
                    for scheduler in (scheduler_g, scheduler_d):
                        scheduler.step()
                step += 1

                losses = {
                    "loss_disc": loss_d, "loss_disc_real": loss_d_real, "loss_disc_fake": loss_d_fake,
                    "loss_adv": loss_adv, "loss_gen_total": loss_g, "loss_fm": loss_fm,
                    "loss_spectral": loss_mel, **spectral_loss.terms,
                }
                for key, value in losses.items():
                    caches[key].append(value.detach().float())
                    epoch_sums[key] += value.detach().float()
                epoch_steps += 1
                if plateau_metric == "train" and plateau_interval > 0 and step > plateau_start:
                    plateau_sum += loss_mel.detach().float()
                    plateau_count += 1
                    # Counted rather than ``step % interval``, so every rank
                    # reaches the reduction together and a resume starts a
                    # whole interval.
                    if plateau_count == plateau_interval:
                        value = ranks.mean(plateau_sum / plateau_count).item()
                        plateau_sum.zero_()
                        plateau_count = 0
                        plateau_update(value)
                if eval_interval and step % eval_interval == 0:
                    val_spectral, val_stft = evaluate()
                    if main_rank:
                        writer.add_scalar("val/spectral", val_spectral, step)
                        writer.add_scalar("val/mrstft", val_stft, step)
                    if plateau_metric == "val" and step > plateau_start:
                        plateau_update(val_spectral)
                adv_cache.append(branch_adv)
                for key, value in (("grad_norm_d", grad_norm_d), ("grad_norm_g", grad_norm_g)):
                    if value is None:
                        continue
                    if torch.isfinite(value):
                        caches[key].append(value.detach().float())
                    elif main_rank:
                        writer.add_scalar(f"Grad_Norm_Diag/{key[-1].upper()}_Skipped", 1, step)
                for key, value in module_metrics.items():
                    if torch.isfinite(value):
                        caches[key].append(value.detach().float())

                if main_rank:
                    if not metrics or (batch_index + 1) % METRICS_INTERVAL == 0:
                        metrics = f"G={loss_g.item():.4f}  D={loss_d.item():.4f}"
                    progress.update(task, advance=1, metrics=metrics)
                    emit_machine_progress(epoch, total_epochs, batch_index + 1, len(loader), step, metrics, 0)
                    if step % rolling == 0:
                        log_rolling()
                    if step % preview_interval == 0:
                        render_preview()
                if ranks.stop_requested():
                    # Nothing is being written at a batch boundary.
                    finish_stop(writer)

        # Averaged over the GPUs; every rank takes part in the reduction.
        epoch_means = {key: ranks.mean(total / max(1, epoch_steps)) for key, total in sorted(epoch_sums.items())}
        if main_rank:
            print(f"{name} | epoch={epoch} | step={step} | {recorder.record()}")
            if epoch_steps:
                for key, mean in epoch_means.items():
                    writer.add_scalar(f"loss_avg/{key}", mean.item(), step)
                writer.add_scalar("learning_rate/lr_d", optim_d.param_groups[0]["lr"] * plateau.scale, step)
                writer.add_scalar("learning_rate/lr_g", optim_g.param_groups[0]["lr"] * plateau.scale, step)
        if not step_schedulers:
            for scheduler in (scheduler_g, scheduler_d):
                if scheduler is not None:
                    scheduler.step()
        if main_rank:
            if skipped and scaler is not None:
                info(f"GradScaler at {scaler.get_scale():.0f}; {skipped} step(s) skipped so far.", tag=TAG)
            if epoch % save_every == 0 or epoch == total_epochs:
                save(epoch)
                render_preview()
            writer.flush()
        epoch += 1

    if main_rank:
        writer.close()
        success("Vocoder training finished.", tag=TAG)
