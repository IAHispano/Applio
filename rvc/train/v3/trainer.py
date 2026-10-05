"""Staged V3 training on cached features, independent of the Gradio interface.

Predictor trains conditioning and deterministic mel prediction. Flow/shortcut
freeze those branches and train residual velocity; shortcut uses a detached EMA
teacher. Adaptation remaps speaker vocabulary and trains adapters or the full
acoustic model. Vocoder training has its own generator and waveform/STFT critics.

Each yielded progress record follows one accumulated update. Rank zero validates
and writes atomic resumable checkpoints; all ranks synchronize. Stage changes
initialize from EMA, whereas exact resume restores live weights, optimizers,
critics, scaler, random states and unchanged data/settings. See docs/README.md.
"""

import json
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rvc.configs.v3 import AcousticConfig, VocoderConfig, require_contract
from rvc.lib.algorithm.v3.acoustic import EMA, AcousticModel, masked_mean
from rvc.lib.algorithm.v3.adapters import install_adapters
from rvc.lib.algorithm.v3.spectral import MelExtractor, spectral_loss
from rvc.lib.algorithm.v3.vocoder import (
    SpectralVocoder,
    WaveformCritics,
    discriminator_loss,
    generator_loss,
)
from rvc.train.process.v3_checkpoints import (
    atomic_save,
    construct,
    header,
    load_payload,
    restore_rng,
    rng_state,
)
from rvc.train.process.v3_tensorboard import log_training
from rvc.train.v3.data import AcousticDataset, collate, condition_batch
from rvc.train.v3.distributed import TrainingGroup


def to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def fit_statistics(model, dataset, device, residual=False):
    """Fit fixed per-band mel statistics or predictor-residual RMS on training data.

    Full cached training segments are used rather than randomly sampled crops.
    Residual fitting runs after predictor initialization; its scale defines the
    coordinate system shared by flow targets, EMA composition and inference.
    """

    total, squares, count = None, None, 0
    for i in range(len(dataset)):
        batch = to_device(collate([dataset[i]]), device)
        values = batch["mel"]
        if residual:
            values = model.normalize(values) - model.predict(
                condition_batch(model, batch), batch["mask"]
            )
        values = values.double()
        summed, squared = values.sum((0, 2)), values.square().sum((0, 2))
        total = summed if total is None else total + summed
        squares = squared if squares is None else squares + squared
        count += values.shape[0] * values.shape[-1]
    if residual:
        model.residual_scale.copy_(
            (squares / count).sqrt().clamp_min(0.1).float()[None, :, None]
        )
    else:
        mean = total / count
        std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(0.1)
        model.mel_mean.copy_(mean.float()[None, :, None])
        model.mel_std.copy_(std.float()[None, :, None])
        model.statistics_fitted.fill_(True)


def training_parameters(model, phase):
    """Select predictor/conditioning or refiner parameters without crossing stage boundaries."""

    if phase in {"flow", "shortcut"}:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(("refiner_", "time_embedding", "step_embedding"))
            )
    elif phase == "predictor":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                not name.startswith(("refiner_", "time_embedding", "step_embedding"))
            )
    return [p for p in model.parameters() if p.requires_grad]


def resolve_device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def amp_settings(device, precision):
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        precision
    ]
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("FP16 training requires CUDA; use fp32 on CPU")
    if (
        device.type == "cuda"
        and precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This CUDA device does not support BF16; choose fp16 or fp32")
    return dtype, torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and precision == "fp16",
        init_scale=256,
    )


def backward_loss(loss, scaler, divisor=1):
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Nonfinite training objective; checkpoint preserved at previous successful save"
        )
    scaler.scale(loss / divisor).backward()


def finish_update(optimizer, scaler, parameters, group, clip=1):
    scaler.unscale_(optimizer)
    group.mean_gradients(parameters)
    gradients = [p.grad for p in parameters if p.grad is not None]
    total_norm = nn.utils.get_total_norm(gradients, error_if_nonfinite=False)
    if not torch.isfinite(total_norm) and scaler.is_enabled():
        # AMP records the overflow at unscale_, skips the update and backs off.
        # Never clip nonfinite gradients or advance EMA/capability flags here.
        scaler.update(new_scale=scaler.get_scale() / 2)
        optimizer.zero_grad(set_to_none=True)
        return None
    norm = nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return float(norm)


@torch.no_grad()
def validate_acoustic(model, dataset, device, shortcut):
    previous_mode = model.training
    model.eval()
    losses = []
    # Stable crop/seed and all held-out examples, not a random training minibatch.
    for i in range(len(dataset)):
        batch = to_device(collate([dataset[i]]), device)
        condition = condition_batch(model, batch)
        output = model.sample(
            condition,
            steps=(4 if shortcut else 8) if bool(model.flow_trained) else 0,
            seed=1234 + i,
            mask=batch["mask"],
            ordinary=not shortcut,
        )
        losses.append(float(masked_mean((output - batch["mel"]).abs(), batch["mask"])))
    model.train(previous_mode)
    return sum(losses) / len(losses)


@torch.no_grad()
def validate_vocoder(model, dataset, extractor, device):
    previous_mode = model.training
    model.eval()
    losses = []
    from rvc.realtime.v3_streaming import coordinate_noise

    for i in range(len(dataset)):
        batch = to_device(collate([dataset[i]]), device)
        length = batch["mel"].shape[-1] * model.config.hop_length
        noise = torch.from_numpy(coordinate_noise(length, seed=1234 + i).T.copy()).to(
            device
        )
        output = (
            model(batch["mel"], batch["f0"], batch["voiced"], noise=noise)
            * batch["waveform_mask"]
        )
        losses.append(
            float(masked_mean((extractor(output) - batch["mel"]).abs(), batch["mask"]))
        )
    model.train(previous_mode)
    return sum(losses) / len(losses)


@log_training
def train(
    manifest,
    output,
    kind="acoustic",
    phase="predictor",
    steps=10000,
    batch_size=2,
    crop_frames=128,
    learning_rate=2e-4,
    device="auto",
    precision="bf16",
    seed=1234,
    checkpoint_every=100,
    pretrained=None,
    resume=None,
    config=None,
    adapter_rank=8,
    adaptation="lora",
    accumulation_steps=1,
    stop_requested=None,
):
    """Yield staged training progress and save exact-resume checkpoints.

    steps is an additional budget, including on resume. pretrained initializes a
    new phase from EMA; resume restores the original phase and complete live state.
    batch_size is per microbatch/rank; accumulation increases effective batch.
    Vocoder alternates detached-fake critic updates and generator updates using
    the same excitation noise. A stop request is handled after a complete update,
    then validation/checkpoint saving finishes before the generator returns.
    Rank zero also writes TensorBoard scalars in output/tensorboard for every
    stage; CLI, GUI and direct callers share this automatic logging path.
    """

    if (
        steps < 1
        or batch_size < 1
        or crop_frames < 1
        or checkpoint_every < 1
        or accumulation_steps < 1
    ):
        raise ValueError(
            "Training budget, batch/crop and checkpoint interval must be positive"
        )
    if kind not in {"acoustic", "vocoder"} or phase not in {
        "predictor",
        "flow",
        "shortcut",
        "adapt",
    }:
        raise ValueError("Unknown training kind/phase")
    if kind == "vocoder":
        phase = "vocoder"
    if pretrained and resume:
        raise ValueError("Choose either phase initialization or exact resume")
    if adaptation not in {"lora", "full"}:
        raise ValueError("Adaptation must be lora or full")
    if (
        device in {"auto", "cuda"}
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
        and torch.cuda.is_available()
    ):
        device = "cuda:" + os.environ.get("LOCAL_RANK", "0")
    device = resolve_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    group = TrainingGroup(device)
    dtype, scaler = amp_settings(device, precision)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    training = AcousticDataset(manifest, "train", crop_frames)
    validation = AcousticDataset(manifest, "validation", crop_frames)
    whole_training = AcousticDataset(manifest, "train", 0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    initialization = (
        load_payload(resume or pretrained) if (resume or pretrained) else None
    )
    adapters = initialization.get("adapters") if initialization else None
    if initialization:
        if initialization["kind"] != kind:
            raise ValueError("Checkpoint kind does not match trainer")
        require_contract(
            initialization["mel"], training.manifest["contract"]["mel"], "training mel"
        )
        require_contract(
            initialization["features"],
            training.manifest["contract"]["features"],
            "training features",
        )
        if resume and initialization.get("inference_only"):
            raise ValueError("Inference packages cannot exactly resume training")
        if resume and (
            initialization["dataset_id"] != training.manifest["dataset_id"]
            or initialization["phase"] != phase
        ):
            raise ValueError(
                "Exact resume requires the original dataset contract and phase"
            )
        model = construct(initialization, device, use_ema=not bool(resume))
        if kind == "acoustic" and phase == "adapt" and not resume:
            if adapters:
                raise ValueError(
                    "Adapt from a merged base package, not an existing adapter checkpoint"
                )
            speakers = len(training.manifest["speakers"])
            embedding = nn.Embedding(speakers, model.config.condition_width).to(device)
            with torch.no_grad():
                embedding.weight.copy_(
                    model.speaker.weight.mean(0, keepdim=True).expand(speakers, -1)
                )
            model.speaker = embedding
            model.config = replace(model.config, speakers=speakers)
            if adaptation == "lora":
                install_adapters(model, adapter_rank, float(adapter_rank))
                adapters = {"rank": adapter_rank, "alpha": float(adapter_rank)}
            else:
                model.requires_grad_(True)
        elif (
            kind == "acoustic"
            and initialization["speakers"] != training.manifest["speakers"]
        ):
            raise ValueError(
                "Speaker vocabulary changed; use adaptation instead of continuation"
            )
    elif kind == "acoustic":
        if phase != "predictor":
            raise ValueError(
                "Flow/shortcut/adaptation phases require a pretrained acoustic checkpoint"
            )
        configuration = config or AcousticConfig(
            content_dim=training.manifest["contract"]["features"]["content_dim"],
            mel_dim=training.config.n_mels,
            speakers=len(training.manifest["speakers"]),
        )
        configuration = replace(
            configuration, speakers=len(training.manifest["speakers"])
        )
        model = AcousticModel(configuration).to(device)
    else:
        configuration = config or VocoderConfig(
            sample_rate=training.config.sample_rate,
            hop_length=training.config.hop_length,
            mel_dim=training.config.n_mels,
        )
        model = SpectralVocoder(configuration).to(device)
    if kind == "acoustic":
        if (model.config.content_dim, model.config.mel_dim, model.config.speakers) != (
            training.manifest["contract"]["features"]["content_dim"],
            training.config.n_mels,
            len(training.manifest["speakers"]),
        ):
            raise ValueError(
                "Acoustic model dimensions disagree with the training contract"
            )
        if phase == "flow" and not bool(model.predictor_trained):
            raise ValueError("Warm up the predictor before ordinary-flow training")
        if phase in {"shortcut", "adapt"} and not bool(model.flow_trained):
            raise ValueError(
                "Shortcut/adaptation requires an ordinary-flow trained base"
            )
        if not resume and phase == "predictor":
            model.flow_trained.fill_(False)
            model.shortcut_trained.fill_(False)
        elif not resume and phase == "flow":
            model.shortcut_trained.fill_(False)
        if not bool(model.statistics_fitted):
            fit_statistics(model, whole_training, device)
        if phase == "flow" and not resume:
            model.eval()
            if model.config.prediction_centered:
                fit_statistics(model, whole_training, device, residual=True)
            else:
                model.residual_scale.fill_(1)
        parameters = training_parameters(model, phase)
    else:
        if (
            model.config.sample_rate,
            model.config.hop_length,
            model.config.mel_dim,
        ) != (
            training.config.sample_rate,
            training.config.hop_length,
            training.config.n_mels,
        ):
            raise ValueError(
                "Vocoder dimensions disagree with the training mel contract"
            )
        parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=(0.9, 0.99))
    settings = {
        "batch_size": batch_size,
        "crop_frames": crop_frames,
        "precision": precision,
        "learning_rate": learning_rate,
        "seed": seed,
        "adaptation": adaptation,
        "adapter_rank": adapter_rank,
        "accumulation_steps": accumulation_steps,
        "world_size": group.world_size,
        "torch_version": str(torch.__version__),
    }
    if resume and initialization.get("training_settings") != settings:
        raise ValueError(
            "Exact resume requires unchanged batch/crop/precision/learning rate/seed"
        )
    ema = EMA(model)
    critics, critic_optimizer, extractor = None, None, None
    if kind == "vocoder":
        critics = WaveformCritics().to(device)
        critic_optimizer = torch.optim.AdamW(
            critics.parameters(), lr=learning_rate, betas=(0.8, 0.99)
        )
        extractor = MelExtractor(training.config).to(device)
    # Identical model/critic initialization; independent rank-local crop/noise RNG.
    torch.manual_seed(seed + group.rank)
    np.random.seed(seed + group.rank)
    random.seed(seed + group.rank)
    start = 0
    if resume:
        optimizer.load_state_dict(initialization["optimizer"])
        ema.model.load_state_dict(initialization["ema"])
        scaler.load_state_dict(initialization["scaler"])
        start = initialization["step"]
        if critics:
            critics.load_state_dict(initialization["critics"])
            critic_optimizer.load_state_dict(initialization["critic_optimizer"])
        restore_rng(initialization["rank_rng"][group.rank])
    model.train()
    for step in range(start, start + steps):
        local_size = batch_size * accumulation_steps
        sampler = torch.Generator().manual_seed(seed + step)
        global_indices = torch.randint(
            len(training), (local_size * group.world_size,), generator=sampler
        )
        indices = global_indices[
            group.rank * local_size : (group.rank + 1) * local_size
        ].tolist()
        # Only features/PCM live across microbatches; activation graphs are released.
        batches = [
            collate([training[i] for i in indices[start : start + batch_size]])
            for start in range(0, local_size, batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)
        losses, d_losses, noises = [], [], []
        bootstrap_seen = False
        if critics:
            critics.requires_grad_(True)
            critic_optimizer.zero_grad(set_to_none=True)
            for cpu_batch in batches:
                batch = to_device(cpu_batch, device)
                noise = torch.randn_like(batch["waveform"])
                noises.append(noise.cpu())
                with torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=precision != "fp32"
                ):
                    with torch.no_grad():
                        fake = (
                            model(
                                batch["mel"], batch["f0"], batch["voiced"], noise=noise
                            )
                            * batch["waveform_mask"]
                        )
                    d_loss = discriminator_loss(
                        critics(batch["waveform"]), critics(fake)
                    )
                backward_loss(d_loss, scaler, accumulation_steps)
                d_losses.append(float(d_loss.detach()))
            critic_norm = finish_update(
                critic_optimizer, scaler, list(critics.parameters()), group, clip=10
            )
            critics.requires_grad_(False)
        for micro, cpu_batch in enumerate(batches):
            batch = to_device(cpu_batch, device)
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=precision != "fp32"
            ):
                if kind == "acoustic":
                    if phase in {"flow", "shortcut"}:
                        with torch.no_grad():
                            condition = condition_batch(model, batch)
                    else:
                        condition = condition_batch(model, batch)
                    if phase == "predictor":
                        loss = model.predictor_loss(
                            condition, batch["mel"], batch["mask"]
                        )
                    else:
                        shortcut = phase == "shortcut" or (
                            phase == "adapt" and bool(model.shortcut_trained)
                        )
                        offset = (
                            step * local_size * group.world_size
                            + group.rank * local_size
                            + micro * batch_size
                        )
                        selected = (
                            torch.arange(batch_size, device=device) + offset
                        ) % 8 == 0
                        bootstrap_seen |= shortcut and bool(selected.any())
                        loss = model.flow_loss(
                            condition,
                            batch["mel"],
                            batch["mask"],
                            ema.model if shortcut else None,
                            bootstrap_select=selected if shortcut else None,
                        )
                        if phase == "adapt":
                            loss = loss + 0.2 * model.predictor_loss(
                                condition, batch["mel"], batch["mask"]
                            )
                else:
                    fake = (
                        model(
                            batch["mel"],
                            batch["f0"],
                            batch["voiced"],
                            noise=noises[micro].to(device),
                        )
                        * batch["waveform_mask"]
                    )
                    real = batch["waveform"]
                    with torch.no_grad():
                        real_scores = critics(real)
                    adversarial, matching = generator_loss(real_scores, critics(fake))
                    mel_loss = masked_mean(
                        (extractor(fake) - batch["mel"]).abs(), batch["mask"]
                    )
                    loss = (
                        15 * mel_loss
                        + spectral_loss(fake, real)
                        + adversarial
                        + 2 * matching
                    )
            backward_loss(loss, scaler, accumulation_steps)
            losses.append(float(loss.detach()))
        norm = finish_update(
            optimizer, scaler, parameters, group, clip=10 if kind == "vocoder" else 1
        )
        bootstrap_seen = group.any(bootstrap_seen)
        if kind == "acoustic" and norm is not None:
            if phase == "predictor":
                model.predictor_trained.fill_(True)
            if phase == "flow":
                model.flow_trained.fill_(True)
            if phase in {"shortcut", "adapt"} and bootstrap_seen:
                model.shortcut_trained.fill_(True)
        if norm is not None:
            ema.update(model)
        progress = {
            "step": step + 1,
            "kind": kind,
            "phase": phase,
            "loss": group.mean(sum(losses) / len(losses)),
            "gradient_norm": norm,
            "optimizer_updated": norm is not None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "amp_scale": scaler.get_scale(),
        }
        if critics:
            progress["critic_optimizer_updated"] = critic_norm is not None
            progress["critic_gradient_norm"] = critic_norm
            progress["discriminator_loss"] = group.mean(sum(d_losses) / len(d_losses))
        stopping = group.any(bool(stop_requested and stop_requested()))
        if stopping:
            progress["stopped"] = True
        if (step + 1) % checkpoint_every == 0 or step + 1 == start + steps or stopping:
            rank_rng = group.gather(rng_state())
            if group.rank == 0:
                if kind == "acoustic":
                    progress["validation_mel_l1"] = validate_acoustic(
                        ema.model, validation, device, bool(model.shortcut_trained)
                    )
                else:
                    progress["validation_mel_l1"] = validate_vocoder(
                        ema.model, validation, extractor, device
                    )
                payload = header(model, kind, training.manifest, adapters)
                payload.update(
                    weights=model.state_dict(),
                    ema=ema.model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),
                    step=step + 1,
                    phase=phase,
                    rng=rank_rng[0],
                    rank_rng=rank_rng,
                    training_settings=settings,
                )
                if critics:
                    payload.update(
                        critics=critics.state_dict(),
                        critic_optimizer=critic_optimizer.state_dict(),
                    )
                atomic_save(output / "last.pt", payload)
                if (
                    not (output / "best.json").exists()
                    or progress["validation_mel_l1"]
                    < json.loads((output / "best.json").read_text())[
                        "validation_mel_l1"
                    ]
                ):
                    atomic_save(output / "best.pt", payload)
                    (output / "best.json").write_text(
                        json.dumps(progress), encoding="utf-8"
                    )
            group.barrier()
        if group.rank == 0:
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(progress) + "\n")
            yield progress
        if stopping:
            break
