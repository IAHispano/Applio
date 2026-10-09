"""Staged V3 training on cached features, independent of the Gradio interface.

For the residual family, predictor trains conditioning and deterministic mel
prediction. Flow/shortcut freeze those branches and train residual velocity;
shortcut uses a detached EMA teacher. The shallow-flow family instead trains
auxiliary prediction and direct-mel velocity jointly in one predictor stage;
it rejects separate flow/shortcut stages. Adaptation remaps speaker vocabulary
and trains adapters or the full acoustic model using that family's objective.
Vocoder training has its own generator and waveform/STFT critics.

Each yielded progress record follows one accumulated update. Rank zero validates
and writes atomic resumable checkpoints; all ranks synchronize. Stage changes
initialize from EMA, whereas exact resume restores live weights, optimizers,
critics, scaler, random states and unchanged data/settings. See docs/README.md.
"""

import json
import math
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rvc.configs.neural import (
    DEFAULT_BATCH_SIZE,
    AcousticConfig,
    VocoderConfig,
    require_contract,
)
from rvc.lib.algorithm.acoustic.model import EMA, create_acoustic, masked_mean
from rvc.lib.algorithm.acoustic.adapters import install_adapters
from rvc.lib.algorithm.acoustic.spectral import MelExtractor, spectral_loss
from rvc.lib.algorithm.acoustic.vocoder import (
    SpectralVocoder,
    WaveformCritics,
    discriminator_loss,
    generator_loss,
)
from rvc.train.process.checkpoints import (
    atomic_save,
    construct,
    header,
    load_payload,
    restore_rng,
    rng_state,
)
from rvc.train.process.tensorboard import log_training
from rvc.train.acoustic.data import AcousticDataset, atomic_json, collate, condition_batch
from rvc.train.acoustic.distributed import TrainingGroup


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
    """Select the family's stage parameters while preserving adaptation freezing.

    Joint predictor training updates both branches. Adaptation retains the
    trainability selected by adapter installation or explicit full fine-tuning;
    it must not silently unfreeze the base when collecting optimizer parameters.
    """

    if getattr(model.config, "family", "residual") == "shallow-flow":
        if phase in {"flow", "shortcut"}:
            raise ValueError(
                "Shallow-flow trains auxiliary and velocity jointly; use predictor or adapt"
            )
        if phase == "predictor":
            model.requires_grad_(True)
    elif phase in {"flow", "shortcut"}:
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


def resolve_precision(device, precision):
    """Choose arithmetic for the device without changing the model architecture.

    Resolve before recording settings so exact resume compares the actual dtype.
    Explicit precision choices remain available for reproducible experiments.
    """
    if precision != "auto":
        return precision
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


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
def validate_vocoder(model, dataset, extractor, device, listening_output=None):
    previous_mode = model.training
    model.eval()
    losses = []
    from rvc.realtime.streaming import coordinate_noise

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
        if i == 0 and listening_output is not None:
            # Reuse the deterministic held-out forward pass. No extra GPU work.
            # The summary wrapper logs this fixed pair at each saved update.
            import soundfile as sf

            directory = Path(listening_output)
            directory.mkdir(parents=True, exist_ok=True)
            rate = dataset.config.sample_rate
            count = min(int(batch["waveform_length"][0]), rate * 8)
            for name, wave in (
                ("reference", batch["waveform"]),
                ("reconstruction", output),
            ):
                sf.write(
                    directory / f"{name}.wav",
                    wave[0, :count].float().cpu().numpy(),
                    rate,
                    subtype="FLOAT",
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
    batch_size=DEFAULT_BATCH_SIZE,
    crop_frames=128,
    learning_rate=2e-4,
    device="auto",
    precision="auto",
    seed=1234,
    checkpoint_every=100,
    pretrained=None,
    resume=None,
    config=None,
    adapter_rank=8,
    adaptation="lora",
    accumulation_steps=1,
    stop_requested=None,
    mel_detail_weight=0.0,
    spectral_vocoder=None,
    waveform_weight=0.0,
    mel_adversarial_weight=0.0,
    pitch_guidance=None,
    waveform_adversarial_weight=0.0,
    sampling_mode="segments",
    validation_limit=0,
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
    if not math.isfinite(mel_detail_weight) or not 0 <= mel_detail_weight <= 1:
        raise ValueError("Mel detail weight must be finite and between zero and one")
    if mel_detail_weight and (
        kind != "acoustic" or phase not in {"predictor", "adapt"}
    ):
        raise ValueError(
            "Mel detail supervision applies to acoustic predictor/adaptation stages"
        )
    if not math.isfinite(waveform_weight) or not 0 <= waveform_weight <= 1:
        raise ValueError(
            "Waveform objective weight must be finite and between zero and one"
        )
    if waveform_weight and (
        not spectral_vocoder
        or kind != "acoustic"
        or phase not in {"predictor", "adapt"}
    ):
        raise ValueError(
            "Waveform supervision requires an acoustic predictor/adaptation stage and a compatible frozen vocoder"
        )
    if (
        not math.isfinite(mel_adversarial_weight)
        or not 0 <= mel_adversarial_weight <= 1
    ):
        raise ValueError(
            "Mel adversarial weight must be finite and between zero and one"
        )
    if mel_adversarial_weight and (
        kind != "acoustic" or phase not in {"predictor", "adapt"}
    ):
        raise ValueError("Mel critics apply only to predictor/adaptation stages")
    if (
        not math.isfinite(waveform_adversarial_weight)
        or not 0 <= waveform_adversarial_weight <= 1
    ):
        raise ValueError("Rendered waveform adversarial weight must be in [0, 1]")
    if waveform_adversarial_weight and (
        not spectral_vocoder
        or kind != "acoustic"
        or phase not in {"predictor", "adapt"}
    ):
        raise ValueError(
            "Rendered waveform critics require a predictor/adaptation stage and frozen vocoder"
        )
    if pitch_guidance is not None and (
        not isinstance(pitch_guidance, bool)
        or kind != "acoustic"
        or phase not in {"predictor", "adapt"}
    ):
        raise ValueError(
            "Pitch guidance is a boolean acoustic predictor/adaptation option"
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
    precision = resolve_precision(device, precision)
    dtype, scaler = amp_settings(device, precision)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    load_waveform = kind == "vocoder" or bool(
        waveform_weight or waveform_adversarial_weight
    )
    training = AcousticDataset(
        manifest, "train", crop_frames, load_waveform=load_waveform
    )
    validation = AcousticDataset(
        manifest, "validation", crop_frames, load_waveform=kind == "vocoder"
    )
    whole_training = AcousticDataset(manifest, "train", 0, load_waveform=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    initialization = (
        load_payload(resume or pretrained) if (resume or pretrained) else None
    )
    adapters = initialization.get("adapters") if initialization else None
    if initialization:
        if kind == "vocoder" and initialization.get(
            "vocoder_backend", "spectral"
        ) not in {"spectral", "wavehax"}:
            raise ValueError(
                "Imported pretrained vocoders are frozen inference backends; select them for acoustic fine-tuning rather than native vocoder training"
            )
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
        if pitch_guidance is False and model.config.pitch_guidance:
            raise ValueError("Cannot remove trained pitch guidance from this model")
        if pitch_guidance and not model.config.pitch_guidance:
            if (
                resume
                or phase not in {"predictor", "adapt"}
                or bool(model.flow_trained)
            ):
                raise ValueError(
                    "Initialize pitch guidance in a new predictor/adaptation run from a predictor-only base"
                )
            if initialization.get("adapters"):
                raise ValueError(
                    "Initialize pitch guidance from merged predictor weights"
                )
            # Canonical parameter order keeps optimizer state aligned on resume.
            # Preserve every old tensor; only the zero projection and physical
            # feature buffers are new.
            upgraded = create_acoustic(
                replace(model.config, pitch_guidance=True), training.config
            ).to(device)
            missing, unexpected = upgraded.load_state_dict(
                model.state_dict(), strict=False
            )
            allowed = {
                "pitch_projection.weight",
                "pitch_features.basis",
                "pitch_features.harmonics",
                "pitch_features.offsets",
            }
            if set(missing) != allowed or unexpected:
                raise ValueError("Pitch initialization changed unrelated model tensors")
            model = upgraded
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
        if pitch_guidance is not None:
            configuration = replace(configuration, pitch_guidance=bool(pitch_guidance))
        model = create_acoustic(configuration, training.config).to(device)
    else:
        configuration = config or VocoderConfig(
            sample_rate=training.config.sample_rate,
            hop_length=training.config.hop_length,
            mel_dim=training.config.n_mels,
        )
        if configuration.backend == "wavehax":
            from rvc.lib.algorithm.acoustic.wavehax import WavehaxVocoder

            model = WavehaxVocoder(configuration).to(device)
        else:
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
        if phase == "shortcut" and not bool(model.flow_trained):
            raise ValueError("Shortcut requires an ordinary-flow trained base")
        if phase == "adapt" and not bool(model.predictor_trained):
            raise ValueError("Fine-tuning requires a trained acoustic predictor")
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
        # Joint shallow flow marks its velocity trained after the first update.
        # That flag does not turn a resumed joint predictor into residual-flow
        # adaptation. The exact settings comparison below still rejects adding
        # or changing objectives on resume.
        resuming_joint = bool(resume) and model.config.family == "shallow-flow"
        if not resuming_joint and (waveform_weight or waveform_adversarial_weight) and bool(
            model.flow_trained
        ):
            raise ValueError(
                "Waveform supervision currently requires a predictor-only base; disable it for refiner adaptation"
            )
        if not resuming_joint and mel_adversarial_weight and bool(model.flow_trained):
            raise ValueError(
                "Mel adversarial supervision requires a predictor-only base"
            )
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
    # Preserve old checkpoint settings byte-for-byte for the default sampler.
    # A changed data mixture must never silently exact-resume old optimizer/RNG.
    from rvc.train.acoustic.sampling import CorpusSampler, validation_indices
    corpus_sampler = CorpusSampler(training, sampling_mode)
    if sampling_mode != "segments":
        settings["sampling_mode"] = sampling_mode
        settings["sampling_version"] = 1
        if group.rank == 0:
            atomic_json(Path(output) / "sampling.json", corpus_sampler.summary)
    if validation_limit:
        if kind != "acoustic":
            raise ValueError("Bounded validation panels currently apply to acoustic training")
        indices = validation_indices(validation, validation_limit)
        settings["validation_limit"] = validation_limit
        settings["validation_selection_version"] = 1
        if group.rank == 0:
            atomic_json(Path(output) / "validation_panel.json", dict(indices=indices,
                examples=len(indices), full_validation_examples=len(validation),
                speakers=sorted({validation.entries[i]["speaker"] for i in indices}),
                selection="Round robin by voice and recording hash, distinct recordings before repeated segments"))
        validation = torch.utils.data.Subset(validation, indices)
    # Keep old checkpoints resumable without silently changing their objective.
    # New nonzero settings are persisted and must match on exact resume.
    if mel_detail_weight or (
        resume and "mel_detail_weight" in initialization.get("training_settings", {})
    ):
        settings["mel_detail_weight"] = mel_detail_weight
    frozen_vocoder = None
    if mel_adversarial_weight:
        settings.update(
            mel_adversarial_weight=mel_adversarial_weight,
            mel_critic_version=1,
            mel_adversarial_warmup=200,
        )
    if waveform_adversarial_weight:
        settings.update(
            waveform_adversarial_weight=waveform_adversarial_weight,
            rendered_critic_version=1,
            rendered_adversarial_warmup=200,
        )
    if waveform_weight or waveform_adversarial_weight:
        from rvc.train.extract.features import file_hash

        package = load_payload(spectral_vocoder)
        if package["kind"] != "vocoder":
            raise ValueError("Waveform supervision requires a vocoder package")
        require_contract(
            package["mel"],
            training.manifest["contract"]["mel"],
            "frozen training vocoder",
        )
        frozen_vocoder = construct(package, device).eval().requires_grad_(False)
        del package
        settings.update(
            waveform_weight=waveform_weight,
            spectral_vocoder_hash=file_hash(Path(spectral_vocoder)),
        )
    if resume and initialization.get("training_settings") != settings:
        raise ValueError(
            "Exact resume requires unchanged training settings, including objectives and frozen vocoder"
        )
    ema = EMA(model)
    critics, critic_optimizer, extractor = None, None, None
    if kind == "vocoder":
        critics = WaveformCritics().to(device)
        critic_optimizer = torch.optim.AdamW(
            critics.parameters(), lr=learning_rate, betas=(0.8, 0.99)
        )
        extractor = MelExtractor(training.config).to(device)
    elif mel_adversarial_weight or waveform_adversarial_weight:
        from rvc.lib.algorithm.acoustic.discriminators import (
            MelCritics,
            RenderedAcousticCritics,
        )

        # Independent of additional acoustic module initialization, paired
        # variants start from exactly the same discriminator weights.
        torch.manual_seed(seed)
        critics = (
            RenderedAcousticCritics(include_mel=bool(mel_adversarial_weight))
            if waveform_adversarial_weight
            else MelCritics()
        ).to(device)
        critic_optimizer = torch.optim.AdamW(
            critics.parameters(), lr=learning_rate * 2, betas=(0.8, 0.99)
        )
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
        global_indices = corpus_sampler.draw(local_size * group.world_size, sampler)
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
        waveform_losses = []
        adversarial_losses, matching_losses = [], []
        rendered_adversarial_losses, rendered_matching_losses = [], []
        bootstrap_seen = False
        if critics:
            critics.requires_grad_(True)
            critics.train()
            critic_optimizer.zero_grad(set_to_none=True)
            for cpu_batch in batches:
                batch = to_device(cpu_batch, device)
                if kind == "vocoder":
                    noise = torch.randn_like(batch["waveform"])
                    noises.append(noise.cpu())
                with torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=precision != "fp32"
                ):
                    with torch.no_grad():
                        if kind == "acoustic":
                            fake = model.predict(
                                condition_batch(model, batch), batch["mask"]
                            )
                        else:
                            fake = (
                                model(
                                    batch["mel"],
                                    batch["f0"],
                                    batch["voiced"],
                                    noise=noise,
                                )
                                * batch["waveform_mask"]
                            )
                    if kind == "acoustic":
                        d_loss = fake.new_zeros((), dtype=torch.float32)
                        if mel_adversarial_weight:
                            mel_critics = (
                                critics.mel if waveform_adversarial_weight else critics
                            )
                            d_loss = discriminator_loss(
                                mel_critics(model.normalize(batch["mel"]), batch),
                                mel_critics(fake, batch),
                            )
                        if waveform_adversarial_weight:
                            with torch.no_grad():
                                rendered = (
                                    frozen_vocoder(
                                        model.denormalize(fake),
                                        batch["f0"],
                                        batch["voiced"],
                                    )
                                    * batch["waveform_mask"]
                                )
                            d_loss = d_loss + discriminator_loss(
                                critics.rendered(batch["waveform"], batch),
                                critics.rendered(rendered, batch),
                            )
                    else:
                        d_loss = discriminator_loss(
                            critics(batch["waveform"]), critics(fake)
                        )
                backward_loss(d_loss, scaler, accumulation_steps)
                d_losses.append(float(d_loss.detach()))
            critic_norm = finish_update(
                critic_optimizer, scaler, list(critics.parameters()), group, clip=10
            )
            critics.requires_grad_(False)
            # Spectral-normalization power iteration is stateful. Freeze it for
            # real/fake feature matching, then restore training mode next update.
            if kind == "acoustic":
                critics.eval()
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
                    if phase == "predictor" or (
                        phase == "adapt"
                        and (
                            model.config.family == "shallow-flow"
                            or not bool(model.flow_trained)
                        )
                    ):
                        prediction = (
                            model.predict(condition, batch["mask"])
                            if frozen_vocoder is not None or mel_adversarial_weight
                            else None
                        )
                        loss = model.predictor_loss(
                            condition,
                            batch["mel"],
                            batch["mask"],
                            mel_detail_weight,
                            prediction=prediction,
                        )
                        if mel_adversarial_weight:
                            mel_critics = (
                                critics.mel if waveform_adversarial_weight else critics
                            )
                            with torch.no_grad():
                                real_scores = mel_critics(
                                    model.normalize(batch["mel"]), batch
                                )
                            adversarial, matching = generator_loss(
                                real_scores, mel_critics(prediction, batch)
                            )
                            # Let the newly initialized critics learn before
                            # applying their full gradient to the predictor.
                            ramp = min((step + 1) / 200, 1.0)
                            loss = loss + ramp * mel_adversarial_weight * (
                                adversarial + 2 * matching
                            )
                            adversarial_losses.append(float(adversarial.detach()))
                            matching_losses.append(float(matching.detach()))
                        if frozen_vocoder is not None:
                            # Freeze vocoder weights, not the input graph: waveform
                            # spectral gradients must reach the acoustic predictor.
                            rendered = (
                                frozen_vocoder(
                                    model.denormalize(prediction),
                                    batch["f0"],
                                    batch["voiced"],
                                )
                                * batch["waveform_mask"]
                            )
                            if waveform_weight:
                                rendered_loss = spectral_loss(
                                    rendered, batch["waveform"] * batch["waveform_mask"]
                                )
                                loss = loss + waveform_weight * rendered_loss
                                waveform_losses.append(float(rendered_loss.detach()))
                            if waveform_adversarial_weight:
                                with torch.no_grad():
                                    real_scores = critics.rendered(
                                        batch["waveform"], batch
                                    )
                                adversarial, matching = generator_loss(
                                    real_scores, critics.rendered(rendered, batch)
                                )
                                ramp = min((step + 1) / 200, 1.0)
                                loss = loss + ramp * waveform_adversarial_weight * (
                                    adversarial + 2 * matching
                                )
                                rendered_adversarial_losses.append(
                                    float(adversarial.detach())
                                )
                                rendered_matching_losses.append(
                                    float(matching.detach())
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
                                condition,
                                batch["mel"],
                                batch["mask"],
                                mel_detail_weight,
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
            if model.config.family == "shallow-flow" and phase in {
                "predictor",
                "adapt",
            }:
                model.predictor_trained.fill_(True)
                model.flow_trained.fill_(True)
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
            "precision": precision,
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
        if waveform_losses:
            progress["waveform_spectral_loss"] = group.mean(
                sum(waveform_losses) / len(waveform_losses)
            )
        if adversarial_losses:
            progress["mel_adversarial_loss"] = group.mean(
                sum(adversarial_losses) / len(adversarial_losses)
            )
            progress["mel_feature_matching_loss"] = group.mean(
                sum(matching_losses) / len(matching_losses)
            )
        if rendered_adversarial_losses:
            progress["rendered_adversarial_loss"] = group.mean(
                sum(rendered_adversarial_losses) / len(rendered_adversarial_losses)
            )
            progress["rendered_feature_matching_loss"] = group.mean(
                sum(rendered_matching_losses) / len(rendered_matching_losses)
            )
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
                        ema.model, validation, extractor, device, output / "listening"
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
                    < json.loads((output / "best.json").read_text(encoding="utf-8"))[
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
