"""Production training & Applio experiment-directory fine-tuning entrypoint for HSVC.

Supports two modes:
  1. Applio Experiment Mode (`--exp-dir logs/<model_name>`):
     Directly reads Applio's standard `sliced_audios/`, `f0_voiced/` (or `f0/`), and `extracted/`
     directories produced by Applio's Preprocess & Extract steps, fine-tunes from `hsvc_base_48k.pth`
     (with frozen GRL content bottleneck + pretrained DualDomainDiscriminator), supports
     Applio's Stop Training PID tracking, Fresh Training (`--cleanup`), Resume, GPU Caching,
     TensorBoard logging, Save-Every-Epoch / Save-Only-Latest, and exports `<exp_dir>/<model_name>.pth`
     ready for auto-detected Inference & Realtime!
  2. Manifest Mode (`--manifest manifest.json --feat-dir features/`):
     Multi-speaker pretraining or custom dataset fine-tuning.
"""

import argparse
import copy
import datetime
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.getcwd())
from backend.architectures.hsvc.pipeline import compute_energy_100hz_torch
from backend.architectures.hsvc import (
    DualDomainDiscriminator,
    HardenedSpectralSpeakerCritic,
    HSVCConfig,
    HSVCGenerator,
    HSVCLoss,
    UtteranceSpeakerAdversary,
    compute_10ms_log_rms,
)
from backend.train.losses import discriminator_loss, feature_loss, generator_loss


def linear_upsample_content(c50: np.ndarray, n_frames: int) -> np.ndarray:
    t = torch.from_numpy(c50.astype(np.float32)).T.unsqueeze(0)
    return F.interpolate(t, size=n_frames, mode="linear", align_corners=False)[0].T.numpy()


def _format_duration(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _resolve_device(device_arg: str, gpu_arg: str) -> str:
    if gpu_arg is not None and str(gpu_arg).strip() != "":
        g = str(gpu_arg).strip()
        if g == "-" or not torch.cuda.is_available():
            return "cpu"
        first_gpu = g.split("-")[0].split(",")[0].strip()
        if first_gpu.isdigit():
            return f"cuda:{first_gpu}"
        return "cuda"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def _register_pid(exp_dir: str, add: bool = True):
    """Register/unregister process PID in Applio's config.json so Stop Training works cleanly."""
    if not exp_dir or not os.path.isdir(exp_dir):
        return
    cfg_path = os.path.join(exp_dir, "config.json")
    try:
        data = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        pids = list(data.get("process_pids", []))
        pid = os.getpid()
        if add:
            if pid not in pids:
                pids.append(pid)
            data["process_pids"] = pids
        else:
            pids = [p for p in pids if p != pid]
            if pids:
                data["process_pids"] = pids
            else:
                data.pop("process_pids", None)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception:
        pass


def _cleanup_exp_dir(exp_dir: str, model_name: str):
    """Delete previous HSVC weights, resume checkpoints, and tensorboard logs for Fresh Training."""
    if not exp_dir or not os.path.isdir(exp_dir):
        return
    patterns = [
        os.path.join(exp_dir, f"{model_name}*.pth"),
        os.path.join(exp_dir, "G_hsvc_latest.pt"),
        os.path.join(exp_dir, "events.out.tfevents.*"),
    ]
    removed = 0
    for pat in patterns:
        for fp in glob.glob(pat):
            try:
                os.remove(fp)
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"[HSVC] Fresh training cleanup removed {removed} previous file(s).", flush=True)


class ApplioExpHSVCDataset(Dataset):
    """Reads directly from an Applio `logs/<model_name>/` experiment directory."""

    def __init__(
        self,
        exp_dir: str,
        sr: int = 48000,
        hop_in: int = 480,
        crop_frames: int = 80,
        silence_aug_prob: float = 0.18,
        sid: int = 0,
        cache_in_ram: bool = False,
    ):
        self.sr = sr
        self.hop_in = hop_in
        self.crop_frames = crop_frames
        self.silence_aug_prob = silence_aug_prob
        self.sid = sid
        self.cache_in_ram = cache_in_ram
        self._cache = {}

        wav_dir = os.path.join(exp_dir, "sliced_audios")
        feat_dir = os.path.join(exp_dir, "extracted")
        f0_dir = os.path.join(exp_dir, "f0_voiced")
        if not os.path.isdir(f0_dir):
            f0_dir = os.path.join(exp_dir, "f0")

        self.items = []
        self.total_duration_s = 0.0
        for wp in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
            stem = os.path.splitext(os.path.basename(wp))[0]
            fp = os.path.join(feat_dir, f"{stem}.npy")
            f0p = os.path.join(f0_dir, f"{stem}.wav.npy")
            if not os.path.exists(f0p):
                f0p = os.path.join(f0_dir, f"{stem}.npy")
            if os.path.exists(fp) and os.path.exists(f0p):
                self.items.append((wp, fp, f0p))
                try:
                    self.total_duration_s += sf.info(wp).duration
                except Exception:
                    pass
        if not self.items:
            raise FileNotFoundError(
                f"No aligned (sliced_audios, extracted, f0_voiced) files found in {exp_dir}. "
                "Run Preprocess and Extract Features first."
            )

    def __len__(self):
        return len(self.items)

    def _load_item(self, idx):
        if self.cache_in_ram and idx in self._cache:
            return self._cache[idx]
        wp, fp, f0p = self.items[idx]
        wav, file_sr = sf.read(wp, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != self.sr:
            wav = soxr.resample(wav, file_sr, self.sr, quality="HQ").astype(np.float32)

        c50 = np.load(fp).astype(np.float32)
        f0 = np.load(f0p).astype(np.float32)
        n_frames = min(len(wav) // self.hop_in, len(f0))
        if n_frames < 10:
            n_frames = max(10, len(wav) // self.hop_in)
            wav = np.pad(wav, (0, max(0, n_frames * self.hop_in - len(wav))))
            f0 = np.pad(f0, (0, max(0, n_frames - len(f0))), mode="edge")
        wav = wav[: n_frames * self.hop_in]
        f0 = f0[:n_frames]
        vuv_conf = (f0 > 1.0).astype(np.float32)
        content = linear_upsample_content(c50, n_frames)

        e_t, hf_t = compute_energy_100hz_torch(torch.from_numpy(wav).unsqueeze(0), self.sr, n_frames)
        energy_db = e_t[0].numpy()
        hf_energy_db = hf_t[0].numpy()
        loaded = (content, f0, vuv_conf, energy_db, hf_energy_db, wav)
        if self.cache_in_ram:
            self._cache[idx] = loaded
        return loaded

    def __getitem__(self, idx):
        content, f0, vuv_conf, energy_db, hf_energy_db, wav = self._load_item(idx)
        return _crop_and_augment(
            content, f0, vuv_conf, energy_db, hf_energy_db, wav, self.sid,
            self.hop_in, self.crop_frames, self.silence_aug_prob,
        )


def _crop_and_augment(content, f0, vuv_conf, energy_db, hf_energy_db, wav, sid, hop_in, C, silence_aug_prob):
    n_frames = len(f0)
    if n_frames >= C:
        start = np.random.randint(0, n_frames - C + 1)
        end = start + C
        content, f0, vuv_conf = content[start:end], f0[start:end], vuv_conf[start:end]
        energy_db, hf_energy_db = energy_db[start:end], hf_energy_db[start:end]
        wav = wav[start * hop_in : end * hop_in]
    else:
        pad = C - n_frames
        content = np.pad(content, ((0, pad), (0, 0)), mode="constant")
        f0 = np.pad(f0, (0, pad), mode="constant")
        vuv_conf = np.pad(vuv_conf, (0, pad), mode="constant")
        energy_db = np.pad(energy_db, (0, pad), mode="constant", constant_values=-100.0)
        hf_energy_db = np.pad(hf_energy_db, (0, pad), mode="constant", constant_values=-100.0)
        wav = np.pad(wav, (0, pad * hop_in), mode="constant")

    if np.random.rand() < silence_aug_prob:
        sil_len = np.random.randint(12, min(28, C - 4))
        sil_start = np.random.randint(0, C - sil_len)
        sil_end = sil_start + sil_len
        f0, vuv_conf = f0.copy(), vuv_conf.copy()
        energy_db, hf_energy_db, wav = energy_db.copy(), hf_energy_db.copy(), wav.copy()
        if np.random.rand() < 0.5:
            content = content.copy()
            content[sil_start:sil_end] = 0.0
        f0[sil_start:sil_end] = 0.0
        vuv_conf[sil_start:sil_end] = 0.0
        floor_db = np.random.uniform(-100.0, -72.0)
        energy_db[sil_start:sil_end] = floor_db
        hf_energy_db[sil_start:sil_end] = floor_db
        fade_n = min(240, (sil_end - sil_start) * hop_in // 4)
        s_samp, e_samp = sil_start * hop_in, sil_end * hop_in
        wav[s_samp:e_samp] = 0.0
        if s_samp >= fade_n:
            wav[s_samp - fade_n : s_samp] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        if e_samp + fade_n <= len(wav):
            wav[e_samp : e_samp + fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)

    return (
        torch.from_numpy(content.copy()),
        torch.from_numpy(f0.copy()),
        torch.from_numpy(vuv_conf.copy()),
        torch.from_numpy(energy_db.copy()),
        torch.from_numpy(hf_energy_db.copy()),
        torch.from_numpy(wav.copy()).unsqueeze(0),
        int(sid),
    )


class CachedHSVCDataset(Dataset):
    """Loads pre-extracted 48 kHz audio and aligned 100 Hz .npz features."""

    def __init__(self, items, feat_dir: str, hop_in: int = 480, crop_frames: int = 80, silence_aug_prob: float = 0.18):
        self.items = [(it, os.path.join(feat_dir, f"{it['id']}.npz")) for it in items if os.path.exists(os.path.join(feat_dir, f"{it['id']}.npz"))]
        self.hop_in = hop_in
        self.crop_frames = crop_frames
        self.silence_aug_prob = silence_aug_prob
        self.total_duration_s = float(sum(it.get("duration", 0.0) for it, _ in self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it, fp = self.items[idx]
        wav, _ = sf.read(it["path"], dtype="float32")
        d = np.load(fp)
        n_frames = int(d["n_frames"])
        wav = wav[: n_frames * self.hop_in]
        content = linear_upsample_content(d["content_50hz"], n_frames)
        f0 = d["f0_100hz"].astype(np.float32)[:n_frames]
        vuv_conf = d["vuv_conf"].astype(np.float32)[:n_frames]
        energy_db = d["energy_db"].astype(np.float32)[:n_frames]
        hf_energy_db = d["hf_energy_db"].astype(np.float32)[:n_frames]
        return _crop_and_augment(
            content, f0, vuv_conf, energy_db, hf_energy_db, wav, int(it["sid"]),
            self.hop_in, self.crop_frames, self.silence_aug_prob,
        )


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.995):
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train_hsvc(
    output_dir: str,
    model_name: str = "hsvc_model",
    exp_dir: str = "",
    manifest_path: str = "",
    feat_dir: str = "",
    config_path: str = "backend/architectures/hsvc/configs/48000_base.json",
    base_package: str = "",
    base_discriminator: str = "",
    epochs: int = 0,
    steps: int = 800,
    save_every_epoch: int = 10,
    save_only_latest: bool = False,
    save_every_weights: bool = True,
    cleanup: bool = False,
    cache_data_in_gpu: bool = False,
    fp16_export: bool = False,
    sample_rate: int = 0,
    batch_size: int = 6,
    device: str = "cuda",
):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif device.startswith("cuda"):
        device = "cpu"

    with open(config_path) as f:
        cfg_json = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    if cleanup:
        _cleanup_exp_dir(output_dir, model_name)

    _register_pid(exp_dir or output_dir, add=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=output_dir)
    except Exception:
        writer = None

    try:
        tcfg = cfg_json["train"]
        mcfg = HSVCConfig.from_dict(cfg_json["model"])
        torch.manual_seed(tcfg["seed"])
        np.random.seed(tcfg["seed"])

        finetune_mode = bool(base_package and os.path.isfile(base_package))
        if finetune_mode:
            net_g, spk_map = HSVCGenerator.from_package(base_package, device=device)
            if sample_rate and int(sample_rate) != net_g.config.sample_rate:
                net_g.adapt_sample_rate(int(sample_rate))
                print(
                    f"[HSVC] Adapted universal pretrained ({net_g.config.ref_sample_rate} Hz) -> "
                    f"{net_g.config.sample_rate} Hz (hop={net_g.config.hop_in}/{net_g.config.hop_dec}, fft={net_g.config.n_fft_dec})",
                    flush=True,
                )
            mcfg = net_g.config
            net_g.train()
            n_trained = max(1, len(spk_map))
            if exp_dir:
                with torch.no_grad():
                    net_g.timbre_bank.global_emb.weight[0] = net_g.timbre_bank.global_emb.weight[:n_trained].mean(dim=0)
                    net_g.timbre_bank.proto_values.weight[0] = net_g.timbre_bank.proto_values.weight[:n_trained].mean(dim=0)
            # Preserve GRL-purified speaker-invariant content bottleneck!
            for mod in (net_g.content_norm, net_g.content_down):
                for p in mod.parameters():
                    p.requires_grad_(False)
            print(f"[HSVC] Fine-tuning from base package: {base_package}", flush=True)
        else:
            if sample_rate and int(sample_rate) != mcfg.sample_rate:
                for k, v in HSVCConfig.stft_params_for_sr(int(sample_rate), ref_sr=mcfg.ref_sample_rate).items():
                    setattr(mcfg, k, v)
            net_g = HSVCGenerator(mcfg).to(device)
            print(f"[HSVC] Training from scratch ({config_path}, sr={mcfg.sample_rate} Hz)", flush=True)

        ema_g = copy.deepcopy(net_g).eval()
        for p in ema_g.parameters():
            p.requires_grad_(False)

        net_d = DualDomainDiscriminator().to(device)
        if base_discriminator and os.path.isfile(base_discriminator):
            d_ck = torch.load(base_discriminator, map_location="cpu", weights_only=False)
            net_d.load_state_dict(d_ck.get("model", d_ck.get("net_d", d_ck)))
            print(f"[HSVC] Loaded pretrained DualDomainDiscriminator: {base_discriminator}", flush=True)

        criterion = HSVCLoss(sr=mcfg.sample_rate, hop_in=mcfg.hop_in).to(device)

        speaker_map = {model_name: 0}
        if exp_dir:
            ds = ApplioExpHSVCDataset(
                exp_dir,
                sr=mcfg.sample_rate,
                hop_in=mcfg.hop_in,
                crop_frames=tcfg["crop_frames"],
                silence_aug_prob=tcfg["silence_aug_prob"],
                sid=0,
                cache_in_ram=bool(cache_data_in_gpu),
            )
            num_workers = 0 if cache_data_in_gpu else 2
            loader = DataLoader(
                ds,
                batch_size=min(batch_size, len(ds)),
                shuffle=True,
                num_workers=num_workers,
                pin_memory=device.startswith("cuda"),
                drop_last=False,
            )
            use_swap = False
            if epochs > 0:
                steps = max(1, epochs * max(1, len(loader)))
        else:
            with open(manifest_path) as f:
                manifest = json.load(f)
            speaker_map = manifest.get("speaker_to_id", {model_name: 0})
            ds = CachedHSVCDataset(manifest["train"], feat_dir, hop_in=mcfg.hop_in, crop_frames=tcfg["crop_frames"], silence_aug_prob=tcfg["silence_aug_prob"])
            weights = [2.5 if it.get("domain") == "singing" else 1.0 for it, _ in ds.items]
            sampler = WeightedRandomSampler(weights, num_samples=max(len(ds) * 4, batch_size * 4), replacement=True)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=device.startswith("cuda"), drop_last=True)
            use_swap = (not finetune_mode) and (len({it["sid"] for it, _ in ds.items}) > 1)

        if use_swap:
            spk_adv = UtteranceSpeakerAdversary(mcfg.bottleneck_dim, mcfg.num_speakers).to(device)
            spk_critic = HardenedSpectralSpeakerCritic(mcfg.num_speakers, sr=mcfg.sample_rate).to(device)
            optim_sc = torch.optim.AdamW(spk_critic.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4)
            spk_f0_lists = {}
            for it, fp in ds.items:
                f0_v = np.load(fp)["f0_100hz"]
                v = f0_v[f0_v > 50.0]
                if len(v) > 5:
                    spk_f0_lists.setdefault(int(it["sid"]), []).append(float(np.median(v)))
            spk_med_f0 = torch.full((mcfg.num_speakers,), 180.0, device=device)
            for k, vals in spk_f0_lists.items():
                spk_med_f0[k] = float(np.median(vals))
            valid_sids_t = torch.tensor(sorted(spk_f0_lists.keys()), dtype=torch.long, device=device)
            optim_g = torch.optim.AdamW(list(net_g.parameters()) + list(spk_adv.parameters()), lr=tcfg["learning_rate"], betas=tuple(tcfg["betas"]), weight_decay=tcfg["weight_decay"])
        else:
            optim_g = torch.optim.AdamW(
                [
                    {"params": net_g.timbre_bank.parameters(), "lr": tcfg["finetune_timbre_lr"]},
                    {"params": [p for n, p in net_g.named_parameters() if p.requires_grad and not n.startswith("timbre_bank.")], "lr": tcfg["finetune_backbone_lr"]},
                ],
                betas=tuple(tcfg["betas"]),
                weight_decay=tcfg["weight_decay"],
            )

        optim_d = torch.optim.AdamW(net_d.parameters(), lr=1.2e-4 if finetune_mode else tcfg["disc_learning_rate"], betas=tuple(tcfg["betas"]), weight_decay=tcfg["weight_decay"])
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=max(steps, 1), eta_min=tcfg["finetune_backbone_lr"] * 0.1)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=max(steps, 1), eta_min=1e-5)

        step = 0
        epoch_idx = 0
        resume_ckpt = os.path.join(output_dir, "G_hsvc_latest.pt")
        if (not cleanup) and os.path.isfile(resume_ckpt):
            try:
                ck = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
                net_g.load_state_dict(ck["net_g"])
                ema_g.load_state_dict(ck["ema_g"])
                net_d.load_state_dict(ck["net_d"])
                optim_g.load_state_dict(ck["optim_g"])
                optim_d.load_state_dict(ck["optim_d"])
                step = int(ck.get("step", 0))
                epoch_idx = int(ck.get("epoch", 0))
                print(f"[HSVC] Resumed training from epoch {epoch_idx}, step {step}.", flush=True)
            except Exception as e:
                print(f"[HSVC] Could not resume from {resume_ckpt} ({e}); starting fresh.", flush=True)

        disc_samples = tcfg["disc_frames"] * mcfg.hop_in
        adv_warmup = 0 if (finetune_mode and base_discriminator and os.path.isfile(base_discriminator)) else (50 if finetune_mode else tcfg["adv_warmup_steps"])
        dataset_length_str = _format_duration(getattr(ds, "total_duration_s", 0.0))
        creation_date = datetime.datetime.now().isoformat()
        last_epoch_export = None

        def _make_meta(ep: int, st: int) -> dict:
            return {
                "model_name": model_name,
                "epoch": ep,
                "step": st,
                "creation_date": creation_date,
                "dataset_length": dataset_length_str,
                "author": "Applio HSVC",
            }

        def _save_resume(ep: int, st: int):
            torch.save(
                {
                    "epoch": ep,
                    "step": st,
                    "net_g": net_g.state_dict(),
                    "ema_g": ema_g.state_dict(),
                    "net_d": net_d.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                },
                resume_ckpt,
            )

        t0 = time.time()
        print(
            f"[HSVC] Training '{model_name}' on {device} for {steps} steps "
            f"({len(ds)} clips, {dataset_length_str}, batch_size={batch_size})...",
            flush=True,
        )

        use_amp = device.startswith("cuda")
        while step < steps:
            epoch_idx += 1
            for batch in loader:
                if step >= steps:
                    break
                content, f0, vuv_conf, energy_db, hf_energy_db, wave, sid = [b.to(device, non_blocking=True) for b in batch]
                active_mask = (energy_db > -55.0).float()

                if use_swap:
                    spk_critic.requires_grad_(True); spk_critic.train()
                    sc_logits_real, _ = spk_critic(wave, active_mask, apply_specaug=True)
                    loss_sc = F.cross_entropy(sc_logits_real, sid, label_smoothing=0.1)
                    optim_sc.zero_grad(set_to_none=True); loss_sc.backward(); optim_sc.step()
                    with torch.no_grad():
                        spk_critic.eval()
                        _, clean_emb = spk_critic(wave, active_mask, apply_specaug=False)
                        spk_critic.update_centroids(clean_emb, sid)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    y_hat, z_content, _ = net_g(content, f0, vuv_conf, energy_db, hf_energy_db, sid, return_bottleneck=True)

                total_s = wave.shape[-1]
                d_start = int(np.random.randint(0, total_s - disc_samples + 1)) if total_s > disc_samples else 0
                wave_d, yhat_d = wave[..., d_start : d_start + disc_samples], y_hat[..., d_start : d_start + disc_samples]

                adv_weight = 0.7 if finetune_mode else min(1.0, max(0.0, (step - adv_warmup) / 400.0))
                loss_d_val = 0.0
                if step >= adv_warmup:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        y_d_r, y_d_g, _, _ = net_d(wave_d, yhat_d.detach())
                        loss_d, _, _ = discriminator_loss(y_d_r, y_d_g)
                    optim_d.zero_grad(set_to_none=True); loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(net_d.parameters(), 10.0); optim_d.step()
                    loss_d_val = float(loss_d.item())

                losses = criterion(wave, y_hat, energy_db=energy_db)
                loss_g1 = (
                    28.0 * losses["mel"] + 8.0 * losses["stft_mag"] + 4.0 * losses["stft_sc"]
                    + 6.0 * losses["delta"] + 3.0 * losses["rms"] + 35.0 * losses["silence"]
                )
                if use_swap:
                    grl_ramp = min(1.0, step / 1500.0)
                    u_logits = spk_adv(z_content.float(), active_mask, lambd=tcfg["grl_weight"] * grl_ramp)
                    spk_critic.requires_grad_(False); spk_critic.eval()
                    sc_logits_r, sc_emb_r = spk_critic(y_hat, active_mask, apply_specaug=False)
                    loss_spk_r = (
                        0.25 * F.cross_entropy(sc_logits_r, sid, label_smoothing=0.1)
                        + (1.0 - F.cosine_similarity(sc_emb_r, spk_critic.centroids[sid], dim=-1)).mean()
                    )
                    loss_g1 = loss_g1 + 0.5 * F.cross_entropy(u_logits, sid) + 0.35 * loss_spk_r

                if step >= adv_warmup and adv_weight > 0.0:
                    net_d.requires_grad_(False)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        _, y_d_g, fmap_r, fmap_g = net_d(wave_d, yhat_d)
                    loss_g1 = loss_g1 + adv_weight * (generator_loss(y_d_g)[0] + 1.5 * feature_loss(fmap_r, fmap_g))
                    net_d.requires_grad_(True)

                optim_g.zero_grad(set_to_none=True); loss_g1.backward()

                if use_swap and step >= tcfg["swap_start_steps"] and (step % tcfg.get("swap_every", 3) == 0):
                    rand_idx = torch.randint(0, len(valid_sids_t), (sid.shape[0],), device=device)
                    tgt_sid = valid_sids_t[rand_idx]
                    tgt_sid = torch.where(tgt_sid == sid, valid_sids_t[(rand_idx + 1) % len(valid_sids_t)], tgt_sid)
                    ratio = (spk_med_f0[tgt_sid] / spk_med_f0[sid].clamp_min(60.0)).clamp(0.65, 1.55)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        y_swap = net_g(content, f0 * ratio.unsqueeze(1), vuv_conf, energy_db, hf_energy_db, tgt_sid)
                    sc_logits_s, sc_emb_s = spk_critic(y_swap, active_mask, apply_specaug=True)
                    loss_swap_ce = F.cross_entropy(sc_logits_s, tgt_sid, label_smoothing=0.1)
                    loss_swap_cos = (1.0 - F.cosine_similarity(sc_emb_s, spk_critic.centroids[tgt_sid], dim=-1)).mean()
                    loss_swap_rms = F.l1_loss(compute_10ms_log_rms(y_swap, mcfg.hop_in), compute_10ms_log_rms(wave, mcfg.hop_in))
                    net_d.requires_grad_(False)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        _, y_d_s, _, _ = net_d(wave_d, y_swap[..., d_start : d_start + disc_samples])
                    loss_swap_adv = generator_loss(y_d_s)[0]
                    net_d.requires_grad_(True)
                    swap_ramp = min(1.0, (step - tcfg["swap_start_steps"]) / 1000.0)
                    (swap_ramp * (0.30 * loss_swap_ce + 1.5 * loss_swap_cos + 2.5 * loss_swap_rms + 0.4 * loss_swap_adv)).backward()

                torch.nn.utils.clip_grad_norm_(net_g.parameters(), 10.0)
                optim_g.step(); sched_g.step()
                if step >= adv_warmup:
                    sched_d.step()
                update_ema(ema_g, net_g, decay=min(0.995 if finetune_mode else tcfg["ema_decay"], (1.0 + step) / (10.0 + step)))
                step += 1

                if writer is not None and (step % 25 == 0 or step == steps):
                    writer.add_scalar("loss/mel", float(losses["mel"].item()), step)
                    writer.add_scalar("loss/stft_mag", float(losses["stft_mag"].item()), step)
                    writer.add_scalar("loss/silence", float(losses["silence"].item()), step)
                    writer.add_scalar("loss/disc", loss_d_val, step)

                if step % 100 == 0 or step == steps:
                    print(
                        f"[HSVC] step={step}/{steps} (epoch {epoch_idx}) mel={losses['mel'].item():.3f} "
                        f"stft={losses['stft_mag'].item():.3f} sil={losses['silence'].item():.5f} elapsed={time.time()-t0:.1f}s",
                        flush=True,
                    )

            if save_every_epoch > 0 and epoch_idx % save_every_epoch == 0:
                _save_resume(epoch_idx, step)
                if save_every_weights:
                    ep_path = os.path.join(output_dir, f"{model_name}_{epoch_idx}e_{step}s.pth")
                    ema_g.export_package(ep_path, speaker_map=speaker_map, fp16=fp16_export, metadata=_make_meta(epoch_idx, step))
                    if save_only_latest and last_epoch_export and os.path.isfile(last_epoch_export) and last_epoch_export != ep_path:
                        try:
                            os.remove(last_epoch_export)
                        except OSError:
                            pass
                    last_epoch_export = ep_path

        _save_resume(epoch_idx, step)
        pkg_path = os.path.join(output_dir, f"{model_name}.pth")
        final_meta = _make_meta(epoch_idx, step)
        ema_g.export_package(pkg_path, speaker_map=speaker_map, fp16=fp16_export, metadata=final_meta)
        try:
            sr_label = "44.1k" if mcfg.sample_rate == 44100 else f"{mcfg.sample_rate // 1000}k"
            with open(os.path.join(output_dir, "model_info.json"), "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        **final_meta,
                        "architecture": "HSVC",
                        "vocoder": "HSVC",
                        "sr": sr_label,
                        "embedder_model": "contentvec",
                    },
                    mf,
                    indent=2,
                )
        except Exception:
            pass
        print(f"[HSVC] Exported final HSVC package -> {pkg_path} ({time.time()-t0:.1f}s)", flush=True)
        return pkg_path
    finally:
        if writer is not None:
            writer.close()
        _register_pid(exp_dir or output_dir, add=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Applio HSVC Trainer")
    p.add_argument("--exp-dir", default="", help="Applio experiment directory (e.g. logs/my-voice)")
    p.add_argument("--model-name", default="hsvc_model")
    p.add_argument("--manifest", default="")
    p.add_argument("--feat-dir", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--config", default="backend/architectures/hsvc/configs/48000_base.json")
    p.add_argument("--base-package", default="models/pretraineds/hsvc/hsvc_base_48k.pth")
    p.add_argument("--base-discriminator", default="models/pretraineds/hsvc/hsvc_base_48k_d.pth")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--save-every-epoch", type=int, default=10)
    p.add_argument("--save-only-latest", action="store_true", default=False)
    p.add_argument("--save-every-weights", action="store_true", default=False)
    p.add_argument("--cleanup", action="store_true", default=False)
    p.add_argument("--cache-data-in-gpu", action="store_true", default=False)
    p.add_argument("--fp16-export", action="store_true", default=False)
    p.add_argument("--sample-rate", type=int, default=0, help="Target sample rate (24000/32000/40000/44100/48000); 0 keeps base SR")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--gpu", default="", help="Applio GPU specifier (e.g. '0', '0-1', or '-' for CPU)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    out_dir = args.output_dir or args.exp_dir or os.path.join("logs", args.model_name)
    resolved_dev = _resolve_device(args.device, args.gpu)
    train_hsvc(
        output_dir=out_dir,
        model_name=args.model_name,
        exp_dir=args.exp_dir,
        manifest_path=args.manifest,
        feat_dir=args.feat_dir,
        config_path=args.config,
        base_package=args.base_package,
        base_discriminator=args.base_discriminator,
        epochs=args.epochs,
        steps=args.steps,
        save_every_epoch=args.save_every_epoch,
        save_only_latest=args.save_only_latest,
        save_every_weights=args.save_every_weights,
        cleanup=args.cleanup,
        cache_data_in_gpu=args.cache_data_in_gpu,
        fp16_export=args.fp16_export,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        device=resolved_dev,
    )
