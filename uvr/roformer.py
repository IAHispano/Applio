"""Roformer-family separator (Mel-Band / BS-Roformer .ckpt + yaml configs)."""

import os

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from uvr.base import BaseSeparator
from uvr.roformer_lib.registry import resolve as resolve_variant_class


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_weights(model, ckpt_path, logger=None):
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for wrapper in ("state", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and wrapper in state:
            state = state[wrapper]
    if isinstance(state, dict):
        state = {
            k.replace("module.", "") if k.startswith("module.") else k: v
            for k, v in state.items()
        }
    model.load_state_dict(state, strict=True)
    return model


def _fade_window(size, fade):
    window = torch.ones(size, dtype=torch.float32)
    if fade > 0 and fade * 2 < size:
        fade_in = torch.linspace(0.0, 1.0, fade)
        fade_out = torch.linspace(1.0, 0.0, fade)
        window[:fade] = fade_in
        window[-fade:] = fade_out
    return window


class RoformerSeparator(BaseSeparator):
    def __init__(self, common_config, arch_config: dict):
        super().__init__(config=common_config)
        self.roformer_class = common_config.get("roformer_class", "mel_band")
        model_cls = resolve_variant_class(self.roformer_class)
        self.config_path = common_config.get("roformer_config")
        if not self.config_path or not os.path.isfile(self.config_path):
            raise ValueError(f"Roformer config not found: {self.config_path}")

        cfg = load_config(self.config_path)
        self.cfg_audio = cfg.get("audio", {})
        self.cfg_infer = cfg.get("inference", {})
        training = cfg.get("training", {}) or {}
        self.instruments = [str(s) for s in training.get("instruments", [])]
        self.target = str(
            training.get("target_instrument")
            or (self.instruments[0] if self.instruments else "vocals")
        )
        others = [s for s in self.instruments if s.lower() != self.target.lower()]
        self.counterpart = others[0] if others else self.secondary_stem(self.target)
        self.primary_stem_name = self.target
        self.secondary_stem_name = self.counterpart

        self.sample_rate = int(self.cfg_audio.get("sample_rate", 44100))
        # Runtime overrides (Roformer settings): chunk length in seconds,
        # overlapping prediction windows, and batched forward passes.
        # Unset values fall back to the model's yaml config.
        chunk_seconds = arch_config.get("chunk_seconds")
        if chunk_seconds:
            self.chunk_size = max(1, int(float(chunk_seconds) * self.sample_rate))
        else:
            self.chunk_size = int(self.cfg_audio.get("chunk_size", 352800))
        overlap_override = arch_config.get("overlap")
        if overlap_override:
            self.overlap = max(1, int(overlap_override))
        else:
            self.overlap = max(1, int(self.cfg_infer.get("num_overlap", 2)))
        self.batch_size = max(1, int(arch_config.get("batch_size", 1) or 1))

        model_kwargs = dict(cfg.get("model", {}) or {})
        self.logger.debug(
            f"Roformer {self.roformer_class} kwargs: {sorted(model_kwargs)}"
        )
        model = model_cls(**model_kwargs)
        load_weights(model, self.model_path, self.logger)
        model.to(self.torch_device)
        model.eval()
        self.model_run = model
        self.logger.info(
            f"Roformer separator initialised ({self.roformer_class}, target: {self.target})"
        )

    def separate(self, audio_file_path, custom_output_names=None):
        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        try:
            self.input_bit_depth = (
                24 if "24" in sf.info(audio_file_path).subtype else 16
            )
        except Exception:
            self.input_bit_depth = 16

        mix = self.prepare_mix(self.audio_file_path)
        peak = float(np.abs(mix).max()) or 1.0
        mix_tensor = torch.tensor(mix / peak, dtype=torch.float32)

        chunk_size = self.chunk_size
        step = max(1, chunk_size // self.overlap)
        fade = chunk_size // 10
        length = mix_tensor.shape[-1]
        window = _fade_window(chunk_size, fade)

        # Per-stem accumulators; stem count is known after the first chunk.
        results = {}
        counters = {}
        order = []

        def _forward(chunk_2ch):
            out = self.model_run(chunk_2ch[None].to(self.torch_device))
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.detach().cpu()
            if out.ndim == 4:
                out = out[0]
            if out.ndim == 2:
                out = out[None]
            return out

        # Collect chunk jobs first so the model can run batched forward
        # passes (batch_size=1 keeps the original one-by-one behavior).
        jobs = []
        for i in range(0, length, step):
            end = min(i + chunk_size, length)
            chunk = mix_tensor[:, i:end]
            cur_len = chunk.shape[-1]
            if cur_len < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - cur_len))
            jobs.append((i, end, cur_len, chunk))

        n_batches = max(1, (len(jobs) + self.batch_size - 1) // self.batch_size)
        with torch.no_grad():
            for b in tqdm(
                range(0, len(jobs), self.batch_size),
                desc="Separating",
                total=n_batches,
            ):
                batch_jobs = jobs[b : b + self.batch_size]
                if len(batch_jobs) == 1:
                    outs = [_forward(batch_jobs[0][3])]
                else:
                    stacked = torch.stack([j[3] for j in batch_jobs], dim=0).to(
                        self.torch_device
                    )
                    out = self.model_run(stacked)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    out = out.detach().cpu()
                    outs = [
                        o[0] if o.ndim == 4 else (o[None] if o.ndim == 2 else o)
                        for o in out
                    ]
                for (i, end, cur_len, _), out in zip(batch_jobs, outs):
                    n_stems = out.shape[0]
                    if not order:
                        order = self._stem_names(n_stems)
                        for name in order:
                            results[name] = torch.zeros_like(mix_tensor)
                            counters[name] = torch.zeros(
                                mix_tensor.shape[-1], dtype=torch.float32
                            )
                    w = window[:cur_len].clone()
                    if i == 0:
                        w[:fade] = 1.0
                    if end >= length:
                        w[-fade:] = 1.0
                    for idx, name in enumerate(order):
                        stem = out[idx] if idx < n_stems else out[0]
                        if stem.shape[0] == 1:
                            stem = stem.repeat(2, 1)
                        stem = stem[:, :cur_len]
                        results[name][:, i:end] += stem * w
                        counters[name][i:end] += w

        stems = {}
        for name in order:
            wav = (results[name] / torch.clamp(counters[name], min=1e-8)).numpy() * peak
            stems[name] = wav

        # Single-mask models predict only the target; derive the counterpart.
        if len(order) == 1:
            only = order[0]
            stems[self.counterpart] = mix - stems[only]

        output_files = []
        for stem_name, source in stems.items():
            if (
                self.output_single_stem
                and self.output_single_stem.lower() != stem_name.lower()
            ):
                continue
            stem_path = self.get_stem_output_path(stem_name, custom_output_names)
            self.logger.info(f"Saving {stem_name} stem to {stem_path}...")
            self.final_process(stem_path, source.T, stem_name)
            output_files.append(stem_path)

        return output_files

    def _stem_names(self, n_stems):
        if self.instruments and len(self.instruments) == n_stems:
            return list(self.instruments)
        if n_stems == 1:
            return [self.target]
        if n_stems == 2:
            return [self.target, self.counterpart]
        raise ValueError(
            f"Roformer model produced {n_stems} stems but the config lists "
            f"{self.instruments}; cannot name the outputs."
        )
