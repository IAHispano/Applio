"""MDXC-architecture separator (TFC-TDF .ckpt + yaml configs)."""

import os

import numpy as np
import torch
from ml_collections import ConfigDict
from tqdm import tqdm

from uvr import spectral as spec_utils
from uvr.base import BaseSeparator
from uvr.tfc_tdf_v3 import TFC_TDF_net


class MdxcSeparator(BaseSeparator):
    def __init__(self, common_config, arch_config: dict):
        super().__init__(config=common_config)
        self.logger.debug(
            f"Model data keys: {list(self.model_data.keys()) if isinstance(self.model_data, dict) else None}"
        )

        self.segment_size = arch_config.get("segment_size", 256)
        inference_config = (
            self.model_data.get("inference", {})
            if isinstance(self.model_data, dict)
            else {}
        )

        overlap = arch_config.get("overlap")
        if overlap is None:
            overlap = inference_config.get("num_overlap")
        if overlap is None:
            overlap = 8
        batch_size = arch_config.get("batch_size")
        if batch_size is None:
            batch_size = inference_config.get("batch_size")
        if batch_size is None:
            batch_size = 1
        if overlap <= 0:
            raise ValueError("MDXC overlap must be greater than zero")
        if batch_size <= 0:
            raise ValueError("MDXC batch size must be greater than zero")
        self.overlap = overlap
        self.batch_size = batch_size
        self.load_model()

        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None
        self.logger.info("MDXC separator initialised")

    def load_model(self):
        self.model_data_cfg = ConfigDict(self.model_data)
        self.model_run = TFC_TDF_net(self.model_data_cfg, device=self.torch_device)
        # Load on CPU first (hardware-accelerated loads are unreliable),
        # then move to the target device.
        self.model_run.load_state_dict(
            torch.load(self.model_path, map_location="cpu", weights_only=True)
        )
        self.model_run.to(self.torch_device).eval()

    def separate(self, audio_file_path, custom_output_names=None):
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        mix = self.prepare_mix(self.audio_file_path)
        mix = spec_utils.normalize(
            wave=mix,
            max_peak=self.normalization_threshold,
            min_peak=self.amplification_threshold,
        )
        source = self.demix(mix)

        output_files = []
        instruments = list(self.model_data_cfg.training.instruments)
        # Single-target checkpoints return one stereo array (not a per-stem
        # stack): treat it as a single pair instead of iterating channels.
        if (
            len(instruments) == 1
            and isinstance(source, np.ndarray)
            and source.ndim == 2
        ):
            pairs = [(instruments[0], source)]
        else:
            pairs = list(zip(instruments, source))
        for key, value in pairs:
            if (
                self.output_single_stem
                and self.output_single_stem.lower() != key.lower()
            ):
                continue
            stem_path = self.get_stem_output_path(key, custom_output_names)
            stem_source = spec_utils.normalize(
                wave=value,
                max_peak=self.normalization_threshold,
                min_peak=self.amplification_threshold,
            ).T
            self.logger.info(f"Saving {key} stem to {stem_path}...")
            self.final_process(stem_path, stem_source, key)
            output_files.append(stem_path)
        return output_files

    def demix(self, mix):
        try:
            num_stems = self.model_run.num_target_instruments
        except AttributeError:
            num_stems = self.model_run.module.num_target_instruments

        hop_length = self.model_data_cfg.audio.hop_length
        chunk_size = hop_length * (self.segment_size - 1)
        hop_size = chunk_size // self.overlap
        mix_len = mix.shape[1]
        pad_size = hop_size - (mix_len - chunk_size) % hop_size
        padded_length = mix_len + pad_size + 2 * (chunk_size - hop_size)

        mix_t = torch.tensor(mix, dtype=torch.float32, device=self.torch_device)
        mix_t = torch.cat(
            [
                torch.zeros(2, chunk_size - hop_size, device=self.torch_device),
                mix_t,
                torch.zeros(
                    2, pad_size + chunk_size - hop_size, device=self.torch_device
                ),
            ],
            1,
        )
        chunks = mix_t.unfold(1, chunk_size, hop_size).transpose(0, 1)
        batches = [
            chunks[i : i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]

        accumulated = (
            torch.zeros(num_stems, *mix_t.shape, device=self.torch_device)
            if num_stems > 1
            else torch.zeros_like(mix_t)
        )
        with torch.no_grad():
            count = 0
            for batch in tqdm(batches, desc="Separating"):
                batch_result = self.model_run(batch.to(self.torch_device))
                for individual in batch_result:
                    individual = (
                        individual.to(self.torch_device)
                        if individual.device != self.torch_device
                        else individual
                    )
                    accumulated[
                        ..., count * hop_size : count * hop_size + chunk_size
                    ] += individual
                    count += 1

        accumulated.div_(self.overlap)
        out = accumulated[
            ..., chunk_size - hop_size : -(pad_size + chunk_size - hop_size)
        ]
        return out.cpu().detach().numpy()
