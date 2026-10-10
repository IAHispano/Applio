"""Demucs v4 separator (4-stem band split, torch/CUDA)."""

import os
from pathlib import Path

import numpy as np
import torch

# The vendored demucs stack self-aliases as ``demucs`` for checkpoint compat.
import uvr.demucs_lib  # noqa: F401
from uvr import spectral as spec_utils
from uvr.base import BaseSeparator
from uvr.demucs_lib.apply import apply_model, demucs_segments
from uvr.demucs_lib.hdemucs import HDemucs
from uvr.demucs_lib.pretrained import get_model as get_demucs_model
from uvr.device import mps_accumulation_budget_bytes, should_accumulate_on_device

DEMUCS_4_SOURCE = ["drums", "bass", "other", "vocals"]
DEMUCS_4_SOURCE_MAPPER = {
    BaseSeparator.BASS_STEM: 0,
    BaseSeparator.DRUM_STEM: 1,
    BaseSeparator.OTHER_STEM: 2,
    BaseSeparator.VOCAL_STEM: 3,
}


def _estimate_buffer_bytes(channels, samples, num_sources, shifts, num_bag_models):
    input_bytes = channels * samples * 4
    output_copies = 2 if shifts > 1 else 1
    if num_bag_models:
        output_copies = max(output_copies, 2)
        if num_bag_models > 1 and shifts > 1:
            output_copies = 3
    return 3 * input_bytes + output_copies * num_sources * input_bytes + samples * 4


class DemucsSeparator(BaseSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)
        self.segment_size = arch_config.get("segment_size", "Default")
        self.shifts = arch_config.get("shifts", 2)
        self.overlap = arch_config.get("overlap", 0.25)
        self.segments_enabled = arch_config.get("segments_enabled", True)
        self.demucs_source_map = dict(DEMUCS_4_SOURCE_MAPPER)
        self.audio_file_path = None
        self.audio_file_base = None
        self.logger.info("Demucs separator initialised")

    def separate(self, audio_file_path, custom_output_names=None):
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        mix = self.prepare_mix(self.audio_file_path)
        self.demucs_model_instance = HDemucs(sources=DEMUCS_4_SOURCE)
        self.demucs_model_instance = get_demucs_model(
            name=os.path.splitext(os.path.basename(self.model_path))[0],
            repo=Path(os.path.dirname(self.model_path)),
        )
        self.demucs_model_instance = demucs_segments(
            self.segment_size, self.demucs_model_instance
        )
        self.demucs_model_instance.to(self.torch_device)
        self.demucs_model_instance.eval()
        try:
            source = self.demix_demucs(mix)
        finally:
            del self.demucs_model_instance
            self.clear_gpu_cache()

        output_files = []
        if isinstance(source, np.ndarray) and len(source) not in (2, 4, 6):
            self.logger.warning(
                f"Unexpected Demucs source count {len(source)}; writing all stems."
            )
        for stem_name, stem_value in self.demucs_source_map.items():
            if (
                self.output_single_stem is not None
                and stem_name.lower() != self.output_single_stem.lower()
            ):
                continue
            stem_path = self.get_stem_output_path(stem_name, custom_output_names)
            self.final_process(stem_path, source[stem_value].T, stem_name)
            output_files.append(stem_path)
        return output_files

    def demix_demucs(self, mix):
        num_sources = len(self.demucs_model_instance.sources)
        estimated = _estimate_buffer_bytes(
            mix.shape[0],
            mix.shape[-1],
            num_sources,
            self.shifts,
            len(getattr(self.demucs_model_instance, "models", ())),
        )
        accumulate = should_accumulate_on_device(self.torch_device, estimated)
        mix_device = self.torch_device if accumulate else torch.device("cpu")
        mix = torch.tensor(mix, dtype=torch.float32, device=mix_device)
        ref = mix.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std()
        if not torch.isfinite(ref_std):
            ref_std = torch.zeros_like(ref_std)
        mix.sub_(ref_mean).div_(ref_std.clamp_min(torch.finfo(mix.dtype).eps))

        with torch.no_grad():
            sources = apply_model(
                model=self.demucs_model_instance,
                mix=mix[None],
                shifts=self.shifts,
                split=self.segments_enabled,
                overlap=self.overlap,
                static_shifts=1 if self.shifts == 0 else self.shifts,
                set_progress_bar=None,
                device=self.torch_device,
                progress=True,
            )[0]

        sources.mul_(ref_std).add_(ref_mean)
        sources = sources.cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]
        return np.concatenate([sources[:, :, 0:None]], axis=-1)
