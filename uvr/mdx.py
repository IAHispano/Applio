"""MDX-architecture separator (.onnx models, onnx2torch + torch/CUDA inference)."""

import os

import numpy as np
import onnx
import onnx2torch
import onnxruntime as ort
import torch
from tqdm import tqdm

from uvr import spectral as spec_utils
from uvr.base import BaseSeparator
from uvr.stft import STFT


class MdxSeparator(BaseSeparator):
    def __init__(self, common_config, arch_config: dict):
        super().__init__(config=common_config)
        self.segment_size = arch_config.get("segment_size")
        self.overlap = arch_config.get("overlap", 0.25)
        self.batch_size = arch_config.get("batch_size", 1)
        self.hop_length = arch_config.get("hop_length", 1024)
        self.enable_denoise = arch_config.get("enable_denoise", False)

        self.compensate = self.model_data["compensate"]
        self.dim_f = self.model_data["mdx_dim_f_set"]
        self.dim_t = 2 ** self.model_data["mdx_dim_t_set"]
        self.n_fft = self.model_data["mdx_n_fft_scale_set"]

        self.load_model()

        self.n_bins = 0
        self.trim = 0
        self.chunk_size = 0
        self.gen_size = 0
        self.stft = None
        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None

    def load_model(self):
        if self.segment_size == self.dim_t:
            self.uses_pytorch_inference = False
            options = ort.SessionOptions()
            options.log_severity_level = 3
            session = ort.InferenceSession(
                self.model_path,
                providers=self.onnx_execution_provider,
                sess_options=options,
            )
            providers = session.get_providers()
            requested = (
                self.onnx_execution_provider[0]
                if self.onnx_execution_provider
                else None
            )
            if requested and requested not in providers:
                self.logger.warning(
                    f"ONNX Runtime could not activate {requested}; using {providers}."
                )
            self.model_run = lambda spek: session.run(
                None, {"input": spek.cpu().numpy()}
            )[0]
        else:
            self.model_run = onnx2torch.convert(self.model_path)
            self.model_run.to(self.torch_device).eval()
            self.logger.warning(
                "Model converted from ONNX to torch (segment size != dim_t); processing may be slower."
            )

    def separate(self, audio_file_path, custom_output_names=None):
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        mix = self.prepare_mix(self.audio_file_path)
        peak = np.abs(mix).max()
        mix = spec_utils.normalize(
            wave=mix,
            max_peak=self.normalization_threshold,
            min_peak=self.amplification_threshold,
        )

        source = self.demix(mix) * peak
        if not isinstance(self.primary_source, np.ndarray):
            self.primary_source = source.T

        output_files = []
        if not isinstance(self.secondary_source, np.ndarray):
            raw_mix = self.demix(mix, is_match_mix=True)
            if getattr(self, "invert_using_spec", False):
                self.secondary_source = spec_utils.invert_stem(
                    raw_mix, self.primary_source * self.compensate
                )
            else:
                self.secondary_source = (-self.primary_source * self.compensate) + mix.T

        if (
            not self.output_single_stem
            or self.output_single_stem.lower() == self.secondary_stem_name.lower()
        ):
            self.secondary_stem_output_path = self.get_stem_output_path(
                self.secondary_stem_name, custom_output_names
            )
            self.logger.info(
                f"Saving {self.secondary_stem_name} stem to {self.secondary_stem_output_path}..."
            )
            self.final_process(
                self.secondary_stem_output_path,
                self.secondary_source,
                self.secondary_stem_name,
            )
            output_files.append(self.secondary_stem_output_path)

        if (
            not self.output_single_stem
            or self.output_single_stem.lower() == self.primary_stem_name.lower()
        ):
            self.primary_stem_output_path = self.get_stem_output_path(
                self.primary_stem_name, custom_output_names
            )
            self.logger.info(
                f"Saving {self.primary_stem_name} stem to {self.primary_stem_output_path}..."
            )
            self.final_process(
                self.primary_stem_output_path,
                self.primary_source,
                self.primary_stem_name,
            )
            output_files.append(self.primary_stem_output_path)

        return output_files

    def initialize_model_settings(self):
        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim
        self.stft = STFT(
            self.logger, self.n_fft, self.hop_length, self.dim_f, self.torch_device
        )

    def demix(self, mix, is_match_mix=False):
        self.initialize_model_settings()
        chunk_size = self.hop_length * (self.segment_size - 1)
        overlap = 0.02 if is_match_mix else self.overlap
        gen_size = chunk_size - 2 * self.trim
        pad = gen_size + self.trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate(
            (
                np.zeros((2, self.trim), dtype="float32"),
                mix,
                np.zeros((2, pad), dtype="float32"),
            ),
            1,
        )
        step = int((1 - overlap) * chunk_size)

        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        tar_waves_ = []
        total_chunks = (mixture.shape[-1] + step - 1) // step

        for i in tqdm(range(0, mixture.shape[-1], step), desc="Separating"):
            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            chunk_size_actual = end - start
            window = None
            if overlap != 0:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))
            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                mix_part_ = np.concatenate(
                    (mix_part_, np.zeros((2, (i + chunk_size) - end), dtype="float32")),
                    axis=-1,
                )
            mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(
                self.torch_device
            )
            with torch.no_grad():
                for mix_wave in mix_part.split(self.batch_size):
                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)
                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1
                    result[..., start:end] += tar_waves[..., : end - start]

        tar_waves = result / divider
        tar_waves_.append(tar_waves)
        # NOTE: concatenate receives the stacked array itself (not wrapped in
        # a list): an (1,2,L) array iterates as one (2,L) frame, yielding (2,L).
        tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim : -self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, : mix.shape[-1]]
        return tar_waves[:, 0:None]

    def run_model(self, mix, is_match_mix=False):
        spek = self.stft(mix.to(self.torch_device))
        spek[:, :, :3, :] *= 0
        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        elif self.enable_denoise:
            spec_pred = (self.model_run(-spek) * -0.5) + (self.model_run(spek) * 0.5)
        else:
            spec_pred = self.model_run(spek)
        return (
            self.stft.inverse(torch.tensor(spec_pred).to(self.torch_device))
            .cpu()
            .detach()
            .numpy()
        )
