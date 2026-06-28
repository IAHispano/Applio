import os
import sys
import types
import numpy as np
import torch
import torch.nn.utils.parametrize
import torch.nn.functional as F
from torch import Tensor
import torchcrepe

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.utils.torch import circular_write, AudioProcessorTorch, IndexWrapper
from rvc.configs.config import Config
from rvc.infer.pipeline import Autotune
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.predictors.f0 import FCPE, RMVPE
from rvc.lib.utils import load_embedding, HubertModelWithFinalProj


class RealtimeVoiceConverter:
    """
    A class for performing realtime voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self, weight_root):
        """
        Initializes the RealtimeVoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load configuration
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.use_f0 = None  # Whether the model uses F0
        # Change this when you need to test FP16, and it may not be faster.
        self.dtype = torch.float32  # torch.float16 if config.is_half else torch.float32
        # load weights and setup model network.
        self.load_model(weight_root)
        self.setup_network()

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.

        Args:
            weight_root (str): Path to the model weights.
        """
        self.cpt = (
            torch.load(weight_root, map_location="cpu", weights_only=True)
            if os.path.isfile(weight_root)
            else None
        )

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                vocoder=self.vocoder,
            )

            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            strip_parametrizations(self.net_g)
            self.net_g = self.net_g.to(self.config.device).to(self.dtype)
            self.net_g.eval()
            # self.net_g.remove_weight_norm()

    def inference(
        self,
        feats: Tensor,
        p_len: Tensor,
        sid: Tensor,
        pitch: Tensor,
        pitchf: Tensor,
        rate: Tensor = None,
    ):
        output = self.net_g.infer(feats, p_len, pitch, pitchf, sid, rate)[0][0, 0]

        return torch.clip(output, -1.0, 1.0, out=output)


class Realtime_Pipeline:
    def __init__(
        self,
        vc: RealtimeVoiceConverter,
        hubert_model: HubertModelWithFinalProj = None,
        index=None,
        big_tsr=None,
        f0_method: str = "rmvpe",
        sid: int = 0,
    ):
        self.vc = vc
        self.hubert_model = hubert_model
        self.index = index
        self.big_tsr = big_tsr
        self.use_f0 = vc.use_f0
        self.version = vc.version
        self.f0_method = f0_method
        self.sample_rate = 16000
        self.tgt_sr = vc.tgt_sr
        self.window = 160
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.device = vc.config.device
        self.sid = sid
        self.torch_sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
        self.autotune = Autotune()
        self.resamplers = {}
        self.f0_model = self.setup_f0(self.f0_method)
        self.dtype = vc.dtype
        # Reuse scalar tensors to avoid per-block allocations.
        self._rate_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._p_len_tensor = torch.zeros(1, device=self.device, dtype=torch.int64)
    
    def autotune_f0(self, f0, f0_autotune_strength):
        notes = torch.as_tensor(self.autotune.note_dict, dtype=f0.dtype, device=f0.device)
        nearest = notes[torch.cdist(f0[:, None], notes[:, None]).argmin(dim=1)]

        return f0 + (nearest - f0) * f0_autotune_strength

    def setup_f0(self, f0_method: str = "fcpe"):
        if f0_method == "rmvpe":
            def _infer_from_audio(self, audio, thred=0.03):
                mel = self.mel_extractor(audio.unsqueeze(0), center=True)
                hidden = self.mel2hidden(mel)
                hidden = hidden.squeeze(0)
                f0 = self.decode(hidden, thred=thred)
                return f0

            def _to_local_average_cents(self, salience, thred=0.05):
                center = torch.argmax(salience, dim=1)
                salience = torch.nn.functional.pad(salience, (4, 4))
                center += 4
                offsets = torch.arange(-4, 5, device=salience.device)
                idx = center[:, None] + offsets[None, :]
                local_salience = salience[torch.arange(salience.shape[0], device=salience.device)[:, None], idx]
                product_sum = (local_salience * self.cents_mapping[idx]).sum(dim=1)
                weight_sum = local_salience.sum(dim=1)
                devided = product_sum / weight_sum
                maxx = salience.max(dim=1).values
                devided = torch.where(maxx <= thred, torch.zeros_like(devided), devided)
                return devided

            f0_model = RMVPE(
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.window,
            )

            f0_model.model.cents_mapping = torch.from_numpy(f0_model.model.cents_mapping).to(self.device)
            f0_model.model.infer_from_audio = types.MethodType(_infer_from_audio, f0_model.model)
            f0_model.model.to_local_average_cents = types.MethodType(_to_local_average_cents, f0_model.model)
        elif f0_method == "fcpe":
            f0_model = FCPE(
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.window,
            )
        elif self.f0_method in ("crepe", "crepe-tiny"):
            f0_model = None

        return f0_model

    def get_f0(
        self,
        x: Tensor,
        pitch: Tensor = None,
        pitchf: Tensor = None,
        f0_up_key: int = 0,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using various methods.
        """

        if self.f0_method == "rmvpe":
            f0 = self.f0_model.get_f0(x, filter_radius=0.03)
        elif self.f0_method == "fcpe":
            f0 = self.f0_model.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.f0_model.sample_rate,
                decoder_mode="local_argmax",
                threshold=0.006
            ).squeeze()
        elif self.f0_method in ("crepe", "crepe-tiny"):
            f0, pd = torchcrepe.predict(
                x.float().to(self.device).unsqueeze(dim=0),
                self.sample_rate,
                self.window,
                self.f0_min,
                self.f0_max,
                model="tiny" if "-tiny" in self.f0_method else "full",
                batch_size=512,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0]

        # f0 adjustments
        if f0_autotune is True:
            f0 = self.autotune_f0(f0, f0_autotune_strength)
        elif proposed_pitch is True:
            limit = 12
            _f0 = f0.cpu().numpy().copy()
            # calculate median f0 of the audio
            valid_f0 = np.where(_f0 > 0)[0]
            if len(valid_f0) < 2:
                # no valid f0 detected
                up_key = 0
            else:
                median_f0 = float(
                    np.median(np.interp(np.arange(len(_f0)), valid_f0, _f0[valid_f0]))
                )
                if median_f0 <= 0 or np.isnan(median_f0):
                    up_key = 0
                else:
                    # calculate proposed shift
                    up_key = max(
                        -limit,
                        min(
                            limit,
                            int(
                                np.round(
                                    12 * np.log2(proposed_pitch_threshold / median_f0)
                                )
                            ),
                        ),
                    )
            print(
                "calculated pitch offset:", up_key
            )  # Might need to hide so terminal output doesn't become a mess
            f0 *= 2 ** ((f0_up_key + up_key) / 12)
        else:
            f0 *= 2 ** (f0_up_key / 12)

        # Convert to Tensor for computational use
        # f0 = torch.from_numpy(f0).to(self.device).float()

        # quantizing f0 to 255 buckets to make coarse f0
        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel = torch.clip(
            (f0_mel - self.f0_min) * 254 / (self.f0_max - self.f0_min) + 1,
            1,
            255,
            out=f0_mel,
        )
        f0_coarse = torch.round(f0_mel, out=f0_mel).long()

        if pitch is not None and pitchf is not None:
            # Trim unreliable boundary frames before writing to cache.
            f0_interior = f0_coarse[3:-1] if f0_coarse.shape[0] > 4 else f0_coarse
            f0f_interior = f0[3:-1] if f0.shape[0] > 4 else f0
            circular_write(f0_interior, pitch)
            circular_write(f0f_interior, pitchf)
        else:
            pitch = f0_coarse
            pitchf = f0

        return pitch.unsqueeze(0), pitchf.unsqueeze(0)

    def voice_conversion(
        self,
        audio: Tensor,
        pitch: Tensor = None,
        pitchf: Tensor = None,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        p_len: int = 0,
        silence_front: int = 0,
        skip_head: int = None,
        return_length: int = None,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        reduced_noise=None,
        board=None,
        block_size_16k: int = None,
    ):
        """
        Performs realtime voice conversion on a given audio segment.
        """
        with torch.no_grad():
            assert audio.dim() == 1, audio.dim()
            feats = audio.view(1, -1).to(self.device)

            formant_length = int(np.ceil(return_length * 1.0))

            if self.use_f0:
                # Extract F0 from the most recent audio window only.
                shift = (block_size_16k or skip_head * self.window) // self.window
                f0_frame = (
                    block_size_16k + 800
                    if block_size_16k
                    else skip_head * self.window + 800
                )
                if self.f0_method == "rmvpe":
                    f0_frame = 5120 * ((f0_frame - 1) // 5120 + 1) - 160
                f0_frame = min(f0_frame, audio.shape[0])

                f0_coarse_new, f0_new = self.get_f0(
                    audio[-f0_frame:],
                    None,
                    None,
                    f0_up_key,
                    f0_autotune,
                    f0_autotune_strength,
                    proposed_pitch,
                    proposed_pitch_threshold,
                )
                # Remove batch dimension.
                f0_coarse_new = f0_coarse_new.squeeze(0)
                f0_new = f0_new.squeeze(0)

                # Shift pitch cache left by one block and append new frames (trimmed [3:-1]).
                if shift > 0:
                    pitch[:-shift] = pitch[shift:].clone()
                    pitchf[:-shift] = pitchf[shift:].clone()
                interior_coarse = (
                    f0_coarse_new[3:-1] if f0_coarse_new.shape[0] > 4 else f0_coarse_new
                )
                interior_f = f0_new[3:-1] if f0_new.shape[0] > 4 else f0_new
                pitch[-interior_coarse.shape[0] :] = interior_coarse
                pitchf[-interior_f.shape[0] :] = interior_f
            else:
                pitch, pitchf = None, None

            # extract features
            feats = self.hubert_model(feats)["last_hidden_state"]
            feats = (
                self.hubert_model.final_proj(feats[0]).unsqueeze(0)
                if self.version == "v1"
                else feats
            )

            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            # make a copy for pitch guidance and protection
            feats0 = feats.detach().clone() if self.use_f0 else None

            try:
                if (
                    self.index and index_rate > 0
                ):  # set by parent function, only true if index is available, loaded, and index rate > 0
                    feats = self._retrieve_speaker_embeddings(
                        skip_head, feats, self.index, self.big_tsr, index_rate
                    )
            except AssertionError:
                print("The index file structure is incompatible with the model.")
                self.index = self.big_tsr = None

            # feature upsampling
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )[:, :p_len, :]

            if self.use_f0:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )[:, :p_len, :]
                pitch_p = pitch[-p_len:].unsqueeze(0)
                pitchf_p = pitchf[-p_len:].unsqueeze(0) * (
                    formant_length / return_length
                )

                # Pitch protection blending
                if protect < 0.5:
                    pitchff = pitchf_p.detach().clone()
                    pitchff[pitchf_p > 0] = 1
                    pitchff[pitchf_p < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                        1 - pitchff.unsqueeze(-1)
                    )
                    feats = feats.to(feats0.dtype)
            else:
                pitch_p, pitchf_p = None, None

            pitchf_p = pitchf_p.to(self.dtype) if self.use_f0 else None
            # Trim oldest context so model output covers only the current block.
            self._rate_tensor.fill_(return_length / p_len)
            self._p_len_tensor.fill_(p_len)
            out_audio = self.vc.inference(
                feats,
                self._p_len_tensor,
                self.torch_sid,
                pitch_p,
                pitchf_p,
                self._rate_tensor,
            )
            # Match output RMS to the current block's input RMS.
            if volume_envelope < 1:
                rms_src = audio[-(return_length * self.window) :]
                out_audio = AudioProcessorTorch.change_rms(
                    rms_src,
                    self.sample_rate,
                    out_audio,
                    self.tgt_sr,
                    volume_envelope,
                    device=self.device,
                    dtype=self.dtype,
                )

            if reduced_noise is not None:
                out_audio = reduced_noise(out_audio.unsqueeze(0)).squeeze(0)
            if board is not None:
                out_audio = torch.as_tensor(
                    board(out_audio.cpu().numpy(), self.tgt_sr),
                    device=self.device,
                )

        return out_audio.float()

    def _retrieve_speaker_embeddings(
        self, skip_head, feats: torch.Tensor, index: IndexWrapper, big_tsr: torch.Tensor, index_rate: float
    ):
        # skip_offset = skip_head // 2
        # npy = feats[0][skip_offset:].cpu().numpy()
        # if self.dtype == torch.float16:
        #     npy = npy.astype(np.float32)
        # score, ix = index.search(npy, k=8)
        # weight = np.square(1 / score)
        # weight /= weight.sum(axis=1, keepdims=True)
        # npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        # if self.dtype == torch.float16:
        #     npy = npy.astype(np.float16)
        # feats[0][skip_offset:] = (
        #     torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
        #     + (1 - index_rate) * feats[0][skip_offset:]
        # )
        skip_offset = skip_head // 2
        tsr = feats[0][skip_offset:]
        score, ix = index.search(tsr, k=8)
        weight = (1 / score).square()
        weight /= weight.sum(dim=1, keepdim=True)
        query = (big_tsr[ix] * weight.unsqueeze(2)).sum(dim=1)

        feats[0][skip_offset :] = (
            query.unsqueeze(0) * index_rate
            + (1.0 - index_rate) * feats[0][skip_offset :]
        )
        return feats


def create_pipeline(
    model_path: str = None,
    index_path: str = None,
    f0_method: str = "rmvpe",
    embedder_model: str = None,
    embedder_model_custom: str = None,
    # device: str = "cuda",
    sid: int = 0,
):
    """
    Initialize real-time voice conversion pipeline.
    """

    vc = RealtimeVoiceConverter(model_path)
    # index, big_npy = load_faiss_index(
    #     index_path.strip()
    #     .strip('"')
    #     .strip("\n")
    #     .strip('"')
    #     .strip()
    #     .replace("trained", "added")
    # )

    index = IndexWrapper(
        index_path.strip()
        .strip('"')
        .strip("\n")
        .strip('"')
        .strip()
        .replace("trained", "added"),
        device=vc.config.device,
        dtype=vc.dtype
    )
    big_tsr, _ = index.read_index_tensor()

    hubert_model = load_embedding(embedder_model, embedder_model_custom)
    hubert_model = hubert_model.to(vc.config.device).to(vc.dtype)
    hubert_model.eval()

    pipeline = Realtime_Pipeline(
        vc,
        hubert_model,
        index,
        big_tsr,
        f0_method,
        sid,
    )

    return pipeline


def strip_parametrizations(module: torch.nn.Module):
    """
    Remove all parametrizations (e.g., weight norm) from a module and log each removal.
    """
    for name, submodule in module.named_modules():
        if hasattr(submodule, "parametrizations"):
            for pname, plist in list(submodule.parametrizations.items()):
                # print(f"Removing parametrizations from {name}.{pname}: {[p.__class__.__name__ for p in plist]}")
                torch.nn.utils.parametrize.remove_parametrizations(
                    submodule, pname, leave_parametrized=True
                )
