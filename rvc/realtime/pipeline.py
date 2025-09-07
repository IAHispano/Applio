import os
import sys
import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
from torch import Tensor

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.utils.torch import circular_write
from rvc.configs.config import Config
from rvc.infer.pipeline import Autotune, AudioProcessor
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.predictors.f0 import FCPE, RMVPE, SWIFT
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
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()
            self.net_g.remove_weight_norm()

    def inference(
        self,
        feats: Tensor,
        p_len: Tensor,
        sid: Tensor,
        pitch: Tensor,
        pitchf: Tensor,
    ):
        output = self.net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]

        return torch.clip(output, -1.0, 1.0, out=output)


class Realtime_Pipeline:
    def __init__(
        self,
        vc: RealtimeVoiceConverter,
        hubert_model: HubertModelWithFinalProj = None,
        index=None,
        big_npy=None,
        f0_method: str = "rmvpe",
        sid: int = 0,
    ):
        self.vc = vc
        self.hubert_model = hubert_model
        self.index = index
        self.big_npy = big_npy
        self.use_f0 = vc.use_f0
        self.version = vc.version
        self.f0_method = f0_method
        self.sample_rate = 16000
        self.tgt_sr = vc.tgt_sr
        self.window = 160
        self.model_window = self.tgt_sr // 100
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.device = vc.config.device
        self.sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
        self.autotune = Autotune()
        self.resamplers = {}
        self.f0_model = None

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

        if torch.is_tensor(x):
            # If the input is a tensor, it will need to be converted to numpy array to calculate with RMVPE and FCPE.
            x = x.cpu().numpy()

        if self.f0_method == "rmvpe":
            if self.f0_model is None:
                self.f0_model = RMVPE(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.f0_model.get_f0(x, filter_radius=0.03)
        elif self.f0_method == "fcpe":
            if self.f0_model is None:
                self.f0_model = FCPE(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.f0_model.get_f0(x, x.shape[0] // self.window, filter_radius=0.006)
        elif self.f0_method == "swift":
            if self.f0_model is None:
                self.f0_model = SWIFT(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.f0_model.get_f0(
                x,
                self.f0_min,
                self.f0_max,
                x.shape[0] // self.window,
                confidence_threshold=0.887,
            )

        # f0 adjustments
        if f0_autotune is True:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        elif proposed_pitch is True:
            limit = 12
            # calculate median f0 of the audio
            valid_f0 = np.where(f0 > 0)[0]
            if len(valid_f0) < 2:
                # no valid f0 detected
                up_key = 0
            else:
                median_f0 = float(
                    np.median(np.interp(np.arange(len(f0)), valid_f0, f0[valid_f0]))
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
            f0 *= pow(2, (f0_up_key + up_key) / 12)
        else:
            f0 *= pow(2, f0_up_key / 12)

        # Convert to Tensor for computational use
        f0 = torch.from_numpy(f0).to(self.device).float()

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
            circular_write(f0_coarse, pitch)
            circular_write(f0, pitchf)
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
    ):
        """
        Performs realtime voice conversion on a given audio segment.
        """
        assert audio.dim() == 1, audio.dim()
        feats = audio.view(1, -1).to(self.device)

        formant_length = int(np.ceil(return_length * 1.0))

        pitch, pitchf = (
            self.get_f0(
                audio[silence_front:],
                pitch,
                pitchf,
                f0_up_key,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
            )
            if self.use_f0
            else (None, None)
        )

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

        if (
            self.index
        ):  # set by parent function, only true if index is available, loaded, and index rate > 0
            feats = self._retrieve_speaker_embeddings(
                skip_head, feats, self.index, self.big_npy, index_rate
            )
        # feature upsampling
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)[
            :, :p_len, :
        ]

        if self.use_f0:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )[:, :p_len, :]
            pitch, pitchf = pitch[:, -p_len:], pitchf[:, -p_len:] * (
                formant_length / return_length
            )

            # Pitch protection blending
            if protect < 0.5:
                pitchff = pitchf.detach().clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                    1 - pitchff.unsqueeze(-1)
                )
                feats = feats.to(feats0.dtype)
        else:
            pitch, pitchf = None, None

        p_len = torch.tensor([p_len], device=self.device, dtype=torch.int64)
        out_audio = self.vc.inference(feats, p_len, self.sid, pitch, pitchf).float()
        if volume_envelope != 1:
            out_audio = AudioProcessor.change_rms(
                audio, self.sample_rate, out_audio, self.tgt_sr, volume_envelope
            )

        scaled_window = int(np.floor(1.0 * self.model_window))

        if scaled_window != self.model_window:
            if scaled_window not in self.resamplers:
                self.resamplers[scaled_window] = tat.Resample(
                    orig_freq=scaled_window,
                    new_freq=self.model_window,
                    dtype=torch.float32,
                ).to(self.device)
            out_audio = self.resamplers[scaled_window](
                out_audio[: return_length * scaled_window]
            )

        return out_audio

    def _retrieve_speaker_embeddings(
        self, skip_head, feats, index, big_npy, index_rate
    ):
        skip_offset = skip_head // 2
        npy = feats[0][skip_offset:].cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats[0][skip_offset:] = (
            torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
            + (1 - index_rate) * feats[0][skip_offset:]
        )
        return feats


def load_faiss_index(file_index):
    if file_index != "" and os.path.exists(file_index):
        try:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as error:
            print(f"An error occurred reading the FAISS index: {error}")
            index = big_npy = None
    else:
        index = big_npy = None

    return index, big_npy


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
    index, big_npy = load_faiss_index(
        index_path.strip()
        .strip('"')
        .strip("\n")
        .strip('"')
        .strip()
        .replace("trained", "added")
    )

    hubert_model = load_embedding(embedder_model, embedder_model_custom)
    hubert_model = hubert_model.to(vc.config.device).float()
    hubert_model.eval()

    pipeline = Realtime_Pipeline(
        vc,
        hubert_model,
        index,
        big_npy,
        f0_method,
        sid,
    )

    return pipeline
