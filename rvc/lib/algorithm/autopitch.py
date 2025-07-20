import os
import sys
import torch
import librosa

import numpy as np
import sklearn.metrics.pairwise as pwise

sys.path.append(os.getcwd())

from rvc.lib.utils import load_embedding


class AutoPitch:
    """
    This class is used to calculate the threshold required to change the pitch automatically.

    Args:
        vc (pipeline): Voice conversion pipeline class.
        rvc_npz_path (str): RVC npz data file path.
        emb_npz_path (str): The path to the npz file containing reference features used for similarity comparison.
        pitch_guidance (bool): Whether to use pitch guidance during voice conversion.
        version (str): RVC model version. Default v1.
        device (str, torch.device): Use if GPU device is present. Default CPU.
    """

    def __init__(
        self, vc, rvc_npz_path, emb_npz_path, pitch_guidance=True, version="v1"
    ):
        self.vc = vc
        self.pitch_guidance = pitch_guidance
        self.version = version
        self.device = vc.device
        self.rvc_loaded = np.load(rvc_npz_path)
        self.emb_loaded = np.load(emb_npz_path)
        self.embedders = (
            load_embedding("contentvec", None).to(self.device).float().eval()
        )
        self.male_threshold_adjustment = 0.7
        self.female_threshold_adjustment = 1
        self.male_euclidean_offset = 1.015
        self.female_euclidean_offset = 1.715

    def conversion(self, model, net_g, sid):
        # Index is disabled because it causes threshold deviation (Unknown reason)
        audio_opt = self.vc.voice_conversion(
            model,
            net_g,
            sid,
            self.rvc_loaded["audio"],
            (
                torch.tensor(self.rvc_loaded["pitch"]).to(self.device)
                if self.pitch_guidance
                else None
            ),
            (
                torch.tensor(self.rvc_loaded["pitchf"]).to(self.device)
                if self.pitch_guidance
                else None
            ),
            None,
            None,
            0.5,
            self.version,
            0.5,
        )[self.vc.t_pad_tgt : -self.vc.t_pad_tgt]

        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max

        return audio_opt

    def get_hubert_feature(self, model, feats):
        with torch.no_grad():
            # take the 9th layer because it gives better value

            feats = model(
                feats.to(self.device), output_hidden_states=True
            )  # ["last_hidden_state"]
            feats = feats.hidden_states[9]

            if feats.dim() == 3:
                feats = feats.squeeze(0)

        return feats[::2].mean(dim=0).cpu().numpy()

    def autopitch(self, model, net_g, sid):
        audio_opt = self.conversion(model, net_g, sid)
        tgt_sr = self.vc.tgt_sr

        if tgt_sr != 16000:
            if audio_opt.ndim > 1:
                audio_opt = audio_opt[0]

            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=16000)
            audio_opt = torch.from_numpy(audio_opt).to(self.device)

        if audio_opt.ndim > 1 and audio_opt.shape[0] > 1:
            audio_opt = audio_opt[:1]

        feats = audio_opt.squeeze(0).view(1, -1)
        emb_rvc = self.get_hubert_feature(self.embedders, feats).reshape(1, -1)

        # Using euclidean_distances instead of cosine_similarity is not as good.

        euclidean_male = (
            pwise.euclidean_distances(emb_rvc, self.emb_loaded["male"]).mean()
            - self.male_threshold_adjustment
        )
        euclidean_female = (
            pwise.euclidean_distances(emb_rvc, self.emb_loaded["female"]).mean()
            - self.female_threshold_adjustment
        )

        # can return 155 or 255 depending on the voice model instead of calculating

        euclidean = (
            (euclidean_male + self.male_euclidean_offset)
            if euclidean_male - euclidean_female > 0.1
            else (euclidean_female + self.female_euclidean_offset)
        )
        freq = round(1127 * np.log(1 + euclidean * 100 / 700), 1)

        return -freq if freq < 0 else freq
