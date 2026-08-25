import os
import json
import torch

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from torchfcpe import spawn_infer_model_from_pt
import torchcrepe
import numpy as np
import librosa

# Defaults for the "rmvpe_high_register" section of assets/config.json
# (edited from the Settings tab; see assets/config_template.json).
HIGH_REGISTER_DEFAULTS = {"enabled": False, "mode": "true_pitch", "f0_ceil": 1250.0}


def load_high_register_settings():
    # Missing file, malformed JSON or missing keys fall back to the defaults
    # (corrector disabled), so RMVPE keeps stock behaviour outside the UI.
    settings = dict(HIGH_REGISTER_DEFAULTS)
    try:
        with open(os.path.join("assets", "config.json"), "r", encoding="utf-8") as f:
            section = json.load(f).get("rmvpe_high_register")
        if isinstance(section, dict):
            settings.update(section)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return settings


class RMVPE:
    def __init__(
        self,
        device,
        model_name="rmvpe.pt",
        sample_rate=16000,
        hop_size=160,
        high_register=None,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        # high_register: settings for the high-register corrector (keys:
        # enabled, mode, f0_ceil). None reads them from assets/config.json;
        # callers pass a dict to override (extraction pins the mode, realtime
        # disables it).
        if high_register is None:
            high_register = load_high_register_settings()
        self.high_register = {**HIGH_REGISTER_DEFAULTS, **high_register}
        self.model = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03):
        f0 = self.model.infer_from_audio(x, thred=filter_radius)
        if self.high_register["enabled"]:
            f0 = self._fix_high_register(x, f0, filter_radius)
        return f0

    def _fix_high_register(self, x, normal, thred, gate=150.0):
        # rmvpe.pt cannot track fundamentals above ~1040 Hz: its training data
        # (vocal pitch corpora) tops out near C6, so for higher notes the
        # salience net confidently reports f/2 or f/3 instead. A second rmvpe
        # pass on octave-down-resampled audio reads those registers correctly;
        # it is used only to fix the octave class of the normal pass, keeping
        # the normal 10 ms contour wherever it already agrees.
        # Mode "true_pitch" (default) writes the true pitch, capped at f0_ceil
        # (default 1250 Hz) because RVC models trained on stock rmvpe labels
        # cannot synthesize above that and collapse; above the ceiling stock's
        # octave-down drive is kept (the vocoder folds its harmonics back up).
        # Mode "fold" is a compatibility mode for such models: it fixes only
        # wrong-pitch-class frames, using half the true pitch so every drive
        # stays in the model's trained register.
        ceil = float(self.high_register["f0_ceil"])
        guide = self._half_speed_guide(x, len(normal), thred)
        out = normal.copy()
        nv, gv = normal > 0, guide > 0
        both = nv & gv
        c_keep = np.full(len(normal), 1e9)
        c_double = np.full(len(normal), 1e9)
        c_keep[both] = np.abs(1200.0 * np.log2(normal[both] / guide[both]))
        c_double[both] = np.abs(1200.0 * np.log2(2.0 * normal[both] / guide[both]))
        agree = both & (c_keep < gate)
        cand_dbl = both & ~agree & (c_double < gate)
        if self.high_register["mode"] == "fold":
            fold_other = (both & ~agree & ~cand_dbl) & (guide >= 990.0)
            fold_fill = (~nv) & gv & (guide >= 900.0)
            out[fold_other] = guide[fold_other] / 2.0
            out[fold_fill] = guide[fold_fill] / 2.0
            return out
        dbl = cand_dbl & (guide >= 460.0) & (2.0 * normal <= ceil)
        other = (both & ~agree & ~cand_dbl) & (guide >= 990.0) & (guide <= ceil)
        filled = (~nv) & gv & (guide >= 460.0) & (guide <= ceil)
        out[dbl] = 2.0 * normal[dbl]
        out[other] = guide[other]
        out[filled] = guide[filled]
        return out

    def _half_speed_guide(self, x, n_target, thred):
        y = np.asarray(x, dtype=np.float32)
        y2 = librosa.resample(
            y,
            orig_sr=self.sample_rate,
            target_sr=2 * self.sample_rate,
            res_type="soxr_vhq",
        )
        f0h = self.model.infer_from_audio(y2, thred=thred)
        pos = np.arange(n_target) * 2.0
        j0 = np.minimum(np.floor(pos).astype(int), len(f0h) - 1)
        j1 = np.minimum(j0 + 1, len(f0h) - 1)
        w = pos - np.floor(pos)
        v0, v1 = f0h[j0] > 0, f0h[j1] > 0
        voiced = np.where(w == 0, v0, v0 & v1)
        lg0 = np.log2(np.where(f0h[j0] > 0, f0h[j0], 1.0))
        lg1 = np.log2(np.where(f0h[j1] > 0, f0h[j1], 1.0))
        lg = (1 - w) * lg0 + w * lg1
        return np.where(voiced, 2.0 ** (lg + 1.0), 0.0)


class CREPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, model="full"):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        batch_size = 512

        f0, pd = torchcrepe.predict(
            x.float().to(self.device).unsqueeze(dim=0),
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            model=model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()

        return f0


class FCPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = spawn_infer_model_from_pt(
            os.path.join("rvc", "models", "predictors", "fcpe.pt"),
            self.device,
            bundled_model=True,
        )

    def get_f0(self, x, p_len=None, filter_radius=0.006):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        f0 = (
            self.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return f0
