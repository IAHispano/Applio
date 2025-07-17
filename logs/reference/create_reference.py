import numpy as np
import torch
import librosa
import soundfile as sf
from rvc.lib.predictors.f0 import RMVPE
from transformers import HubertModel


def cf0(f0):
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    """Convert F0 to coarse F0."""
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel = np.clip(
        (f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1,
        1,
        f0_bin - 1,
    )
    return np.rint(f0_mel).astype(int)


ref = r"reference.wav"
audio, sr = librosa.load(ref, sr=16000)
trimmed_len = (len(audio) // 320) * 320
# to prevent feature and pitch offset mismatch
audio = audio[:trimmed_len]

print("audio", audio.shape)
rmvpe_model = RMVPE(device="cpu", sample_rate=16000, hop_size=160)
f0 = rmvpe_model.get_f0(audio, filter_radius=0.03)
print("f0", f0.shape)
f0c = cf0(f0)
print("f0c", f0c.shape)

cv_path = r"rvc\models\embedders\contentvec"
cv_model = HubertModel.from_pretrained(cv_path)

spin_path = r"rvc\models\embedders\spin"
spin_model = HubertModel.from_pretrained(spin_path)

feats = torch.from_numpy(audio).to(torch.float32).to("cpu")
feats = torch.nn.functional.pad(feats.unsqueeze(0), (40, 40), mode="reflect")
feats = feats.view(1, -1)

with torch.no_grad():
    cv_feats = cv_model(feats)["last_hidden_state"]
    cv_feats = cv_feats.squeeze(0).float().cpu().numpy()
    print("cv", cv_feats.shape)

    spin_feats = spin_model(feats)["last_hidden_state"]
    spin_feats = spin_feats.squeeze(0).float().cpu().numpy()
    print("spin", spin_feats.shape)
np.save(r"logs\reference\contentvec\feats.npy", cv_feats)
np.save(r"logs\reference\spin\feats.npy", spin_feats)
np.save(r"logs\reference\pitch_coarse.npy", f0c)
np.save(r"logs\reference\pitch_fine.npy", f0)
