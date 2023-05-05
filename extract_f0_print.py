import os, traceback, sys, parselmouth
import librosa
import pyworld
from scipy.io import wavfile
import numpy as np, logging
import torchcrepe # Fork Feature. Crepe algo for training and preprocess
import torch
from torch import Tensor # Fork Feature. Used for pitch prediction for torch crepe.

logging.getLogger("numba").setLevel(logging.WARNING)
from multiprocessing import Process

exp_dir = sys.argv[1]
f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


n_p = int(sys.argv[2])
f0method = sys.argv[3]
extraction_crepe_hop_length = int(sys.argv[4])
print("EXTRACTION CREPE HOP LENGTH: " + extraction_crepe_hop_length)
print("EXTRACTION CREPE HOP LENGTH TYPE: " + type(extraction_crepe_hop_length))


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method, crepe_hop_length):
        # default resample type of librosa.resample is "soxr_hq".
        # Quality: soxr_vhq > soxr_hq
        x, sr = librosa.load(path, self.fs)  # , res_type='soxr_vhq'
        p_len = x.shape[0] // self.hop
        f0_min = 50
        f0_max = 1100
        assert sr == self.fs
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0 = (
                parselmouth.Sound(x, sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=sr,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / sr,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=sr,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / sr,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "crepe": # Fork Feature: Added crepe f0 for f0 feature extraction
            print("Performing crepe pitch extraction. (EXPERIMENTAL)")
            print("CREPE PITCH EXTRACTION HOP LENGTH: " + str(crepe_hop_length))
            x = x.astype(np.float32)
            x /= np.quantile(np.abs(x), 0.999)
            torch_device_index = 0
            torch_device = None
            if torch.cuda.is_available():
                torch_device = torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
            elif torch.backends.mps.is_available():
                torch_device = torch.device("mps")
            else:
                torch_device = torch.device("cpu")
            audio = torch.from_numpy(x).to(torch_device, copy=True)
            audio = torch.unsqueeze(audio, dim=0)
            if audio.ndim == 2 and audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True).detach()
            audio = audio.detach()
            print(
                "Initiating f0 Crepe Feature Extraction with an extraction_crepe_hop_length of: " +
                str(crepe_hop_length)
            )
            # Pitch prediction for pitch extraction
            pitch: Tensor = torchcrepe.predict(
                audio,
                sr,
                crepe_hop_length,
                f0_min,
                f0_max,
                "full",
                batch_size=crepe_hop_length * 2,
                device=torch_device,
                pad=True                
            )
            p_len = p_len or x.shape[0] // crepe_hop_length
            # Resize the pitch
            source = np.array(pitch.squeeze(0).cpu().float().numpy())
            source[source < 0.001] = np.nan
            target = np.interp(
                np.arange(0, len(source) * p_len, len(source)) / p_len,
                np.arange(0, len(source)),
                source
            )
            f0 = np.nan_to_num(target)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(np.int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method, crepe_hop_length):
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt("todo-f0-%s" % len(paths))
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method, crepe_hop_length)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


if __name__ == "__main__":
    # exp_dir=r"E:\codes\py39\dataset\mi-test"
    # n_p=16
    # f = open("%s/log_extract_f0.log"%exp_dir, "w")
    printt(sys.argv)
    featureInput = FeatureInput()
    paths = []
    inp_root = "%s/1_16k_wavs" % (exp_dir)
    opt_root1 = "%s/2a_f0" % (exp_dir)
    opt_root2 = "%s/2b-f0nsf" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])

    ps = []
    for i in range(n_p):
        p = Process(
            target=featureInput.go,
            args=(
                paths[i::n_p],
                f0method,
                extraction_crepe_hop_length,
            ),
        )
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
