import os
import sys
import traceback

import parselmouth

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging


import numpy as np
import pyworld
import torchcrepe
import torch
#from torch import Tensor  # Fork Feature. Used for pitch prediction for torch crepe.
import tqdm
from lib.infer.infer_libs.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)
from multiprocessing import Process

exp_dir = sys.argv[1]
f = open("%s/extract_f0_feature.log" % exp_dir, "a+")

DoFormant = False
Quefrency = 1.0
Timbre = 1.0

def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


n_p = int(sys.argv[2])
f0method = sys.argv[3]
extraction_crepe_hop_length = 0
try:
    extraction_crepe_hop_length = int(sys.argv[4])
except:
    print("Temp Issue. echl is not being passed with argument!")
    extraction_crepe_hop_length = 128

class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_method_dict = self.get_f0_method_dict()
        
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def mncrepe(self, method, x, p_len, crepe_hop_length):
        f0 = None
        torch_device_index = 0
        torch_device = torch.device(
            f"cuda:{torch_device_index % torch.cuda.device_count()}"
        ) if torch.cuda.is_available() \
            else torch.device("mps") if torch.backends.mps.is_available() \
            else torch.device("cpu")

        audio = torch.from_numpy(x.astype(np.float32)).to(torch_device, copy=True)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        
        if method == 'mangio-crepe':
            pitch: torch.Tensor = torchcrepe.predict(
                audio,
                self.fs,
                crepe_hop_length,
                self.f0_min,
                self.f0_max,
                "full",
                batch_size=crepe_hop_length * 2,
                device=torch_device,
                pad=True,
            )
            p_len = p_len or x.shape[0] // crepe_hop_length
            # Resize the pitch
            source = np.array(pitch.squeeze(0).cpu().float().numpy())
            source[source < 0.001] = np.nan
            target = np.interp(
                np.arange(0, len(source) * p_len, len(source)) / p_len,
                np.arange(0, len(source)),
                source,
            )
            f0 = np.nan_to_num(target)
            
        elif method == 'crepe':
            batch_size = 512
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.fs,
                160,
                self.f0_min,
                self.f0_max,
                "full",
                batch_size=batch_size,
                device=torch_device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
            f0 = f0[1:]  # Get rid of extra first frame

        return f0

    def get_pm(self, x, p_len):
        f0 = parselmouth.Sound(x, self.fs).to_pitch_ac(
            time_step=160 / 16000,
            voicing_threshold=0.6,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max,
        ).selected_array["frequency"]
        
        return np.pad(
            f0,
            [[max(0, (p_len - len(f0) + 1) // 2), max(0, p_len - len(f0) - (p_len - len(f0) + 1) // 2)]],
            mode="constant"
        )

    def get_harvest(self, x):
        f0_spectral = pyworld.harvest(
            x.astype(np.double),
            fs=self.fs,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=1000 * self.hop / self.fs,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.fs)

    def get_dio(self, x):
        f0_spectral = pyworld.dio(
            x.astype(np.double),
            fs=self.fs,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=1000 * self.hop / self.fs,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.fs)

    def get_rmvpe(self, x):
        if hasattr(self, "model_rmvpe") == False:
                from lib.infer.infer_libs.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=False, device="cpu"
                )
        return self.model_rmvpe.infer_from_audio(x, thred=0.03)
        
    def get_rmvpe_dml(self, x):
        ...

    def get_f0_method_dict(self):
        return {
            "pm": self.get_pm,
            "harvest": self.get_harvest,
            "dio": self.get_dio,
            "rmvpe": self.get_rmvpe
        }

    def get_f0_hybrid_computation(
        self,
        methods_str,
        x,
        p_len,
        crepe_hop_length,
    ):
        # Get various f0 methods from input to use in the computation stack
        s = methods_str
        s = s.split("hybrid")[1]
        s = s.replace("[", "").replace("]", "")
        methods = s.split("+")
        f0_computation_stack = []

        for method in methods:
            if method in self.f0_method_dict:
                f0 = self.f0_method_dict[method](x, p_len) if method == 'pm' else self.f0_method_dict[method](x)
                f0_computation_stack.append(f0)
            elif method == 'crepe' or method == 'mangio-crepe':
                self.the_other_complex_function(x, method, crepe_hop_length)

        if len(f0_computation_stack) != 0:        
            f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0) if len(f0_computation_stack)>1 else f0_computation_stack[0]
            return f0_median_hybrid
        else:
            raise ValueError("No valid methods were provided")

    def compute_f0(self, path, f0_method, crepe_hop_length):
        x = load_audio(path, self.fs, DoFormant, Quefrency, Timbre)
        p_len = x.shape[0] // self.hop

        if f0_method in self.f0_method_dict:
            f0 = self.f0_method_dict[f0_method](x, p_len) if f0_method == 'pm' else self.f0_method_dict[f0_method](x)
        elif f0_method in ['crepe', 'mangio-crepe']:
            f0 = self.mncrepe(f0_method, x, p_len, crepe_hop_length)
        elif "hybrid" in f0_method:  # EXPERIMENTAL
            # Perform hybrid median pitch estimation
            f0 = self.get_f0_hybrid_computation(
                f0_method,
                x,
                p_len,
                crepe_hop_length,
            )
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method, crepe_hop_length, thread_n):
        if len(paths) == 0:
            printt("no-f0-todo")
            return
        with tqdm.tqdm(total=len(paths), leave=True, position=thread_n) as pbar:
            description = f"thread:{thread_n}, f0ing, Hop-Length:{crepe_hop_length}"
            pbar.set_description(description)
                
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if (
                        os.path.exists(opt_path1 + ".npy") 
                        and os.path.exists(opt_path2 + ".npy")
                    ):
                        pbar.update(1)
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
                    pbar.update(1)
                except Exception as e:
                    printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")


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
    print("Using f0 method: " + f0method)
    for i in range(n_p):
        p = Process(
            target=featureInput.go,
            args=(paths[i::n_p], f0method, extraction_crepe_hop_length, i),
        )
        ps.append(p)
        p.start()
    for i in range(n_p):
        ps[i].join()
