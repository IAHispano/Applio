import os
import sys
import numpy as np
import pyworld
import torchcrepe
import torch
import parselmouth
import tqdm
from multiprocessing import Process, cpu_count

current_directory = os.getcwd()
sys.path.append(current_directory)


from rvc.lib.utils import load_audio


exp_dir = sys.argv[1]
f0_method = sys.argv[2]
num_processes = cpu_count()

try:
    hop_length = int(sys.argv[3])
except ValueError:
    hop_length = 128

DoFormant = False
Quefrency = 1.0
Timbre = 1.0


class FeatureInput:
    def __init__(self, sample_rate=16000, hop_size=160):
        self.fs = sample_rate
        self.hop = hop_size

        self.f0_method_dict = self.get_f0_method_dict()

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def mncrepe(self, method, x, p_len, hop_length):
        f0 = None
        torch_device_index = 0
        torch_device = (
            torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )

        audio = torch.from_numpy(x.astype(np.float32)).to(torch_device, copy=True)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()

        if method == "crepe":
            pitch = torchcrepe.predict(
                audio,
                self.fs,
                hop_length,
                self.f0_min,
                self.f0_max,
                "full",
                batch_size=hop_length * 2,
                device=torch_device,
                pad=True,
            )
            p_len = p_len or x.shape[0] // hop_length
            source = np.array(pitch.squeeze(0).cpu().float().numpy())
            source[source < 0.001] = np.nan
            target = np.interp(
                np.arange(0, len(source) * p_len, len(source)) / p_len,
                np.arange(0, len(source)),
                source,
            )
            f0 = np.nan_to_num(target)

        return f0

    def get_pm(self, x, p_len):
        f0 = (
            parselmouth.Sound(x, self.fs)
            .to_pitch_ac(
                time_step=160 / 16000,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            .selected_array["frequency"]
        )

        return np.pad(
            f0,
            [
                [
                    max(0, (p_len - len(f0) + 1) // 2),
                    max(0, p_len - len(f0) - (p_len - len(f0) + 1) // 2),
                ]
            ],
            mode="constant",
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

    def get_rmvpe(self, x, hop_length):
        if not hasattr(self, "model_rmvpe"):
            from rvc.lib.rmvpe import RMVPE

            self.model_rmvpe = RMVPE("rmvpe.pt", is_half=False, device="cpu")
        return self.model_rmvpe.infer_from_audio(x, thred=0.03, hop_length=hop_length)

    def get_f0_method_dict(self):
        return {
            "pm": self.get_pm,
            "harvest": self.get_harvest,
            "dio": self.get_dio
        }

    def compute_f0(self, path, f0_method, hop_length):
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop

        if f0_method in self.f0_method_dict:
            f0 = (
                self.f0_method_dict[f0_method](x, p_len)
                if f0_method == "pm"
                else self.f0_method_dict[f0_method](x)
            )
        elif f0_method == "crepe":
            f0 = self.mncrepe(f0_method, x, p_len, hop_length)
        elif f0_method == "rmvpe":
            f0 = self.get_rmvpe(x, hop_length=hop_length)
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

    def process_paths(self, paths, f0_method, hop_length, thread_n):
        if len(paths) == 0:
            print("There are no paths to process.")
            return
        with tqdm.tqdm(total=len(paths), leave=True, position=thread_n) as pbar:
            description = f"Thread {thread_n} | Hop-Length {hop_length}"
            pbar.set_description(description)

            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if os.path.exists(opt_path1 + ".npy") and os.path.exists(
                        opt_path2 + ".npy"
                    ):
                        pbar.update(1)
                        continue

                    feature_pit = self.compute_f0(inp_path, f0_method, hop_length)
                    np.save(
                        opt_path2,
                        feature_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(feature_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                    pbar.update(1)
                except Exception as error:
                    print(f"f0fail-{idx}-{inp_path}-{error}")


if __name__ == "__main__":
    feature_input = FeatureInput()
    paths = []
    input_root = f"{exp_dir}/1_16k_wavs"
    output_root1 = f"{exp_dir}/2a_f0"
    output_root2 = f"{exp_dir}/2b-f0nsf"

    os.makedirs(output_root1, exist_ok=True)
    os.makedirs(output_root2, exist_ok=True)
    for name in sorted(list(os.listdir(input_root))):
        input_path = f"{input_root}/{name}"
        if "spec" in input_path:
            continue
        output_path1 = f"{output_root1}/{name}"
        output_path2 = f"{output_root2}/{name}"
        paths.append([input_path, output_path1, output_path2])

    processes = []
    print("Using f0 method: " + f0_method)
    for i in range(num_processes):
        p = Process(
            target=feature_input.process_paths,
            args=(paths[i::num_processes], f0_method, hop_length, i),
        )
        processes.append(p)
        p.start()
    for i in range(num_processes):
        processes[i].join()
