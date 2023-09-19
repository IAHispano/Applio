import os
import sys
import traceback
import logging

logger = logging.getLogger(__name__)

from functools import lru_cache
from time import time as ttime
from torch import Tensor
import faiss
import librosa
import numpy as np
import parselmouth
import pyworld
import torch.nn.functional as F
from scipy import signal
from tqdm import tqdm

import random
now_dir = os.getcwd()
sys.path.append(now_dir)
import re
from functools import partial
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}
import torchcrepe  # Fork Feature. Crepe algo for training and preprocess
import torch
from lib.infer.infer_libs.rmvpe import RMVPE

@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device
        self.model_rmvpe = RMVPE("%s/rmvpe.pt" % os.environ["rmvpe_root"], is_half=self.is_half, device=self.device)
        self.f0_method_dict = {
            "pm": self.get_pm,
            "harvest": self.get_harvest,
            "dio": self.get_dio,
            "rmvpe": self.get_rmvpe,
            "rmvpe+": self.get_pitch_dependant_rmvpe,
            "crepe": self.get_f0_official_crepe_computation,
            "crepe-tiny": partial(self.get_f0_official_crepe_computation, model='model'),
            "mangio-crepe": self.get_f0_crepe_computation,
            "mangio-crepe-tiny": partial(self.get_f0_crepe_computation, model='model'),
            
        }
        self.note_dict = [
            65.41, 69.30, 73.42, 77.78, 82.41, 87.31,
            92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
            130.81, 138.59, 146.83, 155.56, 164.81, 174.61,
            185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
            261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
            369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
            523.25, 554.37, 587.33, 622.25, 659.25, 698.46,
            739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
            1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91,
            1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
            2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83,
            2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07
        ]

    # Fork Feature: Get the best torch device to use for f0 algorithms that require a torch device. Will return the type (torch.device)
    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(
                f"cuda:{index % torch.cuda.device_count()}"
            )  # Very fast
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Fork Feature: Compute f0 with the crepe method
    def get_f0_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        p_len,
        *args,  # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
        **kwargs,  # Either use crepe-tiny "tiny" or crepe "full". Default is full
    ):
        x = x.astype(
            np.float32
        )  # fixes the F.conv2D exception. We needed to convert double to float.
        x /= np.quantile(np.abs(x), 0.999)
        torch_device = self.get_optimal_torch_device()
        audio = torch.from_numpy(x).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        hop_length = kwargs.get('crepe_hop_length', 160)
        model = kwargs.get('model', 'full') 
        print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=torch_device,
            pad=True,
        )
        p_len = p_len or x.shape[0] // hop_length
        # Resize the pitch for final f0
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        return f0  # Resized f0
    
    def get_f0_official_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        *args,
        **kwargs
    ):
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using first gpu
        audio = torch.tensor(np.copy(x))[None].float()
        model = kwargs.get('model', 'full') 
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            self.window,
            f0_min,
            f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        return f0

    # Fork Feature: Compute pYIN f0 method
    def get_f0_pyin_computation(self, x, f0_min, f0_max):
        y, sr = librosa.load("saudio/Sidney.wav", self.sr, mono=True)
        f0, _, _ = librosa.pyin(y, sr=self.sr, fmin=f0_min, fmax=f0_max)
        f0 = f0[1:]  # Get rid of extra first frame
        return f0

    def get_pm(self, x, p_len, *args, **kwargs):
        f0 = parselmouth.Sound(x, self.sr).to_pitch_ac(
            time_step=160 / 16000,
            voicing_threshold=0.6,
            pitch_floor=kwargs.get('f0_min'),
            pitch_ceiling=kwargs.get('f0_max'),
        ).selected_array["frequency"]
        
        return np.pad(
            f0,
            [[max(0, (p_len - len(f0) + 1) // 2), max(0, p_len - len(f0) - (p_len - len(f0) + 1) // 2)]],
            mode="constant"
        )

    def get_harvest(self, x, *args, **kwargs):
        f0_spectral = pyworld.harvest(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=kwargs.get('f0_max'),
            f0_floor=kwargs.get('f0_min'),
            frame_period=1000 * kwargs.get('hop_length', 160) / self.sr,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.sr)

    def get_dio(self, x, *args, **kwargs):
        f0_spectral = pyworld.dio(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=kwargs.get('f0_max'),
            f0_floor=kwargs.get('f0_min'),
            frame_period=1000 * kwargs.get('hop_length', 160) / self.sr,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.sr)


    def get_rmvpe(self, x, *args, **kwargs):
        if not hasattr(self, "model_rmvpe"):
            from lib.infer.infer_libs.rmvpe import RMVPE
            
            logger.info(
                "Loading rmvpe model,%s" % "%s/rmvpe.pt" % os.environ["rmvpe_root"]
            )
            self.model_rmvpe = RMVPE(
                "%s/rmvpe.pt" % os.environ["rmvpe_root"],
                is_half=self.is_half,
                device=self.device,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            
        return f0
    

    def get_pitch_dependant_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        return self.model_rmvpe.infer_from_audio_with_pitch(x, thred=0.03, f0_min=f0_min, f0_max=f0_max)

    def autotune_f0(self, f0):
        autotuned_f0 = []
        for freq in f0:
            closest_notes = [x for x in self.note_dict if abs(x - freq) == min(abs(n - freq) for n in self.note_dict)]
            autotuned_f0.append(random.choice(closest_notes))
        return np.array(autotuned_f0, np.float64)

    # Fork Feature: Acquire median hybrid f0 estimation calculation
    def get_f0_hybrid_computation(
        self,
        methods_str,
        input_audio_path,
        x,
        f0_min,
        f0_max,
        p_len,
        filter_radius,
        crepe_hop_length,
        time_step
    ):
        # Get various f0 methods from input to use in the computation stack
        params = {'x': x, 'p_len': p_len, 'f0_min': f0_min, 
          'f0_max': f0_max, 'time_step': time_step, 'filter_radius': filter_radius, 
          'crepe_hop_length': crepe_hop_length, 'model': "full"
        }
        methods_str = re.search('hybrid\[(.+)\]', methods_str)
        if methods_str:  # Ensure a match was found
            methods = [method.strip() for method in methods_str.group(1).split('+')]
        f0_computation_stack = []

        print(f"Calculating f0 pitch estimations for methods: {str(methods)}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        # Get f0 calculations for all methods specified

        for method in methods:
            if method not in self.f0_method_dict:
                print(f"Method {method} not found.")
                continue
            f0 = self.f0_method_dict[method](**params)
            if method == 'harvest' and filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
                f0 = f0[1:]  # Get rid of first frame.
            f0_computation_stack.append(f0)

        for fc in f0_computation_stack:
            print(len(fc))

        print(f"Calculating hybrid median f0 from the stack of: {str(methods)}")
        f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0)
        return f0_median_hybrid
    
    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        crepe_hop_length,
        f0_autotune,
        inp_f0=None,
        f0_min=50,
        f0_max=1100,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        params = {'x': x, 'p_len': p_len, 'f0_up_key': f0_up_key, 'f0_min': f0_min, 
          'f0_max': f0_max, 'time_step': time_step, 'filter_radius': filter_radius, 
          'crepe_hop_length': crepe_hop_length, 'model': "full"
        }

        if "hybrid" in f0_method:
            # Perform hybrid median pitch estimation
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = self.get_f0_hybrid_computation(
                f0_method,+
                input_audio_path,
                x,
                f0_min,
                f0_max,
                p_len,
                filter_radius,
                crepe_hop_length,
                time_step,
            )
        else:
            f0 = self.f0_method_dict[f0_method](**params)

        if "privateuseone" in str(self.device):  # clean ortruntime memory
            del self.model_rmvpe.model
            del self.model_rmvpe
            logger.info("Cleaning ortruntime memory")

        if f0_autotune:
            f0 = self.autotune_f0(f0)

        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak  # 1-0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if (
            not isinstance(index, type(None))
            and not isinstance(big_npy, type(None))
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1
    def process_t(self, t, s, window, audio_pad, pitch, pitchf, times, index, big_npy, index_rate, version, protect, t_pad_tgt, if_f0, sid, model, net_g):
        t = t // window * window
        if if_f0 == 1:
            return self.vc(
                model,
                net_g,
                sid,
                audio_pad[s : t + t_pad_tgt + window],
                pitch[:, s // window : (t + t_pad_tgt) // window],
                pitchf[:, s // window : (t + t_pad_tgt) // window],
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[t_pad_tgt : -t_pad_tgt]
        else:
            return self.vc(
                model,
                net_g,
                sid,
                audio_pad[s : t + t_pad_tgt + window],
                None,
                None,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[t_pad_tgt : -t_pad_tgt]


    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length, 
        f0_autotune, 
        f0_file=None, 
        f0_min=50, 
        f0_max=1100
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius, 
                crepe_hop_length, 
                f0_autotune,
                inp_f0, 
                f0_min, 
                f0_max
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps" or "xpu" in self.device:
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1

        with tqdm(total=len(opt_ts), desc="Processing", unit="window") as pbar:
            for i, t in enumerate(opt_ts):
                t = t // self.window * self.window
                start = s
                end = t + self.t_pad2 + self.window
                audio_slice = audio_pad[start:end]
                pitch_slice = pitch[:, start // self.window:end // self.window] if if_f0 else None
                pitchf_slice = pitchf[:, start // self.window:end // self.window] if if_f0 else None
                audio_opt.append(self.vc(model, net_g, sid, audio_slice, pitch_slice, pitchf_slice, times, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
                s = t
                pbar.update(1)
                pbar.refresh()

        audio_slice = audio_pad[t:]
        pitch_slice = pitch[:, t // self.window:] if if_f0 and t is not None else pitch
        pitchf_slice = pitchf[:, t // self.window:] if if_f0 and t is not None else pitchf
        audio_opt.append(self.vc(model, net_g, sid, audio_slice, pitch_slice, pitchf_slice, times, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Returning completed audio...")
        print("-------------------")
        return audio_opt
