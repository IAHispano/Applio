import numpy as np, parselmouth, torch, pdb
from time import time as ttime
import torch.nn.functional as F
import torchcrepe # Fork feature. Use the crepe f0 algorithm. New dependency (pip install torchcrepe)
import scipy.signal as signal
import pyworld, os, traceback, faiss
from scipy import signal
from torch import Tensor # Fork Feature. Used for pitch prediction for the torchcrepe f0 inference computation

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

class VC(object):
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

    #region f0 Overhaul Region
    # Fork Feature: Get the best torch device to use for f0 algorithms that require a torch device. Will return the type (torch.device)
    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        # Get cuda device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{index % torch.cuda.device_count()}") # Very fast
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        # Insert an else here to grab "xla" devices if available. TO DO later. Requires the torch_xla.core.xla_model library
        # Else wise return the "cpu" as a torch device, 
        return torch.device("cpu")

    # Get the f0 via parselmouth computation
    def get_f0_pm_computation(self, x, time_step, f0_min, f0_max, p_len):
        f0 = (
            parselmouth.Sound(x, self.sr)
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
        return f0

    # Get the f0 via the pyworld computation. Fork Feature +dio along with harvest
    def get_f0_pyworld_computation(self, x, f0_min, f0_max, f0_type):
        if f0_type == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
        elif f0_type == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
        f0 = signal.medfilt(f0, 3) 
        return f0
    
    # Fork Feature: Get the f0 via the crepe algorithm from torchcrepe
    def get_f0_crepe_computation(
            self, 
            x, 
            f0_min,
            f0_max,
            p_len,
            hop_length=128, # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
            model="full", # Either use crepe-tiny "tiny" or crepe "full". Default is full
    ):
        x = x.astype(np.float32) # fixes the F.conv2D exception. We needed to convert double to float.
        x /= np.quantile(np.abs(x), 0.999)
        torch_device = self.get_optimal_torch_device()
        audio = torch.from_numpy(x).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
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
            pad=True
        )
        p_len = p_len or x.shape[0] // hop_length
        # Resize the pitch for final f0
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source
        )
        f0 = np.nan_to_num(target)
        return f0 # Resized f0
    
    #endregion

    def get_f0(self, x, p_len, f0_up_key, f0_method, crepe_hop_length, inp_f0=None):
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = self.get_f0_pm_computation(x, time_step, f0_min, f0_max, p_len)
        elif f0_method == "harvest":
            f0 = self.get_f0_pyworld_computation(x, f0_min, f0_max, "harvest")
        elif f0_method == "dio": # Fork Feature
            f0 = self.get_f0_pyworld_computation(x, f0_min, f0_max, "dio")
        elif f0_method == "crepe": # Fork Feature: Adding a new f0 algorithm called crepe
            f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, crepe_hop_length)
        elif f0_method == "crepe-tiny": # For Feature add crepe-tiny model
            f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, crepe_hop_length, "tiny")

        print("Using the following f0 method: " + f0_method)
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
        f0_coarse = np.rint(f0_mel).astype(np.int)

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
            "output_layer": 9,  # layer 9
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0])

        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
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
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0] * 32768)
                    .data.cpu()
                    .float()
                    .numpy()
                    .astype(np.int16)
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0] * 32768)
                    .data.cpu()
                    .float()
                    .numpy()
                    .astype(np.int16)
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        crepe_hop_length,
        f0_file=None,
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index) == True
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
        if hasattr(f0_file, "name") == True:
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
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method, crepe_hop_length, inp_f0)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
