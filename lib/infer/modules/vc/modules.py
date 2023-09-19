import os, sys
import traceback
import logging
now_dir = os.getcwd()
sys.path.append(now_dir)
logger = logging.getLogger(__name__)
import lib.globals.globals as rvc_globals
import numpy as np
import soundfile as sf
import torch
from io import BytesIO
from lib.infer.infer_libs.audio import load_audio
from lib.infer.infer_libs.audio import wav2
from lib.infer.infer_libs.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.infer.modules.vc.pipeline import Pipeline
from lib.infer.modules.vc.utils import *
import time
import scipy.io.wavfile as wavfile

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}
def note_to_hz(note_name):
        SEMITONES = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
        pitch_class, octave = note_name[:-1], int(note_name[-1])
        semitone = SEMITONES[pitch_class]
        note_number = 12 * (octave - 4) + semitone
        frequency = 440.0 * (2.0 ** (1.0/12)) ** note_number
        return frequency

class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[0]
            if self.if_f0 != 0 and to_return_protect
            else 0.5,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[1]
            if self.if_f0 != 0 and to_return_protect
            else 0.33,
            "__type__": "update",
        }

        if not sid:
            if self.hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (
                    self.net_g,
                    self.n_spk,
                    self.vc,
                    self.hubert_model,
                    self.tgt_sr,
                )  # ,cpt
                self.hubert_model = (
                    self.net_g
                ) = self.n_spk = self.vc = self.hubert_model = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        #person = f'{os.getenv("weight_root")}/{sid}'
        person = f'{sid}'
        #logger.info(f"Loading: {person}")
        logger.info(f"Loading...")
        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": False, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1
            )
            if to_return_protect
            else {"visible": False, "maximum": n_spk, "__type__": "update"}
        )
    

    def vc_single(
        self,
        sid,
        input_audio_path1,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        f0_min,
        note_min,
        f0_max,
        note_max,
        f0_autotune,
    ):
        global total_time
        total_time = 0
        start_time = time.time()
        if not input_audio_path1:
            return "You need to upload an audio", None
        
        if (not os.path.exists(input_audio_path1)) and (not os.path.exists(os.path.join(now_dir, input_audio_path1))):
            return "Audio was not properly selected or doesn't exist", None
        
        print(f"\nStarting inference for '{os.path.basename(input_audio_path1)}'")
        print("-------------------")
        f0_up_key = int(f0_up_key)
        if rvc_globals.NotesOrHertz and f0_method != 'rmvpe':
            f0_min = note_to_hz(note_min) if note_min else 50
            f0_max = note_to_hz(note_max) if note_max else 1100
            print(f"Converted Min pitch: freq - {f0_min}\n"
                  f"Converted Max pitch: freq - {f0_max}")
        else:
            f0_min = f0_min or 50
            f0_max = f0_max or 1100
        try:
            print(f"Attempting to load {input_audio_path1}....")
            audio = load_audio(file=input_audio_path1,
                               sr=16000,
                               DoFormant=rvc_globals.DoFormant,
                               Quefrency=rvc_globals.Quefrency,
                               Timbre=rvc_globals.Timbre)
            
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            try:
                self.if_f0 = self.cpt.get("f0", 1)
            except NameError:
                message = "Model was not properly selected"
                print(message)
                return message, None
            
            file_index = (
                (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
                if file_index != ""
                else file_index2
            )  # 防止小白写错，自动帮他替换掉

            try:
                audio_opt = self.pipeline.pipeline(
                    self.hubert_model,
                    self.net_g,
                    sid,
                    audio,
                    input_audio_path1,
                    times,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    self.if_f0,
                    filter_radius,
                    self.tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    crepe_hop_length,
                    f0_autotune,
                    f0_file=f0_file,
                    f0_min=f0_min,
                    f0_max=f0_max
                    )
            except AssertionError:
                message = "Mismatching index version detected (v1 with v2, or v2 with v1)."
                print(message)
                return message, None
            except NameError:
                message = "RVC libraries are still loading. Please try again in a few seconds."
                print(message)
                return message, None

            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            end_time = time.time()
            total_time = end_time - start_time

            output_folder = "assets/audios/audio-outputs"
            os.makedirs(output_folder, exist_ok=True)  
            output_filename = "generated_audio_{}.wav"
            output_count = 1
            while True:
                current_output_path = os.path.join(output_folder, output_filename.format(output_count))
                if not os.path.exists(current_output_path):
                    break
                output_count += 1
            
            wavfile.write(current_output_path, self.tgt_sr, audio_opt)
            print(f"Generated audio saved to: {current_output_path}")
            
            return (
                "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warn(info)
            return info, (None, None)

    def vc_single_dont_save(
        self,
        sid,
        input_audio_path0,
        input_audio_path1,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        f0_min,
        note_min,
        f0_max,
        note_max,
        f0_autotune,
    ):
        global total_time
        total_time = 0
        start_time = time.time()
        if not input_audio_path0 and not input_audio_path1:
            return "You need to upload an audio", None
        
        if (not os.path.exists(input_audio_path0)) and (not os.path.exists(os.path.join(now_dir, input_audio_path0))):
            return "Audio was not properly selected or doesn't exist", None
        
        input_audio_path1 = input_audio_path1 or input_audio_path0
        print(f"\nStarting inference for '{os.path.basename(input_audio_path1)}'")
        print("-------------------")
        f0_up_key = int(f0_up_key)
        if rvc_globals.NotesOrHertz and f0_method != 'rmvpe':
            f0_min = note_to_hz(note_min) if note_min else 50
            f0_max = note_to_hz(note_max) if note_max else 1100
            print(f"Converted Min pitch: freq - {f0_min}\n"
                  f"Converted Max pitch: freq - {f0_max}")
        else:
            f0_min = f0_min or 50
            f0_max = f0_max or 1100
        try:
            input_audio_path1 = input_audio_path1 or input_audio_path0
            print(f"Attempting to load {input_audio_path1}....")
            audio = load_audio(file=input_audio_path1,
                               sr=16000,
                               DoFormant=rvc_globals.DoFormant,
                               Quefrency=rvc_globals.Quefrency,
                               Timbre=rvc_globals.Timbre)
            
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            try:
                self.if_f0 = self.cpt.get("f0", 1)
            except NameError:
                message = "Model was not properly selected"
                print(message)
                return message, None
            
            file_index = (
                (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
                if file_index != ""
                else file_index2
            )  # 防止小白写错，自动帮他替换掉

            try:
                audio_opt = self.pipeline.pipeline(
                    self.hubert_model,
                    self.net_g,
                    sid,
                    audio,
                    input_audio_path1,
                    times,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    self.if_f0,
                    filter_radius,
                    self.tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    crepe_hop_length,
                    f0_autotune,
                    f0_file=f0_file,
                    f0_min=f0_min,
                    f0_max=f0_max
                    )
            except AssertionError:
                message = "Mismatching index version detected (v1 with v2, or v2 with v1)."
                print(message)
                return message, None
            except NameError:
                message = "RVC libraries are still loading. Please try again in a few seconds."
                print(message)
                return message, None

            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            end_time = time.time()
            total_time = end_time - start_time
            return (
                "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warn(info)
            return info, (None, None)


    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
        crepe_hop_length,
        f0_min,
        note_min,
        f0_max,
        note_max,
        f0_autotune,
    ):
        if rvc_globals.NotesOrHertz and f0_method != 'rmvpe':
            f0_min = note_to_hz(note_min) if note_min else 50
            f0_max = note_to_hz(note_max) if note_max else 1100
            print(f"Converted Min pitch: freq - {f0_min}\n"
                  f"Converted Max pitch: freq - {f0_max}")
        else:
            f0_min = f0_min or 50
            f0_max = f0_max or 1100
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(root, name)
                        for root, _, files in os.walk(dir_path, topdown=False)
                        for name in files
                        if name.endswith(tuple(sup_audioext)) and root == dir_path
                        ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            print(paths)
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                    crepe_hop_length, 
                    f0_min, 
                    note_min, 
                    f0_max, 
                    note_max,
                    f0_autotune,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (opt_root, os.path.basename(path), format1)
                            with BytesIO() as wavf:
                                sf.write(
                                    wavf,
                                    audio_opt,
                                    tgt_sr,
                                    format="wav"
                                )
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
