import os
import time
import torch
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from scipy.io import wavfile
from rvc.infer.pipeline import Pipeline as VC
from audio_upscaler import upscale
from rvc.utils import load_audio, load_embedding
from rvc.tools.split_audio import process_audio, merge_audio
from rvc.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.configs.config import Config

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class VoiceConverter:
    def __init__(self):
        self.config = Config()
        self.hubert_model = None
        self.tgt_sr = None
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = None
        self.n_spk = None

    def load_hubert(self, embedder_model, embedder_model_custom):
        models, _, _ = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = models[0].to(self.config.device)
        self.hubert_model = (
            self.hubert_model.half()
            if self.config.is_half
            else self.hubert_model.float()
        )
        self.hubert_model.eval()

    @staticmethod
    def remove_audio_noise(input_audio_path, reduction_strength=0.7):
        try:
            rate, data = wavfile.read(input_audio_path)
            reduced_noise = nr.reduce_noise(
                y=data, sr=rate, prop_decrease=reduction_strength
            )
            return reduced_noise
        except Exception as error:
            print(f"Error cleaning audio: {error}")
            return None

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        try:
            if output_format != "WAV":
                print(f"Converting audio to {output_format} format...")
                audio, sample_rate = librosa.load(input_path, sr=None)
                common_sample_rates = [
                    8000,
                    11025,
                    12000,
                    16000,
                    22050,
                    24000,
                    32000,
                    44100,
                    48000,
                ]
                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
            return output_path
        except Exception as error:
            print(f"Failed to convert audio to {output_format} format: {error}")

    def voice_conversion(
        self,
        sid=0,
        input_audio_path=None,
        f0_up_key=None,
        f0_file=None,
        f0_method=None,
        file_index=None,
        index_rate=None,
        resample_sr=0,
        rms_mix_rate=None,
        protect=None,
        hop_length=None,
        output_path=None,
        split_audio=False,
        f0autotune=False,
        filter_radius=None,
        embedder_model=None,
        embedder_model_custom=None,
    ):
        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95

            if audio_max > 1:
                audio /= audio_max

            if not self.hubert_model:
                self.load_hubert(embedder_model, embedder_model_custom)
            if_f0 = self.cpt.get("f0", 1)

            file_index = (
                file_index.strip()
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip()
                .replace("trained", "added")
            )
            if self.tgt_sr != resample_sr >= 16000:
                self.tgt_sr = resample_sr

            if split_audio == "True":
                result, new_dir_path = process_audio(input_audio_path)
                if result == "Error":
                    return "Error with Split Audio", None

                dir_path = (
                    new_dir_path.strip().strip('"').strip("\n").strip('"').strip()
                )
                if dir_path:
                    paths = [
                        os.path.join(root, name)
                        for root, _, files in os.walk(dir_path, topdown=False)
                        for name in files
                        if name.endswith(".wav") and root == dir_path
                    ]
                try:
                    for path in paths:
                        self.voice_conversion(
                            sid,
                            path,
                            f0_up_key,
                            None,
                            f0_method,
                            file_index,
                            index_rate,
                            resample_sr,
                            rms_mix_rate,
                            protect,
                            hop_length,
                            path,
                            False,
                            f0autotune,
                            filter_radius,
                            embedder_model,
                            embedder_model_custom,
                        )
                except Exception as error:
                    print(error)
                    return f"Error {error}"
                print("Finished processing segmented audio, now merging audio...")
                merge_timestamps_file = os.path.join(
                    os.path.dirname(new_dir_path),
                    f"{os.path.basename(input_audio_path).split('.')[0]}_timestamps.txt",
                )
                self.tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
                os.remove(merge_timestamps_file)
            else:
                audio_opt = self.vc.pipeline(
                    self.hubert_model,
                    self.net_g,
                    sid,
                    audio,
                    input_audio_path,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    if_f0,
                    filter_radius,
                    self.tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    hop_length,
                    f0autotune,
                    f0_file=f0_file,
                )

            if output_path:
                sf.write(output_path, audio_opt, self.tgt_sr, format="WAV")

            return self.tgt_sr, audio_opt

        except Exception as error:
            print(error)

    def get_vc(self, weight_root, sid):
        if sid == "" or sid == []:
            if self.hubert_model is not None:
                print("clean_empty_cache")
                del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
                self.hubert_model = self.net_g = self.n_spk = self.vc = (
                    self.hubert_model
                ) = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if_f0 = self.cpt.get("f0", 1)
            self.version = self.cpt.get("version", "v1")
            if self.version == "v1":
                if if_f0 == 1:
                    self.net_g = SynthesizerTrnMs256NSFsid(
                        *self.cpt["config"], is_half=self.config.is_half
                    )
                else:
                    self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
            elif self.version == "v2":
                if if_f0 == 1:
                    self.net_g = SynthesizerTrnMs768NSFsid(
                        *self.cpt["config"], is_half=self.config.is_half
                    )
                else:
                    self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
            del self.net_g, self.cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cpt = None

        person = weight_root
        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        if_f0 = self.cpt.get("f0", 1)

        self.version = self.cpt.get("version", "v1")
        if self.version == "v1":
            if if_f0 == 1:
                self.net_g = SynthesizerTrnMs256NSFsid(
                    *self.cpt["config"], is_half=self.config.is_half
                )
            else:
                self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
        elif self.version == "v2":
            if if_f0 == 1:
                self.net_g = SynthesizerTrnMs768NSFsid(
                    *self.cpt["config"], is_half=self.config.is_half
                )
            else:
                self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        self.net_g = self.net_g.half() if self.config.is_half else self.net_g.float()
        self.vc = VC(self.tgt_sr, self.config)
        self.n_spk = self.cpt["config"][-3]

    def infer_pipeline(
        self,
        f0up_key,
        filter_radius,
        index_rate,
        rms_mix_rate,
        protect,
        hop_length,
        f0method,
        audio_input_path,
        audio_output_path,
        model_path,
        index_path,
        split_audio,
        f0autotune,
        clean_audio,
        clean_strength,
        export_format,
        embedder_model,
        embedder_model_custom,
        upscale_audio,
    ):
        self.get_vc(model_path, 0)

        try:
            start_time = time.time()
            print(f"Converting audio '{audio_input_path}'...")
            if upscale_audio == "True":
                upscale(audio_input_path, audio_input_path)
                
            self.voice_conversion(
                sid=0,
                input_audio_path=audio_input_path,
                f0_up_key=f0up_key,
                f0_file=None,
                f0_method=f0method,
                file_index=index_path,
                index_rate=float(index_rate),
                rms_mix_rate=float(rms_mix_rate),
                protect=float(protect),
                hop_length=hop_length,
                output_path=audio_output_path,
                split_audio=split_audio,
                f0autotune=f0autotune,
                filter_radius=filter_radius,
                embedder_model=embedder_model,
                embedder_model_custom=embedder_model_custom,
            )

            if clean_audio == "True":
                cleaned_audio = self.remove_audio_noise(
                    audio_output_path, clean_strength
                )
                if cleaned_audio is not None:
                    sf.write(
                        audio_output_path, cleaned_audio, self.tgt_sr, format="WAV"
                    )

            output_path_format = audio_output_path.replace(
                ".wav", f".{export_format.lower()}"
            )
            audio_output_path = self.convert_audio_format(
                audio_output_path, output_path_format, export_format
            )

            elapsed_time = time.time() - start_time
            print(
                f"Conversion completed at '{audio_output_path}' in {elapsed_time:.2f} seconds."
            )

        except Exception as error:
            print(f"Voice conversion failed: {error}")
