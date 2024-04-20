import os
import sys
import time
import torch
import logging

import numpy as np
import soundfile as sf
import librosa

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.infer.pipeline import VC
from scipy.io import wavfile
import noisereduce as nr
from rvc.lib.utils import load_audio
from rvc.lib.tools.split_audio import process_audio, merge_audio
from fairseq import checkpoint_utils
from rvc.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.configs.config import Config

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

config = Config()
hubert_model = None
tgt_sr = None
net_g = None
vc = None
cpt = None
version = None
n_spk = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def remove_audio_noise(input_audio_path, reduction_strength=0.7):
    try:
        rate, data = wavfile.read(input_audio_path)
        reduced_noise = nr.reduce_noise(
            y=data,
            sr=rate,
            prop_decrease=reduction_strength,
        )
        return reduced_noise
    except Exception as error:
        print(f"Error cleaning audio: {error}")
        return None


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
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
            sf.write(output_path, audio, target_sr, format=output_format.lower())
        return output_path
    except Exception as error:
        print(f"Failed to convert audio to {output_format} format: {error}")


def vc_single(
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
):
    global tgt_sr, net_g, vc, hubert_model, version

    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95

        if audio_max > 1:
            audio /= audio_max

        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)

        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        if split_audio == "True":
            result, new_dir_path = process_audio(input_audio_path)
            if result == "Error":
                return "Error with Split Audio", None
            dir_path = (
                new_dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )
            if dir_path != "":
                paths = [
                    os.path.join(root, name)
                    for root, _, files in os.walk(dir_path, topdown=False)
                    for name in files
                    if name.endswith(".wav") and root == dir_path
                ]
            try:
                for path in paths:
                    vc_single(
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
                    )
            except Exception as error:
                print(error)
                return f"Error {error}"
            print("Finished processing segmented audio, now merging audio...")
            merge_timestamps_file = os.path.join(
                os.path.dirname(new_dir_path),
                f"{os.path.basename(input_audio_path).split('.')[0]}_timestamps.txt",
            )
            tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
            os.remove(merge_timestamps_file)

        else:
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                sid,
                audio,
                input_audio_path,
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
                hop_length,
                f0autotune,
                f0_file=f0_file,
            )
        if output_path is not None:
            sf.write(output_path, audio_opt, tgt_sr, format="WAV")

        return (tgt_sr, audio_opt)

    except Exception as error:
        print(error)


def get_vc(weight_root, sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
    person = weight_root
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)

    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]


def infer_pipeline(
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
):
    global tgt_sr, net_g, vc, cpt

    get_vc(model_path, 0)

    try:
        start_time = time.time()
        vc_single(
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
        )

        if clean_audio == "True":
            cleaned_audio = remove_audio_noise(audio_output_path, clean_strength)
            if cleaned_audio is not None:
                sf.write(audio_output_path, cleaned_audio, tgt_sr, format="WAV")

        output_path_format = audio_output_path.replace(
            ".wav", f".{export_format.lower()}"
        )
        audio_output_path = convert_audio_format(
            audio_output_path, output_path_format, export_format
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Conversion completed. Output file: '{audio_output_path}' in {elapsed_time:.2f} seconds."
        )

    except Exception as error:
        print(f"Voice conversion failed: {error}")
