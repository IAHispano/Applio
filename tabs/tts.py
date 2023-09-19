import sys

sys.path.append("..")
import os

now_dir = os.getcwd()

from dotenv import load_dotenv
from lib.infer.modules.vc.modules import VC
from assets.configs.config import Config

load_dotenv()
config = Config()
vc = VC(config)


import numpy as np
import torch

import soundfile as sf
from gtts import gTTS
import edge_tts
import asyncio
import scipy.io.wavfile as wavfile
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize
from bark import SAMPLE_RATE


import tabs.resources as resources

set_bark_voice = resources.get_bark_voice()
set_edge_voice = resources.get_edge_voice()


def update_tts_methods_voice(select_value):
    # ["Edge-tts", "RVG-tts", "Bark-tts"]
    if select_value == "Edge-tts":
        return {"choices": set_edge_voice, "value": "", "__type__": "update"}
    elif select_value == "Bark-tts":
        return {"choices": set_bark_voice, "value": "", "__type__": "update"}


def custom_voice(
    _values,  # filter indices
    audio_files,  # all audio files
    model_voice_path="",
    transpose=0,
    f0method="pm",
    index_rate_=float(0.66),
    crepe_hop_length_=float(64),
    f0_autotune=False,
    file_index="",
    file_index2="",
):
    vc.get_vc(model_voice_path)

    for _value_item in _values:
        filename = (
            "assets/audios/audio_outputs" + audio_files[_value_item]
            if _value_item != "converted_tts"
            else audio_files[0]
        )
        # filename = "audio2/"+audio_files[_value_item]
        try:
            print(audio_files[_value_item], model_voice_path)
        except:
            pass
        info_, (sample_, audio_output_) = vc.vc_single_dont_save(
            sid=0,
            input_audio_path0=filename,  # f"audio2/{filename}",
            input_audio_path1=filename,  # f"audio2/{filename}",
            f0_up_key=transpose,  # transpose for m to f and reverse 0 12
            f0_file=None,
            f0_method=f0method,
            file_index=file_index,  # dir pwd?
            file_index2=file_index2,
            # file_big_npy1,
            index_rate=index_rate_,
            filter_radius=int(3),
            resample_sr=int(0),
            rms_mix_rate=float(0.25),
            protect=float(0.33),
            crepe_hop_length=crepe_hop_length_,
            f0_autotune=f0_autotune,
            f0_min=50,
            note_min=50,
            f0_max=1100,
            note_max=1100,
        )

        sf.write(
            file=filename,  # f"audio2/{filename}",
            samplerate=sample_,
            data=audio_output_,
        )


def cast_to_device(tensor, device):
    try:
        return tensor.to(device)
    except Exception as e:
        print(e)
        return tensor


def __bark__(text, voice_preset):
    os.makedirs(os.path.join(now_dir, "tts"), exist_ok=True)
    from transformers import AutoProcessor, BarkModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if "cpu" in device else torch.float16
    bark_processor = AutoProcessor.from_pretrained(
        "suno/bark",
        cache_dir=os.path.join(now_dir, "tts", "suno/bark"),
        torch_dtype=dtype,
    )
    bark_model = BarkModel.from_pretrained(
        "suno/bark",
        cache_dir=os.path.join(now_dir, "tts", "suno/bark"),
        torch_dtype=dtype,
    ).to(device)
    # bark_model.enable_cpu_offload()
    inputs = bark_processor(text=[text], return_tensors="pt", voice_preset=voice_preset)
    tensor_dict = {
        k: cast_to_device(v, device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }
    speech_values = bark_model.generate(**tensor_dict, do_sample=True)
    sampling_rate = bark_model.generation_config.sample_rate
    speech = speech_values.cpu().numpy().squeeze()
    return speech, sampling_rate


def use_tts(
    tts_text,
    tts_voice,
    model_path,
    index_path,
    transpose,
    f0_method,
    index_rate,
    crepe_hop_length,
    f0_autotune,
    tts_method,
):
    if tts_voice == None:
        return

    filename = os.path.join(now_dir, "assets", "audios", "audio-outputs", "converted_tts.wav")
    if "SET_LIMIT" == os.getenv("DEMO"):
        if len(tts_text) > 60:
            tts_text = tts_text[:60]
            print("DEMO; limit to 60 characters")

    language = tts_voice[:2]
    if tts_method == "Edge-tts":
        try:
            # nest_asyncio.apply() # gradio;not
            asyncio.run(
                edge_tts.Communicate(
                    tts_text, "-".join(tts_voice.split("-")[:-1])
                ).save(filename)
            )
        except:
            try:
                tts = gTTS(tts_text, lang=language)
                tts.save(filename)
                tts.save
                print(
                    f"No audio was received. Please change the tts voice for {tts_voice}. USING gTTS."
                )
            except:
                tts = gTTS("a", lang=language)
                tts.save(filename)
                print("Error: Audio will be replaced.")

        os.system("copy assets\\audios\\audio-outputs\\converted_tts.wav assets\\audios\\audio-outputs\\real_tts.wav")

        custom_voice(
            ["converted_tts"],  # filter indices
            ["assets/audios/audio-outputs/converted_tts.wav"],  # all audio files
            model_voice_path=model_path,
            transpose=transpose,
            f0method=f0_method,
            index_rate_=index_rate,
            crepe_hop_length_=crepe_hop_length,
            f0_autotune=f0_autotune,
            file_index="",
            file_index2=index_path,
        )
        return os.path.join(
            now_dir, "assets", "audios", "audio-outputs", "converted_tts.wav"
        ), os.path.join(now_dir, "assets", "audios", "audio-outputs", "real_tts.wav")
    elif tts_method == "Bark-tts":
        try:
            script = tts_text.replace("\n", " ").strip()
            sentences = sent_tokenize(script)
            print(sentences)
            silence = np.zeros(int(0.25 * SAMPLE_RATE))
            pieces = []
            nombre_archivo = os.path.join(now_dir, "assets", "audios", "audio-outputs", "bark_out.wav")
            for sentence in sentences:
                audio_array, _ = __bark__(sentence, tts_voice.split("-")[0])
                pieces += [audio_array, silence.copy()]

            sf.write(
                file=nombre_archivo, samplerate=SAMPLE_RATE, data=np.concatenate(pieces)
            )
            vc.get_vc(model_path)
            info_, (sample_, audio_output_) = vc.vc_single_dont_save(
                sid=0,
                input_audio_path0=os.path.join(
                    now_dir, "assets", "audios", "audio-outputs", "bark_out.wav"
                ),  # f"audio2/{filename}",
                input_audio_path1=os.path.join(
                    now_dir, "assets", "audios", "audio-outputs", "bark_out.wav"
                ),  # f"audio2/{filename}",
                f0_up_key=transpose,  # transpose for m to f and reverse 0 12
                f0_file=None,
                f0_method=f0_method,
                file_index="",  # dir pwd?
                file_index2=index_path,
                # file_big_npy1,
                index_rate=index_rate,
                filter_radius=int(3),
                resample_sr=int(0),
                rms_mix_rate=float(0.25),
                protect=float(0.33),
                crepe_hop_length=crepe_hop_length,
                f0_autotune=f0_autotune,
                f0_min=50,
                note_min=50,
                f0_max=1100,
                note_max=1100,
            )
            wavfile.write(
                os.path.join(now_dir, "assets", "audios", "audio-outputs", "converted_bark.wav"),
                rate=sample_,
                data=audio_output_,
            )
            return (
                os.path.join(now_dir, "assets", "audios", "audio-outputs", "converted_bark.wav"),
                nombre_archivo,
            )

        except Exception as e:
            print(f"{e}")
            return None, None
