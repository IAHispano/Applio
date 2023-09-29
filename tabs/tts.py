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

import shutil
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

import json
import ssl
from typing import Any, Dict, List, Optional
import asyncio
import aiohttp
import certifi

VOICE_LIST = (
    "https://speech.platform.bing.com/consumer/speech/synthesize/"
    + "readaloud/voices/list?trustedclienttoken="
    + "6A5AA1D4EAFF4E9FB37E23D68491D6F4"
)
def get_bark_voice():
    mensaje = """
v2/en_speaker_0	English	Male
v2/en_speaker_1	English	Male
v2/en_speaker_2	English	Male
v2/en_speaker_3	English	Male
v2/en_speaker_4	English	Male
v2/en_speaker_5	English	Male
v2/en_speaker_6	English	Male
v2/en_speaker_7	English	Male
v2/en_speaker_8	English	Male
v2/en_speaker_9	English	Female
v2/zh_speaker_0	Chinese (Simplified)	Male
v2/zh_speaker_1	Chinese (Simplified)	Male
v2/zh_speaker_2	Chinese (Simplified)	Male
v2/zh_speaker_3	Chinese (Simplified)	Male
v2/zh_speaker_4	Chinese (Simplified)	Female
v2/zh_speaker_5	Chinese (Simplified)	Male
v2/zh_speaker_6	Chinese (Simplified)	Female
v2/zh_speaker_7	Chinese (Simplified)	Female
v2/zh_speaker_8	Chinese (Simplified)	Male
v2/zh_speaker_9	Chinese (Simplified)	Female
v2/fr_speaker_0	French	Male
v2/fr_speaker_1	French	Female
v2/fr_speaker_2	French	Female
v2/fr_speaker_3	French	Male
v2/fr_speaker_4	French	Male
v2/fr_speaker_5	French	Female
v2/fr_speaker_6	French	Male
v2/fr_speaker_7	French	Male
v2/fr_speaker_8	French	Male
v2/fr_speaker_9	French	Male
v2/de_speaker_0	German	Male
v2/de_speaker_1	German	Male
v2/de_speaker_2	German	Male
v2/de_speaker_3	German	Female
v2/de_speaker_4	German	Male
v2/de_speaker_5	German	Male
v2/de_speaker_6	German	Male
v2/de_speaker_7	German	Male
v2/de_speaker_8	German	Female
v2/de_speaker_9	German	Male
v2/hi_speaker_0	Hindi	Female
v2/hi_speaker_1	Hindi	Female
v2/hi_speaker_2	Hindi	Male
v2/hi_speaker_3	Hindi	Female
v2/hi_speaker_4	Hindi	Female
v2/hi_speaker_5	Hindi	Male
v2/hi_speaker_6	Hindi	Male
v2/hi_speaker_7	Hindi	Male
v2/hi_speaker_8	Hindi	Male
v2/hi_speaker_9	Hindi	Female
v2/it_speaker_0	Italian	Male
v2/it_speaker_1	Italian	Male
v2/it_speaker_2	Italian	Female
v2/it_speaker_3	Italian	Male
v2/it_speaker_4	Italian	Male
v2/it_speaker_5	Italian	Male
v2/it_speaker_6	Italian	Male
v2/it_speaker_7	Italian	Female
v2/it_speaker_8	Italian	Male
v2/it_speaker_9	Italian	Female
v2/ja_speaker_0	Japanese	Female
v2/ja_speaker_1	Japanese	Female
v2/ja_speaker_2	Japanese	Male
v2/ja_speaker_3	Japanese	Female
v2/ja_speaker_4	Japanese	Female
v2/ja_speaker_5	Japanese	Female
v2/ja_speaker_6	Japanese	Male
v2/ja_speaker_7	Japanese	Female
v2/ja_speaker_8	Japanese	Female
v2/ja_speaker_9	Japanese	Female
v2/ko_speaker_0	Korean	Female
v2/ko_speaker_1	Korean	Male
v2/ko_speaker_2	Korean	Male
v2/ko_speaker_3	Korean	Male
v2/ko_speaker_4	Korean	Male
v2/ko_speaker_5	Korean	Male
v2/ko_speaker_6	Korean	Male
v2/ko_speaker_7	Korean	Male
v2/ko_speaker_8	Korean	Male
v2/ko_speaker_9	Korean	Male
v2/pl_speaker_0	Polish	Male
v2/pl_speaker_1	Polish	Male
v2/pl_speaker_2	Polish	Male
v2/pl_speaker_3	Polish	Male
v2/pl_speaker_4	Polish	Female
v2/pl_speaker_5	Polish	Male
v2/pl_speaker_6	Polish	Female
v2/pl_speaker_7	Polish	Male
v2/pl_speaker_8	Polish	Male
v2/pl_speaker_9	Polish	Female
v2/pt_speaker_0	Portuguese	Male
v2/pt_speaker_1	Portuguese	Male
v2/pt_speaker_2	Portuguese	Male
v2/pt_speaker_3	Portuguese	Male
v2/pt_speaker_4	Portuguese	Male
v2/pt_speaker_5	Portuguese	Male
v2/pt_speaker_6	Portuguese	Male
v2/pt_speaker_7	Portuguese	Male
v2/pt_speaker_8	Portuguese	Male
v2/pt_speaker_9	Portuguese	Male
v2/ru_speaker_0	Russian	Male
v2/ru_speaker_1	Russian	Male
v2/ru_speaker_2	Russian	Male
v2/ru_speaker_3	Russian	Male
v2/ru_speaker_4	Russian	Male
v2/ru_speaker_5	Russian	Female
v2/ru_speaker_6	Russian	Female
v2/ru_speaker_7	Russian	Male
v2/ru_speaker_8	Russian	Male
v2/ru_speaker_9	Russian	Female
v2/es_speaker_0	Spanish	Male
v2/es_speaker_1	Spanish	Male
v2/es_speaker_2	Spanish	Male
v2/es_speaker_3	Spanish	Male
v2/es_speaker_4	Spanish	Male
v2/es_speaker_5	Spanish	Male
v2/es_speaker_6	Spanish	Male
v2/es_speaker_7	Spanish	Male
v2/es_speaker_8	Spanish	Female
v2/es_speaker_9	Spanish	Female
v2/tr_speaker_0	Turkish	Male
v2/tr_speaker_1	Turkish	Male
v2/tr_speaker_2	Turkish	Male
v2/tr_speaker_3	Turkish	Male
v2/tr_speaker_4	Turkish	Female
v2/tr_speaker_5	Turkish	Female
v2/tr_speaker_6	Turkish	Male
v2/tr_speaker_7	Turkish	Male
v2/tr_speaker_8	Turkish	Male
v2/tr_speaker_9	Turkish	Male
    """
    # Dividir el mensaje en líneas
    lineas = mensaje.split("\n")
    datos_deseados = []
    for linea in lineas:
        partes = linea.split("\t")
        if len(partes) == 3:
            clave, _, genero = partes
            datos_deseados.append(f"{clave}-{genero}")

    return datos_deseados

# ||-----------------------------------------------------------------------------------||
# ||                         Obtained from dependency edge_tts                         ||
# ||-----------------------------------------------------------------------------------||

async def list_voices(*, proxy: Optional[str] = None) -> Any:
    """
    List all available voices and their attributes.

    This pulls data from the URL used by Microsoft Edge to return a list of
    all available voices.

    Returns:
        dict: A dictionary of voice attributes.
    """
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(
            VOICE_LIST,
            headers={
                "Authority": "speech.platform.bing.com",
                "Sec-CH-UA": '" Not;A Brand";v="99", "Microsoft Edge";v="91", "Chromium";v="91"',
                "Sec-CH-UA-Mobile": "?0",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36 Edg/91.0.864.41",
                "Accept": "*/*",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Dest": "empty",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
            },
            proxy=proxy,
            ssl=ssl_ctx,
        ) as url:
            data = json.loads(await url.text())
    return data
async def create(custom_voices: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Creates a list of voices with all available voices and their attributes.
    """
    voices = await list_voices() if custom_voices is None else custom_voices
    voices = [
        {**voice, **{"Language": voice["Locale"].split("-")[0]}}
        for voice in voices
    ]
    simplified_voices = [
        {'ShortName': voice['ShortName'], 'Gender': voice['Gender']}
        for voice in voices
    ]
    return simplified_voices

async def loop_main():
    voices = await create()
    voices_json = json.dumps(voices)
    return voices_json

def get_edge_voice():
    loop = asyncio.get_event_loop()
    voices_json = loop.run_until_complete(loop_main())
    voices = json.loads(voices_json)
    tts_voice = []
    for voice in voices:
        short_name = voice['ShortName']
        gender = voice['Gender']
        formatted_entry = f"{short_name}-{gender}"
        tts_voice.append(formatted_entry)
       # print(f"{short_name}-{gender}")
    return tts_voice

set_bark_voice = get_bark_voice()
set_edge_voice = get_edge_voice()

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

    output_folder = "assets/audios/audio-outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_count = 1  # Contador para nombres de archivo únicos

    while True:
        converted_tts_filename = os.path.join(output_folder, f"tts_out_{output_count}.wav")
        bark_out_filename = os.path.join(output_folder, f"bark_out_{output_count}.wav")
        
        if not os.path.exists(converted_tts_filename) and not os.path.exists(bark_out_filename):
            break
        output_count += 1
    
    
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
                ).save(converted_tts_filename)
            )
        except:
            try:
                tts = gTTS(tts_text, lang=language)
                tts.save(converted_tts_filename)
                tts.save
                print(
                    f"No audio was received. Please change the tts voice for {tts_voice}. USING gTTS."
                )
            except:
                tts = gTTS("a", lang=language)
                tts.save(converted_tts_filename)
                print("Error: Audio will be replaced.")
        
        try:
            vc.get_vc(model_path)
            info_, (sample_, audio_output_) = vc.vc_single_dont_save(
                sid=0,
                input_audio_path1=converted_tts_filename,
                f0_up_key=transpose,
                f0_file=None,
                f0_method=f0_method,
                file_index="",
                file_index2=index_path,
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

            # Genera un nombre de archivo único para el archivo procesado por vc.vc_single_dont_save
            vc_output_filename = os.path.join(output_folder, f"converted_tts_{output_count}.wav")
            
            # Guarda el archivo de audio procesado por vc.vc_single_dont_save
            wavfile.write(
                vc_output_filename,
                rate=sample_,
                data=audio_output_,
            )

            return vc_output_filename,converted_tts_filename
        except Exception as e:
            print(f"{e}")
            return None, None

    elif tts_method == "Bark-tts":
        try:
            script = tts_text.replace("\n", " ").strip()
            sentences = sent_tokenize(script)
            print(sentences)
            silence = np.zeros(int(0.25 * SAMPLE_RATE))
            pieces = []
            for sentence in sentences:
                audio_array, _ = __bark__(sentence, tts_voice.split("-")[0])
                pieces += [audio_array, silence.copy()]

            sf.write(
                file=bark_out_filename, samplerate=SAMPLE_RATE, data=np.concatenate(pieces)
            )
            vc.get_vc(model_path)
            info_, (sample_, audio_output_) = vc.vc_single_dont_save(
                sid=0,
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
            
            vc_output_filename = os.path.join(output_folder, f"converted_bark_{output_count}.wav")
            
            # Guarda el archivo de audio procesado por vc.vc_single_dont_save
            wavfile.write(
                vc_output_filename,
                rate=sample_,
                data=audio_output_,
            )

            return vc_output_filename, bark_out_filename

        except Exception as e:
            print(f"{e}")
            return None, None
