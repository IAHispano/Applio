import sys

sys.path.append("..")
import os
now_dir = os.getcwd()

from dotenv import load_dotenv
from lib.modules.vc.modules import VC
from assets.configs.config import Config

load_dotenv()
config = Config()
vc = VC(config)

import hashlib
import urllib.parse
import soundfile as sf
from gtts import gTTS
import edge_tts
import scipy.io.wavfile as wavfile
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

import logging
import json
import ssl
from typing import Any, Dict, List, Optional
import asyncio
logging.getLogger('asyncio').setLevel(logging.FATAL)

import aiohttp
import certifi

VOICE_LIST = (
    "https://speech.platform.bing.com/consumer/speech/synthesize/"
    + "readaloud/voices/list?trustedclienttoken="
    + "6A5AA1D4EAFF4E9FB37E23D68491D6F4"
)
class Speech:
    LANGUAGE_ABBREVIATIONS = {'afrikaans': 'af', 'albanian': 'sq', 'arabic': 'ar', 'bengali': 'bn', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'chinese (simplified)': 'zh-CN', 'chinese (traditional)': 'zh-TW', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dutch': 'nl', 'english': 'en', 'estonian': 'et', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'german': 'de', 'greek': 'el', 'gujarati': 'gu', 'hebrew': 'iw', 'hindi': 'hi', 'hungarian': 'hu', 'icelandic': 'is', 'indonesian': 'id', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'khmer': 'km', 'korean': 'ko', 'latin': 'la', 'latvian': 'lv', 'malay': 'ms', 'malayalam': 'ml', 'marathi': 'mr', 'myanmar (burmese)': 'my', 'nepali': 'ne', 'norwegian': 'no', 'polish': 'pl', 'portuguese': 'pt', 'romanian': 'ro', 'russian': 'ru', 'serbian': 'sr', 'sinhala': 'si', 'slovak': 'sk', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr', 'ukrainian': 'uk', 'urdu': 'ur', 'vietnamese': 'vi'}

    def get_language_abbreviation(self, language):
        return self.LANGUAGE_ABBREVIATIONS.get(language, language)

    def __init__(self, output, language, proxy, handler):
        self.folder = output
        self.language = self.get_language_abbreviation(language.lower())
        self.proxy = proxy
        self.handler = handler

    async def create_speech_file(self, text, file_name):
        try:

            err = await self.download_if_not_exists(file_name, text)
            if err:
                return "", err

            return file_name, None
        except Exception as e:
            return "", e

    async def speak(self, text):
        try:
            file_name, err = await self.create_speech_file(text, self.folder)
            if err:
                raise err
            return file_name
        except Exception as e:
            return e


    async def download_if_not_exists(self, file_name, text):
        try:
            if not os.path.exists(file_name):
                dl_url = f"http://translate.google.com/translate_tts?ie=UTF-8&total=1&idx=0&textlen=32&client=tw-ob&q={urllib.parse.quote(text)}&tl={self.language}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(dl_url) as response:
                        if response.status != 200:
                            raise Exception(f"HTTP error: {response.status}")
                        content = await response.read()
                        with open(file_name, "wb") as output:
                            output.write(content)
                print("Downloaded file:", file_name)
            return None
        except Exception as e:
            print("Exception during download:", e)
            return e

def get_google_voice():
    return list(Speech.LANGUAGE_ABBREVIATIONS.keys())

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
    try:
        loop = asyncio.get_event_loop()
        voices_json = loop.run_until_complete(loop_main())
        #loop.close()
        voices = json.loads(voices_json)
        tts_voice = []
        for voice in voices:
            short_name = voice['ShortName']
            gender = voice['Gender']
            formatted_entry = f"{short_name}-{gender}"
            tts_voice.append(formatted_entry)
            # print(f"{short_name}-{gender}")
        return tts_voice
    except Exception as e:
        if isinstance(e, asyncio.exceptions.CancelledError):
            pass
        else:
            return []

set_google_voice = get_google_voice()
set_edge_voice = get_edge_voice()

def update_tts_methods_voice(select_value):
    # ["Edge-tts", "RVG-tts", "Bark-tts"]
    if select_value == "Edge-tts":
        return {"choices": set_edge_voice, "value": "", "__type__": "update"}
    elif select_value == "Google-tts":
        return {"choices": set_google_voice, "value": "", "__type__": "update"}


async def process_google_tts(language, tts_text, output_folder):
    try:
        script = tts_text.replace("\n", " ").strip()
        speech = Speech(output=output_folder, language=language, proxy=None, handler=None)
        google_out_filename = await speech.speak(script)
        print(google_out_filename)
        return google_out_filename

    except Exception as e:
        print(f"{e}")
        return None
    
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
    output_count = 1

    while True:
        converted_tts_filename = os.path.join(output_folder, f"tts_out_{output_count}.wav")
        
        if not os.path.exists(converted_tts_filename):
            break
        output_count += 1
    output_count = 1
    while True:
        google_out_filename = os.path.join(output_folder, f"google_out_{output_count}.wav")
        
        if not os.path.exists(google_out_filename):
            break
        output_count += 1
    
    if len(tts_text) > 3900 and tts_method == "Google-tts":
        tts_text = tts_text[:3900]
        print("Google Traductor; limit to 3900 characters")

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
            vc_output_filename = os.path.join(output_folder, f"converted_tts_{output_count}.wav")
            wavfile.write(
                vc_output_filename,
                rate=sample_,
                data=audio_output_,
            )

            return vc_output_filename,converted_tts_filename
        except Exception as e:
            print(f"{e}")
            return None, None

    elif tts_method == "Google-tts":
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            google_out_filename = loop.run_until_complete(process_google_tts(tts_voice, tts_text, google_out_filename))
            loop.close()

            vc.get_vc(model_path)
            info_, (sample_, audio_output_) = vc.vc_single_dont_save(
                sid=0,
                input_audio_path1=google_out_filename,  # f"audio2/{filename}",
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
            
            vc_output_filename = os.path.join(output_folder, f"converted_google_{output_count}.wav")
            wavfile.write(
                vc_output_filename,
                rate=sample_,
                data=audio_output_,
            )

            return vc_output_filename, google_out_filename

        except Exception as e:
            print(f"{e}")
            return None, None
