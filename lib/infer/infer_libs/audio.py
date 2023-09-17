import librosa
import numpy as np
import av
from io import BytesIO
import ffmpeg
import os
import sys

import random
from lib.infer.infer_libs.csvutil import CSVutil
#import csv

platform_stft_mapping = {
    'linux': 'stftpitchshift',
    'darwin': 'stftpitchshift',
    'win32': 'stftpitchshift.exe',
}

stft = platform_stft_mapping.get(sys.platform)

def wav2(i, o, format):
    inp = av.open(i, 'rb')
    if format == "m4a": format = "mp4"
    out = av.open(o, 'wb', format=format)
    if format == "ogg": format = "libvorbis"
    if format == "mp4": format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame): out.mux(p)

    for p in ostream.encode(None): out.mux(p)

    out.close()
    inp.close()

def audio2(i, o, format, sr):
    inp = av.open(i, 'rb')
    out = av.open(o, 'wb', format=format)
    if format == "ogg": format = "libvorbis"
    if format == "f32le": format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame): out.mux(p)

    out.close()
    inp.close()

def load_audion(file, sr):
    try:
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        with open(file, "rb") as f:
            with BytesIO() as out:
                audio2(f, out, "f32le", sr)
                return np.frombuffer(out.getvalue(), np.float32).flatten()

    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file[0], target_sr=16000)

    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")




def load_audio(file, sr, DoFormant=False, Quefrency=1.0, Timbre=1.0):
    converted = False
    DoFormant, Quefrency, Timbre = CSVutil("lib/csvdb/formanting.csv", "r", "formanting")
    try:
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        file_formanted = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

        if (
            lambda DoFormant: True
            if DoFormant.lower() == "true"
            else (False if DoFormant.lower() == "false" else DoFormant)
        )(DoFormant):
            numerator = round(random.uniform(1, 4), 4)

            if not file.endswith(".wav"):
                if not os.path.isfile(f"{file_formanted}.wav"):
                    converted = True
                    converting = (
                        ffmpeg.input(file_formanted, threads=0)
                        .output(f"{file_formanted}.wav")
                        .run(
                            cmd=["ffmpeg", "-nostdin"],
                            capture_stdout=True,
                            capture_stderr=True,
                        )
                    )
                else:
                    pass

            file_formanted = (
                f"{file_formanted}.wav"
                if not file_formanted.endswith(".wav")
                else file_formanted
            )

            print(f" · Formanting {file_formanted}...\n")

            os.system(
                '%s -i "%s" -q "%s" -t "%s" -o "%sFORMANTED_%s.wav"'
                % (
                    stft,
                    file_formanted,
                    Quefrency,
                    Timbre,
                    file_formanted,
                    str(numerator),
                )
            )

            print(f" · Formanted {file_formanted}!\n")

            out, _ = (
                ffmpeg.input(
                    "%sFORMANTED_%s.wav" % (file_formanted, str(numerator)), threads=0
                )
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )

            try:
                os.remove("%sFORMANTED_%s.wav" % (file_formanted, str(numerator)))
            except Exception:
                pass
                print("couldn't remove formanted type of file")

        else:
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    if converted:
        try:
            os.remove(file_formanted)
        except Exception:
            pass
            print("couldn't remove converted type of file")
        converted = False

    return np.frombuffer(out, np.float32).flatten()


def check_audio_duration(file):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

        probe = ffmpeg.probe(file)

        duration = float(probe['streams'][0]['duration'])

        if duration < 0.76:
            print(
                f"\n------------\n"
                f"Audio file, {file.split('/')[-1]}, under ~0.76s detected - file is too short. Target at least 1-2s for best results."
                f"\n------------\n\n"
            )
            return False

        return True
    except Exception as e:
        raise RuntimeError(f"Failed to check audio duration: {e}")