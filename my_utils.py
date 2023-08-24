import ffmpeg
import numpy as np

import os
import sys

import random

#import csv

platform_stft_mapping = {
    'linux': 'stftpitchshift',
    'darwin': 'stftpitchshift',
    'win32': 'stftpitchshift.exe',
}

stft = platform_stft_mapping.get(sys.platform)

def load_audio(file, sr, DoFormant=False, Quefrency=1.0, Timbre=1.0):
    converted = False
    try:
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        file_formanted = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        
        if DoFormant:
            numerator = round(random.uniform(1,4), 4)
            if not file.endswith(".wav"):
                if not os.path.isfile(f"{file_formanted}.wav"):
                    converted = True
                    converting = (
                        ffmpeg.input(file_formanted, threads = 0)
                        .output(f"{file_formanted}.wav")
                        .run(
                            cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                        )
                    )
            file_formanted = f"{file_formanted}.wav" if not file_formanted.endswith(".wav") else file_formanted
            print(f" · Formanting {file_formanted}...\n")
            
            command = (
                f'{stft} -i "{file_formanted}" -q "{Quefrency}" '
                f'-t "{Timbre}" -o "{file_formanted}FORMANTED_{str(numerator)}.wav"'
            )

            os.system(command)
            
            print(f" · Formanted {file_formanted}!\n")
            
            out, _ = (
                ffmpeg.input(f"{file_formanted}FORMANTED_{str(numerator)}.wav", threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )

            try: os.remove(f"{file_formanted}FORMANTED_{str(numerator)}.wav")
            except Exception as e: pass; print(f"couldn't remove formanted type of file due to {e}")
            
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
        try: os.remove(file_formanted)
        except Exception as e: pass; print(f"Couldn't remove converted type of file due to {e}")
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