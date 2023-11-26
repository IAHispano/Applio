import os
import subprocess
from io import BytesIO
import traceback
import sys
import av
import librosa
import numpy as np
import ffmpeg

platform_stft_mapping = {
    "linux": "stftpitchshift",
    "darwin": "stftpitchshift",
    "win32": "stftpitchshift.exe",
}

stft = platform_stft_mapping.get(sys.platform)


def wav2(input_file, output_file, format):
    inp = av.open(input_file, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(output_file, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(input_file, output_file, format, sr):
    inp = av.open(input_file, "rb")
    out = av.open(output_file, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file, sr, DoFormant=False, Quefrency=1.0, Timbre=1.0):
    file = file.strip(' \n"')
    if not os.path.exists(file):
        raise RuntimeError(
            "Wrong audio path, that does not exist."
        )

    converted = False
    try:
        if not file.endswith(".wav"):
            converted = True
            subprocess.run(
                ["ffmpeg", "-nostdin", "-i", file, f"{file}.wav"],
                capture_output=True,
                text=True,
            )
            file = f"{file}.wav"
            print(f"File formatted to wav format: {file}\n")

        if DoFormant:
            command = (
                f'{stft} -i "{file}" -q "{Quefrency}" '
                f'-t "{Timbre}" -o "{file}_formatted.wav"'
            )
            subprocess.run(command, shell=True)
            file = f"{file}_FORMATTED.wav"
            print(f"Formatted {file}\n")

        with open(file, "rb") as f:
            with BytesIO() as out:
                audio2(f, out, "f32le", sr)
                audio_data = np.frombuffer(out.getvalue(), np.float32).flatten()

        if converted:
            try:
                os.remove(file)
            except Exception as error:
                print(f"Couldn't remove converted type of file due to {error}")
                error = None
            converted = False

        return audio_data

    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file[0], target_sr=16000)
    except Exception:
        raise RuntimeError(traceback.format_exc())


def check_audio_duration(file):
    try:
        file = file.strip(' \n"')
        probe = ffmpeg.probe(file)
        duration = float(probe["streams"][0]["duration"])

        if duration < 0.76:
            print(
                f"Audio file, {os.path.basename(file)}, under ~0.76s detected - file is too short. Target at least 1-2s for best results."
            )
            return False

        return True
    except Exception as error:
        raise RuntimeError(f"Failed to check audio duration: {error}")
