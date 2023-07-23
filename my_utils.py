import ffmpeg
import numpy as np

# import praatio
# import praatio.praat_scripts
import os

# from os.path import join

# praatEXE = join('.',os.path.abspath(os.getcwd()) + r"\Praat.exe")


def load_audio(file, sr, DoFormant, Quefrency, Timbre):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        file_formanted = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        with open("formanting.txt", "r") as fvf:
            content = fvf.readlines()
            if "True" in content[0].split("\n")[0]:
                # print("true")
                DoFormant = True
                Quefrency, Timbre = content[1].split("\n")[0], content[2].split("\n")[0]

            else:
                # print("not true")
                DoFormant = False

        if DoFormant:
            # os.system(f"stftpitchshift -i {file} -q {Quefrency} -t {Timbre} -o {file_formanted}")
            # print('stftpitchshift -i "%s" -p 1.0 --rms -w 128 -v 8 -q %s -t %s -o "%s"' % (file, Quefrency, Timbre, file_formanted))
            print("formanting...")

            os.system(
                'stftpitchshift -i "%s" -q %s -t %s -o "%sFORMANTED"'
                % (file, Quefrency, Timbre, file_formanted)
            )
            print("formanted!")
            # filepraat = (os.path.abspath(os.getcwd()) + '\\' + file).replace('/','\\')
            # file_formantedpraat = ('"' + os.path.abspath(os.getcwd()) + '/' + 'formanted'.join(file_formanted) + '"').replace('/','\\')

            out, _ = (
                ffmpeg.input("%sFORMANTED%s" % (file_formanted, ".wav"), threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )

            os.remove("%sFORMANTED%s" % (file_formanted, ".wav"))
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

    return np.frombuffer(out, np.float32).flatten()
