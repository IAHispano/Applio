import ffmpeg
import numpy as np

# import praatio
# import praatio.praat_scripts
import os
import sys

import random

import csv

platform_stft_mapping = {
    'linux': 'stftpitchshift',
    'darwin': 'stftpitchshift',
    'win32': 'stftpitchshift.exe',
}

stft = platform_stft_mapping.get(sys.platform)
# praatEXE = join('.',os.path.abspath(os.getcwd()) + r"\Praat.exe")

def CSVutil(file, rw, type, *args):
    if type == 'formanting':
        if rw == 'r':
            with open(file) as fileCSVread:
                csv_reader = list(csv.reader(fileCSVread))
                return (
                    csv_reader[0][0], csv_reader[0][1], csv_reader[0][2]
                ) if csv_reader is not None else (lambda: exec('raise ValueError("No data")'))()
        else:
            if args:
                doformnt = args[0]
            else:
                doformnt = False
            qfr = args[1] if len(args) > 1 else 1.0
            tmb = args[2] if len(args) > 2 else 1.0
            with open(file, rw, newline='') as fileCSVwrite:
                csv_writer = csv.writer(fileCSVwrite, delimiter=',')
                csv_writer.writerow([doformnt, qfr, tmb])
    elif type == 'stop':
        stop = args[0] if args else False
        with open(file, rw, newline='') as fileCSVwrite:
            csv_writer = csv.writer(fileCSVwrite, delimiter=',')
            csv_writer.writerow([stop])

def load_audio(file, sr, DoFormant, Quefrency, Timbre):
    converted = False
    DoFormant, Quefrency, Timbre = CSVutil('csvdb/formanting.csv', 'r', 'formanting')    
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        file_formanted = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        
        #print(f"dofor={bool(DoFormant)} timbr={Timbre} quef={Quefrency}\n")
        
        if (lambda DoFormant: True if DoFormant.lower() == 'true' else (False if DoFormant.lower() == 'false' else DoFormant))(DoFormant):
            numerator = round(random.uniform(1,4), 4)
            # os.system(f"stftpitchshift -i {file} -q {Quefrency} -t {Timbre} -o {file_formanted}")
            # print('stftpitchshift -i "%s" -p 1.0 --rms -w 128 -v 8 -q %s -t %s -o "%s"' % (file, Quefrency, Timbre, file_formanted))
            
            if not file.endswith(".wav"):
                
                if not os.path.isfile(f"{file_formanted}.wav"):
                    converted = True
                    #print(f"\nfile = {file}\n")
                    #print(f"\nfile_formanted = {file_formanted}\n")
                    converting = (
                        ffmpeg.input(file_formanted, threads = 0)
                        .output(f"{file_formanted}.wav")
                        .run(
                            cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                        )
                    )
                else:
                    pass
            
            
            
            file_formanted = f"{file_formanted}.wav" if not file_formanted.endswith(".wav") else file_formanted
            
            
            
            print(f" · Formanting {file_formanted}...\n")
            
            
            
            os.system(
                '%s -i "%s" -q "%s" -t "%s" -o "%sFORMANTED_%s.wav"'
                % (stft, file_formanted, Quefrency, Timbre, file_formanted, str(numerator))
            )
            
            
            
            print(f" · Formanted {file_formanted}!\n")
            
            
            
            # filepraat = (os.path.abspath(os.getcwd()) + '\\' + file).replace('/','\\')
            # file_formantedpraat = ('"' + os.path.abspath(os.getcwd()) + '/' + 'formanted'.join(file_formanted) + '"').replace('/','\\')
            #print("%sFORMANTED_%s.wav" % (file_formanted, str(numerator)))
            
            out, _ = (
                ffmpeg.input("%sFORMANTED_%s.wav" % (file_formanted, str(numerator)), threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )

            try: os.remove("%sFORMANTED_%s.wav" % (file_formanted, str(numerator)))
            except Exception: pass; print("couldn't remove formanted type of file")
            
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
        except Exception: pass; print("couldn't remove converted type of file")
        converted = False
    
    return np.frombuffer(out, np.float32).flatten()
