import os
import shutil
from random import shuffle

from rvc.configs.config import Config

config = Config()
current_directory = os.getcwd()


def generate_config(rvc_version: str, sample_rate: int, model_path: str):
    config_path = os.path.join("rvc", "configs", rvc_version, f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)


def generate_filelist(
    pitch_guidance: bool, model_path: str, rvc_version: str, sample_rate: int
):
    gt_wavs_dir = f"{model_path}/sliced_audios"
    feature_dir = (
        f"{model_path}/v1_extracted"
        if rvc_version == "v1"
        else f"{model_path}/v2_extracted"
    )
    if pitch_guidance == True:
        f0_dir = f"{model_path}/f0"
        f0nsf_dir = f"{model_path}/f0_voiced"
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    elif pitch_guidance == False:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    options = []
    for name in names:
        if pitch_guidance == 1:
            options.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0"
            )
        else:
            options.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0")
    if pitch_guidance == 1:
        for _ in range(2):
            options.append(
                f"{current_directory}/logs/mute/sliced_audios/mute{sample_rate}.wav|{current_directory}/logs/mute/{rvc_version}_extracted/mute.npy|{current_directory}/logs/mute/f0/mute.wav.npy|{current_directory}/logs/mute/f0_voiced/mute.wav.npy|0"
            )
    else:
        for _ in range(2):
            options.append(
                f"{current_directory}/logs/mute/sliced_audios/mute{sample_rate}.wav|{current_directory}/logs/mute/{rvc_version}_extracted/mute.npy|0"
            )
    shuffle(options)
    with open(f"{model_path}/filelist.txt", "w") as f:
        f.write("\n".join(options))
