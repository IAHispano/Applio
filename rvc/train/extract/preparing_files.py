import os
import shutil
from random import shuffle
from rvc.configs.config import Config
import json

config = Config()
current_directory = os.getcwd()


def generate_config(sample_rate: int, model_path: str):
    config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    # slices may be .wav or .flac, and the f0 files inherit that extension
    gt_wavs_files = {
        name.split(".")[0]: name
        for name in os.listdir(gt_wavs_dir)
        if name.lower().endswith((".wav", ".flac"))
    }
    feature_files = {
        name.split(".")[0]: name
        for name in os.listdir(feature_dir)
        if name.lower().endswith(".npy")
    }
    f0_files = {
        name.split(".")[0]: name
        for name in os.listdir(f0_dir)
        if name.lower().endswith(".npy")
    }
    f0nsf_files = {
        name.split(".")[0]: name
        for name in os.listdir(f0nsf_dir)
        if name.lower().endswith(".npy")
    }
    names = (
        gt_wavs_files.keys()
        & feature_files.keys()
        & f0_files.keys()
        & f0nsf_files.keys()
    )

    try:
        model_info_path = os.path.join(model_path, "model_info.json")
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
    except:
        embedder_name = "contentvec"

    if embedder_name == "spin":
        mute_base_path = os.path.join(current_directory, "logs", "mute_spin")
    elif embedder_name == "spin-v2":
        mute_base_path = os.path.join(current_directory, "logs", "mute_spin-v2")
    else:
        mute_base_path = os.path.join(current_directory, "logs", "mute")

    options = []
    sids = []
    for name in names:
        sid = name.split("_")[0]
        if sid not in sids:
            sids.append(sid)

        # Calculate relative pathing
        rel_wav = os.path.relpath(os.path.join(gt_wavs_dir, gt_wavs_files[name]))
        rel_feat = os.path.relpath(os.path.join(feature_dir, feature_files[name]))
        rel_f0 = os.path.relpath(os.path.join(f0_dir, f0_files[name]))
        rel_f0nsf = os.path.relpath(os.path.join(f0nsf_dir, f0nsf_files[name]))

        options.append(
            f"{rel_wav}|{rel_feat}|{rel_f0}|{rel_f0nsf}|{sid}".replace("\\", "/")
        )

    if include_mutes > 0:
        mute_audio_path = os.path.relpath(
            os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
        )
        mute_feature_path = os.path.relpath(
            os.path.join(mute_base_path, f"extracted", "mute.npy")
        )
        mute_f0_path = os.path.relpath(
            os.path.join(mute_base_path, "f0", "mute.wav.npy")
        )
        mute_f0nsf_path = os.path.relpath(
            os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")
        )

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
            )

    file_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(
        {
            "speakers_id": len(sids),
        }
    )
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    shuffle(options)

    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))
