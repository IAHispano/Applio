import os
import shutil
from random import shuffle
from rvc.configs.config import Config
from rvc.lib.user_config import get_logs_dir
import json

config = Config()
current_directory = os.getcwd()


def generate_config(sample_rate: int, model_path: str):
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", f"{sample_rate}.json"
    )
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))

    f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir))
    f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
    names = gt_wavs_files & feature_files & f0_files & f0nsf_files

    try:
        model_info_path = os.path.join(model_path, "model_info.json")
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
    except:
        embedder_name = "contentvec"

    def find_mute_base(sub_dir: str) -> str:
        candidates = [
            os.path.join(get_logs_dir(), sub_dir),
            os.path.join(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..")
                ),
                "logs",
                sub_dir,
            ),
        ]
        code_root = os.environ.get("APPLIO_CODE_ROOT") or os.environ.get("APPLIO_ROOT")
        if code_root:
            candidates.insert(1, os.path.join(code_root, "logs", sub_dir))
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return candidates[0]

    if embedder_name == "spin":
        mute_base_path = find_mute_base("mute_spin")
    elif embedder_name == "spin-v2":
        mute_base_path = find_mute_base("mute_spin-v2")
    else:
        mute_base_path = find_mute_base("mute")

    options = []
    sids = []
    for name in names:
        sid = name.split("_")[0]
        if sid not in sids:
            sids.append(sid)

        # Absolute paths support datasets and features on a different drive.
        rel_wav = os.path.abspath(f"{os.path.join(gt_wavs_dir, name)}.wav")
        rel_feat = os.path.abspath(f"{os.path.join(feature_dir, name)}.npy")
        rel_f0 = os.path.abspath(f"{os.path.join(f0_dir, name)}.wav.npy")
        rel_f0nsf = os.path.abspath(f"{os.path.join(f0nsf_dir, name)}.wav.npy")

        options.append(
            f"{rel_wav}|{rel_feat}|{rel_f0}|{rel_f0nsf}|{sid}".replace("\\", "/")
        )

    if include_mutes > 0:
        mute_audio_raw = os.path.join(
            mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"
        )
        mute_feature_raw = os.path.join(mute_base_path, "extracted", "mute.npy")
        mute_f0_raw = os.path.join(mute_base_path, "f0", "mute.wav.npy")
        mute_f0nsf_raw = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        if (
            os.path.exists(mute_audio_raw)
            and os.path.exists(mute_feature_raw)
            and os.path.exists(mute_f0_raw)
            and os.path.exists(mute_f0nsf_raw)
        ):
            mute_audio_path = os.path.abspath(mute_audio_raw).replace("\\", "/")
            mute_feature_path = os.path.abspath(mute_feature_raw).replace("\\", "/")
            mute_f0_path = os.path.abspath(mute_f0_raw).replace("\\", "/")
            mute_f0nsf_path = os.path.abspath(mute_f0nsf_raw).replace("\\", "/")

            # adding x files per sid
            for sid in sids * include_mutes:
                options.append(
                    f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
                )
        else:
            print(
                f"Warning: Mute files missing in '{mute_base_path}', skipping mute inclusion."
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
