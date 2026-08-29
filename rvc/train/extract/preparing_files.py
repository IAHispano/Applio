import os
import shutil
from random import shuffle
from rvc.configs.config import Config
import json

config = Config()
current_directory = os.getcwd()

# Filelist paths are stored relative to this. Absolute paths bake in the
# machine that ran the extraction, and paths relative to the working directory
# break whenever training is launched from elsewhere. Derived from this file
# rather than os.getcwd() because the extractor runs as a subprocess.
APPLICATION_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def relative_to_root(path: str) -> str:
    """
    Returns a path relative to the application root, with forward slashes.

    POSIX separators keep a preprocessed logs directory portable between
    platforms. A path outside the root is returned absolute.

    Args:
        path (str): The path to rewrite.
    """
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, APPLICATION_ROOT)
    except ValueError:
        # Different drive on Windows; no relative path exists.
        return absolute.replace(os.sep, "/")
    if relative.startswith(os.pardir):
        return absolute.replace(os.sep, "/")
    return relative.replace(os.sep, "/")


def _by_stem(directory: str) -> dict:
    """
    Maps every file in a directory to its slice name.

    Slice names never contain a dot, so 0_0_0.wav, 0_0_0.flac, 0_0_0.wav.npy
    and 0_0_0.npy all resolve to 0_0_0. Keeping the real filename is what lets
    the dataset be written in a format other than wav.

    Args:
        directory (str): Directory to list.
    """
    return {name.split(".")[0]: name for name in os.listdir(directory)}


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

    gt_wavs_files = _by_stem(gt_wavs_dir)
    feature_files = _by_stem(feature_dir)
    f0_files = _by_stem(f0_dir)
    f0nsf_files = _by_stem(f0nsf_dir)

    names = set(gt_wavs_files) & set(feature_files) & set(f0_files) & set(f0nsf_files)

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

        options.append(
            "|".join(
                (
                    relative_to_root(os.path.join(gt_wavs_dir, gt_wavs_files[name])),
                    relative_to_root(os.path.join(feature_dir, feature_files[name])),
                    relative_to_root(os.path.join(f0_dir, f0_files[name])),
                    relative_to_root(os.path.join(f0nsf_dir, f0nsf_files[name])),
                    sid,
                )
            )
        )

    if include_mutes > 0:
        mute_entry = "|".join(
            (
                relative_to_root(
                    os.path.join(
                        mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"
                    )
                ),
                relative_to_root(
                    os.path.join(mute_base_path, f"extracted", "mute.npy")
                ),
                relative_to_root(os.path.join(mute_base_path, "f0", "mute.wav.npy")),
                relative_to_root(
                    os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")
                ),
            )
        )

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(f"{mute_entry}|{sid}")

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
