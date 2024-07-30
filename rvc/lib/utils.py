import os, sys
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
from fairseq import checkpoint_utils
import wget

import logging

logging.getLogger("fairseq").setLevel(logging.WARNING)

now_dir = os.getcwd()
sys.path.append(now_dir)


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"Failed to load audio: {error}")

    return audio.flatten()


def format_title(title):
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec_base.pt"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese-hubert-base.pt"),
        "chinese-hubert-large": os.path.join(embedder_root, "chinese-hubert-large.pt"),
    }

    online_embedders = {
        "japanese-hubert-base": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
        "chinese-hubert-large": "https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt",
    }

    if embedder_model == "custom":
        model_path = custom_embedder
        if not custom_embedder and os.path.exists(custom_embedder):
            model_path = embedding_list["contentvec"]
    else:
        model_path = embedding_list[embedder_model]
        if embedder_model in online_embedders:
            model_path = embedding_list[embedder_model]
            url = online_embedders[embedder_model]
            print(f"\nDownloading {url} to {model_path}...")
            wget.download(url, out=model_path)
        else:
            model_path = embedding_list["contentvec"]

    models = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )

    # print(f"Embedding model {embedder_model} loaded successfully.")
    return models
