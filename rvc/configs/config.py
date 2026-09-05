import torch
import json
import os

version_config_paths = [
    os.path.join("48000.json"),
    os.path.join("40000.json"),
    os.path.join("32000.json"),
    os.path.join("24000.json"),
]


#: Vocoders that ship settings of their own, and the folder they live in
#: beside the shared ``{sample_rate}.json``.
VOCODER_CONFIG_DIRS = {"RefineGAN2": "refinegan2"}


def vocoder_config(vocoder: str, sample_rate: int):
    """The ``model`` keys a vocoder ships for a sample rate, or ``{}``.

    Only the keys that vocoder owns; the shared ``{sample_rate}.json`` still
    carries everything the rest of the model reads.  These are loaded where the
    decoder is built rather than passed down from the training config, because
    the exported checkpoint's ``config`` is a fixed positional list with no room
    for them -- reading them in one place is what keeps training and inference
    building the same decoder.
    """

    folder = VOCODER_CONFIG_DIRS.get(vocoder)
    if folder is None:
        return {}
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), folder, f"{int(sample_rate)}.json"
    )
    if not os.path.exists(config_path):
        # Falling back to the generator's own defaults would render a trained
        # model through a decoder it was not trained for, and nothing would say
        # so: none of these settings appear in the weights.
        raise ValueError(
            f"{vocoder} has no settings for {sample_rate} Hz; expected "
            f"{config_path}."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f).get("model", {})


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_name = (
            torch.cuda.get_device_name(int(self.device.split(":")[-1]))
            if self.device.startswith("cuda")
            else None
        )
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self):
        configs = {}
        for config_file in version_config_paths:
            config_path = os.path.join("rvc", "configs", config_file)
            with open(config_path, "r", encoding="utf-8") as f:
                configs[config_file] = json.load(f)
        return configs

    def device_config(self):
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        else:
            self.device = "cpu"

        # Configuration for 6GB GPU memory
        x_pad, x_query, x_center, x_max = (1, 6, 38, 41)
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Configuration for 5GB GPU memory
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (
            1024**3
        )


def max_vram_gpu(gpu):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)
        return total_memory_gb
    else:
        return "8"


def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4
            )
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)")
    if len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
    else:
        gpu_info = "Unfortunately, there is no compatible GPU available to support your training."
    return gpu_info


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    else:
        return "-"
