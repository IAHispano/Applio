import os
import glob
import torch
import numpy as np
from scipy.io.wavfile import read
from collections import OrderedDict
import matplotlib.pylab as plt

MATPLOTLIB_FLAG = False


def replace_keys_in_dict(d, old_key_part, new_key_part):
    """
    Replaces keys in a dictionary recursively.

    Args:
        d (dict or OrderedDict): The dictionary to update.
        old_key_part (str): The part of the key to replace.
        new_key_part (str): The new part of the key.
    """
    if isinstance(d, OrderedDict):
        updated_dict = OrderedDict()
    else:
        updated_dict = {}
    for key, value in d.items():
        if isinstance(key, str):
            new_key = key.replace(old_key_part, new_key_part)
        else:
            new_key = key
        if isinstance(value, dict):
            value = replace_keys_in_dict(value, old_key_part, new_key_part)
        updated_dict[new_key] = value
    return updated_dict


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    """
    Loads a checkpoint from a file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state from. Defaults to None.
        load_opt (int, optional): Whether to load the optimizer state. Defaults to 1.
    """
    assert os.path.isfile(checkpoint_path)
    checkpoint_old_dict = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_new_version_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"{os.path.splitext(os.path.basename(checkpoint_path))[0]}_new_version.pth",
    )

    torch.save(
        replace_keys_in_dict(
            replace_keys_in_dict(
                checkpoint_old_dict, ".weight_v", ".parametrizations.weight.original1"
            ),
            ".weight_g",
            ".parametrizations.weight.original0",
        ),
        checkpoint_new_version_path,
    )

    os.remove(checkpoint_path)
    os.rename(checkpoint_new_version_path, checkpoint_path)

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                print(
                    "shape-%s-mismatch|need-%s|get-%s",
                    k,
                    state_dict[k].shape,
                    saved_state_dict[k].shape,
                )
                raise KeyError
        except:
            print("%s is not in the checkpoint", k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    """
    Saves a checkpoint to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save the state of.
        learning_rate (float): The current learning rate.
        iteration (int): The current iteration.
        checkpoint_path (str): The path to save the checkpoint to.
    """
    print(f"Saved model '{checkpoint_path}' (epoch {iteration})")
    checkpoint_old_version_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"{os.path.splitext(os.path.basename(checkpoint_path))[0]}_old_version.pth",
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    torch.save(
        replace_keys_in_dict(
            replace_keys_in_dict(
                checkpoint, ".parametrizations.weight.original1", ".weight_v"
            ),
            ".parametrizations.weight.original0",
            ".weight_g",
        ),
        checkpoint_old_version_path,
    )
    os.remove(checkpoint_path)
    os.rename(checkpoint_old_version_path, checkpoint_path)


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050,
):
    """
    Summarizes training statistics and logs them to a TensorBoard writer.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        global_step (int): The current global step.
        scalars (dict, optional): Dictionary of scalar values to log. Defaults to {}.
        histograms (dict, optional): Dictionary of histogram values to log. Defaults to {}.
        images (dict, optional): Dictionary of image values to log. Defaults to {}.
        audios (dict, optional): Dictionary of audio values to log. Defaults to {}.
        audio_sample_rate (int, optional): Sampling rate of the audio data. Defaults to 22050.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Returns the path to the latest checkpoint file in a directory.

    Args:
        dir_path (str): The directory to search for checkpoints.
        regex (str, optional): The regular expression to match checkpoint files. Defaults to "G_*.pth".
    """
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def plot_spectrogram_to_numpy(spectrogram):
    """
    Plots a spectrogram to a NumPy array.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to plot.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    """
    Loads a WAV file into a PyTorch tensor.

    Args:
        full_path (str): The path to the WAV file.
    """
    sample_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sample_rate


def load_filepaths_and_text(filename, split="|"):
    """
    Loads filepaths and text from a file.

    Args:
        filename (str): The path to the file.
        split (str, optional): The delimiter used to split the lines. Defaults to "|".
    """
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


class HParams:
    """
    A class for storing and accessing hyperparameters.

    Attributes:
        **kwargs: Keyword arguments representing hyperparameters and their values.
    """

    def __init__(self, **kwargs):
        """
        Initializes an HParams object.

        Args:
            **kwargs: Keyword arguments representing hyperparameters and their values.
        """
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        """
        Returns a list of hyperparameter keys.
        """
        return self.__dict__.keys()

    def items(self):
        """
        Returns a list of (key, value) pairs for each hyperparameter.
        """
        return self.__dict__.items()

    def values(self):
        """
        Returns a list of hyperparameter values.
        """
        return self.__dict__.values()

    def __len__(self):
        """
        Returns the number of hyperparameters.
        """
        return len(self.__dict__)

    def __getitem__(self, key):
        """
        Gets the value of a hyperparameter.
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """
        Sets the value of a hyperparameter.
        """
        return setattr(self, key, value)

    def __contains__(self, key):
        """
        Checks if a hyperparameter key exists.
        """
        return key in self.__dict__

    def __repr__(self):
        """
        Returns a string representation of the HParams object.
        """
        return self.__dict__.__repr__()
