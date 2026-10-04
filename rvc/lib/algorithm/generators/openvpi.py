import json
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# NSF-HiFiGAN and PC-NSF-HiFiGAN generators, for inference. Ported from
# SingingVocoders by Team OpenVPI (MIT): https://github.com/openvpi/SingingVocoders

LRELU_SLOPE = 0.1
# `architecture` of a rectified vocoder export holding an OpenVPI generator.
ARCHITECTURE = "openvpi_nsf_hifigan"

# The mel of the 44.1 kHz releases.
DEFAULT_MEL = {
    "sample_rate": 44100,
    "hop_length": 512,
    "n_fft": 2048,
    "win_length": 2048,
    "n_mels": 128,
    "mel_fmin": 40.0,
    "mel_fmax": 16000.0,
}
# OpenVPI's `config.json` keys -> the rectified flow config's.
CONFIG_MEL_KEYS = {
    "sampling_rate": "sample_rate",
    "hop_size": "hop_length",
    "n_fft": "n_fft",
    "win_size": "win_length",
    "num_mels": "n_mels",
    "fmin": "mel_fmin",
    "fmax": "mel_fmax",
}


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    """
    Residual block with pairs of dilated and plain convolutions.

    Args:
        channels (int): Number of channels.
        kernel_size (int): Kernel size of the convolutions.
        dilation (list[int]): Dilation rates.
    """

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels, kernel_size, padding=get_padding(kernel_size)
                )
                for _ in dilation
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            x = c2(F.leaky_relu(xt, LRELU_SLOPE)) + x
        return x


class ResBlock2(nn.Module):
    """
    Residual block with dilated convolutions.

    Args:
        channels (int): Number of channels.
        kernel_size (int): Kernel size of the convolutions.
        dilation (list[int]): Dilation rates.
    """

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )

    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


class SourceModuleHnNSF(nn.Module):
    """
    Sine plus harmonics merged to one excitation.

    Args:
        sample_rate (int): Sampling rate of the audio.
        harmonic_num (int): Number of harmonics above the fundamental.
        sine_amp (float, optional): Amplitude of the sines. Defaults to 0.1.
        noise_std (float, optional): Noise added to the voiced frames. Defaults to 0.003.
    """

    def __init__(self, sample_rate, harmonic_num, sine_amp=0.1, noise_std=0.003):
        super().__init__()
        self.sample_rate = sample_rate
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.l_linear = nn.Linear(self.dim, 1)

    def forward(self, f0, upp):
        f0 = f0.unsqueeze(-1)
        rad = f0 / self.sample_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1) * torch.arange(
            1, self.dim + 1, device=f0.device
        )
        start = torch.rand(1, 1, self.dim, device=f0.device)
        start[..., 0] = 0
        sines = torch.sin(2 * np.pi * (rad + start)) * self.sine_amp
        uv = F.interpolate(
            (f0 > 0).float().transpose(2, 1), scale_factor=upp, mode="nearest"
        ).transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sines)
        return torch.tanh(self.l_linear(sines * uv + noise))


class NSFHiFiGAN(nn.Module):
    """
    OpenVPI's NSF-HiFiGAN generator, plain and pitch-controllable (`mini_nsf`).

    Args:
        sample_rate (int): Sampling rate of the audio.
        num_mels (int): Number of mel bins.
        upsample_initial_channel (int): Number of channels before the upsampling.
        upsample_rates (list[int]): Upsampling rates.
        upsample_kernel_sizes (list[int]): Kernel sizes of the upsampling layers.
        resblock (str): Type of the residual blocks, "1" or "2".
        resblock_kernel_sizes (list[int]): Kernel sizes of the residual blocks.
        resblock_dilation_sizes (list[list[int]]): Dilation rates of the residual blocks.
        mini_nsf (bool): Whether the source joins after the second stage.
        harmonic_num (int, optional): Number of harmonics of the source. Defaults to 8.
        noise_sigma (float, optional): Noise added after the first convolution. Defaults to 0.0.
    """

    def __init__(
        self,
        sample_rate,
        num_mels,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        mini_nsf,
        harmonic_num=8,
        noise_sigma=0.0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.mini_nsf = mini_nsf
        self.noise_sigma = noise_sigma
        if mini_nsf:
            self.source_sr = sample_rate / int(np.prod(upsample_rates[2:]))
            self.upp = int(np.prod(upsample_rates[:2]))
        else:
            self.upp = int(np.prod(upsample_rates))
            self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)
            self.noise_convs = nn.ModuleList()

        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        block = ResBlock1 if resblock == "1" else ResBlock2
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(nn.ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(block(ch, kernel, dilation))
            if not mini_nsf:
                if i + 1 < len(upsample_rates):
                    stride = int(np.prod(upsample_rates[i + 1 :]))
                    self.noise_convs.append(
                        nn.Conv1d(
                            1,
                            ch,
                            kernel_size=stride * 2,
                            stride=stride,
                            padding=stride // 2,
                        )
                    )
                else:
                    self.noise_convs.append(nn.Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = nn.Conv1d(1, ch, 1)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

    def _fast_sine(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        return torch.sin(2 * np.pi * rad.reshape(f0.shape[0], 1, -1))

    def forward(self, mel, f0):
        """
        Args:
            mel (torch.Tensor): Log mel, shape (batch, num_mels, frames).
            f0 (torch.Tensor): Pitch in Hz, shape (batch, frames).
        """
        f0 = f0.float()
        if self.mini_nsf:
            source = self._fast_sine(f0)
        else:
            source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(mel)
        if self.noise_sigma > 0:
            x = x + self.noise_sigma * torch.randn_like(x)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            if not self.mini_nsf:
                x = x + self.noise_convs[i](source)
            elif i == 1:
                x = x + self.source_conv(source)
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


def generator_state(checkpoint):
    """
    Get the generator weights of an OpenVPI checkpoint, None when it is not one.

    Args:
        checkpoint (dict): The loaded checkpoint.
    """
    if isinstance(checkpoint.get("generator"), dict):
        return checkpoint["generator"]
    state = checkpoint.get("state_dict")
    if isinstance(state, dict) and "generator.conv_pre.bias" in state:
        return {
            k[len("generator.") :]: v
            for k, v in state.items()
            if k.startswith("generator.")
        }
    return None


def fold_weight_norm(state):
    """
    Fold the `weight_g` / `weight_v` pairs of a state dict into plain weights.

    Args:
        state (dict): Weights of a generator.
    """
    folded = {}
    for key, value in state.items():
        if key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            base = key[: -len("_v")]
            g = state[base + "_g"]
            norm = value.flatten(1).norm(dim=1).view(-1, *([1] * (value.dim() - 1)))
            folded[base] = value * (g / norm)
        else:
            folded[key] = value
    return folded


def openvpi_spec(path, state):
    """
    Get the NSFHiFiGAN arguments, the mel settings and the folded weights of an
    OpenVPI generator. What the weights cannot tell comes from a `config.json`
    beside the file, else from the 44.1 kHz release defaults.

    Args:
        path (str): Path to the checkpoint.
        state (dict): Weights of its generator.
    """
    config = {}
    config_path = os.path.join(os.path.dirname(path), "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    state = fold_weight_norm(state)
    mel = dict(DEFAULT_MEL)
    for source, target in CONFIG_MEL_KEYS.items():
        if source in config:
            mel[target] = config[source]

    count = sum(1 for k in state if k.startswith("ups.") and k.endswith(".weight"))
    kernels = [state[f"ups.{i}.weight"].shape[-1] for i in range(count)]
    rates = config.get("upsample_rates") or [k // 2 for k in kernels]
    resblock = "1" if "resblocks.0.convs2.0.weight" in state else "2"
    blocks = {int(k.split(".")[1]) for k in state if k.startswith("resblocks.")}
    per_stage = len(blocks) // count
    first_conv = "convs1.0.weight" if resblock == "1" else "convs.0.weight"
    resblock_kernels = [
        state[f"resblocks.{j}.{first_conv}"].shape[-1] for j in range(per_stage)
    ]
    default_dilation = [1, 3, 5] if resblock == "1" else [1, 3]
    dilations = config.get("resblock_dilation_sizes") or [default_dilation] * per_stage
    mini_nsf = "source_conv.weight" in state
    harmonics = 8 if mini_nsf else state["m_source.l_linear.weight"].shape[1] - 1

    hop = int(np.prod(rates))
    if hop != int(mel["hop_length"]):
        raise ValueError(
            f"{path} upsamples by {hop}, but its mel hop is {mel['hop_length']}."
        )
    if state["conv_pre.weight"].shape[1] != int(mel["n_mels"]):
        raise ValueError(
            f"{path} takes {state['conv_pre.weight'].shape[1]} mel bins, not {mel['n_mels']}."
        )

    hparams = dict(
        sample_rate=int(mel["sample_rate"]),
        num_mels=int(mel["n_mels"]),
        upsample_initial_channel=state["conv_pre.weight"].shape[0],
        upsample_rates=rates,
        upsample_kernel_sizes=kernels,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernels,
        resblock_dilation_sizes=dilations,
        mini_nsf=mini_nsf,
        harmonic_num=harmonics,
        noise_sigma=float(config.get("noise_sigma") or 0.0),
    )
    return hparams, mel, state
