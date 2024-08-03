import math
import torch

import sys
import os

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.algorithm.commons import sequence_mask
from rvc.lib.algorithm.encoders.encoder import Encoder


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        embedding_dim,
        vocoder_type="hifigan",
        f0=True,
    ):
        super(TextEncoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.vocoder_type = vocoder_type
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.encoder = Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            vocoder_type=vocoder_type,
        )
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)
        if f0:
            self.emb_pitch = torch.nn.Embedding(256, hidden_channels)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor, lengths: torch.Tensor):
        if self.vocoder_type == "hifigan":
            if pitch is None:
                x = self.emb_phone(phone)
            else:
                x = self.emb_phone(phone) + self.emb_pitch(pitch)
            x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
            x = self.lrelu(x)
            x = torch.transpose(x, 1, -1)  # [b, h, t]
            x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
            x = self.encoder(x * x_mask, x_mask)
            stats = self.proj(x) * x_mask
            m, logs = torch.split(stats, self.out_channels, dim=1)

        elif self.vocoder_type in ["bigvgan", "bigvsan"]:
            if pitch is None:
                x = self.emb_phone(phone)
            else:
                x = self.emb_phone(phone) + self.emb_pitch(pitch)

            x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
            x = self.lrelu(x)
            x_mask = sequence_mask(lengths, x.size(1)).to(x.dtype)
            x = self.encoder(x, x_mask)
            x_mask = x_mask.unsqueeze(-1)
            stats = self.proj(x.transpose(1, 2)).transpose(1, 2) * x_mask
            stats = stats.transpose(1, 2)
            x_mask = x_mask.transpose(1, 2)
            m, logs = torch.split(stats, self.out_channels, dim=1)

        return m, logs, x_mask
