"""Training-only mel critics for local spectral texture and pitch conditioning.

Pointwise reconstruction rewards averaged spectral envelopes. Overlapping low,
middle and high mel-band critics instead judge local frequency/time structure.
This is inspired by sub-frequency adversarial acoustic training in HiFiSinger
(arXiv:2009.01776), not a reproduction of its networks or published results.
The critics are omitted from exports and never run during voice conversion.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


class MelBandCritic(nn.Module):
    """Small spectral-normalized patch critic with pitch/voicing/energy controls."""

    def __init__(self):
        super().__init__()
        channels = (4, 16, 32, 64)
        self.layers = nn.ModuleList(
            spectral_norm(nn.Conv2d(a, b, (3, 5), stride=(2, 2), padding=(1, 2)))
            for a, b in zip(channels[:-1], channels[1:])
        )
        self.output = spectral_norm(nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, mel, controls):
        x = torch.cat((mel[:, None], controls.expand(-1, -1, mel.shape[1], -1)), 1)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        return self.output(x), features


class MelCritics(nn.Module):
    """Judge three overlapping bands without treating padded tails as recordings.

    Inputs are normalized mels. Each example is trimmed to its valid frame count
    before convolutions; generator feature matching cannot learn padded targets.
    Controls are aligned continuous F0 gated by voicing, voicing and log energy.
    Return the same score/feature structure used by the waveform GAN losses.
    """

    def __init__(self):
        super().__init__()
        self.bands = nn.ModuleList(MelBandCritic() for _ in range(3))

    def forward(self, mel, batch):
        channels = mel.shape[1]
        width = max(1, (channels + 1) // 2)
        starts = (0, (channels - width) // 2, channels - width)
        scores = []
        for index in range(mel.shape[0]):
            frames = int(batch["length"][index])
            if frames < 1 or frames > mel.shape[-1]:
                raise ValueError("Mel critic requires valid recording frame lengths")
            voiced = batch["voiced"][index : index + 1, :frames].detach()
            pitch = batch["f0"][index : index + 1, :frames].detach()
            energy = batch["energy"][index : index + 1, :frames].detach()
            controls = torch.stack(
                (
                    torch.log2(pitch.clamp_min(1) / 220) * voiced / 3,
                    voiced,
                    energy.clamp(-12, 2) / 6,
                ),
                dim=1,
            )[:, :, None]
            for start, critic in zip(starts, self.bands):
                scores.append(
                    critic(
                        mel[index : index + 1, start : start + width, :frames], controls
                    )
                )
        return scores


class RenderedAcousticCritics(nn.Module):
    """Judge acoustic mels and rendered PCM without owning the frozen vocoder.

    The trainer renders the predicted mel with gradients for generator updates
    and without gradients for critic updates. Only these small critics enter
    their optimizer/checkpoint; the shared vocoder remains a frozen dependency.
    Waveform examples are trimmed individually so padded tails cannot become
    discriminator cues or generator feature-matching targets.
    """

    def __init__(self, include_mel=False):
        super().__init__()
        from rvc.lib.algorithm.acoustic.vocoder import PeriodCritic

        self.mel = MelCritics() if include_mel else None
        # The frozen mel vocoder chooses waveform phase. Do not add a paired
        # real/imaginary STFT feature target that requires it to reproduce the
        # recording's phase; judge periodic waveform structure instead.
        self.waveform = nn.ModuleList(PeriodCritic(p) for p in (2, 3, 5, 7, 11))

    def rendered(self, audio, batch):
        scores = []
        for index in range(audio.shape[0]):
            samples = int(batch["waveform_mask"][index].sum())
            if not 1 <= samples <= audio.shape[-1]:
                raise ValueError("Rendered critics require valid waveform lengths")
            trimmed = audio[index : index + 1, :samples]
            scores.extend(critic(trimmed) for critic in self.waveform)
        return scores
