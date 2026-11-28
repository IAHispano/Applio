"""FCPE's fixed inference mel transform without diagnostic GPU synchronizations."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def local_decoder(self, y, threshold=0.05, mask=True):
    """The same nine-bin decoder, with all index construction on the device."""
    batch, frames, _ = y.shape
    cents = self.cent_table[None, None, :].expand(batch, frames, -1)
    confidence, peak = torch.max(y, dim=-1, keepdim=True)
    indices = (torch.arange(9, device=y.device) + peak - 4).clamp(0, self.out_dims - 1)
    selected_cents = torch.gather(cents, -1, indices)
    selected = torch.gather(y, -1, indices)
    result = torch.sum(selected_cents * selected, dim=-1, keepdim=True) / torch.sum(
        selected, dim=-1, keepdim=True
    )
    if mask:
        result = result * torch.where(confidence <= threshold, float("-inf"), 1.0)
    return result


class RealtimeMel(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        # FCPE constructs its window on the CPU before transferring it. Keep
        # exactly that window (rather than using CUDA's rounded coefficients).
        self.register_buffer(
            "window", torch.hann_window(original.win_size).to(original.mel_basis.device)
        )

    @torch.no_grad()
    def forward(self, y, key_shift=0, speed=1, center=False, no_cache_window=False):
        if key_shift != 0 or speed != 1 or no_cache_window:
            return self.original(y, key_shift, speed, center, no_cache_window)
        mel = self.original
        y = y.squeeze(-1)
        left = (mel.win_size - mel.hop_length) // 2
        right = max(
            (mel.win_size - mel.hop_length + 1) // 2, mel.win_size - y.shape[-1] - left
        )
        y = F.pad(
            y.unsqueeze(1),
            (left, right),
            mode="reflect" if right < y.shape[-1] else "constant",
        ).squeeze(1)
        spectrum = torch.stft(
            y,
            mel.n_fft,
            hop_length=mel.hop_length,
            win_length=mel.win_size,
            window=self.window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(spectrum.real.pow(2) + spectrum.imag.pow(2) + 1e-9)
        magnitude = (
            magnitude[:, :512, :]
            if mel.out_stft
            else torch.matmul(mel.mel_basis, magnitude)
        )
        # The upstream min/max checks only print warnings; removing them does
        # not clip, normalize, or otherwise change the samples or pitch output.
        return torch.log(torch.clamp(magnitude, min=mel.clip_val) * 1).transpose(-1, -2)
