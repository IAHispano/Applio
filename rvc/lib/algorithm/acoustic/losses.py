"""Training-only constraints on acoustic mel detail; inference stays unchanged."""

import torch


def masked_mean(value, mask):
    """Normalize by valid elements, including broadcast feature channels."""
    mask = torch.broadcast_to(mask, value.shape).to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def mel_detail_loss(prediction, target, mask):
    """Match spectral contrast and transitions in physical log-mel units.

    Pointwise L1 can tolerate broad, smooth spectral envelopes. Differences at
    several band spacings also constrain local peaks/valleys without sharpening
    inference output or inventing harmonics. Temporal pairs require both frames
    to be valid, so padded tails never become training targets. This objective
    costs no extra encoder or vocoder passes and adds no inference parameters.
    It is a reconstruction constraint, not a perceptual quality guarantee.
    """
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.detach().float()
        spectral = []
        for spacing in (1, 2, 4):
            if prediction.shape[1] > spacing:
                difference = (prediction[:, spacing:] - prediction[:, :-spacing]) - (
                    target[:, spacing:] - target[:, :-spacing]
                )
                spectral.append(masked_mean(difference.abs(), mask))
        temporal = []
        for spacing in (1, 2):
            if prediction.shape[-1] > spacing:
                difference = (
                    prediction[..., spacing:] - prediction[..., :-spacing]
                ) - (target[..., spacing:] - target[..., :-spacing])
                valid = mask[..., spacing:] * mask[..., :-spacing]
                temporal.append(masked_mean(difference.abs(), valid))
        zero = prediction.sum() * 0
        frequency = sum(spectral, zero) / max(len(spectral), 1)
        transitions = sum(temporal, zero) / max(len(temporal), 1)
        return frequency + 0.2 * transitions
