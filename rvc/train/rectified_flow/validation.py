import torch

from rvc.lib.algorithm.rectified_flow.features import denormalize_mel, normalize_mel
from rvc.train.utils import plot_spectrogram_to_numpy, summarize

# Sampling steps of the validation audio
PREVIEW_STEPS = 16
# Where in the trained time range the validation loss is taken
EVAL_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def evaluate(hps, net_flow, ema, eval_loader, writer, device, step, amp_dtype):
    """
    Logs the flow loss of the held out clips through the averaged weights, from
    the same noise and at the same times on every call.

    Args:
        hps (dict): Hyperparameters.
        net_flow (RectifiedFlow): The flow model.
        ema (WeightEMA): Average of the flow weights.
        eval_loader (DataLoader): Dataloader of the held out clips.
        writer (SummaryWriter): The TensorBoard writer.
        device (torch.device): The device of the model.
        step (int): The current step.
        amp_dtype (torch.dtype): Precision of the automatic mixed precision, None for none.
    """
    data_config = hps["config"]["data"]
    totals = torch.zeros(len(EVAL_FRACTIONS), device=device)
    aux_total = 0.0
    items = 0
    with ema.applied(net_flow):
        net_flow.eval()
        for batch_idx, (mel, inputs) in enumerate(eval_loader):
            inputs = inputs.to(device)
            mel = normalize_mel(mel.to(device), data_config) * inputs.mask
            generator = torch.Generator(device=device).manual_seed(batch_idx)
            noise = torch.randn(mel.shape, device=device, generator=generator)
            with torch.amp.autocast(
                device_type="cuda", enabled=amp_dtype is not None, dtype=amp_dtype
            ):
                losses, loss_aux = net_flow.validation_losses(
                    mel, inputs, noise, EVAL_FRACTIONS
                )
            totals += losses.float() * mel.shape[0]
            if loss_aux is not None:
                aux_total += loss_aux.item() * mel.shape[0]
            items += mel.shape[0]
        net_flow.train()

    totals /= max(1, items)
    scalar_dict = {"loss/val/flow": totals.mean()}
    for fraction, value in zip(EVAL_FRACTIONS, totals):
        scalar_dict[f"loss/val/flow_t{fraction:g}"] = value
    if net_flow.aux is not None:
        scalar_dict["loss/val/aux_mel"] = aux_total / max(1, items)
    summarize(writer=writer, global_step=step, scalars=scalar_dict)


def generate_validation(
    hps, net_flow, ema, references, vocoder, writer, step, log_real_mel=False
):
    """
    Logs the mel and the audio of the validation clips sampled through the
    averaged weights. Each clip is also rendered from the mel of the aux
    decoder alone and, when it was held out, from the flow started at its real
    mel on the same noise, which tells the error of the aux decoder from the
    error of the flow.

    Args:
        hps (dict): Hyperparameters.
        net_flow (RectifiedFlow): The flow model.
        ema (WeightEMA): Average of the flow weights.
        references (list): Name, mel, flow inputs and whether it was held out, for each validation clip.
        vocoder (torch.nn.Module): The vocoder that renders the validation audio.
        writer (SummaryWriter): The TensorBoard writer.
        step (int): The current step.
        log_real_mel (bool, optional): Whether to also render the real mel through the vocoder. Defaults to False.
    """
    data_config = hps["config"]["data"]
    image_dict = {}
    audio_dict = {}
    with ema.applied(net_flow), torch.no_grad():
        net_flow.eval()
        for name, ref_mel, inputs, held_out in references:
            real_mel = normalize_mel(ref_mel, data_config)
            noise = torch.randn_like(real_mel)
            mels = {"": net_flow.sample(inputs, steps=PREVIEW_STEPS, noise=noise)}
            if net_flow.aux is not None:
                mels["_aux"] = net_flow.aux_mel(inputs)
            if net_flow.starts_from_aux and held_out:
                mels["_from_real_mel"] = net_flow.sample(
                    inputs, steps=PREVIEW_STEPS, noise=noise, start_mel=real_mel
                )
            if net_flow.backbone.span_mlp is not None:
                mels["_1_step"] = net_flow.sample(
                    inputs, steps=1, method="mean", noise=noise
                )
            if log_real_mel:
                mels["_real_mel"] = real_mel

            image_dict[f"slice/{name}mel_org"] = plot_spectrogram_to_numpy(
                ref_mel[0].data.cpu().numpy()
            )
            image_dict[f"slice/{name}mel_gen"] = plot_spectrogram_to_numpy(
                denormalize_mel(mels[""], data_config)[0].float().data.cpu().numpy()
            )
            if vocoder is not None:
                for kind, mel in mels.items():
                    o = vocoder(mel.float(), inputs.f0)
                    audio_dict[f"gen/{name}audio{kind}_{step:07d}"] = o[0, :, :]
        net_flow.train()

    summarize(
        writer=writer,
        global_step=step,
        images=image_dict,
        audios=audio_dict,
        audio_sample_rate=data_config["sample_rate"],
    )
