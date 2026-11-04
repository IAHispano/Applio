import os
import io
import sys
import onnx
import json
import types
import torch

try:
    import onnxslim
except:
    onnxslim = None
import warnings
import traceback

from typing import Optional
from datetime import datetime

sys.path.append(os.getcwd())

from rvc.lib.algorithm.onnx import Encoder
from rvc.lib.algorithm.synthesizers import Synthesizer

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")

FEATS_LENGTH = 200


def autocast(model):
    """
    This function is designed to allow a model—after being converted to the ONNX format—to accept Float-type inputs even when the model is in Half-precision.

    Args:
        model (torch.nn.Module): The PyTorch model to apply autocasting to.

    Returns:
        torch.nn.Module: The modified model with wrapped forward logic.
    """

    orig_forward = model.forward

    def _forward(*args, **kwargs):
        # Fetch the model's internal data type (e.g., float16 or float32)
        dtype = next(model.parameters()).dtype
        # Cast floating-point tensors dynamically, keeping integer or non-tensor inputs untouched
        return orig_forward(
            *[
                (
                    a.to(dtype)
                    if isinstance(a, torch.Tensor) and a.dtype.is_floating_point
                    else a
                )
                for a in args
            ],
            **kwargs,
        )

    model.forward = _forward
    return model


def prettify_date(date_str):
    if date_str is None:
        return "None"
    try:
        date_time_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        return date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        # Creation dates from community checkpoints vary; keep the raw value
        # instead of failing the whole inspection.
        try:
            return str(date_str)
        except Exception:
            return "None"


def onnx_infer(
    self: Synthesizer,
    phone: torch.Tensor,
    phone_lengths: torch.Tensor,
    sid: torch.Tensor = None,
    rate: Optional[torch.Tensor] = None,
    pitch: Optional[torch.Tensor] = None,
    nsff0: Optional[torch.Tensor] = None,
):
    """Alternative target inference route utilizing ONNX-exportable tracing steps."""

    g = self.emb_g(sid).unsqueeze(-1)
    m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

    # Sample latent vectors from source content priors using reparameterization
    z_p = (m_p + logs_p.exp() * torch.randn_like(m_p) * 0.66666) * x_mask

    # Floor and casting are explicitly typed for static tensor indexing compliance in ONNX runtimes
    head = torch.floor(z_p.size(2) * (1.0 - rate)).long()
    z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
    if self.use_f0:
        nsff0 = nsff0[:, head:]

    # Pass priors through Inverse Flow to reconstruct converted target latents
    z = self.flow(z_p, x_mask, g=g, reverse=True)
    return self.dec(z * x_mask, nsff0, g=g)


def strip_parametrizations(module: torch.nn.Module):
    """
    Remove all parametrizations (e.g., weight norm) from a module and log each removal.
    """
    for name, submodule in module.named_modules():
        if hasattr(submodule, "weight_g") and hasattr(submodule, "weight_v"):
            torch.nn.utils.remove_weight_norm(submodule)
        if hasattr(submodule, "parametrizations"):
            for pname, plist in list(submodule.parametrizations.items()):
                torch.nn.utils.parametrize.remove_parametrizations(
                    submodule, pname, leave_parametrized=True
                )


def onnx_exporter(input_path, output_path, is_half=False, device="cpu"):
    """
    Exports a trained RVC synthesizer
    checkpoint to an optimized ONNX model structure, attaching custom metadata.

    Args:
        input_path (str): File path to the PyTorch checkpoint (.pth).
        output_path (str): File path where the output .onnx model will be stored.
        is_half (bool, optional): If True, converts the model weights to Float16. Defaults to False.
        device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to "cpu".
    """

    # Load checkpoint safely avoiding arbitrary code execution hazards
    cpt = (
        torch.load(input_path, map_location="cpu", weights_only=True)
        if os.path.isfile(input_path)
        else None
    )
    # Adjust dynamic speaker/embedding layer configuration from the loaded weight matrix
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    model_name = cpt.get("model_name", "None")
    epochs = cpt.get("epoch", "None")
    steps = cpt.get("step", "None")
    sr = cpt.get("sr", "None")
    f0 = cpt.get("f0", "None")
    dataset_length = cpt.get("dataset_length", "None")
    vocoder = cpt.get("vocoder", "HiFi-GAN")
    creation_date = cpt.get("creation_date", "None")
    model_hash = cpt.get("model_hash", None)
    model_author = cpt.get("author", "None")
    embedder_model = cpt.get("embedder_model", "None")
    speakers_id = cpt.get("speakers_id", 0)
    version = cpt.get("version", "v1")

    # Set up basic parameters based on model generation version
    text_enc_hidden_dim = 768 if version == "v2" else 256
    tgt_sr = cpt["config"][-1]

    # Instantiate the correct Model Architecture depending on the model specification
    net_g = Synthesizer(
        *cpt["config"],
        use_f0=f0,
        text_enc_hidden_dim=text_enc_hidden_dim,
        vocoder=vocoder,
    )
    net_g.enc_p.encoder = Encoder(
        net_g.enc_p.encoder.hidden_channels,
        768,
        2,
        net_g.enc_p.encoder.n_layers,
        3,
        net_g.enc_p.encoder.drop.p,
    )

    # Re-route the forward pass to the dedicated ONNX inference method
    net_g.forward = types.MethodType(onnx_infer, net_g)
    net_g.load_state_dict(cpt["weight"], strict=False)
    # Switch model to evaluation state, transfer to target device, and assign float precision
    net_g.eval().to(device).to(torch.float16 if is_half else torch.float32)
    net_g.remove_weight_norm()
    strip_parametrizations(net_g)
    del net_g.enc_q
    # Apply global dynamic casting wrapper if working with float16 targets
    if is_half:
        net_g = autocast(net_g)

    # Initialize mock structural dummy input variables for PyTorch's tracing engine
    phone = torch.rand(1, FEATS_LENGTH, text_enc_hidden_dim).to(device)
    phone_length = torch.LongTensor([FEATS_LENGTH]).to(device)
    sid = torch.LongTensor([0]).to(device)
    rate = torch.FloatTensor([1.0]).to(device)

    args = [phone, phone_length, sid, rate]
    input_names = ["phone", "phone_lengths", "sid", "rate"]

    # Set dynamic axes dimensions to allow flexible batch (B) and sequence (T) runtime evaluation
    dynamic_axes = {
        "phone": {0: "B", 1: "T"},
        "phone_lengths": {0: "B"},
        "sid": {0: "B"},
        "rate": {0: "B"},
    }

    if (
        f0
    ):  # Append pitch-related parameters if the model variant relies on F0 fundamental frequency
        pitch = torch.randint(size=(1, FEATS_LENGTH), low=5, high=255).to(device)
        nsff0 = torch.rand(1, FEATS_LENGTH).to(device)

        args += [pitch, nsff0]
        input_names += ["pitch", "nsff0"]
        dynamic_axes.update(
            {
                "pitch": {0: "B", 1: "T"},
                "nsff0": {0: "B", 1: "T"},
            }
        )

    try:
        # Export model graph via byte stream pipeline to perform memory-contained optimization
        with io.BytesIO() as model:
            torch.onnx.export(
                net_g,
                tuple(args),
                model,
                do_constant_folding=True,  # Pre-calculate constant nodes at compile time
                opset_version=17,
                verbose=False,
                input_names=input_names,
                output_names=["audio"],
                dynamic_axes=dynamic_axes,
                dynamo=False,  # Use classic tracing engine
            )

            # Prune unused node structures, fuse blocks and squeeze overhead via onnxslim
            model = onnx.load_model_from_string(model.getvalue())
            if onnxslim is not None:
                model = onnxslim.slim(model)
            # Embed structured architecture details back directly into the ONNX model metadata block
            model.metadata_props.append(
                onnx.StringStringEntryProto(
                    key="model_info",
                    value=json.dumps(
                        {
                            "model_name": model_name,
                            "epoch": epochs,
                            "step": steps,
                            "version": version,
                            "sr": tgt_sr,
                            "f0": f0,
                            "dataset_length": dataset_length,
                            "vocoder": vocoder,
                            "creation_date": creation_date,
                            "model_hash": model_hash,
                            "author": model_author,
                            "embedder_model": embedder_model,
                            "speakers_id": speakers_id,
                        }
                    ),
                )
            )

        # Save optimized model graph structure out into persistent disk storage
        onnx.save(model, output_path)
        # Clear heavy model instances from RAM memory
        del net_g, model

        return f"Model conversion to ONNX format complete; the model path is: {output_path}!"
    except:
        print(traceback.format_exc())
        return f"An error occurred while converting the model to ONNX; check the Terminal for more information!"
