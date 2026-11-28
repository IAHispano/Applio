"""Convert any HSVC checkpoint (pretrained or fine-tuned) to another target sample rate and/or precision.

Because HSVC operates at constant 100 Hz / 200 Hz frame rates and constant 23.4375 Hz STFT
bin spacing, a single 48 kHz checkpoint can be converted losslessly to 44.1 kHz, 40 kHz,
32 kHz, or 24 kHz (and optionally quantized to compact ~39 MB FP16) without retraining.

CLI examples:
    # Convert 48 kHz pretrained to 40 kHz FP16
    python -m backend.lib.tools.hsvc_convert_sr \
        --input models/pretraineds/hsvc/hsvc_base_48k.pth \
        --output hsvc_base_40k.pth \
        --target-sr 40000 --fp16

    # Batch-export 32k, 40k, 44.1k, and 48k pretrained files into a directory
    python -m backend.lib.tools.hsvc_convert_sr \
        --input models/pretraineds/hsvc/hsvc_base_48k.pth \
        --out-dir logs/pretrain/hf_upload_hsvc \
        --rates 32000 40000 44100 48000 --fp16
"""

import argparse
import os
from typing import List, Optional

import torch

from backend.architectures.hsvc.model import HSVCGenerator, is_hsvc_checkpoint


def convert_hsvc_checkpoint(
    input_path: str,
    output_path: str,
    target_sr: int,
    fp16: bool = False,
) -> str:
    """
    Convert an HSVC ``.pth`` checkpoint to ``target_sr`` and save it to ``output_path``.

    Preserves all learned weights, ``speaker_map``, and Applio checkpoint metadata while
    updating the model's STFT synthesis configuration to match ``target_sr``.

    Args:
        input_path (str): Path to the source HSVC ``.pth`` file.
        output_path (str): Destination ``.pth`` path.
        target_sr (int): Target sampling rate in Hz (16000 <= target_sr <= ref_sample_rate).
        fp16 (bool): If True, save floating-point weights in float16 (~39 MB vs ~78 MB FP32).

    Returns:
        str: Absolute path to the converted checkpoint file.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    pkg = torch.load(input_path, map_location="cpu", weights_only=True)
    if not is_hsvc_checkpoint(pkg):
        raise ValueError(f"'{input_path}' is not an HSVC checkpoint.")

    model, speaker_map = HSVCGenerator.from_checkpoint_dict(
        pkg, device="cpu", target_sr=int(target_sr)
    )

    preserve_keys = (
        "model_name",
        "epoch",
        "step",
        "creation_date",
        "dataset_length",
        "author",
        "info",
    )
    metadata = {k: pkg[k] for k in preserve_keys if k in pkg}

    out_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    model.export_package(out_abs, speaker_map=speaker_map, fp16=fp16, metadata=metadata)
    return out_abs


def batch_convert_hsvc_checkpoint(
    input_path: str,
    out_dir: str,
    rates: List[int],
    prefix: Optional[str] = None,
    fp16: bool = False,
) -> List[str]:
    """
    Export multiple sample-rate variants of an HSVC checkpoint into ``out_dir``.

    Args:
        input_path (str): Source HSVC ``.pth`` checkpoint path.
        out_dir (str): Output directory for generated ``.pth`` files.
        rates (List[int]): Target sample rates in Hz (e.g. ``[32000, 40000, 44100, 48000]``).
        prefix (Optional[str]): Filename prefix (defaults to base name stripped of ``_48k``).
        fp16 (bool): If True, export compact float16 weights.

    Returns:
        List[str]: Paths to all generated checkpoint files.
    """
    if prefix is None:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        for suf in ("_48k", "_44k", "_44.1k", "_40k", "_32k", "_24k"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        prefix = stem

    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    for sr in rates:
        sr_tag = "44.1k" if int(sr) == 44100 else f"{int(sr) // 1000}k"
        out_path = os.path.join(out_dir, f"{prefix}_{sr_tag}.pth")
        saved = convert_hsvc_checkpoint(input_path, out_path, target_sr=int(sr), fp16=fp16)
        size_mb = os.path.getsize(saved) / 1e6
        print(f"[HSVC Convert] {sr} Hz ({'FP16' if fp16 else 'FP32'}, {size_mb:.1f} MB) -> {saved}")
        outputs.append(saved)
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Convert an HSVC checkpoint to another sample rate and/or FP16 precision."
    )
    parser.add_argument("--input", required=True, help="Path to source HSVC .pth checkpoint.")
    parser.add_argument("--output", default="", help="Output .pth file path (single-rate mode).")
    parser.add_argument("--target-sr", type=int, default=48000, help="Target sample rate in Hz (single-rate mode).")
    parser.add_argument("--out-dir", default="", help="Output directory (batch multi-rate mode).")
    parser.add_argument(
        "--rates",
        type=int,
        nargs="+",
        default=[32000, 40000, 44100, 48000],
        help="Target sample rates in Hz for --out-dir batch mode.",
    )
    parser.add_argument("--prefix", default=None, help="Optional filename prefix for --out-dir mode.")
    parser.add_argument("--fp16", action="store_true", help="Export compact float16 weights (~39 MB).")
    args = parser.parse_args()

    if args.out_dir:
        batch_convert_hsvc_checkpoint(
            args.input, args.out_dir, rates=args.rates, prefix=args.prefix, fp16=args.fp16
        )
    else:
        if not args.output:
            parser.error("Provide either --output (single-rate) or --out-dir (batch multi-rate).")
        saved = convert_hsvc_checkpoint(args.input, args.output, target_sr=args.target_sr, fp16=args.fp16)
        print(f"[HSVC Convert] Saved {args.target_sr} Hz ({'FP16' if args.fp16 else 'FP32'}) -> {saved}")


if __name__ == "__main__":
    main()
