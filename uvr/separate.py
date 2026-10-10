"""Applio UVR separation entry point (engine utility, no UI dependencies).

Prints progress to stderr and a single JSON document to stdout with the
relative stem paths, following the backend job convention.
"""

import json
import logging
import os
import sys

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)

from uvr import models as catalog


def _engine():
    # Imported lazily so --list-models never pays the torch import cost.
    from uvr.separator import OUTPUT_MODELS_DIRNAME, separate_stems

    return OUTPUT_MODELS_DIRNAME, separate_stems


logging.basicConfig(level=logging.INFO, format="[uvr] %(message)s")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Applio UVR stem separator")
    parser.add_argument("--input-path", default=None, help="Input audio file")
    parser.add_argument(
        "--model", default=None, help="UVR model key or weight filename"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for stems (defaults to outputs dir)",
    )
    parser.add_argument(
        "--output-format",
        default="wav",
        choices=["wav", "flac", "mp3"],
        help="Stem file format",
    )
    parser.add_argument(
        "--device", default="auto", help="Compute device: 'auto', 'cpu' or a CUDA index"
    )
    parser.add_argument(
        "--single-stem", default="all", help="Write only this stem, or 'all'"
    )
    parser.add_argument(
        "--vr-aggression", type=int, default=None, help="VR extraction intensity 1-20"
    )
    parser.add_argument("--vr-window", type=int, default=None, help="VR window size")
    parser.add_argument("--vr-batch", type=int, default=None, help="VR batch size")
    parser.add_argument(
        "--vr-tta",
        action="store_true",
        help="VR test-time augmentation (slow, cleaner)",
    )
    parser.add_argument(
        "--vr-high-end",
        action="store_true",
        help="VR high-end frequency mirroring process",
    )
    parser.add_argument(
        "--vr-post-process",
        action="store_true",
        help="VR post-process: mute low-volume vocals left in the stem",
    )
    parser.add_argument(
        "--vr-post-threshold",
        type=float,
        default=None,
        help="VR post-process threshold (lower removes more residues)",
    )
    parser.add_argument(
        "--mdx-segment", type=int, default=None, help="MDX segment size"
    )
    parser.add_argument(
        "--mdx-overlap", type=float, default=None, help="MDX window overlap 0-0.99"
    )
    parser.add_argument("--mdx-batch", type=int, default=None, help="MDX batch size")
    parser.add_argument(
        "--mdx-hop", type=int, default=None, help="MDX hop length (stride)"
    )
    parser.add_argument(
        "--mdx-denoise",
        action="store_true",
        help="MDX denoising pass during separation",
    )
    parser.add_argument(
        "--mdxc-segment", type=int, default=None, help="MDXC segment size"
    )
    parser.add_argument(
        "--mdxc-overlap",
        type=int,
        default=None,
        help="MDXC overlapping prediction windows (>= 1)",
    )
    parser.add_argument("--mdxc-batch", type=int, default=None, help="MDXC batch size")
    parser.add_argument(
        "--roformer-chunk",
        type=float,
        default=None,
        help="Roformer chunk length in seconds (model config default)",
    )
    parser.add_argument(
        "--roformer-overlap",
        type=int,
        default=None,
        help="Roformer overlapping prediction windows (>= 1)",
    )
    parser.add_argument(
        "--roformer-batch", type=int, default=None, help="Roformer batch size"
    )
    parser.add_argument(
        "--demucs-segment",
        default=None,
        help="Demucs segment seconds, or 'Default' for the optimal size",
    )
    parser.add_argument(
        "--demucs-shifts", type=int, default=None, help="Demucs prediction shifts"
    )
    parser.add_argument(
        "--demucs-overlap",
        type=float,
        default=None,
        help="Demucs window overlap 0-0.99",
    )
    parser.add_argument(
        "--demucs-no-split",
        action="store_true",
        help="Disable Demucs chunk splitting (slower, more VRAM)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the model catalog as JSON and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print(
            "APPLIO_JSON:"
            + json.dumps({"models": [m.to_dict() for m in catalog.MODELS]})
        )
        return

    if not args.input_path or not args.model:
        parser.error(
            "--input-path and --model are required unless --list-models is used"
        )

    repo_root = os.getcwd()
    output_dir = args.output_dir or os.path.join(repo_root, "assets", "audios", "uvr")
    os.makedirs(output_dir, exist_ok=True)

    resolved = catalog.resolve(args.model)
    if resolved is None:
        parser.error(f"Unknown UVR model '{args.model}'")
    if resolved.arch == "vr":
        overrides = {
            "aggression": args.vr_aggression,
            "window_size": args.vr_window,
            "batch_size": args.vr_batch,
            "enable_tta": True if args.vr_tta else None,
            "high_end_process": True if args.vr_high_end else None,
            "enable_post_process": True if args.vr_post_process else None,
            "post_process_threshold": args.vr_post_threshold,
        }
    elif resolved.arch == "mdx":
        overrides = {
            "segment_size": args.mdx_segment,
            "overlap": args.mdx_overlap,
            "batch_size": args.mdx_batch,
            "hop_length": args.mdx_hop,
            "enable_denoise": True if args.mdx_denoise else None,
        }
    elif resolved.arch == "mdxc":
        if args.mdxc_overlap is not None and args.mdxc_overlap < 1:
            parser.error("--mdxc-overlap must be >= 1")
        overrides = {
            "segment_size": args.mdxc_segment,
            "overlap": args.mdxc_overlap,
            "batch_size": args.mdxc_batch,
        }
    elif resolved.arch == "demucs":
        demucs_segment = args.demucs_segment
        if demucs_segment is not None and demucs_segment != "Default":
            try:
                demucs_segment = int(demucs_segment)
            except (TypeError, ValueError):
                parser.error("--demucs-segment must be 'Default' or seconds")
            if demucs_segment < 1:
                parser.error("--demucs-segment must be >= 1 second")
        else:
            demucs_segment = None
        overrides = {
            "segment_size": demucs_segment,
            "shifts": args.demucs_shifts,
            "overlap": args.demucs_overlap,
            "segments_enabled": False if args.demucs_no_split else None,
        }
    elif resolved.arch == "roformer":
        if args.roformer_chunk is not None and args.roformer_chunk < 1:
            parser.error("--roformer-chunk must be >= 1 second")
        if args.roformer_overlap is not None and args.roformer_overlap < 1:
            parser.error("--roformer-overlap must be >= 1")
        overrides = {
            "chunk_seconds": args.roformer_chunk,
            "overlap": args.roformer_overlap,
            "batch_size": args.roformer_batch,
        }
    else:
        overrides = {}
    OUTPUT_MODELS_DIRNAME, separate_stems = _engine()
    models_dir = os.path.join(repo_root, OUTPUT_MODELS_DIRNAME, resolved.key)

    device = (args.device or "auto").lower()
    if device == "cpu":
        use_gpu = False
    elif device == "auto":
        use_gpu = True
    elif device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        use_gpu = True
    else:
        parser.error("--device must be 'auto', 'cpu' or a CUDA index")
    stems = separate_stems(
        input_path=args.input_path,
        model_key=args.model,
        models_dir=models_dir,
        output_dir=output_dir,
        output_format=args.output_format,
        use_gpu=use_gpu,
        single_stem=args.single_stem,
        arch_overrides=overrides,
        logger=logging.getLogger("uvr"),
    )
    rel = {
        name: os.path.relpath(path, repo_root).replace("\\", "/")
        for name, path in stems.items()
    }
    lowered = {name.lower(): path for name, path in rel.items()}
    want = (resolved.target or "").lower()
    primary = lowered.get(want, next(iter(rel.values()), None))
    print("APPLIO_JSON:" + json.dumps({"stems": rel, "outputFile": primary}))


if __name__ == "__main__":
    main()
