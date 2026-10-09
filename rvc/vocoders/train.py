"""Train a mel vocoder: NSF-BigVGAN or NSF-HiFiGAN, against the v3 discriminator.

python rvc/vocoders/train.py --vocoder nsf-bigvgan --name my_vocoder --filelist logs/my_model/filelist.txt --epochs 100
"""

import argparse
import json
import os
import sys

# The repository root in place of this folder, whose module names are generic.
sys.path[0] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvc.vocoders.build import ARCHITECTURES
from rvc.vocoders.data import CHECKPOINT_MODES, run_dir
from rvc.vocoders.trainer import main


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vocoder", required=True, choices=ARCHITECTURES, help="Architecture to train.")
    parser.add_argument("--name", required=True, help="Run name; everything is written to logs/<name>/vocoder.")
    parser.add_argument("--filelist", required=True,
                        help="Rows of 'audio|f0.npy', or an RVC filelist.txt (audio|features|f0|f0_voiced|speaker).")
    parser.add_argument("--data-root", default=None,
                        help="Folder the filelist's relative paths start from. Default: the working directory.")
    parser.add_argument("--config", default=None,
                        help="Config copied to the run's config.json on the first run. "
                             "Default: rvc/vocoders/configs/<vocoder>.json.")
    parser.add_argument("--epochs", type=int, required=True, help="Total epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--save-every", type=int, default=1, help="Save every N epochs.")
    parser.add_argument("--gpu", default="0", help="GPU ids joined by '-', e.g. 0-1.")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--checkpoints", choices=CHECKPOINT_MODES, default="latest",
                        help="Training checkpoints kept; exports are always kept.")
    parser.add_argument("--fresh", action="store_true", help="Ignore the run's checkpoints instead of resuming.")
    parser.add_argument("--pretrained-g", default="",
                        help="Generator to start from: a checkpoint or export of this trainer or of SingingVocoders.")
    parser.add_argument("--pretrained-d", default="", help="Discriminator checkpoint of this trainer to start from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spec = {
        "model_name": args.name,
        "architecture": args.vocoder,
        "filelist": os.path.abspath(args.filelist),
        "data_root": os.path.abspath(args.data_root) if args.data_root else os.getcwd(),
        "config": os.path.abspath(args.config) if args.config else None,
        "total_epochs": args.epochs,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "gpu": args.gpu,
        "precision": args.precision,
        "checkpoints": args.checkpoints,
        "fresh": args.fresh,
        "pretrained_g": args.pretrained_g,
        "pretrained_d": args.pretrained_d,
    }
    os.makedirs(run_dir(args.name), exist_ok=True)
    spec_path = os.path.join(run_dir(args.name), "spec.json")
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=4)
    main(spec_path)
