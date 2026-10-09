"""Multi-GPU launch for the rectified trainers: one process per GPU under DDP.

With a single GPU (or none) the trainer runs in the calling process with no
process group, so nothing here costs the single-GPU path anything.
"""

import os
import socket
import sys

import torch
import torch.distributed as dist

from rvc.vocoders.stop import stop_was_requested


def parse_gpus(value) -> list[int]:
    """``"0-1"`` -> ``[0, 1]``, the RVC trainer's format."""
    return [int(item) for item in str(value or "0").split("-") if item.strip()] or [0]


class Ranks:
    """This process's place in the run. ``world == 1`` is the plain trainer."""

    def __init__(self, rank: int, gpus: list[int]):
        self.rank = rank
        self.world = len(gpus)
        self.main = rank == 0
        self.device = (
            torch.device(f"cuda:{gpus[rank]}") if torch.cuda.is_available() else torch.device("cpu")
        )

    def setup(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if self.world > 1:
            dist.init_process_group(
                backend="gloo" if sys.platform == "win32" else "nccl",
                init_method="env://",
                world_size=self.world,
                rank=self.rank,
            )
            # Every rank starts from the same default seed; rank 0 keeps it,
            # so its draws match a single-GPU run's.
            torch.manual_seed(torch.initial_seed() + self.rank)

    def wrap(self, module, **kwargs):
        """``module`` under DDP, or itself on one GPU."""
        if self.world == 1:
            return module
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(module, device_ids=device_ids, **kwargs)

    def sampler(self, dataset):
        if self.world == 1:
            return None
        from torch.utils.data import DistributedSampler

        return DistributedSampler(
            dataset, num_replicas=self.world, rank=self.rank, shuffle=True, drop_last=True
        )

    def mean(self, value: torch.Tensor) -> torch.Tensor:
        """``value`` averaged over the ranks. Every rank must call it."""
        if self.world == 1:
            return value
        value = value.detach().clone()
        dist.all_reduce(value)
        return value / self.world

    def stop_requested(self) -> bool:
        """Whether any rank was asked to stop, agreed on by all of them, so no
        rank leaves another waiting in a collective. Every rank must call it."""
        if self.world == 1:
            return stop_was_requested()
        flag = torch.tensor(float(stop_was_requested()), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


def _run_rank(rank: int, target, spec_path: str, gpus: list[int]) -> None:
    target(Ranks(rank, gpus), spec_path)


def launch(target, spec_path: str, gpus: list[int]) -> None:
    """Run ``target(ranks, spec_path)`` once per GPU in ``gpus``."""
    if len(gpus) == 1 or not torch.cuda.is_available():
        target(Ranks(0, gpus[:1]), spec_path)
        return
    # An explicit IPv4 loopback and a port the OS says is free, as in the
    # RVC trainer.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.multiprocessing.spawn(_run_rank, args=(target, spec_path, gpus), nprocs=len(gpus), join=True)
