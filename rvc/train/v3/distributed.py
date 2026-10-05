"""Synchronous data parallel updates for method-based acoustic/GAN training.

Each rank owns a shard of the global sampled batch. Gradients are averaged once
per optimizer update after accumulation; only rank zero validates/writes files.
"""

import os

import torch
import torch.distributed as dist


class TrainingGroup:
    """Synchronize method-based acoustic/GAN training without wrapping model forwards.

    Each rank samples its part of the global batch. Accumulated gradients are
    averaged once per optimizer update; rank zero owns validation and file writes.
    NCCL uses device collectives; Gloo can stage CUDA gradients through CPU.
    """

    def __init__(self, device):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0")) if self.world_size > 1 else 0
        self.device = device
        self.active = self.world_size > 1
        if self.active and not dist.is_initialized():
            backend = (
                "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
            )
            # Gloo on Windows uses the ordinary TCP store, without libuv.
            os.environ.setdefault("USE_LIBUV", "0")
            dist.init_process_group(backend=backend, init_method="env://")
        self.cpu_collectives = self.active and dist.get_backend() != "nccl"

    def mean_gradients(self, parameters):
        if not self.active:
            return
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            gradient = parameter.grad
            value = gradient.cpu() if self.cpu_collectives else gradient
            dist.all_reduce(value)
            value.div_(self.world_size)
            if value.device != gradient.device:
                gradient.copy_(value.to(gradient.device))

    def any(self, flag):
        if not self.active:
            return bool(flag)
        value = torch.tensor(
            int(flag), device="cpu" if self.cpu_collectives else self.device
        )
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())

    def mean(self, value):
        if not self.active:
            return float(value)
        scalar = torch.tensor(
            float(value),
            device="cpu" if self.cpu_collectives else self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(scalar)
        return float(scalar / self.world_size)

    def gather(self, value):
        if not self.active:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def barrier(self):
        if self.active:
            dist.barrier()
