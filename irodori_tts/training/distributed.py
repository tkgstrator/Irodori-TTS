"""Distributed setup, collective reductions and CUDA input prefetch."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def resolve_dist_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def setup_distributed(device_arg: str) -> tuple[int, int, int, bool, torch.device]:
    rank, world_size, local_rank = resolve_dist_env()
    distributed = world_size > 1
    if distributed:
        if not str(device_arg).startswith("cuda"):
            raise ValueError(
                f"WORLD_SIZE={world_size} detected, but --device={device_arg!r}. "
                "DDP multi-GPU training requires --device cuda."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE>1 detected, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    return rank, world_size, local_rank, distributed, device


def reduce_mean(value: torch.Tensor, world_size: int, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if not distributed:
        return reduced
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= float(world_size)
    return reduced


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _record_batch_stream(batch: dict, stream: torch.cuda.Stream) -> None:
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            value.record_stream(stream)


def cuda_prefetch_batches(loader, *, device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        yield from loader
        return

    stream = torch.cuda.Stream(device=device)
    iterator = iter(loader)
    next_batch = None

    def preload() -> None:
        nonlocal next_batch
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            next_batch = None
            return
        with torch.cuda.stream(stream):
            next_batch = _move_batch_to_device(cpu_batch, device)

    preload()
    while next_batch is not None:
        current_stream = torch.cuda.current_stream(device)
        current_stream.wait_stream(stream)
        batch = next_batch
        _record_batch_stream(batch, current_stream)
        preload()
        yield batch
