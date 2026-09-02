"""Length-grouped batch sampling and the train/valid index split."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler

VALID_MIN_COUNT = 50
VALID_MAX_COUNT = 100


class LengthGroupedSampler(Sampler[int]):
    """
    Randomly samples the full dataset while grouping nearby-length examples into batches.

    Each epoch starts from a global random permutation. The permutation is split into
    random windows; only examples inside each window are sorted by length, then the
    resulting batches are shuffled again. This avoids a short-to-long curriculum while
    reducing per-batch padding.
    """

    def __init__(  # noqa: PLR0913
        self,
        lengths: torch.Tensor,
        *,
        batch_size: int,
        window_batches: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if lengths.ndim != 1:
            raise ValueError(f"lengths must be 1D, got shape={tuple(lengths.shape)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if window_batches <= 0:
            raise ValueError(f"window_batches must be > 0, got {window_batches}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if not (0 <= rank < num_replicas):
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.lengths = lengths.detach().to(device="cpu", dtype=torch.int64).contiguous()
        self.batch_size = int(batch_size)
        self.window_batches = int(window_batches)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_batch_size(self) -> int:
        return self.batch_size * self.num_replicas

    def _target_size(self) -> int:
        dataset_size = int(self.lengths.numel())
        global_batch_size = self._global_batch_size()
        if self.drop_last:
            return (dataset_size // global_batch_size) * global_batch_size
        if self.num_replicas == 1:
            return dataset_size
        return ((dataset_size + global_batch_size - 1) // global_batch_size) * global_batch_size

    def __len__(self) -> int:
        target_size = self._target_size()
        if self.num_replicas == 1:
            return target_size
        return target_size // self.num_replicas

    @staticmethod
    def _take_permutation(size: int, *, generator: torch.Generator) -> list[int]:
        if size <= 0:
            return []
        return torch.randperm(size, generator=generator, dtype=torch.int64).tolist()

    def __iter__(self):
        dataset_size = int(self.lengths.numel())
        if dataset_size <= 0:
            return iter(())
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        indices = self._take_permutation(dataset_size, generator=generator)
        target_size = self._target_size()
        if target_size <= 0:
            return iter(())
        if target_size < len(indices):
            indices = indices[:target_size]
        elif target_size > len(indices):
            repeats = target_size - len(indices)
            base_indices = list(indices)
            while repeats > 0:
                take = min(repeats, len(base_indices))
                indices.extend(base_indices[:take])
                repeats -= take

        global_batch_size = self._global_batch_size()
        window_size = max(global_batch_size, self.window_batches * global_batch_size)
        rank_start = self.rank * self.batch_size
        rank_end = rank_start + self.batch_size
        local_indices: list[int] = []

        for window_start in range(0, len(indices), window_size):
            window = indices[window_start : window_start + window_size]
            window.sort(key=lambda idx: int(self.lengths[idx]), reverse=True)
            batches = [
                window[i : i + global_batch_size]
                for i in range(0, len(window), global_batch_size)
                if len(window[i : i + global_batch_size]) > 0
            ]
            batch_order = self._take_permutation(len(batches), generator=generator)
            for batch_index in batch_order:
                batch = batches[batch_index]
                if self.num_replicas > 1 and len(batch) < global_batch_size:
                    continue
                if len(batch) > 1:
                    order = torch.randperm(
                        len(batch),
                        generator=generator,
                        dtype=torch.int64,
                    ).tolist()
                    batch = [batch[i] for i in order]
                local_indices.extend(batch[rank_start:rank_end])

        return iter(local_indices)


def split_train_valid_indices(
    *,
    num_samples: int,
    valid_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if valid_ratio <= 0.0:
        return torch.arange(num_samples, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    if num_samples < 2:
        raise ValueError(
            f"Validation split requires at least 2 samples in manifest, got {num_samples}."
        )

    valid_count = int(num_samples * valid_ratio)
    valid_count = max(VALID_MIN_COUNT, min(VALID_MAX_COUNT, valid_count))
    if valid_count >= num_samples:
        valid_count = num_samples - 1
    valid_count = max(1, valid_count)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    valid_indices = torch.sort(perm[:valid_count]).values
    train_indices = torch.sort(perm[valid_count:]).values
    if train_indices.numel() == 0 or valid_indices.numel() == 0:
        raise ValueError(
            "Failed to create non-empty train/valid split. "
            f"num_samples={num_samples} valid_ratio={valid_ratio}"
        )
    return train_indices, valid_indices
