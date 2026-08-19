from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def load_three_domain_metadata(root: str | Path) -> dict:
    with (Path(root) / "metadata.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_item_text_embeddings(root: str | Path) -> torch.Tensor:
    path = Path(root) / "item_text_embeddings.pt"
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"Expected a rank-2 tensor in {path}")
    if tensor.shape[0] < 2 or not torch.all(tensor[0] == 0):
        raise ValueError("Item text row 0 must be the all-zero padding vector")
    return tensor.float()


class ThreeDomainBatchDataset(IterableDataset):
    """Streams compact pre-batched multi-domain ranking contexts from NPZ shards."""

    def __init__(
        self,
        root: str | Path,
        split: str | Sequence[str],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.root = Path(root)
        self.splits = (split,) if isinstance(split, str) else tuple(split)
        if not self.splits:
            raise ValueError("At least one split is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        files = [
            path
            for split in self.splits
            for path in sorted((self.root / split).glob("part-*.npz"))
        ]
        if not files:
            raise FileNotFoundError(f"No shards found for {self.splits} under {self.root}")
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(files)
        for path in files:
            with np.load(path, allow_pickle=False) as shard:
                size = int(shard["user"].shape[0])
                indices = np.arange(size)
                if self.shuffle:
                    rng.shuffle(indices)
                for start in range(0, size, self.batch_size):
                    selected = indices[start : start + self.batch_size]
                    yield {
                        "user": torch.from_numpy(
                            shard["user"][selected].astype(np.int64, copy=False)
                        ),
                        "domain": torch.from_numpy(
                            shard["domain"][selected].astype(np.int64, copy=False)
                        ),
                        "history": torch.from_numpy(
                            shard["history"][selected].astype(np.int64, copy=False)
                        ),
                        "time_features": torch.from_numpy(
                            shard["time_features"][selected].astype(np.float32, copy=False)
                        ),
                        "candidates": torch.from_numpy(
                            shard["candidates"][selected].astype(np.int64, copy=False)
                        ),
                    }

