from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def load_metadata(root: str | Path) -> dict:
    with (Path(root) / "metadata.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


class NpzBatchDataset(IterableDataset):
    """Streams already-batched tensors from Ali-CCP NPZ shards."""

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
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        files = [
            path
            for split in self.splits
            for path in sorted((self.root / split).glob("part-*.npz"))
        ]
        if not files:
            raise FileNotFoundError(f"No shards found for splits {self.splits} under {self.root}")
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(files)
        for path in files:
            with np.load(path, allow_pickle=False) as shard:
                categorical = shard["categorical"]
                dense = shard["dense"]
                scenario = shard["scenario"]
                label = shard["label"]
                indices = np.arange(label.shape[0])
                if self.shuffle:
                    rng.shuffle(indices)
                for start in range(0, len(indices), self.batch_size):
                    chosen = indices[start : start + self.batch_size]
                    yield (
                        torch.from_numpy(categorical[chosen].astype(np.int64, copy=False)),
                        torch.from_numpy(dense[chosen].astype(np.float32, copy=False)),
                        torch.from_numpy(scenario[chosen].astype(np.int64, copy=False)),
                        torch.from_numpy(label[chosen].astype(np.float32, copy=False)),
                    )
