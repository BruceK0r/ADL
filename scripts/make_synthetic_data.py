from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": args.rows, "validation": args.rows // 4, "test": args.rows // 4}
    for split, count in counts.items():
        categorical = np.column_stack([rng.integers(0, 31, count), rng.integers(0, 17, count)]).astype(np.int64)
        dense = rng.normal(size=(count, 2)).astype(np.float32)
        scenario = (categorical[:, 0] % 3).astype(np.int64)
        logit = 0.8 * dense[:, 0] - 0.3 * dense[:, 1] + 0.2 * scenario + 0.03 * categorical[:, 1]
        label = (rng.random(count) < 1.0 / (1.0 + np.exp(-logit))).astype(np.uint8)
        split_dir = args.output_dir / split
        split_dir.mkdir(exist_ok=True)
        np.savez(split_dir / "part-00000.npz", categorical=categorical, dense=dense, scenario=scenario, label=label)
    metadata = {
        "scenario_mode": "synthetic",
        "scenario_columns": ["synthetic_scenario"],
        "scenario_num": 3,
        "categorical_columns": ["cat_a", "cat_b"],
        "cardinalities": [31, 17],
        "dense_columns": ["dense_a", "dense_b"],
        "rows": counts,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


if __name__ == "__main__":
    main()

