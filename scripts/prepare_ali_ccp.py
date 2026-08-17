from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DENSE_COLUMNS = ["D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853"]
LABEL_COLUMNS = {"click", "purchase"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert preprocessed Ali-CCP CSV files to streaming NPZ shards")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-mode", choices=["repository3", "paper29"], default="paper29")
    parser.add_argument(
        "--scenario-columns",
        default="",
        help="Comma-separated source columns. paper29 requires the exact columns used to form the 29 scenario tuples.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--force-scenario-count", type=int, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2023)
    parser.add_argument("--expected-train-rows", type=int, default=None)
    parser.add_argument("--expected-test-rows", type=int, default=None)
    return parser.parse_args()


def find_files(root: Path) -> dict[str, Path]:
    alternatives = {
        "train": ["ali_ccp_train.csv", "ali_ccp_train_sample.csv", "train.csv"],
        "validation": ["ali_ccp_val.csv", "ali_ccp_val_sample.csv", "validation.csv", "val.csv"],
        "test": ["ali_ccp_test.csv", "ali_ccp_test_sample.csv", "test.csv"],
    }
    result = {}
    for split, names in alternatives.items():
        for name in names:
            candidate = root / name
            if candidate.exists():
                result[split] = candidate
                break
        if split == "validation" and split not in result:
            continue
        if split not in result:
            raise FileNotFoundError(f"Could not find a {split} CSV under {root}; tried {names}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validation_mask(start_row: int, size: int, ratio: float, seed: int) -> np.ndarray:
    """Stable row-id hash split independent of chunk size and processing order."""
    row_id = np.arange(start_row, start_row + size, dtype=np.uint64)
    value = row_id + np.uint64(seed)
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    threshold = np.uint64(ratio * np.iinfo(np.uint64).max)
    return value <= threshold


def main() -> None:
    args = parse_args()
    files = find_files(args.input_dir)
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")
    derive_validation = "validation" not in files
    existing_outputs = list(args.output_dir.glob("*/part-*.npz")) + list(
        args.output_dir.glob("metadata.json")
    )
    if existing_outputs:
        raise FileExistsError(
            f"Output directory already contains processed data: {args.output_dir}. "
            "Use a new directory to prevent stale shards from contaminating the experiment."
        )
    columns = pd.read_csv(files["train"], nrows=0).columns.tolist()
    if "click" not in columns:
        raise ValueError("Ali-CCP CTR label column 'click' is missing")
    dense_columns = [name for name in DENSE_COLUMNS if name in columns]
    categorical_columns = [name for name in columns if name not in LABEL_COLUMNS and name not in dense_columns]

    if args.scenario_mode == "repository3":
        scenario_columns = ["301"]
        scenario_map = {(1,): 0, (2,): 1, (3,): 2}
        expected_scenarios = 3
    else:
        scenario_columns = [name.strip() for name in args.scenario_columns.split(",") if name.strip()]
        if not scenario_columns:
            raise ValueError(
                "paper29 requires --scenario-columns. The ADL PDF names scenario indicator, user age, and user city "
                "but does not disclose their exact released Ali-CCP field IDs."
            )
        missing = set(scenario_columns) - set(columns)
        if missing:
            raise ValueError(f"Scenario columns are absent from the CSV: {sorted(missing)}")
        keys: set[tuple[int, ...]] = set()
        scenario_fit_paths = [files["train"]]
        if "validation" in files:
            scenario_fit_paths.append(files["validation"])
        for path in scenario_fit_paths:
            for chunk in pd.read_csv(path, usecols=scenario_columns, chunksize=args.chunk_size):
                keys.update(map(tuple, chunk.fillna(0).astype(np.int64).itertuples(index=False, name=None)))
        ordered = sorted(keys)
        scenario_map = {key: index for index, key in enumerate(ordered)}
        expected_scenarios = args.force_scenario_count or 29
        if len(scenario_map) != expected_scenarios:
            raise ValueError(
                f"Scenario tuple construction produced {len(scenario_map)} scenarios, expected {expected_scenarios}. "
                "This guard prevents reporting an experiment that is not comparable with Table 1."
            )

    cardinalities = np.zeros(len(categorical_columns), dtype=np.int64)
    split_maxima: dict[str, np.ndarray] = {}
    rows = {split: 0 for split in ["train", "validation", "test"]}
    for split, path in files.items():
        maximum = np.zeros(len(categorical_columns), dtype=np.int64)
        for chunk in pd.read_csv(path, usecols=categorical_columns, chunksize=args.chunk_size):
            values = chunk.fillna(0).to_numpy(dtype=np.int64)
            if np.any(values < 0):
                raise ValueError("Categorical values must be non-negative integer IDs")
            maximum = np.maximum(maximum, values.max(axis=0) + 1)
            rows[split] += len(chunk)
        split_maxima[split] = maximum
        if split != "test":
            cardinalities = np.maximum(cardinalities, maximum)
    unseen_test_columns = [
        categorical_columns[index]
        for index, (test_size, fitted_size) in enumerate(
            zip(split_maxima["test"], cardinalities)
        )
        if test_size > fitted_size
    ]
    if unseen_test_columns:
        raise ValueError(
            "Test contains categorical IDs outside the train/validation vocabulary in columns "
            f"{unseen_test_columns}. Re-encode unknown values to 0 before preparing shards."
        )
    observed_original_train = rows["train"]
    if args.expected_train_rows is not None and observed_original_train != args.expected_train_rows:
        raise ValueError(
            f"Observed {observed_original_train} original training rows, expected {args.expected_train_rows}"
        )
    if args.expected_test_rows is not None and rows["test"] != args.expected_test_rows:
        raise ValueError(f"Observed {rows['test']} test rows, expected {args.expected_test_rows}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written_rows = {split: 0 for split in ["train", "validation", "test"]}

    def write_shard(split: str, part: int, chunk: pd.DataFrame) -> None:
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        categorical = chunk[categorical_columns].fillna(0).to_numpy(dtype=np.int64)
        dense = chunk[dense_columns].fillna(0).to_numpy(dtype=np.float32)
        label = chunk["click"].to_numpy(dtype=np.uint8)
        tuples = map(
            tuple,
            chunk[scenario_columns].fillna(0).astype(np.int64).itertuples(index=False, name=None),
        )
        try:
            scenario = np.fromiter((scenario_map[key] for key in tuples), dtype=np.int64, count=len(chunk))
        except KeyError as error:
            raise ValueError(f"Unknown scenario tuple: {error.args[0]}") from error
        np.savez(
            split_dir / f"part-{part:05d}.npz",
            categorical=categorical,
            dense=dense,
            scenario=scenario,
            label=label,
        )
        written_rows[split] += len(chunk)

    for split, path in files.items():
        if split == "train" and derive_validation:
            train_part = validation_part = row_offset = 0
            for chunk in pd.read_csv(path, chunksize=args.chunk_size):
                is_validation = validation_mask(
                    row_offset, len(chunk), args.validation_ratio, args.split_seed
                )
                train_chunk = chunk.loc[~is_validation]
                validation_chunk = chunk.loc[is_validation]
                if len(train_chunk):
                    write_shard("train", train_part, train_chunk)
                    train_part += 1
                if len(validation_chunk):
                    write_shard("validation", validation_part, validation_chunk)
                    validation_part += 1
                row_offset += len(chunk)
            continue
        for part, chunk in enumerate(pd.read_csv(path, chunksize=args.chunk_size)):
            write_shard(split, part, chunk)

    metadata = {
        "source_files": {key: str(value.resolve()) for key, value in files.items()},
        "source_sha256": {key: sha256(value) for key, value in files.items()},
        "validation_derived_from_train": derive_validation,
        "validation_ratio": args.validation_ratio if derive_validation else None,
        "split_seed": args.split_seed if derive_validation else None,
        "scenario_mode": args.scenario_mode,
        "scenario_columns": scenario_columns,
        "scenario_num": len(scenario_map),
        "scenario_map": {"|".join(map(str, key)): value for key, value in scenario_map.items()},
        "categorical_columns": categorical_columns,
        "cardinalities": cardinalities.tolist(),
        "dense_columns": dense_columns,
        "source_rows": rows,
        "rows": written_rows,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
