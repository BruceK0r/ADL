from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


DEFAULT_FIELDS = (
    "101",
    "121",
    "122",
    "124",
    "125",
    "126",
    "127",
    "128",
    "129",
    "205",
    "206",
    "207",
    "216",
    "508",
    "509",
    "702",
    "853",
    "301",
)
SEPARATOR = re.compile("\x01|\x02|\x03")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join and encode the four official Ali-CCP raw files into wide CSV files"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=12,
        help="Training frequency required for a value to receive a nonzero ID",
    )
    parser.add_argument(
        "--preserve-fields",
        default="301",
        help="Comma-separated fields kept from frequency 1 (include every scenario-definition field)",
    )
    parser.add_argument("--commit-every", type=int, default=100_000)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--allow-missing-common",
        action="store_true",
        help="Keep skeleton rows whose referenced common-feature ID is absent",
    )
    return parser.parse_args()


def raw_paths(root: Path, split: str) -> tuple[Path, Path]:
    skeleton = root / f"sample_skeleton_{split}.csv"
    common = root / f"common_features_{split}.csv"
    missing = [str(path) for path in (skeleton, common) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing official Ali-CCP files: {missing}")
    return skeleton, common


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def feature_dict(raw: str) -> dict[str, str]:
    tokens = SEPARATOR.split(raw)
    usable = len(tokens) - len(tokens) % 3
    return {tokens[index]: tokens[index + 1] for index in range(0, usable, 3)}


def build_common_database(source: Path, database: Path, commit_every: int) -> int:
    if database.exists():
        raise FileExistsError(f"Intermediate database already exists: {database}")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE common (common_id TEXT PRIMARY KEY, features TEXT NOT NULL)")
    pending: list[tuple[str, str]] = []
    rows = 0
    with source.open("r", encoding="utf-8", newline="") as stream:
        for line in stream:
            columns = line.rstrip("\r\n").split(",")
            if len(columns) < 3:
                raise ValueError(f"Malformed common-feature row {rows + 1} in {source}")
            pending.append((columns[0], columns[2]))
            rows += 1
            if len(pending) >= commit_every:
                connection.executemany("INSERT INTO common VALUES (?, ?)", pending)
                connection.commit()
                pending.clear()
    if pending:
        connection.executemany("INSERT INTO common VALUES (?, ?)", pending)
        connection.commit()
    connection.close()
    return rows


def join_split(
    skeleton: Path,
    database: Path,
    joined: Path,
    fields: tuple[str, ...],
    vocabulary: dict[str, Counter[str]] | None,
) -> dict[str, int]:
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    cursor = connection.cursor()
    counts = {"source": 0, "written": 0, "invalid_label": 0, "missing_common": 0}
    with skeleton.open("r", encoding="utf-8", newline="") as source, joined.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target)
        writer.writerow(("click", "purchase", *fields))
        for line in source:
            counts["source"] += 1
            columns = line.rstrip("\r\n").split(",")
            if len(columns) < 6:
                raise ValueError(f"Malformed skeleton row {counts['source']} in {skeleton}")
            if columns[1] == "0" and columns[2] == "1":
                counts["invalid_label"] += 1
                continue
            common_row = cursor.execute(
                "SELECT features FROM common WHERE common_id = ?", (columns[3],)
            ).fetchone()
            if common_row is None:
                counts["missing_common"] += 1
                common_features = {}
            else:
                common_features = feature_dict(common_row[0])
            features = feature_dict(columns[5])
            features.update(common_features)
            values = [features.get(field, "0") for field in fields]
            writer.writerow((columns[1], columns[2], *values))
            if vocabulary is not None:
                for field, value in zip(fields, values):
                    if value != "0":
                        vocabulary[field][value] += 1
            counts["written"] += 1
    connection.close()
    return counts


def encode_joined(
    source: Path,
    target: Path,
    fields: tuple[str, ...],
    mappings: dict[str, dict[str, int]],
) -> int:
    rows = 0
    with source.open("r", encoding="utf-8", newline="") as input_stream, target.open(
        "w", encoding="utf-8", newline=""
    ) as output_stream:
        reader = csv.reader(input_stream)
        writer = csv.writer(output_stream)
        header = next(reader)
        writer.writerow(header)
        for columns in reader:
            encoded = [
                mappings[field].get(value, 0)
                for field, value in zip(fields, columns[2:])
            ]
            writer.writerow((columns[0], columns[1], *encoded))
            rows += 1
    return rows


def main() -> None:
    args = parse_args()
    fields = tuple(value.strip() for value in args.fields.split(",") if value.strip())
    preserve_fields = {
        value.strip() for value in args.preserve_fields.split(",") if value.strip()
    }
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("--fields must contain distinct field IDs")
    if args.min_frequency < 1:
        raise ValueError("--min-frequency must be positive")
    unknown_preserved = preserve_fields - set(fields)
    if unknown_preserved:
        raise ValueError(f"--preserve-fields are absent from --fields: {sorted(unknown_preserved)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = [args.output_dir / "ali_ccp_train.csv", args.output_dir / "ali_ccp_test.csv"]
    if any(path.exists() for path in protected):
        raise FileExistsError("Encoded output already exists; use a new --output-dir")

    report: dict[str, object] = {
        "fields": list(fields),
        "min_frequency": args.min_frequency,
        "preserve_fields": sorted(preserve_fields),
        "source_files": {},
        "source_sha256": {},
        "splits": {},
    }
    vocabulary = {field: Counter() for field in fields}
    joined_paths: dict[str, Path] = {}
    database_paths: dict[str, Path] = {}

    for split in ("train", "test"):
        skeleton, common = raw_paths(args.input_dir, split)
        report["source_files"][f"skeleton_{split}"] = str(skeleton.resolve())
        report["source_files"][f"common_{split}"] = str(common.resolve())
        database = args.output_dir / f"common_{split}.sqlite3"
        joined = args.output_dir / f"joined_{split}.csv"
        common_rows = build_common_database(common, database, args.commit_every)
        counts = join_split(
            skeleton,
            database,
            joined,
            fields,
            vocabulary if split == "train" else None,
        )
        counts["common_rows"] = common_rows
        if counts["missing_common"] and not args.allow_missing_common:
            raise ValueError(
                f"{split} contains {counts['missing_common']} skeleton rows with missing common features"
            )
        report["splits"][split] = counts
        joined_paths[split] = joined
        database_paths[split] = database

    for name, path in report["source_files"].items():
        report["source_sha256"][name] = sha256(Path(path))

    mappings = {
        field: {
            value: index
            for index, value in enumerate(
                sorted(
                    value
                    for value, count in vocabulary[field].items()
                    if count >= (1 if field in preserve_fields else args.min_frequency)
                ),
                start=1,
            )
        }
        for field in fields
    }
    report["cardinalities_including_oov"] = {
        field: len(mapping) + 1 for field, mapping in mappings.items()
    }
    for split in ("train", "test"):
        target = args.output_dir / f"ali_ccp_{split}.csv"
        report["splits"][split]["encoded_rows"] = encode_joined(
            joined_paths[split], target, fields, mappings
        )

    with (args.output_dir / "raw_conversion_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    with (args.output_dir / "vocabulary.json").open("w", encoding="utf-8") as stream:
        json.dump(mappings, stream, ensure_ascii=False)
    if not args.keep_intermediate:
        for path in (*joined_paths.values(), *database_paths.values()):
            path.unlink(missing_ok=True)
        for database in database_paths.values():
            Path(str(database) + "-wal").unlink(missing_ok=True)
            Path(str(database) + "-shm").unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
