from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CUTOFFS = (1, 5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine two-domain and three-domain ADL metrics")
    parser.add_argument("--three-domain", type=Path, required=True)
    parser.add_argument("--two-domain", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path, experiment: str) -> tuple[list[dict], dict]:
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    rows = [
        {"experiment": experiment, "domain": domain, **values}
        for domain, values in result["test"]["ranking"].items()
    ]
    protocol = {
        "experiment": experiment,
        "result_path": str(path),
        "selected_epoch": result["selected_epoch"],
        "sampled_ranking_protocol": result["sampled_ranking_protocol"],
        "runtime": result["runtime"],
    }
    return rows, protocol


def main() -> None:
    args = parse_args()
    three_rows, three_protocol = load_rows(args.three_domain, "Beauty+Electronic+Phone")
    two_rows, two_protocol = load_rows(args.two_domain, "Electronic+Phone")
    rows = three_rows + two_rows
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(
            {"protocols": [three_protocol, two_protocol], "metrics": rows},
            stream,
            ensure_ascii=False,
            indent=2,
        )
    fieldnames = ["experiment", "domain", "examples"] + [
        metric
        for cutoff in CUTOFFS
        for metric in (f"Recall@{cutoff}", f"NDCG@{cutoff}")
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

