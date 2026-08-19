from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


MODELS = ("sharedbottom", "adasparse", "adl")
PAPER_AUC = {"sharedbottom": 0.5948, "adasparse": 0.6165, "adl": 0.6179}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run validation selection, train+validation refit, and multi-seed aggregation"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--selection-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args, forwarded = parser.parse_known_args()
    return args, forwarded


def invoke(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    args, forwarded = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    if args.selection_epochs <= 0 or args.patience <= 0:
        raise ValueError("selection-epochs and patience must be positive")
    runner = Path(__file__).with_name("run_experiment.py")
    all_results: dict[str, dict[str, dict]] = {}

    for seed in seeds:
        seed_root = args.output_dir / f"seed-{seed}"
        selection_dir = seed_root / "selection"
        selection_command = [
            sys.executable,
            str(runner),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(selection_dir),
            "--model",
            "all",
            "--protocol",
            "select",
            "--epochs",
            str(args.selection_epochs),
            "--patience",
            str(args.patience),
            "--seed",
            str(seed),
            "--skip-test",
            *forwarded,
        ]
        if args.resume:
            selection_command.append("--resume")
        invoke(selection_command)
        selection = read_json(selection_dir / "results.json")
        all_results[str(seed)] = {}

        for model in MODELS:
            selected_epoch = int(selection[model]["selected_epoch"])
            refit_dir = seed_root / "refit" / model
            refit_command = [
                sys.executable,
                str(runner),
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(refit_dir),
                "--model",
                model,
                "--protocol",
                "refit",
                "--epochs",
                str(selected_epoch),
                "--seed",
                str(seed),
                *forwarded,
            ]
            if args.resume:
                refit_command.append("--resume")
            invoke(refit_command)
            final = read_json(refit_dir / "results.json")[model]
            all_results[str(seed)][model] = {
                "selected_epoch": selected_epoch,
                "test_auc": final["test"]["auc"],
                "paper_auc": PAPER_AUC[model],
                "parameters": final["parameters"],
                "result_file": str((refit_dir / "results.json").resolve()),
            }

    summary = {}
    for model in MODELS:
        values = np.asarray(
            [all_results[str(seed)][model]["test_auc"] for seed in seeds], dtype=np.float64
        )
        summary[model] = {
            "seeds": seeds,
            "test_auc_values": values.tolist(),
            "mean_test_auc": float(values.mean()),
            "sample_std_test_auc": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "paper_auc": PAPER_AUC[model],
            "mean_difference_from_paper": float(values.mean() - PAPER_AUC[model]),
        }
    baseline = summary["sharedbottom"]["mean_test_auc"]
    for result in summary.values():
        result["relative_improvement_vs_sharedbottom"] = (
            (result["mean_test_auc"] - 0.5) / (baseline - 0.5) - 1.0
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"per_seed": all_results, "summary": summary}
    output = args.output_dir / "aggregate_results.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
