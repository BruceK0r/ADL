from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = (
    ("SharedBottom", "sharedbottom"),
    ("ADL K=1", "adl_k1"),
    ("ADL K=3", "adl_k3"),
    ("ADL K=5", "adl_k5"),
    ("ADL K=7", "adl_k7"),
    ("ADL K=9", "adl_k9"),
    ("ADL without Qwen", "adl_without_qwen"),
    ("ADL without cross-domain history", "adl_without_cross_history"),
    ("ADL router with domain", "adl_router_with_domain"),
)
METRICS = tuple(
    f"{name}@{cutoff}"
    for cutoff in (1, 5, 10, 20, 50)
    for name in ("Recall", "NDCG")
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the three-domain comparison suite")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    rows = []
    protocol = None
    for label, directory in EXPERIMENTS:
        path = args.root / directory / "results.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        current_protocol = result["sampled_ranking_protocol"]
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError(f"Candidate protocol differs in {path}")
        test = result["test"]
        summaries.append(
            {
                "experiment": label,
                "directory": directory,
                "model": result.get("model", result.get("arguments", {}).get("model", "adl")),
                "selected_epoch": result["selected_epoch"],
                "validation_macro_NDCG@10": result["validation_at_selected_epoch"][
                    "macro_NDCG@10"
                ],
                "test_macro_NDCG@10": test["macro_NDCG@10"],
                "parameters_excluding_frozen_text_table": result[
                    "parameters_excluding_frozen_text_table"
                ],
                "config": result["config"],
                "cluster_counts": test.get("cluster_counts"),
                "mean_routing_entropy": test.get("mean_routing_entropy"),
                "domain_cluster_nmi": test.get("domain_cluster_nmi"),
                "domain_cluster_counts": test.get("domain_cluster_counts"),
                "domain_cluster_fractions": test.get("domain_cluster_fractions"),
            }
        )
        for domain, values in test["ranking"].items():
            rows.append(
                {
                    "experiment": label,
                    "domain": domain,
                    "examples": values["examples"],
                    **{metric: values[metric] for metric in METRICS},
                }
            )

    payload = {"sampled_ranking_protocol": protocol, "experiments": summaries, "metrics": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("experiment", "domain", "examples", *METRICS)
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
