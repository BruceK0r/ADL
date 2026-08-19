from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from adl_repro.three_domain_data import (
    ThreeDomainBatchDataset,
    load_item_text_embeddings,
    load_three_domain_metadata,
)
from adl_repro.three_domain_models import (
    ThreeDomainModelConfig,
    build_three_domain_model,
)
from run_3domain_experiment import evaluate, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a selected three-domain checkpoint with routing diagnostics"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ThreeDomainModelConfig(**checkpoint["config"])
    metadata = load_three_domain_metadata(args.data_dir)
    item_text = load_item_text_embeddings(args.data_dir)
    item_domains = torch.from_numpy(np.load(args.data_dir / "item_domains.npy"))
    train_seen_users = torch.from_numpy(np.load(args.data_dir / "train_seen_users.npy"))
    train_seen_items = torch.from_numpy(np.load(args.data_dir / "train_seen_items.npy"))
    model = build_three_domain_model(
        "adl", config, item_text, item_domains, train_seen_users, train_seen_items
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    dataset = ThreeDomainBatchDataset(
        args.data_dir, "test", args.batch_size, shuffle=False, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    test_result = evaluate(
        model,
        loader,
        device,
        device.type == "cuda" and not args.no_amp,
        tuple(metadata["domain_names"]),
        None,
    )
    with args.source_results.open("r", encoding="utf-8") as stream:
        results = json.load(stream)
    results["test"] = test_result
    results["routing_diagnostics_recomputed"] = True
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(json.dumps(test_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
