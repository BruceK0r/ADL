from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adl_repro.data import NpzBatchDataset
from adl_repro.models import ModelConfig, build_model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect whether ADL learned non-collapsed latent distributions")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["model"] != "adl":
        raise ValueError("The checkpoint is not an ADL model")
    raw_config = checkpoint["config"]
    config = ModelConfig(**raw_config)
    model = build_model("adl", config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = NpzBatchDataset(args.data_dir, args.split, args.batch_size, shuffle=False, seed=0)
    loader = DataLoader(dataset, batch_size=None)
    counts = np.zeros(config.cluster_num, dtype=np.int64)
    scenario_cluster = np.zeros((config.scenario_num, config.cluster_num), dtype=np.int64)
    entropy_sum = 0.0
    examples = 0
    for step, (categorical, dense, scenario, _) in enumerate(loader):
        categorical, dense = categorical.to(device), dense.to(device)
        output = model(categorical, dense, scenario.to(device), return_routing=True)
        route = output.diagnostics["route"]
        coefficient = output.diagnostics["coefficients"]
        route_np = route.cpu().numpy()
        scenario_np = scenario.numpy()
        counts += np.bincount(route_np, minlength=config.cluster_num)
        np.add.at(scenario_cluster, (scenario_np, route_np), 1)
        entropy_sum += float((-(coefficient * coefficient.clamp_min(1e-12).log()).sum(1)).sum())
        examples += len(route_np)
        if args.max_steps is not None and step + 1 >= args.max_steps:
            break

    centers = F.normalize(model.router.centers, dim=1)
    center_cosine = (centers @ centers.T).cpu().numpy()
    report = {
        "examples": examples,
        "cluster_counts": counts.tolist(),
        "cluster_fraction": (counts / max(examples, 1)).tolist(),
        "mean_routing_entropy": entropy_sum / max(examples, 1),
        "center_cosine_similarity": center_cosine.tolist(),
        "scenario_by_cluster_counts": scenario_cluster.tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
