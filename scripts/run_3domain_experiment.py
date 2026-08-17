from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from adl_repro.ranking import DomainRankingAccumulator, macro_average_observed
from adl_repro.three_domain_data import (
    ThreeDomainBatchDataset,
    load_item_text_embeddings,
    load_three_domain_metadata,
)
from adl_repro.three_domain_models import ThreeDomainADL, ThreeDomainModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ADL on 3-domain or Electronic+Phone data")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048, help="Number of ranking contexts")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--user-dim", type=int, default=32)
    parser.add_argument("--item-dim", type=int, default=32)
    parser.add_argument("--domain-dim", type=int, default=8)
    parser.add_argument("--text-projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", default="512,256,128,64,32")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--routing-iterations", type=int, default=3)
    parser.add_argument(
        "--normalize-router-input",
        action="store_true",
        help="L2-normalize routing features (ablation; literal dot-product routing is the default)",
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug-only step limit")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(root: Path, split: str, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = ThreeDomainBatchDataset(
        root, split, args.batch_size, shuffle=shuffle, seed=args.seed
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def labels_for(candidates: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(candidates, dtype=torch.float32)
    labels[:, 0] = 1.0
    return labels


def train_epoch(
    model: ThreeDomainADL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp: bool,
    max_steps: int | None,
) -> dict:
    model.train()
    total_loss = 0.0
    examples = 0
    cluster_counts = np.zeros(model.config.cluster_num, dtype=np.int64)
    entropy_sum = 0.0
    started = time.perf_counter()
    for step, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        labels = labels_for(batch["candidates"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(batch)
            positive_weight = float(batch["candidates"].shape[1] - 1)
            weights = torch.ones_like(labels)
            weights[:, 0] = positive_weight
            loss = nn.functional.binary_cross_entropy_with_logits(
                output.logits, labels, weight=weights
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = int(labels.shape[0])
        total_loss += float(loss.detach()) * count
        examples += count
        cluster_counts += output.diagnostics["cluster_counts"].detach().cpu().numpy()
        entropy_sum += float(output.diagnostics["mean_routing_entropy"]) * count
        if max_steps is not None and step + 1 >= max_steps:
            break
    if examples == 0:
        raise RuntimeError("Training loader produced no contexts")
    return {
        "loss": total_loss / examples,
        "contexts": examples,
        "seconds": time.perf_counter() - started,
        "cluster_counts": cluster_counts.tolist(),
        "mean_routing_entropy": entropy_sum / examples,
    }


@torch.no_grad()
def evaluate(
    model: ThreeDomainADL,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    domain_names: tuple[str, ...],
    max_steps: int | None,
) -> dict:
    model.eval()
    metric = DomainRankingAccumulator(domain_names)
    total_loss = 0.0
    examples = 0
    cluster_counts = np.zeros(model.config.cluster_num, dtype=np.int64)
    entropy_sum = 0.0
    started = time.perf_counter()
    for step, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        labels = labels_for(batch["candidates"])
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(batch)
            positive_weight = float(batch["candidates"].shape[1] - 1)
            weights = torch.ones_like(labels)
            weights[:, 0] = positive_weight
            loss = nn.functional.binary_cross_entropy_with_logits(
                output.logits, labels, weight=weights
            )
        count = int(labels.shape[0])
        total_loss += float(loss) * count
        examples += count
        metric.update(
            batch["domain"].cpu().numpy(), output.logits.float().cpu().numpy()
        )
        cluster_counts += output.diagnostics["cluster_counts"].cpu().numpy()
        entropy_sum += float(output.diagnostics["mean_routing_entropy"]) * count
        if max_steps is not None and step + 1 >= max_steps:
            break
    if examples == 0:
        raise RuntimeError("Evaluation loader produced no contexts")
    ranking = metric.compute()
    macro_ndcg10 = macro_average_observed(ranking, "NDCG@10")
    return {
        "loss": total_loss / examples,
        "contexts": examples,
        "seconds": time.perf_counter() - started,
        "macro_NDCG@10": macro_ndcg10,
        "ranking": ranking,
        "cluster_counts": cluster_counts.tolist(),
        "mean_routing_entropy": entropy_sum / examples,
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience and batch-size must be positive")
    seed_everything(args.seed)
    metadata = load_three_domain_metadata(args.data_dir)
    if metadata.get("eval_negatives", 0) < 50:
        raise ValueError("Prepared evaluation data needs at least 50 negatives")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp = device.type == "cuda" and not args.no_amp
    hidden_dims = tuple(int(value) for value in args.hidden_dims.split(",") if value)
    if len(hidden_dims) != 5:
        raise ValueError("ADL comparison requires five hidden layers")

    item_text = load_item_text_embeddings(args.data_dir)
    item_domains = torch.from_numpy(np.load(args.data_dir / "item_domains.npy"))
    train_seen_users = torch.from_numpy(np.load(args.data_dir / "train_seen_users.npy"))
    train_seen_items = torch.from_numpy(np.load(args.data_dir / "train_seen_items.npy"))
    config = ThreeDomainModelConfig(
        user_num=int(metadata["user_num"]),
        item_num=int(metadata["item_num"]),
        domain_num=int(metadata["domain_num"]),
        time_dim=int(metadata["time_dim"]),
        user_dim=args.user_dim,
        item_dim=args.item_dim,
        domain_dim=args.domain_dim,
        text_projection_dim=args.text_projection_dim,
        hidden_dims=hidden_dims,
        cluster_num=args.clusters,
        beta=args.beta,
        routing_iterations=args.routing_iterations,
        normalize_router_input=args.normalize_router_input,
    )
    model = ThreeDomainADL(
        config, item_text, item_domains, train_seen_users, train_seen_items
    ).to(device)
    del item_text, item_domains, train_seen_users, train_seen_items
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    checkpoint_path = args.output_dir / "adl.last.pth"
    history = []
    best_score = -float("inf")
    best_state = None
    stale = 0
    start_epoch = 0
    serialized_config = {
        key: list(value) if isinstance(value, tuple) else value for key, value in vars(config).items()
    }
    checkpoint_metadata = {
        "data_domains": metadata["domains"],
        "data_seed": metadata["seed"],
        "config": serialized_config,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["checkpoint_metadata"] != checkpoint_metadata:
            raise ValueError("Checkpoint metadata differs; use another output directory")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        history = saved["history"]
        best_score = float(saved["best_score"])
        best_state = saved["best_state"]
        stale = int(saved["stale"])
        start_epoch = int(saved["epoch"])

    train_loader = make_loader(args.data_dir, "train", args, True)
    validation_loader = make_loader(args.data_dir, "validation", args, False)
    domain_names = tuple(metadata["domain_names"])
    for epoch in range(start_epoch, args.epochs):
        train_loader.dataset.set_epoch(epoch)
        train_result = train_epoch(
            model, train_loader, optimizer, scaler, device, amp, args.max_steps
        )
        validation_result = evaluate(
            model, validation_loader, device, amp, domain_names, args.max_steps
        )
        row = {
            "epoch": epoch + 1,
            "train": train_result,
            "validation": validation_result,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = float(validation_result["macro_NDCG@10"])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_score": best_score,
                "best_state": best_state,
                "stale": stale,
                "checkpoint_metadata": checkpoint_metadata,
            },
            checkpoint_path,
        )
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No best model state was selected")
    model.load_state_dict(best_state)
    test_result = evaluate(
        model,
        make_loader(args.data_dir, "test", args, False),
        device,
        amp,
        domain_names,
        args.max_steps,
    )
    selected = max(history, key=lambda row: row["validation"]["macro_NDCG@10"])
    results = {
        "experiment_domains": metadata["domain_names"],
        "sampled_ranking_protocol": {
            "positive_candidates": 1,
            "negative_candidates": metadata["eval_negatives"],
            "negative_sampling": metadata["negative_sampling"],
        },
        "selected_epoch": selected["epoch"],
        "validation_at_selected_epoch": selected["validation"],
        "test": test_result,
        "parameters_excluding_frozen_text_table": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "config": serialized_config,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    torch.save(
        {
            "config": serialized_config,
            "metadata": metadata,
            "state_dict": best_state,
        },
        args.output_dir / "adl.best.pth",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
