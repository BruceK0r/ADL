from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from adl_repro.data import NpzBatchDataset, load_metadata
from adl_repro.metrics import relative_improvement
from adl_repro.models import ModelConfig, build_model
from adl_repro.train import evaluate, fit


PAPER_AUC = {"sharedbottom": 0.5948, "adasparse": 0.6165, "adl": 0.6179}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ali-CCP ADL reproduction")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", choices=["sharedbottom", "adasparse", "adl", "all"], default="all")
    parser.add_argument(
        "--protocol",
        choices=["select", "refit"],
        default="select",
        help="select uses validation early stopping; refit trains train+validation for exactly --epochs",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", default="512,256,128,64,32")
    parser.add_argument("--shared-layer-count", type=int, default=1)
    parser.add_argument("--clusters", type=int, default=9)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--routing-iterations", type=int, default=3)
    parser.add_argument("--adasparse-epsilon", type=float, default=0.25)
    parser.add_argument("--adasparse-beta", type=float, default=2.0)
    parser.add_argument("--adasparse-alpha", type=float, default=0.1)
    parser.add_argument("--adasparse-delta-alpha", type=float, default=1e-4)
    parser.add_argument("--adasparse-alpha-max", type=float, default=5.0)
    parser.add_argument("--adasparse-sparsity-min", type=float, default=0.15)
    parser.add_argument("--adasparse-sparsity-max", type=float, default=0.25)
    parser.add_argument("--adasparse-regularization-weight", type=float, default=0.01)
    parser.add_argument("--adasparse-regularization-growth", type=float, default=1e-5)
    parser.add_argument("--adasparse-regularization-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auc-bins", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke-test limiter; omit for a real run")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not inspect the test split (use during validation-based model selection)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume each model from <output-dir>/<model>.last.pth")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(data_dir: Path, split, args, shuffle: bool) -> DataLoader:
    dataset = NpzBatchDataset(data_dir, split, args.batch_size, shuffle=shuffle, seed=args.seed)
    return DataLoader(dataset, batch_size=None, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if args.patience <= 0 or args.num_workers < 0 or args.auc_bins < 2:
        raise ValueError("patience must be positive, num-workers non-negative, and auc-bins >= 2")
    seed_everything(args.seed)
    metadata = load_metadata(args.data_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is false")
    names = ["sharedbottom", "adasparse", "adl"] if args.model == "all" else [args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hidden_dims = tuple(int(value) for value in args.hidden_dims.split(",") if value.strip())
    if len(hidden_dims) != 5:
        raise ValueError("ADL paper comparison requires exactly five --hidden-dims values")
    non_categorical_scenario_columns = sorted(
        set(metadata["scenario_columns"]) - set(metadata["categorical_columns"])
    )
    if non_categorical_scenario_columns:
        raise ValueError(
            "AdaSparse requires every domain-aware scenario field to be categorical; "
            f"not categorical: {non_categorical_scenario_columns}"
        )
    domain_aware_indices = tuple(
        metadata["categorical_columns"].index(column) for column in metadata["scenario_columns"]
    )
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "data_metadata": metadata,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    results = {}

    for name in names:
        seed_everything(args.seed)
        config = ModelConfig(
            cardinalities=metadata["cardinalities"],
            dense_dim=len(metadata["dense_columns"]),
            scenario_num=metadata["scenario_num"],
            embedding_dim=args.embedding_dim,
            hidden_dims=hidden_dims,
            cluster_num=args.clusters,
            beta=args.beta,
            routing_iterations=args.routing_iterations,
            shared_layer_count=args.shared_layer_count,
            domain_aware_indices=domain_aware_indices,
            adasparse_epsilon=args.adasparse_epsilon,
            adasparse_beta=args.adasparse_beta,
            adasparse_alpha=args.adasparse_alpha,
            adasparse_delta_alpha=args.adasparse_delta_alpha,
            adasparse_alpha_max=args.adasparse_alpha_max,
            adasparse_sparsity_min=args.adasparse_sparsity_min,
            adasparse_sparsity_max=args.adasparse_sparsity_max,
            adasparse_regularization_weight=args.adasparse_regularization_weight,
            adasparse_regularization_growth=args.adasparse_regularization_growth,
            adasparse_regularization_max=args.adasparse_regularization_max,
        )
        model = build_model(name, config).to(device)
        serialized_config = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in vars(config).items()
        }
        training_splits = ("train", "validation") if args.protocol == "refit" else "train"
        validation_loader = (
            None
            if args.protocol == "refit"
            else loader(args.data_dir, "validation", args, False)
        )
        history, best_state = fit(
            model,
            loader(args.data_dir, training_splits, args, True),
            validation_loader,
            device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            max_steps=args.max_steps,
            auc_bins=args.auc_bins,
            checkpoint_path=args.output_dir / f"{name}.last.pth",
            checkpoint_metadata={
                "model": name,
                "model_config": serialized_config,
                "data_source_sha256": metadata.get("source_sha256"),
                "protocol": args.protocol,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            resume=args.resume,
        )
        test = (
            None
            if args.skip_test
            else evaluate(
                model,
                loader(args.data_dir, "test", args, False),
                device,
                max_steps=args.max_steps,
                auc_bins=args.auc_bins,
            )
        )
        torch.save(
            {"model": name, "config": serialized_config, "metadata": metadata, "state_dict": best_state},
            args.output_dir / f"{name}.pth",
        )
        results[name] = {
            "test": vars(test) if test is not None else None,
            "paper_auc": PAPER_AUC[name],
            "auc_difference": test.auc - PAPER_AUC[name] if test is not None else None,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
            "selected_epoch": max(
                history,
                key=lambda row: (
                    row["validation"]["auc"]
                    if row["validation"] is not None
                    else row["epoch"]
                ),
            )["epoch"],
        }

    if "sharedbottom" in results and results["sharedbottom"]["test"] is not None:
        baseline = results["sharedbottom"]["test"]["auc"]
        for result in results.values():
            if result["test"] is not None:
                result["relative_improvement_vs_run_sharedbottom"] = relative_improvement(
                    result["test"]["auc"], baseline
                )
    output = args.output_dir / "results.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
