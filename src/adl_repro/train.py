from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import HistogramAUC


@dataclass
class EpochResult:
    loss: float
    prediction_loss: float
    auxiliary_loss: float
    auc: float | None
    examples: int
    seconds: float
    diagnostics: dict[str, float]


def _move(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _diagnostics_to_float(diagnostics: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {}
    for name, value in diagnostics.items():
        if value.numel() == 1:
            result[name] = float(value.detach().cpu())
    return result


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_steps: int | None = None,
    auc_bins: int = 1_000_000,
    track_auc: bool = False,
) -> EpochResult:
    model.train()
    metric = HistogramAUC(auc_bins) if track_auc else None
    loss_sum = 0.0
    prediction_loss_sum = 0.0
    auxiliary_loss_sum = 0.0
    examples = 0
    diagnostics_sum: dict[str, float] = {}
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        categorical, dense, scenario, label = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(categorical, dense, scenario)
        prediction_loss = nn.functional.binary_cross_entropy_with_logits(output.logits, label)
        loss = prediction_loss + output.auxiliary_loss
        loss.backward()
        optimizer.step()
        count = label.numel()
        loss_sum += float(loss.detach()) * count
        prediction_loss_sum += float(prediction_loss.detach()) * count
        auxiliary_loss_sum += float(output.auxiliary_loss.detach()) * count
        examples += count
        for name, value in _diagnostics_to_float(output.diagnostics).items():
            diagnostics_sum[name] = diagnostics_sum.get(name, 0.0) + value * count
        if metric is not None:
            metric.update(
                label.detach().cpu().numpy(),
                output.logits.detach().sigmoid().cpu().numpy(),
            )
        if max_steps is not None and step + 1 >= max_steps:
            break
    if examples == 0:
        raise RuntimeError("The training loader produced no examples")
    return EpochResult(
        loss_sum / examples,
        prediction_loss_sum / examples,
        auxiliary_loss_sum / examples,
        metric.compute() if metric is not None else None,
        examples,
        time.perf_counter() - started,
        {name: value / examples for name, value in diagnostics_sum.items()},
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    max_steps: int | None = None,
    auc_bins: int = 1_000_000,
) -> EpochResult:
    model.eval()
    metric = HistogramAUC(auc_bins)
    loss_sum = 0.0
    prediction_loss_sum = 0.0
    auxiliary_loss_sum = 0.0
    examples = 0
    diagnostics_sum: dict[str, float] = {}
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        categorical, dense, scenario, label = _move(batch, device)
        output = model(categorical, dense, scenario)
        prediction_loss = nn.functional.binary_cross_entropy_with_logits(output.logits, label)
        loss = prediction_loss + output.auxiliary_loss
        count = label.numel()
        loss_sum += float(loss) * count
        prediction_loss_sum += float(prediction_loss) * count
        auxiliary_loss_sum += float(output.auxiliary_loss) * count
        examples += count
        for name, value in _diagnostics_to_float(output.diagnostics).items():
            diagnostics_sum[name] = diagnostics_sum.get(name, 0.0) + value * count
        metric.update(label.cpu().numpy(), output.logits.sigmoid().cpu().numpy())
        if max_steps is not None and step + 1 >= max_steps:
            break
    if examples == 0:
        raise RuntimeError("The evaluation loader produced no examples")
    return EpochResult(
        loss_sum / examples,
        prediction_loss_sum / examples,
        auxiliary_loss_sum / examples,
        metric.compute(),
        examples,
        time.perf_counter() - started,
        {name: value / examples for name, value in diagnostics_sum.items()},
    )


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
    max_steps: int | None = None,
    auc_bins: int = 1_000_000,
    checkpoint_path: Path | None = None,
    checkpoint_metadata: dict | None = None,
    resume: bool = False,
) -> tuple[list[dict], dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_auc = -math.inf
    best_state = None
    stale = 0
    history: list[dict] = []
    start_epoch = 0
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("checkpoint_metadata") != checkpoint_metadata:
            raise ValueError(
                "Checkpoint metadata differs from this run. Use a new output directory "
                "or restore the original data/model arguments."
            )
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        best_auc = float(saved["best_auc"])
        best_state = saved["best_state"]
        stale = int(saved["stale"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"])
        if val_loader is not None and stale >= patience:
            assert best_state is not None
            model.load_state_dict(best_state)
            return history, best_state
    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_steps=max_steps,
            auc_bins=auc_bins,
            track_auc=False,
        )
        val_result = (
            evaluate(model, val_loader, device, max_steps=max_steps, auc_bins=auc_bins)
            if val_loader is not None
            else None
        )
        row = {
            "epoch": epoch + 1,
            "train": vars(train_result),
            "validation": vars(val_result) if val_result is not None else None,
        }
        history.append(row)
        print(row, flush=True)
        score = val_result.auc if val_result is not None else float(epoch + 1)
        assert score is not None
        if val_result is None or score > best_auc:
            best_auc = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "best_state": best_state,
                    "stale": stale,
                    "history": history,
                    "checkpoint_metadata": checkpoint_metadata,
                },
                checkpoint_path,
            )
        if val_loader is not None and stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return history, best_state
