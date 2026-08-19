from __future__ import annotations

import numpy as np


def exact_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with correct average ranks for ties."""
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have identical shapes")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("AUC labels must be binary")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both positive and negative examples")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


class HistogramAUC:
    """Streaming AUC approximation; use many bins for large Ali-CCP evaluation."""

    def __init__(self, bins: int = 1_000_000):
        self.bins = int(bins)
        if self.bins < 2:
            raise ValueError("HistogramAUC requires at least two bins")
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def update(self, labels: np.ndarray, probabilities: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=np.uint8)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if labels.shape != probabilities.shape:
            raise ValueError("labels and probabilities must have identical shapes")
        if np.any((labels != 0) & (labels != 1)):
            raise ValueError("AUC labels must be binary")
        index = np.minimum((np.clip(probabilities, 0.0, 1.0) * self.bins).astype(np.int64), self.bins - 1)
        positive_index, positive_count = np.unique(index[labels == 1], return_counts=True)
        negative_index, negative_count = np.unique(index[labels == 0], return_counts=True)
        self.positive[positive_index] += positive_count
        self.negative[negative_index] += negative_count

    def compute(self) -> float:
        positives = int(self.positive.sum())
        negatives = int(self.negative.sum())
        if positives == 0 or negatives == 0:
            raise ValueError("AUC requires both positive and negative examples")
        negatives_below = np.cumsum(self.negative) - self.negative
        concordant = (self.positive * negatives_below).sum(dtype=np.float64)
        ties = 0.5 * (self.positive * self.negative).sum(dtype=np.float64)
        return float((concordant + ties) / (positives * negatives))


def relative_improvement(auc: float, baseline_auc: float) -> float:
    if baseline_auc == 0.5:
        raise ValueError("Relative improvement is undefined when baseline AUC is 0.5")
    return (auc - 0.5) / (baseline_auc - 0.5) - 1.0
