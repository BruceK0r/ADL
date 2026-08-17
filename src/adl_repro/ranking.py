from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


RANKING_CUTOFFS = (1, 5, 10, 20, 50)


@dataclass
class DomainRankingAccumulator:
    domain_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self.counts = np.zeros(len(self.domain_names), dtype=np.int64)
        self.recalls = np.zeros((len(self.domain_names), len(RANKING_CUTOFFS)), dtype=np.float64)
        self.ndcgs = np.zeros_like(self.recalls)

    def update(self, domains: np.ndarray, scores: np.ndarray) -> None:
        if scores.ndim != 2 or scores.shape[1] < max(RANKING_CUTOFFS) + 1:
            raise ValueError("Ranking evaluation requires one positive and at least 50 negatives")
        positive = scores[:, :1]
        ranks = 1 + (scores[:, 1:] >= positive).sum(axis=1)
        for domain in range(len(self.domain_names)):
            selected = ranks[domains == domain]
            if selected.size == 0:
                continue
            self.counts[domain] += selected.size
            for index, cutoff in enumerate(RANKING_CUTOFFS):
                hit = selected <= cutoff
                self.recalls[domain, index] += float(hit.sum())
                self.ndcgs[domain, index] += float(
                    sum(1.0 / math.log2(int(rank) + 1) for rank in selected[hit])
                )

    def compute(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for domain, name in enumerate(self.domain_names):
            count = int(self.counts[domain])
            values: dict[str, float | int] = {"examples": count}
            for index, cutoff in enumerate(RANKING_CUTOFFS):
                values[f"Recall@{cutoff}"] = (
                    float(self.recalls[domain, index] / count) if count else float("nan")
                )
                values[f"NDCG@{cutoff}"] = (
                    float(self.ndcgs[domain, index] / count) if count else float("nan")
                )
            result[name] = values
        return result


def macro_average_observed(
    ranking: dict[str, dict[str, float | int]], metric: str
) -> float:
    values = [
        float(domain_values[metric])
        for domain_values in ranking.values()
        if int(domain_values["examples"]) > 0
    ]
    if not values:
        raise RuntimeError("Ranking result contains no observed domains")
    return float(np.mean(values))
