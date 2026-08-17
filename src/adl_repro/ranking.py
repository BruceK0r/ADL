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


def routing_domain_diagnostics(
    counts: np.ndarray, domain_names: tuple[str, ...]
) -> dict[str, object]:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != len(domain_names):
        raise ValueError("counts must be [domain_num, cluster_num]")
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("routing diagnostics require at least one assignment")
    joint = counts / total
    domain_probability = joint.sum(axis=1, keepdims=True)
    cluster_probability = joint.sum(axis=0, keepdims=True)
    expected = domain_probability @ cluster_probability
    valid = joint > 0
    mutual_information = float((joint[valid] * np.log(joint[valid] / expected[valid])).sum())
    domain_nonzero = domain_probability[domain_probability > 0]
    cluster_nonzero = cluster_probability[cluster_probability > 0]
    domain_entropy = float(-(domain_nonzero * np.log(domain_nonzero)).sum())
    cluster_entropy = float(-(cluster_nonzero * np.log(cluster_nonzero)).sum())
    denominator = float(np.sqrt(domain_entropy * cluster_entropy))
    nmi = mutual_information / denominator if denominator > 0 else 0.0

    domain_totals = counts.sum(axis=1, keepdims=True)
    cluster_totals = counts.sum(axis=0, keepdims=True)
    domain_cluster_fractions = np.divide(
        counts, domain_totals, out=np.zeros_like(counts), where=domain_totals > 0
    )
    cluster_domain_fractions = np.divide(
        counts, cluster_totals, out=np.zeros_like(counts), where=cluster_totals > 0
    )
    integer_counts = counts.astype(np.int64)
    return {
        "domain_cluster_counts": {
            name: integer_counts[index].tolist() for index, name in enumerate(domain_names)
        },
        "domain_cluster_fractions": {
            name: domain_cluster_fractions[index].tolist()
            for index, name in enumerate(domain_names)
        },
        "cluster_domain_fractions": cluster_domain_fractions.tolist(),
        "domain_cluster_nmi": float(nmi),
    }
