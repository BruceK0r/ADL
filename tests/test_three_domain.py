import numpy as np
import torch

from adl_repro.ranking import DomainRankingAccumulator, macro_average_observed
from adl_repro.three_domain_models import ThreeDomainADL, ThreeDomainModelConfig


def test_domain_ranking_metrics_cover_requested_cutoffs():
    scores = np.zeros((2, 51), dtype=np.float32)
    scores[0, 0] = 2.0  # rank 1 in Electronic
    scores[1, 0] = 1.0
    scores[1, 1:5] = 2.0  # rank 5 in Phone
    metric = DomainRankingAccumulator(("Electronic", "Phone"))
    metric.update(np.asarray([0, 1]), scores)
    result = metric.compute()
    assert result["Electronic"]["Recall@1"] == 1.0
    assert result["Electronic"]["NDCG@1"] == 1.0
    assert result["Phone"]["Recall@1"] == 0.0
    assert result["Phone"]["Recall@5"] == 1.0
    assert set(result["Phone"]) == {
        "examples",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "Recall@50",
        "NDCG@1",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
        "NDCG@50",
    }


def test_macro_metric_ignores_domains_absent_from_debug_subset():
    ranking = {
        "Electronic": {"examples": 10, "NDCG@10": 0.25},
        "Phone": {"examples": 0, "NDCG@10": float("nan")},
    }
    assert macro_average_observed(ranking, "NDCG@10") == 0.25


def test_three_domain_adl_forward_and_eval_center_freeze():
    config = ThreeDomainModelConfig(
        user_num=3,
        item_num=6,
        domain_num=2,
        time_dim=8,
        user_dim=2,
        item_dim=2,
        domain_dim=2,
        text_projection_dim=3,
        hidden_dims=(8, 7, 6, 5, 4),
        cluster_num=2,
    )
    text = torch.randn(7, 4)
    text[0].zero_()
    item_domains = torch.tensor([-1, 0, 0, 0, 1, 1, 1])
    seen_users = torch.ones(4)
    seen_items = torch.ones(7)
    model = ThreeDomainADL(config, text, item_domains, seen_users, seen_items)
    batch = {
        "user": torch.tensor([1, 2]),
        "domain": torch.tensor([0, 1]),
        "history": torch.tensor([[0, 1, 2], [0, 4, 5]]),
        "time_features": torch.zeros(2, 8),
        "candidates": torch.tensor([[3, 2, 1], [6, 5, 4]]),
    }
    model.train()
    output = model(batch)
    assert output.logits.shape == (2, 3)
    output.logits.sum().backward()
    assert model.router.centers.grad is None
    trained_centers = model.router.centers.clone()
    model.eval()
    with torch.no_grad():
        repeated = model(batch)
    assert repeated.logits.shape == (2, 3)
    assert torch.equal(model.router.centers, trained_centers)
