# Amazon ADL experiment report

These are single-seed (`2023`) sampled-ranking results. Every test row contains
20,000 contexts from one domain. Each context ranks one held-out positive against
100 fixed, same-domain negatives; these numbers are therefore not full-catalog
retrieval metrics or CTR/AUC results.

## Official result: literal ADL dot-product routing

The official runs use the routing score in Algorithm 1, `center dot representation`,
without L2-normalizing the representation. Early stopping selects the checkpoint by
validation macro `NDCG@10` (patience 2, at most 10 epochs).

| Experiment | Domain | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3-domain | Beauty | 0.04860 | 0.18630 | 0.29195 | 0.45640 | 0.82245 | 0.04860 | 0.11772 | 0.15170 | 0.19294 | 0.26518 |
| 3-domain | Electronic | 0.04345 | 0.17400 | 0.29650 | 0.48230 | 0.81330 | 0.04345 | 0.10814 | 0.14743 | 0.19403 | 0.25972 |
| 3-domain | Phone | 0.02070 | 0.10620 | 0.22225 | 0.46065 | 0.84910 | 0.02070 | 0.06191 | 0.09885 | 0.15851 | 0.23593 |
| 2-domain | Electronic | 0.05920 | 0.21520 | 0.34935 | 0.53685 | 0.83290 | 0.05920 | 0.13747 | 0.18053 | 0.22773 | 0.28651 |
| 2-domain | Phone | 0.03240 | 0.16235 | 0.31665 | 0.53605 | 0.85840 | 0.03240 | 0.09589 | 0.14536 | 0.20052 | 0.26473 |

The 3-domain run selected epoch 2 (validation macro `NDCG@10=0.143498`);
the 2-domain run selected epoch 1 (`0.158569`). Their test macro `NDCG@10`
values are `0.132661` and `0.162943`, respectively.

## Routing-normalization audit

We also ran a cosine-style ablation using L2-normalized routing representations.
Its hard candidate assignments nearly collapsed to one cluster, so it is retained
for diagnosis but is not treated as the official ADL result.

| Experiment | Routing | Test macro NDCG@10 | Test hard-assignment counts (K=3) |
|---|---|---:|---|
| 3-domain | dot product (official) | 0.132661 | 1,193,300 / 1,975,598 / 2,891,102 |
| 3-domain | L2-normalized ablation | 0.159077 | 5,433,944 / 0 / 626,056 |
| 2-domain | dot product (official) | 0.162943 | 1,179,695 / 2,762,788 / 97,517 |
| 2-domain | L2-normalized ablation | 0.140647 | 4,036,671 / 0 / 3,329 |

`amazon_adl_metrics.json` and `amazon_adl_metrics.csv` contain the official compact
table. The four `*.results.json` files preserve the full run configuration,
validation selection, runtime, routing diagnostics, and test metrics.

## SharedBottom, K, and ablation comparisons

The complete comparison artifacts are committed in two complementary forms:

- `three_domain_comparison.json/csv` and `two_domain_comparison.json/csv` are compact
  top-level summaries.
- `experiment_runs/compare_3domains_seed2023/` contains the three-domain summary plus
  each method's `results.json` and, when produced by training, `history.json`.
- `experiment_runs/compare_ele_phone_seed2023/` contains the corresponding two-domain
  files.

The comparison methods are SharedBottom, ADL K=1/3/5/7/9, ADL without Qwen, ADL
without cross-domain history, and ADL routing with an explicit domain embedding.
Human-readable settings, tables, and analysis are in `../exp_compare.md`.

Checkpoints (`*.pth`), source tensors (`*.pt`), datasets, and logs are excluded from
Git because they are large or regenerable; the committed JSON files retain model
configuration, selected epoch, test metrics, and routing diagnostics.
