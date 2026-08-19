# ADL reproduction and Amazon multi-domain adaptation

This repository contains a PyTorch implementation of **ADL: Adaptive Distribution
Learning Framework for Multi-Scenario CTR Prediction** and a reproducible adaptation
for the Amazon Beauty, Electronic, and Phone implicit-feedback domains.

The Amazon experiment is not a literal CTR reproduction: the source data contains
observed ratings/interactions but no non-click impressions. It therefore trains ADL
with fixed sampled negatives and reports sampled-candidate ranking metrics.

## Implemented experiment views

Two datasets are prepared independently so excluded domains cannot leak into history:

1. `Beauty + Electronic + Phone` with three-domain user histories.
2. `Electronic + Phone` with Beauty completely excluded before histories, temporal
   cutoffs, catalog statistics, and negatives are generated.

For validation and test, every positive is ranked against 100 fixed same-domain
negatives. By default a deterministic reservoir retains at most 20,000 contexts per
domain and split. Each domain records:

- Recall@1, Recall@5, Recall@10, Recall@20, Recall@50
- NDCG@1, NDCG@5, NDCG@10, NDCG@20, NDCG@50

## Amazon data contract

The input root must contain `beauty`, `ele`, and `phone` directories. Every selected
domain needs:

```text
all_item_seqs.json
all_rating_seqs.json
all_timestamp_seqs.json
id_mapping.json
qwen_embeddings_1024.pt
qwen_embeddings_1024.manifest.json
```

The preparer verifies sequence alignment, item mappings, embedding shape, and embedding
row order. It builds a shared user vocabulary and domain-offset item vocabulary.

The target interaction is never used in its history. Target rating is not an input.
Data is split by global timestamp quantiles (80% train, 10% validation, 10% test).
Negatives are fixed per user-domain, excluded from all known user positives, and drawn
half uniformly and half according to training popularity raised to `0.75`.

## Environment

Use Python 3.10 or newer. Install a CUDA-compatible PyTorch build first when necessary,
then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
```

## Prepare both Amazon experiment views

```bash
python scripts/prepare_3domains.py \
  --input-root data/raw/processed_3domains \
  --output-dir data/processed/amazon_3domains \
  --domains beauty,ele,phone \
  --train-negatives 4 \
  --eval-negatives 100 \
  --max-eval-contexts-per-domain 20000 \
  --seed 2023

python scripts/prepare_3domains.py \
  --input-root data/raw/processed_3domains \
  --output-dir data/processed/amazon_ele_phone \
  --domains ele,phone \
  --train-negatives 4 \
  --eval-negatives 100 \
  --max-eval-contexts-per-domain 20000 \
  --seed 2023
```

Prepared shards contain compact contexts. The positive candidate is always column 0;
the remaining columns are fixed negatives. The model receives global and same-domain
history pools derived only from earlier items.

## Run both ADL experiments

The paper settings retained here are five hidden layers, three routing iterations,
EWMA `beta=0.9`, Adam, and learning rate `1e-3`. The classifier sees domain identity,
while the DLM routing representation excludes the explicit domain embedding. Routing
uses the paper's literal `center dot representation` score by default. Pass
`--normalize-router-input` only for the cosine-style ablation.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_3domain_experiment.py \
  --data-dir data/processed/amazon_3domains \
  --output-dir outputs/amazon_3domains_dot_seed2023 \
  --clusters 3 --epochs 10 --patience 2 \
  --batch-size 2048 --learning-rate 0.001 --seed 2023

CUDA_VISIBLE_DEVICES=1 python scripts/run_3domain_experiment.py \
  --data-dir data/processed/amazon_ele_phone \
  --output-dir outputs/amazon_ele_phone_dot_seed2023 \
  --clusters 3 --epochs 10 --patience 2 \
  --batch-size 2048 --learning-rate 0.001 --seed 2023
```

Each output directory contains a resumable checkpoint, selected checkpoint,
`history.json`, and `results.json`. Summarize both final test results with:

```bash
python scripts/summarize_3domain_results.py \
  --three-domain outputs/amazon_3domains_dot_seed2023/results.json \
  --two-domain outputs/amazon_ele_phone_dot_seed2023/results.json \
  --output-json outputs/amazon_adl_metrics.json \
  --output-csv outputs/amazon_adl_metrics.csv
```

## Experiment reports and committed results

- [`exp_explore.md`](exp_explore.md) documents preprocessing, temporal splitting,
  negative sampling, the ADL architecture, and the first two-/three-domain runs.
- [`exp_compare.md`](exp_compare.md) reports SharedBottom, ADL K=1/3/5/7/9, and the
  Qwen, cross-domain-history, and router-domain ablations on both dataset views.
- [`reports/`](reports/) contains the machine-readable CSV/JSON summaries and each
  comparison run's `results.json` and `history.json`.

Model checkpoints, raw/processed datasets, tensor files, and logs are intentionally
not committed because of their size. They can be regenerated with the commands above
and the settings recorded in the reports.

## Leakage and reproducibility checks

- Histories contain only events earlier than the target event.
- Beauty is absent from the two-domain prepared view.
- Training-visible masks suppress random ID embeddings for temporally cold IDs; frozen
  Qwen content embeddings remain available.
- Evaluation negatives and reservoir samples are deterministic for the preparation seed.
- DLM centers update only in training, are checkpointed, and remain frozen in eval.
- DLM center updates and routing softmax run in float32 under AMP.
- `metadata.json` records source SHA-256 digests, cutoffs, protocol, and sample counts.

## Original Ali-CCP components

The repository also retains `prepare_ali_ccp.py`, `run_experiment.py`, and
`run_paper_protocol.py`. Those use a different CTR data contract and must not be
compared as if they shared the Amazon sampled-ranking labels or candidate universe.
