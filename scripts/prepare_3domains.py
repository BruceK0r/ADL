from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch


DOMAIN_DISPLAY = {"beauty": "Beauty", "ele": "Electronic", "phone": "Phone"}
DAY_MS = 86_400_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Beauty/Electronic/Phone for ADL")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domains", default="beauty,ele,phone")
    parser.add_argument("--history-length", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--train-negatives", type=int, default=4)
    parser.add_argument("--eval-negatives", type=int, default=100)
    parser.add_argument(
        "--max-eval-contexts-per-domain",
        type=int,
        default=20_000,
        help="Deterministic reservoir size for each domain in validation and test; 0 keeps all",
    )
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max-users", type=int, default=None, help="Debug-only deterministic subset")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShardWriter:
    def __init__(self, root: Path, split: str, history_length: int, shard_size: int):
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=False)
        self.split = split
        self.history_length = history_length
        self.shard_size = shard_size
        self.parts = 0
        self.examples = 0
        self.buffer: dict[str, list] = defaultdict(list)

    def add(
        self,
        user: int,
        domain: int,
        history: list[int],
        time_features: list[float],
        candidates: np.ndarray,
    ) -> None:
        padded = np.zeros(self.history_length, dtype=np.int32)
        retained = history[-self.history_length :]
        if retained:
            padded[-len(retained) :] = retained
        self.buffer["user"].append(user)
        self.buffer["domain"].append(domain)
        self.buffer["history"].append(padded)
        self.buffer["time_features"].append(time_features)
        self.buffer["candidates"].append(candidates.astype(np.int32, copy=False))
        if len(self.buffer["user"]) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer["user"]:
            return
        arrays = {
            "user": np.asarray(self.buffer["user"], dtype=np.int32),
            "domain": np.asarray(self.buffer["domain"], dtype=np.int8),
            "history": np.stack(self.buffer["history"]),
            "time_features": np.asarray(self.buffer["time_features"], dtype=np.float32),
            "candidates": np.stack(self.buffer["candidates"]),
        }
        path = self.root / f"part-{self.parts:05d}.npz"
        np.savez(path, **arrays)
        count = len(self.buffer["user"])
        self.examples += count
        self.parts += 1
        self.buffer.clear()
        print(f"wrote {path} ({count} contexts)", flush=True)

    def close(self) -> dict[str, int]:
        self.flush()
        return {"examples": self.examples, "shards": self.parts}


class DomainSampler:
    def __init__(self, items: np.ndarray, counts: np.ndarray, rng: np.random.Generator):
        if items.size == 0:
            raise ValueError("A domain has no training-visible items")
        self.items = items.astype(np.int32, copy=False)
        probabilities = counts.astype(np.float64) ** 0.75
        self.cdf = np.cumsum(probabilities / probabilities.sum())
        self.rng = rng

    def draw(self, forbidden: set[int], count: int, prefix: np.ndarray | None = None) -> np.ndarray:
        chosen = [] if prefix is None else [int(value) for value in prefix]
        used = set(chosen)
        attempts = 0
        while len(chosen) < count:
            needed = count - len(chosen)
            draw_count = max(needed * 3, 16)
            uniform = self.items[self.rng.integers(0, len(self.items), size=(draw_count + 1) // 2)]
            popular_indices = np.searchsorted(
                self.cdf, self.rng.random(draw_count // 2), side="right"
            )
            popular = self.items[np.minimum(popular_indices, len(self.items) - 1)]
            block = np.concatenate([uniform, popular])
            self.rng.shuffle(block)
            for value in block:
                item = int(value)
                if item not in forbidden and item not in used:
                    chosen.append(item)
                    used.add(item)
                    if len(chosen) == count:
                        break
            attempts += draw_count
            if attempts > max(10_000, count * 1_000):
                raise RuntimeError("Unable to sample enough unique negatives")
        return np.asarray(chosen, dtype=np.int32)


def split_name(timestamp: int, train_cutoff: int, validation_cutoff: int) -> str:
    if timestamp <= train_cutoff:
        return "train"
    if timestamp <= validation_cutoff:
        return "validation"
    return "test"


def time_features(
    timestamp: int,
    global_count: int,
    domain_count: int,
    last_global: int | None,
    last_domain: int | None,
) -> list[float]:
    day = timestamp / DAY_MS
    global_gap = 0.0 if last_global is None else max(0.0, (timestamp - last_global) / DAY_MS)
    domain_gap = 0.0 if last_domain is None else max(0.0, (timestamp - last_domain) / DAY_MS)
    return [
        math.log1p(global_count),
        math.log1p(domain_count),
        math.log1p(global_gap),
        math.log1p(domain_gap),
        math.sin(2.0 * math.pi * day / 7.0),
        math.cos(2.0 * math.pi * day / 7.0),
        math.sin(2.0 * math.pi * day / 365.25),
        math.cos(2.0 * math.pi * day / 365.25),
    ]


def main() -> None:
    args = parse_args()
    domains = tuple(part.strip() for part in args.domains.split(",") if part.strip())
    if not domains or any(domain not in DOMAIN_DISPLAY for domain in domains):
        raise ValueError(f"domains must be selected from {sorted(DOMAIN_DISPLAY)}")
    if len(set(domains)) != len(domains):
        raise ValueError("domains must not contain duplicates")
    if args.history_length <= 0 or args.min_history < 0:
        raise ValueError("history-length must be positive and min-history non-negative")
    if args.train_negatives <= 0 or args.eval_negatives < 50:
        raise ValueError("train-negatives must be positive and eval-negatives at least 50")
    if not 0.0 < args.train_ratio < 1.0 or not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("split ratios must be inside (0, 1)")
    if args.train_ratio + args.validation_ratio >= 1.0:
        raise ValueError("train-ratio + validation-ratio must be below 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events_by_user: dict[str, list[tuple[int, int, int, float]]] = defaultdict(list)
    positives: dict[tuple[str, int], set[int]] = defaultdict(set)
    item_offsets: dict[str, int] = {}
    item_counts: dict[str, int] = {}
    source_files: dict[str, dict[str, object]] = {}
    embedding_parts = []
    offset = 0

    for domain_index, domain in enumerate(domains):
        root = args.input_root / domain
        item_offsets[domain] = offset
        with (root / "id_mapping.json").open("r", encoding="utf-8") as stream:
            mapping = json.load(stream)
        item_count = len(mapping["item2id"])
        item_counts[domain] = item_count
        with (root / "qwen_embeddings_1024.manifest.json").open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        embeddings = torch.load(root / "qwen_embeddings_1024.pt", map_location="cpu", weights_only=True)
        if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != (item_count, 1024):
            raise ValueError(f"Unexpected embedding shape for {domain}: {getattr(embeddings, 'shape', None)}")
        if manifest["item_order"] != "id_mapping.json:id2item[1:]":
            raise ValueError(f"Unexpected embedding order for {domain}")
        embedding_parts.append(embeddings.float())

        paths = {
            name: root / name
            for name in ("all_item_seqs.json", "all_rating_seqs.json", "all_timestamp_seqs.json")
        }
        with paths["all_item_seqs.json"].open("r", encoding="utf-8") as stream:
            item_sequences = json.load(stream)
        with paths["all_rating_seqs.json"].open("r", encoding="utf-8") as stream:
            rating_sequences = json.load(stream)
        with paths["all_timestamp_seqs.json"].open("r", encoding="utf-8") as stream:
            timestamp_sequences = json.load(stream)
        if set(item_sequences) != set(rating_sequences) or set(item_sequences) != set(timestamp_sequences):
            raise ValueError(f"Sequence user keys are not aligned in {domain}")
        for raw_user, raw_items in item_sequences.items():
            ratings = rating_sequences[raw_user]
            timestamps = timestamp_sequences[raw_user]
            if not (len(raw_items) == len(ratings) == len(timestamps)):
                raise ValueError(f"Sequence lengths differ for {domain}/{raw_user}")
            for raw_item, rating, timestamp in zip(raw_items, ratings, timestamps):
                global_item = offset + int(mapping["item2id"][raw_item])
                event = (int(timestamp), domain_index, global_item, float(rating))
                events_by_user[raw_user].append(event)
                positives[(raw_user, domain_index)].add(global_item)
        source_files[domain] = {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in paths.items()
        }
        source_files[domain]["id_mapping.json"] = {
            "bytes": (root / "id_mapping.json").stat().st_size,
            "sha256": sha256(root / "id_mapping.json"),
        }
        source_files[domain]["qwen_embeddings_1024.pt"] = {
            "bytes": (root / "qwen_embeddings_1024.pt").stat().st_size,
            "sha256": sha256(root / "qwen_embeddings_1024.pt"),
        }
        offset += item_count
        print(f"loaded {domain}: {len(item_sequences)} users, {item_count} items", flush=True)
        del item_sequences, rating_sequences, timestamp_sequences, mapping, embeddings

    raw_users = sorted(events_by_user)
    if args.max_users is not None:
        raw_users = raw_users[: args.max_users]
        events_by_user = {user: events_by_user[user] for user in raw_users}
        positives = {
            key: value for key, value in positives.items() if key[0] in events_by_user
        }
    user2id = {user: index + 1 for index, user in enumerate(raw_users)}
    all_timestamps = np.fromiter(
        (event[0] for user in raw_users for event in events_by_user[user]), dtype=np.int64
    )
    train_cutoff, validation_cutoff = np.quantile(
        all_timestamps,
        [args.train_ratio, args.train_ratio + args.validation_ratio],
        method="higher",
    ).astype(np.int64)

    item_num = offset
    user_num = len(raw_users)
    train_item_counts = [Counter() for _ in domains]
    train_seen_users = np.zeros(user_num + 1, dtype=np.uint8)
    train_seen_items = np.zeros(item_num + 1, dtype=np.uint8)
    item_domains = np.full(item_num + 1, -1, dtype=np.int16)
    for domain_index, domain in enumerate(domains):
        start = item_offsets[domain] + 1
        item_domains[start : start + item_counts[domain]] = domain_index
    for raw_user in raw_users:
        user_id = user2id[raw_user]
        for timestamp, domain, item, _ in events_by_user[raw_user]:
            if timestamp <= train_cutoff:
                train_seen_users[user_id] = 1
                train_seen_items[item] = 1
                train_item_counts[domain][item] += 1

    rng = np.random.default_rng(args.seed)
    samplers = []
    for domain, counts in enumerate(train_item_counts):
        ordered = sorted(counts)
        samplers.append(
            DomainSampler(
                np.asarray(ordered, dtype=np.int32),
                np.asarray([counts[item] for item in ordered], dtype=np.int64),
                rng,
            )
        )

    padding = torch.zeros((1, embedding_parts[0].shape[1]), dtype=torch.float32)
    item_text_embeddings = torch.cat([padding, *embedding_parts], dim=0)
    torch.save(item_text_embeddings, args.output_dir / "item_text_embeddings.pt")
    np.save(args.output_dir / "item_domains.npy", item_domains)
    np.save(args.output_dir / "train_seen_users.npy", train_seen_users)
    np.save(args.output_dir / "train_seen_items.npy", train_seen_items)

    writers = {
        split: ShardWriter(args.output_dir, split, args.history_length, args.shard_size)
        for split in ("train", "validation", "test")
    }
    negative_cache: dict[tuple[str, int], np.ndarray] = {}
    eval_cap = args.max_eval_contexts_per_domain or None
    reservoir_rng = np.random.default_rng(args.seed + 1)
    reservoirs: dict[str, list[list[tuple[int, int, list[int], list[float], np.ndarray]]]] = {
        split: [[] for _ in domains] for split in ("validation", "test")
    }
    reservoir_seen = {
        split: np.zeros(len(domains), dtype=np.int64) for split in ("validation", "test")
    }

    def add_context(
        split: str,
        user_id: int,
        domain: int,
        history_value: list[int],
        time_value: list[float],
        candidates_value: np.ndarray,
    ) -> None:
        if split == "train" or eval_cap is None:
            writers[split].add(
                user_id, domain, history_value, time_value, candidates_value
            )
            return
        reservoir_seen[split][domain] += 1
        seen = int(reservoir_seen[split][domain])
        row = (user_id, domain, history_value, time_value, candidates_value)
        bucket = reservoirs[split][domain]
        if len(bucket) < eval_cap:
            bucket.append(row)
        else:
            replacement = int(reservoir_rng.integers(0, seen))
            if replacement < eval_cap:
                bucket[replacement] = row

    for user_number, raw_user in enumerate(raw_users, start=1):
        user_id = user2id[raw_user]
        events = sorted(events_by_user[raw_user], key=lambda value: (value[0], value[1], value[2]))
        history: deque[int] = deque(maxlen=args.history_length)
        global_count = 0
        domain_counts = [0] * len(domains)
        last_domain: list[int | None] = [None] * len(domains)
        last_global: int | None = None
        for timestamp, domain, item, _rating in events:
            if len(history) >= args.min_history:
                split = split_name(timestamp, int(train_cutoff), int(validation_cutoff))
                negative_count = (
                    args.train_negatives if split == "train" else args.eval_negatives
                )
                key = (raw_user, domain)
                cached = negative_cache.get(key)
                if cached is None or len(cached) < negative_count:
                    cached = samplers[domain].draw(
                        positives[key], negative_count, prefix=cached
                    )
                    negative_cache[key] = cached
                candidates = np.concatenate(
                    [np.asarray([item], dtype=np.int32), cached[:negative_count]]
                )
                add_context(
                    split,
                    user_id,
                    domain,
                    list(history),
                    time_features(
                        timestamp,
                        global_count,
                        domain_counts[domain],
                        last_global,
                        last_domain[domain],
                    ),
                    candidates,
                )
            history.append(item)
            global_count += 1
            domain_counts[domain] += 1
            last_domain[domain] = timestamp
            last_global = timestamp
        if user_number % 25_000 == 0:
            print(f"processed {user_number}/{user_num} users", flush=True)

    if eval_cap is not None:
        for split in ("validation", "test"):
            for domain in range(len(domains)):
                for row in reservoirs[split][domain]:
                    writers[split].add(*row)
    split_stats = {split: writer.close() for split, writer in writers.items()}
    metadata = {
        "format": "three_domain_ranking_v1",
        "domains": list(domains),
        "domain_names": [DOMAIN_DISPLAY[domain] for domain in domains],
        "domain_num": len(domains),
        "user_num": user_num,
        "item_num": item_num,
        "item_counts": item_counts,
        "item_offsets": item_offsets,
        "text_dim": int(item_text_embeddings.shape[1]),
        "time_dim": 8,
        "history_length": args.history_length,
        "min_history": args.min_history,
        "train_negatives": args.train_negatives,
        "eval_negatives": args.eval_negatives,
        "max_eval_contexts_per_domain": eval_cap,
        "eval_contexts_before_reservoir": {
            split: {
                DOMAIN_DISPLAY[domains[domain]]: int(reservoir_seen[split][domain])
                for domain in range(len(domains))
            }
            for split in ("validation", "test")
        },
        "split": {
            "method": "global_timestamp_quantiles",
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "train_cutoff_ms": int(train_cutoff),
            "validation_cutoff_ms": int(validation_cutoff),
        },
        "negative_sampling": "50% uniform + 50% popularity^0.75; fixed per user-domain",
        "seed": args.seed,
        "max_users": args.max_users,
        "splits": split_stats,
        "source_files": source_files,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
