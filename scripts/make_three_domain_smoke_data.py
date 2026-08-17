from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny synthetic 3-domain raw data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", type=int, default=24)
    parser.add_argument("--items", type=int, default=160)
    parser.add_argument("--interactions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")
    if args.items < 60 or args.interactions < 11 or args.users <= 0:
        raise ValueError("Smoke data needs >=60 items, >=11 interactions, and positive users")
    generator = torch.Generator().manual_seed(args.seed)
    for domain_offset, domain in enumerate(("beauty", "ele", "phone")):
        directory = args.output_dir / domain
        directory.mkdir(parents=True, exist_ok=False)
        items = [f"{domain}-item-{index:04d}" for index in range(args.items)]
        users = [f"user-{index:04d}" for index in range(args.users)]
        item_sequences = {}
        rating_sequences = {}
        timestamp_sequences = {}
        for user_index, user in enumerate(users):
            chosen = [
                items[(domain_offset * 13 + user_index * 7 + step) % args.items]
                for step in range(args.interactions)
            ]
            item_sequences[user] = chosen
            rating_sequences[user] = [float(3 + (step % 3)) for step in range(args.interactions)]
            timestamp_sequences[user] = [
                1_600_000_000_000 + step * 10_000 for step in range(args.interactions)
            ]
        mapping = {
            "user2id": {user: index + 1 for index, user in enumerate(users)},
            "item2id": {item: index + 1 for index, item in enumerate(items)},
            "id2user": ["<pad>", *users],
            "id2item": ["<pad>", *items],
        }
        for name, value in {
            "all_item_seqs.json": item_sequences,
            "all_rating_seqs.json": rating_sequences,
            "all_timestamp_seqs.json": timestamp_sequences,
            "id_mapping.json": mapping,
        }.items():
            with (directory / name).open("w", encoding="utf-8") as stream:
                json.dump(value, stream)
        torch.save(
            torch.nn.functional.normalize(
                torch.randn(args.items, 1024, generator=generator), p=2, dim=1
            ),
            directory / "qwen_embeddings_1024.pt",
        )
        with (directory / "qwen_embeddings_1024.manifest.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "domain": domain,
                    "item_order": "id_mapping.json:id2item[1:]",
                    "item_count": args.items,
                    "embedding_shape": [args.items, 1024],
                },
                stream,
            )
    print(args.output_dir)


if __name__ == "__main__":
    main()

