import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _make_domain(root: Path, domain: str, domain_offset: int) -> None:
    directory = root / domain
    directory.mkdir(parents=True)
    raw_items = [f"{domain}-item-{index:03d}" for index in range(120)]
    users = [f"user-{index:03d}" for index in range(12)]
    item_sequences = {}
    rating_sequences = {}
    timestamp_sequences = {}
    for user_index, user in enumerate(users):
        chosen = [raw_items[(domain_offset + user_index * 7 + step) % 120] for step in range(11)]
        item_sequences[user] = chosen
        rating_sequences[user] = [5.0] * 11
        timestamp_sequences[user] = [1_600_000_000_000 + step * 10_000 for step in range(11)]
    mapping = {
        "user2id": {user: index + 1 for index, user in enumerate(users)},
        "item2id": {item: index + 1 for index, item in enumerate(raw_items)},
        "id2user": ["<pad>", *users],
        "id2item": ["<pad>", *raw_items],
    }
    payloads = {
        "all_item_seqs.json": item_sequences,
        "all_rating_seqs.json": rating_sequences,
        "all_timestamp_seqs.json": timestamp_sequences,
        "id_mapping.json": mapping,
    }
    for name, payload in payloads.items():
        with (directory / name).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream)
    torch.save(torch.randn(120, 1024), directory / "qwen_embeddings_1024.pt")
    with (directory / "qwen_embeddings_1024.manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "item_order": "id_mapping.json:id2item[1:]",
                "item_count": 120,
                "embedding_shape": [120, 1024],
            },
            stream,
        )


def test_end_to_end_preparer_has_independent_two_domain_view(tmp_path: Path):
    raw = tmp_path / "raw"
    for offset, domain in enumerate(("beauty", "ele", "phone")):
        _make_domain(raw, domain, offset * 11)
    output = tmp_path / "prepared"
    script = Path(__file__).parents[1] / "scripts" / "prepare_3domains.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-root",
            str(raw),
            "--output-dir",
            str(output),
            "--domains",
            "ele,phone",
            "--eval-negatives",
            "50",
            "--max-eval-contexts-per-domain",
            "5",
            "--shard-size",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (output / "metadata.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    assert metadata["domains"] == ["ele", "phone"]
    assert metadata["domain_names"] == ["Electronic", "Phone"]
    assert metadata["domain_num"] == 2
    assert metadata["item_num"] == 240
    assert metadata["splits"]["validation"]["examples"] == 10
    assert metadata["splits"]["test"]["examples"] == 10
    validation_file = sorted((output / "validation").glob("part-*.npz"))[0]
    with np.load(validation_file, allow_pickle=False) as shard:
        assert shard["candidates"].shape[1] == 51
        assert np.all(shard["history"][:, -1] != 0)
        assert np.all(shard["candidates"][:, 0, None] != shard["candidates"][:, 1:])
