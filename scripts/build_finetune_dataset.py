"""
Build and validate the fine-tuning dataset.

Usage:
    python scripts/build_finetune_dataset.py

Output:
    evals/finetune/train.json   (80%)
    evals/finetune/val.json     (10%)
    evals/finetune/test.json    (10%)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REQUIRED_KEYS = {"instruction", "response"}
RISK_TIERS = {"Unacceptable", "High", "Limited", "Minimal"}


def validate_example(example: dict) -> None:
    if not REQUIRED_KEYS.issubset(example):
        raise ValueError(f"Missing keys: {REQUIRED_KEYS - example.keys()}")
    response = json.loads(example["response"])
    if response.get("risk_tier") not in RISK_TIERS:
        raise ValueError(f"Invalid risk_tier: {response.get('risk_tier')}")


def build_dataset(source_path: Path, out_dir: Path, seed: int = 42) -> None:
    raw = json.loads(source_path.read_text())
    for ex in raw:
        validate_example(ex)

    random.seed(seed)
    random.shuffle(raw)

    n = len(raw)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    splits = {
        "train": raw[:train_end],
        "val": raw[train_end:val_end],
        "test": raw[val_end:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        (out_dir / f"{name}.json").write_text(json.dumps(data, indent=2))
        print(f"{name}: {len(data)} examples")


if __name__ == "__main__":
    build_dataset(
        source_path=Path("evals/finetune/examples.json"),
        out_dir=Path("evals/finetune"),
    )
