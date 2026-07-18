import json
from pathlib import Path

import pytest


@pytest.fixture
def train_data():
    path = Path("evals/finetune/train.json")
    if not path.exists():
        pytest.skip("Fine-tuning dataset not yet built — run scripts/build_finetune_dataset.py")
    return json.loads(path.read_text())


def test_train_split_not_empty(train_data):
    assert len(train_data) > 50, "Training set should have >50 examples"


def test_every_example_has_required_keys(train_data):
    for ex in train_data:
        assert "instruction" in ex
        assert "response" in ex


def test_every_response_is_valid_json(train_data):
    for ex in train_data:
        parsed = json.loads(ex["response"])
        assert parsed["risk_tier"] in {"Unacceptable", "High", "Limited", "Minimal"}


def test_no_duplicate_instructions_in_train_set(train_data):
    """
    Regression test: an earlier version of this dataset padded its count by
    prefixing "[Scenario variant N]" onto ~40 repeated sentences, giving the
    model far less real signal than its size suggested. Every instruction in
    the training set must be textually unique.
    """
    instructions = [ex["instruction"] for ex in train_data]
    assert len(instructions) == len(set(instructions))
