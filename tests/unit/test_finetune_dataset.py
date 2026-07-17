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
    assert len(train_data) > 100, "Training set should have >100 examples"


def test_every_example_has_required_keys(train_data):
    for ex in train_data:
        assert "instruction" in ex
        assert "response" in ex


def test_every_response_is_valid_json(train_data):
    for ex in train_data:
        parsed = json.loads(ex["response"])
        assert parsed["risk_tier"] in {"Unacceptable", "High", "Limited", "Minimal"}
