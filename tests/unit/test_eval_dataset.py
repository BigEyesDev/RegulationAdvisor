"""Sanity checks for the evaluation dataset — run before any evaluation."""
import json
from pathlib import Path

QA_PATH = Path("evals/qa_pairs.json")
REQUIRED_KEYS = {"question", "ground_truth_answer", "expected_article"}


def load_pairs() -> list[dict]:
    with open(QA_PATH) as f:
        return json.load(f)


def test_minimum_pair_count():
    assert len(load_pairs()) >= 20


def test_all_pairs_have_required_keys():
    for pair in load_pairs():
        assert REQUIRED_KEYS <= pair.keys(), f"Missing keys in: {pair}"


def test_no_empty_fields():
    for pair in load_pairs():
        for key in REQUIRED_KEYS:
            assert pair[key].strip(), f"Empty field '{key}' in: {pair['question']}"
