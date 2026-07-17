"""
Publish fine-tuned RegClassifier adapter to HuggingFace Hub.

Prerequisites:
    huggingface-cli login    # one-time setup
    # or: set HF_TOKEN in .env

Usage:
    python scripts/publish_to_hub.py \
        --checkpoint outputs/reg_classifier/final \
        --repo BigEyesDev/reg-classifier-qwen3-1.7b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def publish(checkpoint: str, repo_id: str) -> None:
    from unsloth import FastLanguageModel

    print(f"Loading checkpoint: {checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    print(f"Pushing to Hub: {repo_id}")
    model.push_to_hub(repo_id, private=False)
    tokenizer.push_to_hub(repo_id, private=False)
    print(f"Published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/reg_classifier/final")
    parser.add_argument("--repo", default="BigEyesDev/reg-classifier-qwen3-1.7b")
    args = parser.parse_args()
    publish(args.checkpoint, args.repo)
