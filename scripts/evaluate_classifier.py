"""
Compare base (prompted) vs fine-tuned RegClassifier on the held-out test set.

Usage:
    python scripts/evaluate_classifier.py \
        --checkpoint outputs/reg_classifier/final \
        --output evals/classifier_eval.json

Prints a sklearn classification_report for both models.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


CLASSIFY_PROMPT = """\
Classify the following EU AI Act regulation finding.
Respond ONLY with valid JSON matching this schema:
{{"risk_tier": "Unacceptable|High|Limited|Minimal",
  "obligation_type": "PROHIBITED|TRANSPARENCY|CONFORMITY|REGISTRATION|GENERAL_PURPOSE",
  "urgency": "IMMEDIATE|2025|2026|2027",
  "article_reference": "Article X",
  "reasoning": "one sentence"}}

Text: {text}"""


def classify_with_base_llm(text: str) -> str:
    from regulation_advisor.llm import build_llm

    llm = build_llm()
    response = llm.invoke(CLASSIFY_PROMPT.format(text=text))
    raw = response.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())["risk_tier"]
    return "Unknown"


def classify_with_finetuned(text: str, pipeline) -> str:
    prompt = (
        f"<|im_start|>user\n"
        f"Classify the following EU AI Act regulation finding by risk tier, obligation type, and urgency.\n\nText: {text}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    output = pipeline(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    generated = output[len(prompt):]
    match = re.search(r"\{.*\}", generated, re.DOTALL)
    if match:
        return json.loads(match.group())["risk_tier"]
    return "Unknown"


def main(checkpoint: str, output_path: Path) -> None:
    from transformers import pipeline as hf_pipeline
    from unsloth import FastLanguageModel
    from sklearn.metrics import classification_report

    test_data = json.loads(Path("evals/finetune/test.json").read_text())
    texts = [ex["instruction"].split("Text: ", 1)[1] for ex in test_data]
    labels = [json.loads(ex["response"])["risk_tier"] for ex in test_data]

    print("Evaluating base model (prompted)...")
    base_preds = [classify_with_base_llm(t) for t in texts]
    print("\nBase model results:")
    print(classification_report(labels, base_preds, zero_division=0))

    print("Loading fine-tuned model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # enable optimised inference mode
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print("Evaluating fine-tuned model...")
    ft_preds = [classify_with_finetuned(t, pipe) for t in texts]
    print("\nFine-tuned model results:")
    print(classification_report(labels, ft_preds, zero_division=0))

    result = {
        "base_model": classification_report(labels, base_preds, output_dict=True, zero_division=0),
        "finetuned_model": classification_report(labels, ft_preds, output_dict=True, zero_division=0),
        "test_size": len(test_data),
    }
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/reg_classifier/final")
    parser.add_argument("--output", default="evals/classifier_eval.json")
    args = parser.parse_args()
    main(args.checkpoint, Path(args.output))
