---
language: en
license: apache-2.0
base_model: Qwen/Qwen3-1.7B-Instruct
tags:
  - peft
  - qlora
  - legal
  - eu-ai-act
  - text-classification
---

# reg-classifier-qwen3-1.7b

QLoRA fine-tuned adapter on top of Qwen3-1.7B-Instruct.
Classifies EU AI Act regulation text into structured findings.

## Task

Given a description of an AI system or practice, returns:
- `risk_tier`: Unacceptable | High | Limited | Minimal
- `obligation_type`: PROHIBITED | TRANSPARENCY | CONFORMITY | REGISTRATION | GENERAL_PURPOSE
- `urgency`: IMMEDIATE | 2025 | 2026 | 2027
- `article_reference`: e.g. "Article 5(1)(d)"
- `reasoning`: one-sentence explanation

## Training

- Base model: Qwen/Qwen3-1.7B-Instruct
- Training examples: 160 (80/10/10 split)
- LoRA rank r=16, alpha=32, target: q_proj, k_proj, v_proj, o_proj
- Epochs: 3, lr: 2e-4 cosine, batch size 16 (4x4 accumulation)

## Evaluation (test set, 20 examples)

Fill in with your own run of `scripts/evaluate_classifier.py` before publishing —
see `evals/classifier_eval.json` for the generated numbers.

| Metric | Base model | Fine-tuned |
|--------|-----------|-----------|
| Accuracy | _pending_ | _pending_ |
| Unacceptable F1 | _pending_ | _pending_ |
| High F1 | _pending_ | _pending_ |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Instruct")
model = PeftModel.from_pretrained(base, "BigEyesDev/reg-classifier-qwen3-1.7b")
tokenizer = AutoTokenizer.from_pretrained("BigEyesDev/reg-classifier-qwen3-1.7b")
```

## Limitations

Trained on 160 synthetically generated examples reviewed by a human.
Do not use for actual legal compliance decisions. Output is AI-generated guidance only.
