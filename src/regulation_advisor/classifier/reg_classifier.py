"""
RegClassifier — real inference using the fine-tuned Qwen3-1.7B LoRA adapter.

The model is loaded lazily on first construction (not at import time) to keep startup fast.
Falls back to a deterministic base-model prompt if no checkpoint is configured or loadable.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from regulation_advisor.models import RegulationFinding

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """\
Classify the following EU AI Act regulation finding by risk tier, obligation type, and urgency.
Respond ONLY with valid JSON. No extra text.

Schema: {{"risk_tier": "Unacceptable|High|Limited|Minimal",
  "obligation_type": "PROHIBITED|TRANSPARENCY|CONFORMITY|REGISTRATION|GENERAL_PURPOSE",
  "urgency": "IMMEDIATE|2025|2026|2027",
  "article_reference": "Article X",
  "reasoning": "one sentence"}}

Text: {text}"""

_RISK_TIERS = {"Unacceptable", "High", "Limited", "Minimal"}
_OBLIGATION_TYPES = {"PROHIBITED", "TRANSPARENCY", "CONFORMITY", "REGISTRATION", "GENERAL_PURPOSE"}
_URGENCIES = {"IMMEDIATE", "2025", "2026", "2027"}


def _parse_json_response(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output: {raw[:200]}")
    return json.loads(match.group())


class RegClassifier:
    """
    Classifies regulation text into risk tier, obligation type, and urgency.

    Two operating modes:
    - Fine-tuned: loads the LoRA adapter from checkpoint_path (fast, consistent)
    - Fallback:   prompts the configured LLM via build_llm() (slower, variable)
    """

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self._checkpoint_path = checkpoint_path
        self._pipeline = None
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load(checkpoint_path)

    def _load(self, path: str) -> None:
        try:
            from unsloth import FastLanguageModel
            from transformers import pipeline as hf_pipeline

            logger.info("Loading RegClassifier from %s", path)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            self._pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)
            logger.info("RegClassifier loaded")
        except Exception:
            logger.exception("Failed to load fine-tuned model — using LLM fallback")

    def classify(self, text: str) -> RegulationFinding:
        if self._pipeline is not None:
            return self._classify_finetuned(text)
        return self._classify_with_llm(text)

    def _classify_finetuned(self, text: str) -> RegulationFinding:
        prompt = (
            f"<|im_start|>user\n"
            f"Classify the following EU AI Act regulation finding by risk tier, "
            f"obligation type, and urgency.\n\nText: {text}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        raw_out = self._pipeline(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        parsed = _parse_json_response(raw_out[len(prompt):])
        return self._to_finding(parsed, confidence=0.92)

    def _classify_with_llm(self, text: str) -> RegulationFinding:
        from regulation_advisor.llm import build_llm

        llm = build_llm()
        response = llm.invoke(_CLASSIFY_PROMPT.format(text=text))
        parsed = _parse_json_response(response.content)
        return self._to_finding(parsed, confidence=0.60)

    @staticmethod
    def _to_finding(parsed: dict, confidence: float) -> RegulationFinding:
        risk_tier = parsed.get("risk_tier")
        obligation_type = parsed.get("obligation_type")
        urgency = parsed.get("urgency")
        return RegulationFinding(
            article=parsed.get("article_reference", "unknown"),
            risk_tier=risk_tier if risk_tier in _RISK_TIERS else "Minimal",
            obligation_type=(
                obligation_type if obligation_type in _OBLIGATION_TYPES else "TRANSPARENCY"
            ),
            urgency=urgency if urgency in _URGENCIES else "2026",
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
        )
