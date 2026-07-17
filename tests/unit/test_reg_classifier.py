import json
from unittest.mock import MagicMock, patch

from regulation_advisor.classifier.reg_classifier import RegClassifier
from regulation_advisor.models import RegulationFinding

_FAKE_RESPONSE = json.dumps(
    {
        "risk_tier": "High",
        "obligation_type": "CONFORMITY",
        "urgency": "2026",
        "article_reference": "Annex III(4)",
        "reasoning": "Used in recruitment screening.",
    }
)


def test_llm_fallback_used_when_no_checkpoint_configured():
    clf = RegClassifier()
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=_FAKE_RESPONSE)
    with patch("regulation_advisor.llm.build_llm", return_value=fake_llm):
        result = clf.classify("An automated CV-screening tool.")
    assert isinstance(result, RegulationFinding)
    assert result.risk_tier == "High"
    assert result.confidence == 0.60


def test_llm_fallback_used_when_checkpoint_path_does_not_exist():
    clf = RegClassifier(checkpoint_path="outputs/does-not-exist")
    assert clf._pipeline is None


def test_load_failure_is_caught_and_falls_back(tmp_path):
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    clf = RegClassifier(checkpoint_path=str(checkpoint))
    # unsloth/transformers are not installed in this environment, so _load must
    # catch the ImportError and leave the pipeline unset rather than crashing.
    assert clf._pipeline is None


def test_finetuned_pipeline_used_when_loaded():
    clf = RegClassifier()
    fake_output = (
        "<|im_start|>user\nClassify the following EU AI Act regulation finding by risk tier, "
        "obligation type, and urgency.\n\nText: irrelevant<|im_end|>\n<|im_start|>assistant\n"
        + _FAKE_RESPONSE
    )
    clf._pipeline = MagicMock(return_value=[{"generated_text": fake_output}])
    result = clf.classify("irrelevant")
    assert result.risk_tier == "High"
    assert result.confidence == 0.92


def test_invalid_values_fall_back_to_safe_defaults():
    clf = RegClassifier()
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({"risk_tier": "Nonsense"}))
    with patch("regulation_advisor.llm.build_llm", return_value=fake_llm):
        result = clf.classify("some text")
    assert result.risk_tier == "Minimal"
    assert result.obligation_type == "TRANSPARENCY"
    assert result.urgency == "2026"
