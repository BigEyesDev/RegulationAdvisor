from unittest.mock import patch

from regulation_advisor.classifier.reg_classifier import RegClassifier
from regulation_advisor.models import RegulationFinding


def test_stub_returns_regulation_finding():
    clf = RegClassifier()
    result = clf.classify("An AI system for CV screening.")
    assert isinstance(result, RegulationFinding)
    assert result.risk_tier in {"Unacceptable", "High", "Limited", "Minimal"}


def test_stub_confidence_is_zero_without_model():
    clf = RegClassifier()
    assert clf.classify("any text").confidence == 0.0


def test_classifier_with_model_path_calls_load():
    with patch.object(RegClassifier, "_load") as mock_load:
        RegClassifier(model_path="some/path")
        mock_load.assert_called_once_with("some/path")
