from regulation_advisor.evaluation.guardrails import build_guardrail_chain
from regulation_advisor.models import RegulationChunk

def _chunk(article: str) -> RegulationChunk:
    return RegulationChunk(content="test", article_number=article,
                           article_title="", source_document="test.pdf")

def test_low_confidence_fails():
    result = build_guardrail_chain(0.7).check("answer", [_chunk("5")], 0.5)
    assert result.passed is False

def test_hallucinated_article_flagged():
    result = build_guardrail_chain(0.0).check("See Article 99.", [_chunk("5")], 0.9)
    assert any("not in retrieved" in w.lower() for w in result.warnings)

def test_legal_claim_flagged():
    result = build_guardrail_chain(0.0).check("You must comply.", [_chunk("5")], 0.9)
    assert any("not legal advice" in w for w in result.warnings)
