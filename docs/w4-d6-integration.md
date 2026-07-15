# W4-D6 — Integration + Cleanup

**Branch:** `feat/w4-d6-integration`  
**Files changed:** `tests/integration/test_api_integration.py` (new), `pyproject.toml`, `notebooks/week4_notes.md`  
**Tests:** 71 total (53 unit + 18 integration), all pass

---

## What we built

Integration tests that verify all four Week 4 features work together correctly. A version bump. A retrospective.

---

## What is an integration test?

**Unit test:** Tests one function or class in complete isolation. No real dependencies — everything is mocked.

**Integration test:** Tests multiple components working together. Uses real code paths but still mocks infrastructure (LLM, index, external services).

**Analogy:**
- Unit test: Does the engine start when I turn the key?
- Integration test: Does the car move when I press the accelerator?
- End-to-end test: Can I drive from Berlin to Munich?

We do unit + integration in this codebase. True end-to-end (real LLM call, real index) is what `scripts/run_evaluation.py` does — you run that manually, not in CI.

---

## Test structure — classes grouping related cases

```python
class TestHealth:
    def test_returns_ok(self, client): ...
    def test_version_is_0_4_0(self, client): ...

class TestChatSync:
    def test_200_with_valid_message(self, client): ...
    def test_422_on_empty_message(self, client): ...
    ...
```

Grouping tests into classes is a style choice for large test files. pytest collects them the same way it collects top-level functions, but the class makes it visually clear what's being tested. When a test fails, the class name appears in the output: `TestChatSync::test_422_on_empty_message FAILED`.

---

## The shared fixtures

```python
@pytest.fixture()
def mock_agent():
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Article 5 prohibits social scoring.")],
        "retrieved_chunks": [],
        "confidence_score": 0.92,
    })
    return agent


@pytest.fixture()
def client(mock_agent):
    app = FastAPI()
    app.include_router(router)
    original = routes._agent
    routes.set_agent(mock_agent)
    with TestClient(app) as c:
        yield c
    routes.set_agent(original)
```

**How pytest fixtures work:**

A fixture is a function decorated with `@pytest.fixture()`. When a test function lists a fixture's name as a parameter, pytest calls the fixture and passes its return value.

```python
def test_200_with_valid_message(self, client):  # pytest sees "client" parameter
    # pytest calls the client() fixture → builds the app and returns TestClient
    r = client.post("/api/chat/sync", json={"message": "What is Article 5?"})
    assert r.status_code == 200
```

The `client` fixture depends on `mock_agent` — pytest calls them in order automatically. You never write:
```python
# DON'T do this in test functions
mock_agent = MagicMock(...)
client = build_test_client(mock_agent)
```

Fixtures are reusable. Multiple tests share the same setup logic without repeating it.

**The `yield` pattern in fixtures:**

```python
@pytest.fixture()
def client(mock_agent):
    ...
    routes.set_agent(mock_agent)
    with TestClient(app) as c:
        yield c                    # ← test runs HERE
    routes.set_agent(original)     # ← cleanup runs AFTER test
```

Code before `yield` is **setup**. Code after `yield` is **teardown**. The teardown runs even if the test fails — guaranteed cleanup. This is the same pattern as Python's context manager (`with` statement).

---

## Edge case tests

```python
def test_422_on_message_too_long(self, client):
    r = client.post("/api/chat/sync", json={"message": "x" * 2001})
    assert r.status_code == 422
```

`Field(max_length=2000)` in `ChatRequest` rejects messages over 2000 characters. The 422 is returned by FastAPI automatically — we don't write any "if len(message) > 2000" code. Pydantic handles it. This test verifies the constraint is actually enforced.

```python
def test_503_when_agent_not_ready(self):
    app = FastAPI()
    app.include_router(router)
    original = routes._agent
    routes.set_agent(None)
    try:
        with TestClient(app) as c:
            r = c.post("/api/chat/sync", json={"message": "hello"})
            assert r.status_code == 503
    finally:
        routes.set_agent(original)
```

This test creates its own client (without the shared fixture) because it needs to set `_agent = None` — which would break the other tests. The `try/finally` ensures the original value is restored even if the assertion fails.

---

## `TestMetricsStore` — testing pure logic

```python
class TestMetricsStore:
    def test_acceptable_true_when_above_thresholds(self, scores_path):
        metrics_store.save(faithfulness=0.81, answer_relevancy=0.71, ...)
        assert metrics_store.load().acceptable is True

    def test_acceptable_false_when_below_faithfulness(self, scores_path):
        metrics_store.save(faithfulness=0.79, answer_relevancy=0.75, ...)
        assert metrics_store.load().acceptable is False
```

These test the business rule: "acceptable means faithfulness ≥ 0.80 AND answer_relevancy ≥ 0.70." The threshold logic lives in `metrics_store.save()`:

```python
acceptable = faithfulness >= 0.80 and answer_relevancy >= 0.70
```

Both tests are needed. Test 1 proves the `True` case works. Test 2 proves it correctly rejects values just below the threshold (0.79 < 0.80). A single test that only checks the passing case might pass even if the logic was `acceptable = True` (a broken implementation that always passes).

---

## Version bump

```toml
# pyproject.toml
version = "0.4.0"
description = "Agentic RAG advisor for EU AI Act compliance — FastAPI REST API + streaming + ChromaDB + Gradio Evaluation Dashboard"
```

Version numbers follow **semantic versioning**: `MAJOR.MINOR.PATCH`.
- `MAJOR`: breaking changes (old clients stop working)
- `MINOR`: new features (old clients keep working)
- `PATCH`: bug fixes

0.2.1 → 0.4.0: We skipped 0.3.x because Weeks 1–3 built the agent+eval stack. Week 4 is a significant milestone (API + production storage), so we jump to 0.4.0.

---

## The full test run

```bash
pytest tests/ -v -q

# Output:
# 53 unit tests    (individual module checks)
# 18 integration   (API layer + metrics store end-to-end)
# ─────────────────
# 71 total, 0 failed
```

---

## Gate check

```bash
pytest tests/ -q                         # 71/71 pass
uvicorn regulation_advisor.api.app:app --port 8000
curl http://localhost:8000/api/health    # {"status":"ok","version":"0.4.0",...}
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Article 5?"}' | python -m json.tool
# Full ChatResponse JSON with answer, sources, session_id
```
