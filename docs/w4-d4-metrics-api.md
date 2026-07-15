# W4-D4 — Metrics API

**Branch:** `feat/w4-d4-metrics-api`  
**Files changed:** `api/metrics_store.py` (new), `api/routes.py`, `api/schemas.py`  
**Tests:** `tests/unit/test_api_metrics.py`

---

## What we built

Two new endpoints:

| Endpoint | What it does |
|---|---|
| `GET /api/metrics` | Returns the latest RAGAS scores from disk |
| `POST /api/evaluate` | Starts a RAGAS evaluation in the background, returns immediately |

And a new module `api/metrics_store.py` that reads/writes `evals/baseline_scores.json`.

---

## The problem: evaluation takes 3–5 minutes

Running RAGAS against 20 Q&A pairs makes 20 LLM calls (one per question) plus additional calls for each metric judge. That takes 3–5 minutes.

If `POST /api/evaluate` waited for it to finish, the HTTP request would time out after 30–60 seconds. The client would get a "connection timed out" error even if the evaluation succeeded.

**Solution: background tasks.** The endpoint kicks off the work and immediately returns "started". The client polls `GET /api/metrics` later to see the results.

**Analogy:** You call a pharmacy and ask them to prepare a prescription. They say "it'll be ready in 20 minutes, come back then." They don't make you wait on the phone while they prepare it.

---

## `metrics_store.py` — file-based cache

```python
_SCORES_PATH = Path("evals/baseline_scores.json")

def load() -> MetricsResponse | None:
    if not _SCORES_PATH.exists():
        return None
    with open(_SCORES_PATH) as f:
        data = json.load(f)
    ...

def save(faithfulness, answer_relevancy, ...) -> None:
    payload = {"version": "v0.4", "metrics": {...}, "evaluated_at": ...}
    _SCORES_PATH.write_text(json.dumps(payload, indent=2))
```

This module knows nothing about HTTP or FastAPI. It only reads and writes a JSON file. This separation keeps responsibilities clear:
- `metrics_store.py` = persistence (read/write file)
- `routes.py` = HTTP interface (when to read/write, what to return)

**Why a file and not a database?** RAGAS scores change infrequently (once per evaluation run). A file is sufficient, zero infrastructure needed, and it persists across container restarts just like ChromaDB does.

---

## The background task

```python
@router.post("/api/evaluate", response_model=EvaluateResponse)
async def trigger_evaluation(background_tasks: BackgroundTasks) -> EvaluateResponse:
    global _evaluation_running
    if _evaluation_running:
        return EvaluateResponse(status="already_running", ...)
    background_tasks.add_task(_run_evaluation)
    return EvaluateResponse(status="started", ...)
```

`BackgroundTasks` is a FastAPI concept. You add a function to it with `background_tasks.add_task(fn)`, and FastAPI runs that function after sending the HTTP response. The caller gets the response in milliseconds; the work happens afterward.

`_evaluation_running` is a flag that prevents two evaluations from running at the same time. If someone calls `POST /api/evaluate` twice quickly, the second call returns `"already_running"` instead of starting a duplicate run.

**The gate pattern:** `if _evaluation_running: return early`. This is called a **guard clause**. Instead of nesting all the logic inside `if not _evaluation_running:`, we return early when the condition is true. Keeps the happy path unindented and readable.

---

## Inside `_run_evaluation()`

```python
async def _run_evaluation() -> None:
    global _evaluation_running
    _evaluation_running = True
    try:
        harness = EvaluationHarness(Path("evals/qa_pairs.json"))

        def pipeline_fn(question: str) -> tuple[str, list[str]]:
            import asyncio
            config = {"configurable": {"thread_id": f"eval_{hash(question)}"}}
            result = asyncio.get_event_loop().run_until_complete(
                _agent.ainvoke({"messages": [("human", question)]}, config=config)
            )
            answer = result["messages"][-1].content
            contexts = [c.content for c in result.get("retrieved_chunks", [])]
            return answer, contexts

        scores = harness.run(pipeline_fn)
        metrics_store.save(
            faithfulness=scores.faithfulness,
            ...
        )
    except Exception:
        logger.exception("Evaluation failed")
    finally:
        _evaluation_running = False
```

**Three things to notice:**

1. **`try / finally`** — `_evaluation_running = False` is in `finally`, not `except`. This means it runs whether the evaluation succeeds or fails. Without this, a failed evaluation would permanently block future runs.

2. **`asyncio.get_event_loop().run_until_complete(...)`** — `harness.run()` is synchronous (it was built in Week 3 without async in mind). But `_agent.ainvoke()` is async. `run_until_complete()` bridges the two — it runs an async function from synchronous code. Not ideal for production (you'd want a full async harness), but correct for now.

3. **`logger.exception()`** — logs the exception with the full stack trace. Never silently swallow exceptions in background tasks — without logging, a failure is invisible and the caller just never sees updated metrics.

---

## `GET /api/metrics` — what 404 means

```python
@router.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    result = metrics_store.load()
    if result is None:
        raise HTTPException(status_code=404, detail="No evaluation scores yet.")
    return result
```

`404 Not Found` normally means "the URL doesn't exist". Here we use it to mean "the resource exists (the endpoint) but the data doesn't (no evaluation has run)". This is standard REST practice — a resource that hasn't been created yet is "not found".

The alternative would be returning 200 with null values. But that's ambiguous — did the evaluation run and produce nulls, or hasn't it run at all? A 404 is unambiguous.

---

## Tests

```python
def test_metrics_404_when_no_file(client, scores_file):
    response = client.get("/api/metrics")
    assert response.status_code == 404
```

`scores_file` is a pytest fixture that creates an empty temp directory and monkeypatches `metrics_store._SCORES_PATH` to point there. Because the file doesn't exist yet, load() returns None, and the route returns 404.

```python
def test_metrics_store_save_and_load_roundtrip(scores_file, monkeypatch):
    monkeypatch.setattr(metrics_store, "_SCORES_PATH", scores_file)
    metrics_store.save(faithfulness=0.85, ...)
    result = metrics_store.load()
    assert result.faithfulness == 0.85
    assert result.acceptable is True  # 0.85 >= 0.80 threshold
```

A **roundtrip test** — save data, load it back, check it survived intact. This verifies the JSON serialization and the threshold logic together.

```python
def test_evaluate_returns_already_running_when_busy(client, monkeypatch):
    monkeypatch.setattr(routes, "_evaluation_running", True)
    response = client.post("/api/evaluate")
    assert response.json()["status"] == "already_running"
```

`monkeypatch.setattr` directly sets the module-level variable to `True`, simulating the in-progress state. No need to actually start a real evaluation.

---

## How to test manually

```bash
# Check metrics (404 expected before first run)
curl http://localhost:8000/api/metrics

# Start evaluation (returns immediately)
curl -X POST http://localhost:8000/api/evaluate
# {"status": "started", "message": "..."}

# Wait 3-5 minutes, then check metrics
curl http://localhost:8000/api/metrics | python -m json.tool
```

---

## Gate check

```bash
pytest tests/unit/test_api_metrics.py -v   # 5/5 pass
curl http://localhost:8000/api/metrics      # 404 (or scores if previously run)
curl -X POST http://localhost:8000/api/evaluate  # {"status": "started"}
```
