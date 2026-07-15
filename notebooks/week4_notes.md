# Week 4 Retrospective

**Version:** v0.4.0  
**Date:** 2026-07-15

---

## What shipped

| Feature | Status |
|---|---|
| FastAPI foundation — health, schemas, lifespan | ✅ |
| Streaming /api/chat (SSE) + sync /api/chat/sync | ✅ |
| ChromaDB migration via config switch | ✅ |
| /api/metrics + /api/evaluate background task | ✅ |
| Evaluation Dashboard tab in Gradio | ✅ |
| Integration tests (18 new, 71 total) | ✅ |

---

## Architecture decisions made

**Why `_agent = None` with `set_agent()` instead of FastAPI `Depends()`?**

The `Depends()` pattern is cleaner for large apps but requires restructuring every route signature. The module-level singleton follows the same pattern already used in `tools.py` for the retriever — consistency across the codebase is more valuable than using FastAPI's preferred pattern for a solo project.

**Why SSE instead of WebSockets for streaming?**

SSE is simpler and sufficient. WebSockets are bidirectional — needed for real-time collaboration or multiplayer. For token streaming, SSE (one-directional server → client) is exactly right. Less infrastructure, works through HTTP/2 proxies, simpler client code.

**Why a file cache for metrics instead of a database?**

Evaluation results change once per eval run (every few days at most). A JSON file is zero-infrastructure persistence. A SQLite or Postgres table would be the right answer if you tracked per-run history, but right now "latest run" is all that's needed.

**Did the lazy `_get_agent()` pattern create any bugs?**

No. The key insight: Gradio Blocks object is created at module import time (needed for `gr.mount_gradio_app`), but Gradio only calls `respond()` when a user sends a message. By then, the FastAPI lifespan has already run and `routes._agent` is set. The timing works out naturally.

---

## What was harder than expected

**SSE streaming with LangGraph's `astream_events()`:** LangGraph emits many event types. Initially I got all events including tool starts/ends and node transitions. Filtering to only `on_chat_model_stream` required understanding the event taxonomy. Once you know that's the event for LLM token output, it's simple.

**The Gradio + asyncio interaction in `trigger_eval()`:** Gradio runs its own event loop. Running `asyncio.get_event_loop().run_until_complete()` from inside a Gradio handler raises `RuntimeError: This event loop is already running`. The fix — `threading.Thread` with a fresh event loop — is correct but non-obvious. This is a known Gradio limitation.

---

## RAGAS scores (run after this week's changes)

Fill in after running: `python scripts/run_evaluation.py`

| Metric | Week 3 (FAISS) | Week 4 (FAISS default) | Δ |
|---|---|---|---|
| Faithfulness | TBD | TBD | — |
| Answer Relevancy | TBD | TBD | — |
| Context Precision | TBD | TBD | — |
| Context Recall | TBD | TBD | — |

*Note: Update `evals/baseline_scores.json` after running evaluation. The Gradio dashboard will show these scores automatically.*

---

## What to improve in Week 5 (Docker)

- Add ChromaDB to Docker Compose so it starts automatically with the app
- Replace the in-process `trigger_eval()` threading hack with a proper Celery/RQ task queue (optional)
- Update the HuggingFace Space to use the FastAPI entry point instead of `gradio_app.py`
