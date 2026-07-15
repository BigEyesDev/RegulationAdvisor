# W4-D5 — Evaluation Dashboard

**Branch:** `feat/w4-d5-eval-dashboard`  
**Files changed:** `ui/gradio_app.py`  
**Tests:** All 53 unit tests pass (no regression)

---

## What we built

The Gradio UI now has two tabs:
1. **Chat** — the existing conversation interface (unchanged)
2. **Evaluation Dashboard** — shows RAGAS scores, a run button, and a refresh button

---

## Why a UI for metrics?

The `GET /api/metrics` endpoint returns JSON — useful for CI, scripts, and programmatic access. But for demos and interviews, you want to show a live number on a screen, not paste curl output.

Clicking "Run Evaluation" in front of someone and watching the scores appear is a demo moment. It makes the eval infrastructure visible and real.

---

## The two-tab layout

```python
with gr.Blocks(title="RegulationAdvisor v0.4") as demo:

    with gr.Tab("Chat"):
        gr.Markdown("## EU AI Act Compliance Advisor\n...")
        gr.ChatInterface(fn=respond, title="")

    with gr.Tab("Evaluation Dashboard"):
        gr.Markdown("## RAGAS Evaluation Scores\n...")
        with gr.Row():
            run_btn = gr.Button("Run Evaluation", variant="primary")
            refresh_btn = gr.Button("Refresh Scores")
        status_box = gr.Markdown("*No scores loaded yet.*")
        with gr.Row():
            faith_num  = gr.Number(label="Faithfulness", precision=3)
            ...
```

**Gradio `Blocks`** is the layout API — like HTML `<div>` but declarative. Inside it:
- `gr.Tab()` creates a named tab
- `gr.Row()` puts components side by side
- `gr.Number()` is a read-only number display
- `gr.Markdown()` renders formatted text

---

## Event wiring — how buttons connect to functions

```python
score_outputs = [status_box, faith_num, rel_num, prec_num, recall_num]

run_btn.click(fn=trigger_eval, outputs=[status_box])
refresh_btn.click(fn=refresh_scores, outputs=score_outputs)
demo.load(fn=refresh_scores, outputs=score_outputs)
```

**How Gradio events work:**
- `.click(fn=..., outputs=[...])` — when this button is clicked, call `fn()`, and write its return values into the listed components
- `demo.load(fn=...)` — call `fn()` when the page loads (so scores appear immediately without clicking)

The function's return values must match the number and order of `outputs`. `refresh_scores()` returns 5 values → maps to 5 outputs:

```python
def refresh_scores():
    data = _fetch_metrics()
    if data is None:
        return ("No scores yet...", None, None, None, None)
    return status, faithfulness, answer_relevancy, context_precision, context_recall
```

---

## `_fetch_metrics()` — reads from disk, not from the API endpoint

```python
def _fetch_metrics() -> dict | None:
    from regulation_advisor.api.metrics_store import load
    result = load()
    if result is None:
        return None
    return result.model_dump()
```

Since Gradio and FastAPI run in the same process, calling the Python function directly (`metrics_store.load()`) is simpler and faster than making an HTTP request to `/api/metrics`. HTTP is for external callers. Internal callers use the module directly.

**Analogy:** If you work in the same building as the kitchen, you walk to the kitchen. You don't call DoorDash.

---

## `trigger_eval()` — running evaluation from the UI

```python
def trigger_eval():
    from regulation_advisor.api import routes
    import asyncio
    if routes._evaluation_running:
        return "⏳ Already running..."
    routes._evaluation_running = True
    try:
        asyncio.get_event_loop().run_until_complete(routes._run_evaluation())
    except RuntimeError:
        # Already in a running event loop — run in a background thread instead
        import threading
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(routes._run_evaluation())
            loop.close()
        threading.Thread(target=run).start()
        return "⏳ Evaluation started. Click Refresh in a few minutes."
    return "✅ Evaluation complete. Click Refresh to see scores."
```

**The event loop problem explained:**

`routes._run_evaluation()` is `async def`. To call an async function from sync code, you need `asyncio.get_event_loop().run_until_complete(...)`.

But Gradio runs its own event loop. If you call `run_until_complete()` from inside Gradio's loop, Python raises a `RuntimeError: This event loop is already running`. The `except RuntimeError` block handles this by spinning up a new thread with its own event loop.

**Why threads work:** Each thread can have its own asyncio event loop. By creating a new thread with `threading.Thread`, we escape Gradio's loop and can create a fresh one for the evaluation.

This is not ideal (the UI says "check back in a few minutes" instead of waiting), but it's correct and safe for an eval run that takes minutes anyway.

---

## Understanding `gr.Number` vs `gr.Textbox`

```python
faith_num = gr.Number(label="Faithfulness", precision=3)
```

`gr.Number` displays a float with `precision=3` (three decimal places). When the function returns `None`, the component shows blank. When it receives `0.843`, it shows `0.843`.

We don't use `gr.Textbox` because a Number component enforces numeric display and formatting. A Textbox would require us to format the number as a string manually.

---

## What the dashboard looks like

```
┌─────────────────────────────────────────────────────────┐
│ [Chat] [Evaluation Dashboard]                           │
├─────────────────────────────────────────────────────────┤
│ ## RAGAS Evaluation Scores                              │
│ Targets: Faithfulness ≥ 0.80 · All others ≥ 0.70      │
│                                                         │
│ [Run Evaluation ▶]  [Refresh Scores ↻]                 │
│                                                         │
│ *v0.4 — evaluated 2026-07-15 — ✅ PASS*                │
│                                                         │
│ Faithfulness  Answer Relevancy  Context Precision  ...  │
│   0.843           0.791              0.734          ...  │
└─────────────────────────────────────────────────────────┘
```

---

## Gate check

```bash
# All unit tests still pass
pytest tests/unit/ -q   # 53/53 pass

# Manual: start the server
uvicorn regulation_advisor.api.app:app --port 8000

# Open http://localhost:8000
# - "Chat" tab: type a question
# - "Evaluation Dashboard" tab: click Refresh, see scores (or "No scores yet")
# - Click "Run Evaluation": wait 3-5 min, click Refresh, see updated scores
```
