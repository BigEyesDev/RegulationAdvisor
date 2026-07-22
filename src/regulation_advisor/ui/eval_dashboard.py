"""
RAGAS Evaluation Dashboard tab — parked, not currently wired into build_ui().

Pulled out of the public Gradio app because its "Run Evaluation" trigger
bypasses the BYOK-required guard: it always calls out through the shared
default agent with no per-visitor key check, so any visitor could burn
through the whole RAGAS QA set on the deployer's key (or fail confusingly
in a BYOK-only deployment). See version_Plan.md for the eval-history rework
that should replace trigger_eval() with a read-only history view before
this tab comes back.

To re-enable once that's done: call ``add_eval_dashboard_tab(demo)`` inside
the ``with gr.Blocks(...) as demo:`` block in gradio_app.py, next to the
Chat tab.
"""
from __future__ import annotations

import gradio as gr


def _fetch_metrics() -> dict | None:
    """Read the latest scores from disk. Returns None if no scores yet."""
    from regulation_advisor.api.metrics_store import load
    result = load()
    if result is None:
        return None
    return result.model_dump()


def refresh_scores():
    data = _fetch_metrics()
    if data is None:
        return ("No scores yet — click **Run Evaluation** to generate them.",
                None, None, None, None)
    m = data
    status = (
        f"**v{m['version']}** — evaluated {m['evaluated_at'][:10]} — "
        f"{'✅ PASS' if m['acceptable'] else '❌ FAIL'}"
    )
    return (
        status, m["faithfulness"], m["answer_relevancy"],
        m["context_precision"], m["context_recall"],
    )


def trigger_eval():
    import asyncio

    from regulation_advisor.api import routes
    if routes._evaluation_running:
        return "⏳ Evaluation already running — check back in a few minutes."
    routes._evaluation_running = True
    try:
        asyncio.get_event_loop().run_until_complete(routes._run_evaluation())
    except RuntimeError:
        # Already in a running event loop (can happen in Gradio's thread)
        import threading
        def run():
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            loop.run_until_complete(routes._run_evaluation())
            loop.close()
        t = threading.Thread(target=run)
        t.start()
        return "⏳ Evaluation started in background. Click **Refresh** in a few minutes."
    return "✅ Evaluation complete. Click **Refresh** to see updated scores."


def add_eval_dashboard_tab(demo: gr.Blocks) -> None:
    """Build the Evaluation Dashboard tab. Call from within a gr.Blocks context."""
    with gr.Tab("Evaluation Dashboard"):
        gr.Markdown(
            "## RAGAS Evaluation Scores\n"
            "Measures how faithfully the agent answers from the regulation text.\n\n"
            "**Targets:** Faithfulness ≥ 0.80 · All others ≥ 0.70"
        )

        with gr.Row():
            run_btn = gr.Button("Run Evaluation", variant="primary")
            refresh_btn = gr.Button("Refresh Scores")

        status_box = gr.Markdown("*No scores loaded yet.*")

        with gr.Row():
            faith_num  = gr.Number(label="Faithfulness",      precision=3)
            rel_num    = gr.Number(label="Answer Relevancy",  precision=3)
            prec_num   = gr.Number(label="Context Precision", precision=3)
            recall_num = gr.Number(label="Context Recall",    precision=3)

        score_outputs = [status_box, faith_num, rel_num, prec_num, recall_num]

        run_btn.click(fn=trigger_eval, outputs=[status_box])
        refresh_btn.click(fn=refresh_scores, outputs=score_outputs)
        demo.load(fn=refresh_scores, outputs=score_outputs)
