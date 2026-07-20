"""
Gradio UI — RegulationAdvisor.

Two-tab interface: Chat (streaming LangGraph agent) and Evaluation Dashboard
(RAGAS scores with on-demand evaluation trigger).

build_ui() reads the agent lazily from api.routes._agent so the Gradio app
can be mounted on FastAPI before the lifespan fires.
"""
from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Generator
from datetime import date

import gradio as gr
from langchain_core.messages import AIMessageChunk, SystemMessage

from regulation_advisor.agent.state import CRITICAL_KEYWORDS
from regulation_advisor.evaluation.guardrails import build_guardrail_chain
from regulation_advisor.models import RegulationChunk

logger = logging.getLogger(__name__)

_guardrails = build_guardrail_chain()

_CRITICAL_WARNING = (
    "\n\n---\n⚠️ **Critical finding** — this topic involves prohibited practices "
    "or significant penalties. Verify with a qualified legal professional before acting."
)

_DATE_CONTEXT_TEMPLATE = (
    "Today's date is {today}. "
    "When discussing regulatory deadlines, always calculate exactly how many days, "
    "weeks, or months remain from today's date — do not use approximate or "
    "static figures when calculating dates and times."
)


def _context_chunks_from_state(agent: object, config: dict) -> list[RegulationChunk]:
    """
    Pull the article numbers the agent actually retrieved during this turn.

    After agent.stream() finishes, the LangGraph checkpointer holds the full
    message history. Tool messages contain the regulation text returned by
    search_regulations. We parse article numbers from that text and build
    lightweight RegulationChunk objects so the citation guardrail can check
    whether the LLM cited an article that was never retrieved.
    """
    try:
        state = agent.get_state(config)
        tool_texts = " ".join(
            m.content for m in state.values.get("messages", [])
            if hasattr(m, "type") and m.type == "tool"
        )
        article_numbers = set(re.findall(r"Article\s+(\d+[a-z]?)", tool_texts, re.IGNORECASE))
        return [
            RegulationChunk(content="", article_number=a, article_title="", source_document="")
            for a in article_numbers
        ]
    except Exception:
        return []


_RISK_BADGE = {
    "Unacceptable": "🔴 **Risk: Unacceptable** — prohibited practice",
    "High": "🟠 **Risk: High** — conformity assessment required",
    "Limited": "🟡 **Risk: Limited** — transparency obligations apply",
    "Minimal": "🟢 **Risk: Minimal** — no specific obligation",
}


def _get_agent():
    """Lazy agent accessor — reads from routes at call time, not import time."""
    from regulation_advisor.api.routes import _agent
    return _agent


def _get_classifier():
    """Lazy classifier accessor — reads from routes at call time, not import time."""
    from regulation_advisor.api.routes import _classifier
    return _classifier


def _fetch_metrics() -> dict | None:
    """Read the latest scores from disk. Returns None if no scores yet."""
    from regulation_advisor.api.metrics_store import load
    result = load()
    if result is None:
        return None
    return result.model_dump()


def build_ui() -> gr.Blocks:
    """Build the two-tab Gradio UI. Agent is read lazily at request time."""
    session_id = str(uuid.uuid4())

    # ── Chat tab ──────────────────────────────────────────────────────────────

    def respond(message: str, history: list) -> Generator[str, None, None]:
        agent = _get_agent()
        if agent is None:
            yield "Service not ready — agent is still loading. Please retry in a moment."
            return

        config = {"configurable": {"thread_id": session_id}}

        messages: list = [("human", message)]
        if not history:
            today = date.today().strftime("%B %d, %Y")
            date_message = SystemMessage(content=_DATE_CONTEXT_TEMPLATE.format(today=today))
            messages = [date_message] + messages

        partial = ""
        for chunk, _ in agent.stream({"messages": messages}, config=config, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                partial += chunk.content
                yield partial

        if any(kw.lower() in partial.lower() for kw in CRITICAL_KEYWORDS):
            partial = partial + _CRITICAL_WARNING
            yield partial

        chunks = _context_chunks_from_state(agent, config)
        guard = _guardrails.check(partial, chunks, confidence=1.0)
        final_text = partial
        if guard.warnings:
            final_text = partial + "\n\n" + "\n\n".join(guard.warnings)
            yield final_text

        try:
            finding = _get_classifier().classify(partial)
            badge = _RISK_BADGE.get(finding.risk_tier, "")
            if badge:
                yield final_text + f"\n\n---\n{badge}"
        except Exception:
            logger.exception("RegClassifier failed — answer shown without a risk badge")

        logger.info(
            "Streamed answer (%d chars, guardrail_passed=%s, warnings=%d)",
            len(partial), guard.passed, len(guard.warnings),
        )

    # ── Evaluation Dashboard tab ──────────────────────────────────────────────

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

    # ── Layout ────────────────────────────────────────────────────────────────

    with gr.Blocks(title="RegulationAdvisor v0.4") as demo:

        with gr.Tab("Chat"):
            gr.Markdown(
                "## EU AI Act Compliance Advisor\n"
                "Ask any question about the EU AI Act or GDPR. "
                "Every answer cites the relevant Article. "
                "Critical findings are flagged for legal review."
            )
            gr.ChatInterface(fn=respond, title="")

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

    return demo
