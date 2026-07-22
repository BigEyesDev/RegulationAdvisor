"""
Gradio UI — RegulationAdvisor.

Chat tab: streaming LangGraph agent with an optional BYOK provider/model/key
selector. The Evaluation Dashboard tab is parked in ui/eval_dashboard.py, not
currently mounted here — see that module's docstring and version_Plan.md.

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

# Curated model choices per BYOK provider — a free-text model field lets
# visitors hit deprecated/misspelled slugs (we hit this ourselves picking a
# free-tier default; see CHANGELOG v0.6.5-0.6.6), so the UI only offers a
# short list instead. Not every entry has been live-verified against a real
# key in this account (openai/deepseek-v4-flash were; the rest are current
# vendor-documented model names as of this writing) — worth spot-checking
# again if one starts erroring for everyone.
_BYOK_MODELS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"],
    "google": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
    "openrouter": [
        "deepseek/deepseek-v4-flash", "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"
    ],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-4-8"],
}


def _get_agent():
    """Lazy agent accessor — reads from routes at call time, not import time."""
    from regulation_advisor.api.routes import _agent
    return _agent


class _ByokRequiredError(Exception):
    """Raised when no key was supplied and the deployment has no default key."""


def _agent_for_call(api_key: str | None, provider: str | None = None, model: str | None = None):
    """
    Mirrors api/routes.py's ``_agent_for_request`` for the Gradio chat tab.

    A key builds a throwaway agent for this call only, using the caller's
    chosen ``provider``/``model`` rather than the deployment's own default —
    otherwise a key for e.g. Groq would get sent to whatever provider this
    deployment happens to be configured with (an auth failure, not a
    security issue: the caller's key always wins over the deployment's, see
    llm.py's ``api_key or settings.*`` — but a confusing one). The agent is
    held in a local variable and discarded when ``respond()`` returns —
    nothing is written to disk or kept beyond the lifetime of this one turn.
    With no key, the shared default agent is used only if the deployment
    actually has a default key configured — a BYOK-only deployment (empty
    keys in its secrets, so no visitor's usage is ever billed to the
    deployer) raises instead of silently calling out with an empty key.
    """
    if api_key:
        from regulation_advisor.agent.graph import build_agent_graph

        return build_agent_graph(provider=provider, model=model, api_key=api_key)
    from regulation_advisor.config import settings

    if not settings.has_default_llm_key:
        raise _ByokRequiredError
    return _get_agent()


def _get_classifier():
    """Lazy classifier accessor — reads from routes at call time, not import time."""
    from regulation_advisor.api.routes import _classifier
    return _classifier


def build_ui() -> gr.Blocks:
    """Build the two-tab Gradio UI. Agent is read lazily at request time."""
    session_id = str(uuid.uuid4())

    # ── Chat tab ──────────────────────────────────────────────────────────────

    def respond(
        message: str,
        history: list,
        provider: str = "openai",
        model: str = "",
        api_key: str = "",
    ) -> Generator[str, None, None]:
        try:
            agent = _agent_for_call(
                api_key.strip() or None, provider=provider, model=model.strip() or None
            )
        except _ByokRequiredError:
            yield (
                "This deployment has no default API key configured — "
                "paste your own key above to chat."
            )
            return
        if agent is None:
            yield "Service not ready — agent is still loading. Please retry in a moment."
            return

        config = {"configurable": {"thread_id": session_id}}

        messages: list = [("human", message)]
        if not history:
            today = date.today().strftime("%B %d, %Y")
            date_message = SystemMessage(content=_DATE_CONTEXT_TEMPLATE.format(today=today))
            messages = [date_message] + messages

        try:
            partial = ""
            for chunk, _ in agent.stream(
                {"messages": messages}, config=config, stream_mode="messages"
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    partial += chunk.content
                    yield partial
        except Exception as exc:
            logger.warning("Chat turn failed: %s", type(exc).__name__)
            yield (
                "The configured LLM provider rejected the request. "
                "Check your API key and try again."
            )
            return

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

    # ── Layout ────────────────────────────────────────────────────────────────

    with gr.Blocks(title="RegulationAdvisor v0.4") as demo:

        with gr.Tab("Chat"):
            gr.Markdown(
                "## EU AI Act Compliance Advisor\n"
                "Ask any question about the EU AI Act or GDPR. "
                "Every answer cites the relevant Article. "
                "Critical findings are flagged for legal review."
            )
            with gr.Accordion("Use your own API key (optional)", open=False):
                gr.Markdown(
                    "Supported: **OpenAI**, **Anthropic (Claude)**, **Groq**, "
                    "**Google Gemini**, **OpenRouter** (OpenRouter can also reach "
                    "Claude, GPT-4o, etc. by model name). Pick the provider your key "
                    "is for, then a model for that provider — used only for your "
                    "messages in this browser tab, never written to disk, and gone "
                    "the moment you refresh or close."
                )
                provider_dropdown = gr.Dropdown(
                    choices=["openai", "anthropic", "groq", "google", "openrouter"],
                    value="openai",
                    label="Provider",
                )
                model_dropdown = gr.Dropdown(
                    choices=_BYOK_MODELS["openai"],
                    value=_BYOK_MODELS["openai"][0],
                    label="Model",
                )
                api_key_box = gr.Textbox(
                    label="API key",
                    type="password",
                    placeholder="Leave blank to use the free default",
                )

                def _update_model_choices(provider: str):
                    choices = _BYOK_MODELS[provider]
                    return gr.Dropdown(choices=choices, value=choices[0])

                provider_dropdown.change(
                    fn=_update_model_choices,
                    inputs=[provider_dropdown],
                    outputs=[model_dropdown],
                )

            gr.ChatInterface(
                fn=respond,
                title="",
                additional_inputs=[provider_dropdown, model_dropdown, api_key_box],
            )

        # Evaluation Dashboard tab intentionally not mounted here — parked in
        # ui/eval_dashboard.py pending the eval-history rework (version_Plan.md).

    return demo
