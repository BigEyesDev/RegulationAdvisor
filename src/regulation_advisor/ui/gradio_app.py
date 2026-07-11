"""
Gradio UI — ChatInterface with citation display.
v0.1 built in Week 1 Day 6 (simple chain).
v0.2 updated in Week 2 Day 5 (LangGraph agent).
v0.3 updated in Week 3 Day 5 (evaluation dashboard tab).
v0.4 updated in Week 4 Day 5 (mounted on FastAPI).
"""
from __future__ import annotations

import logging
import gradio as gr

logger = logging.getLogger(__name__)


def build_ui(agent) -> gr.Blocks:
    """
    Build the Gradio interface.
    Pass in the agent/chain — UI does not care which version it is.
    """
    # TODO Week 1 Day 6: implement the ChatInterface
    # See master plan Week 1 Day 6 for full implementation

    with gr.Blocks(title="RegulationAdvisor") as demo:
        gr.Markdown("## EU AI Act Compliance Advisor\n*Coming soon — Week 1 Day 6*")

    return demo
