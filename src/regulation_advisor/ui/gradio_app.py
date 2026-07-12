"""
Gradio UI — ChatInterface with citation display.
v0.1 built in Week 1 Day 6 (simple chain).
v0.2 updated in Week 2 Day 5 (LangGraph agent).
v0.3 updated in Week 3 Day 5 (evaluation dashboard tab).
v0.4 updated in Week 4 Day 5 (mounted on FastAPI).
"""
from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from regulation_advisor.config import settings
from regulation_advisor.llm import build_llm
from regulation_advisor.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _build_chain():
    """
    Assemble the LangChain RAG chain:
        ChatPromptTemplate | <LLM> | StrOutputParser
    The chain accepts {context} and {question} as inputs and returns a plain string.
    """
    system_prompt = (_PROMPTS_DIR / "system_prompt.txt").read_text()
    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def _format_context(chunks) -> str:
    """
    Turn a list of RegulationChunk objects into a readable context block for the LLM.
    Each chunk is prefixed with its source and article number so the LLM can cite it.

    Example output:
        [eu_ai_act.pdf — Article 5]
        The following AI practices shall be prohibited: ...

        ---

        [eu_ai_act.pdf — Article 6]
        Classification rules for high-risk AI systems ...
    """
    parts = []
    for chunk in chunks:
        header = f"[{chunk.source_document} — Article {chunk.article_number}]"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def build_ui(retriever: Retriever) -> gr.Blocks:
    """
    Build the Gradio ChatInterface.

    Args:
        retriever: A loaded Retriever (wraps FAISS store + embedder).
                   Called on every user message to fetch relevant chunks.

    Returns:
        A gr.Blocks object ready to be launched with demo.launch().
    """
    chain = _build_chain()

    def respond(message: str, history: list) -> str:
        """
        Called by Gradio on every user message.
        1. Retrieve k most relevant regulation chunks.
        2. Format them into a context string.
        3. Invoke the LLM chain.
        4. Return the generated answer (Gradio displays it automatically).
        """
        logger.info("User query: %s", message)
        result = retriever.search(message, k=settings.retrieval_k)
        context = _format_context(result.chunks)
        answer = chain.invoke({"context": context, "question": message})
        logger.info("Answer generated (%d chars)", len(answer))
        return answer

    with gr.Blocks(title="RegulationAdvisor v0.1") as demo:
        gr.Markdown(
            "## EU AI Act Compliance Advisor\n"
            "Ask any question about the EU AI Act or GDPR. "
            "Every answer cites the relevant Article."
        )
        gr.ChatInterface(fn=respond, title="")

    return demo
