# RegulationAdvisor — AI Engineering Master Plan

**Goal:** Become a production-grade AI engineer. Build one real, deployed, interview-ready application
from start to finish. Learn clean Python and design patterns through the project, not separately.

**Timeline:** 7 weeks at 3–4 hours/day  
**Your background:** Senior ML engineer, strong Python, no prior AI engineering stack experience  
**Output:** RegulationAdvisor v1.0 — a live, deployed Agentic RAG system for EU AI Act compliance

---

## Before You Start: Setup Everything Once

### 1. Free API Keys (get all of these now — takes 30 minutes total)

| Provider | What You Get Free | Best For | Sign Up |
|----------|------------------|----------|---------|
| **Groq** | Qwen3-32B, Llama 4 Scout, DeepSeek R1 — 30K TPM | Fast inference during development | console.groq.com |
| **Google AI Studio** | Gemini 2.5 Flash — 1,500 req/day | Long-context tasks, multimodal | aistudio.google.com |
| **OpenRouter** | 35+ free models via one key | Model variety, GLM-5.1 access | openrouter.ai |
| **NVIDIA NIM** | GLM-4.7, Llama — 40 RPM | Showcasing GLM in your portfolio | build.nvidia.com |
| **HuggingFace** | Inference API — Qwen, Llama, Gemma | HF ecosystem integration | huggingface.co |
| **Tavily** | Web search API — 1,000 req/month free | Agent web search tool | tavily.com |

**Important:** Free tiers change. Never hardcode a single provider. Design your code to swap providers
with one config change — this is also a core Python design pattern lesson (Strategy pattern).

### 2. Model Strategy for This Project

Use different models for different roles:

| Role | Model | Why |
|------|-------|-----|
| **Main conversation LLM** | `Qwen3-32B` via Groq (free) | Strong reasoning, tool use, free |
| **Fallback / quality-critical** | `Gemini 2.5 Flash` via AI Studio (free) | Long context (1M tokens), reliable |
| **Evaluation / judging** | `GLM-5.1` via OpenRouter (free) | Strong at structured evaluation |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local) | No API cost, fully local |
| **Fine-tuning base** | `Qwen3-1.7B` or `Gemma 3 4B` (local) | Fits your 16GB GPU comfortably |

Check `huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard` before you start
Week 6 — the fine-tuning model landscape moves fast.

### 3. GPU Strategy

| Phase | Where to Run | Why |
|-------|-------------|-----|
| Weeks 1–5 | Your local machine (CPU is fine) | No GPU needed for RAG, agents, evaluation |
| Week 6 (fine-tuning) | Your local 16GB GPU | Qwen3-1.7B with Unsloth fits easily |
| If you need a larger model | AWS `g5.xlarge` (A10G 24GB, ~$1.00/hour) | Rent only for training — not always-on |
| Deployment | AWS ECS (CPU) or HuggingFace Spaces (free) | Inference via API, no GPU needed at runtime |

**AWS learning path:** Start with ECS for deployment (Week 5). You get Docker, ECR, load balancers,
and IAM all at once — the relevant skills — without paying for always-on GPU.

### 4. Project Structure (create this on Day 1)

```
regulation-advisor/
├── src/
│   └── regulation_advisor/
│       ├── __init__.py
│       ├── config.py               # Pydantic BaseSettings — all config here
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py          # Factory pattern: loads PDF, CSV, MD
│       │   ├── chunkers.py         # Strategy pattern: pluggable chunking
│       │   └── pipeline.py         # Orchestrates ingestion
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── embeddings.py       # Strategy pattern: swappable embedding model
│       │   ├── store.py            # Repository pattern: FAISS → ChromaDB swap
│       │   └── retriever.py        # Retrieval logic
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── state.py            # LangGraph TypedDict state
│       │   ├── tools.py            # Tool definitions with type annotations
│       │   └── graph.py            # StateGraph construction
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── harness.py          # RAGAS evaluation runner
│       │   └── guardrails.py       # Chain of Responsibility pattern
│       ├── classifier/
│       │   ├── __init__.py
│       │   └── reg_classifier.py   # Fine-tuned model inference wrapper
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py              # FastAPI + Gradio mount
│       │   ├── routes.py           # Route definitions
│       │   └── schemas.py          # Pydantic request/response models
│       └── ui/
│           ├── __init__.py
│           └── gradio_app.py       # Gradio ChatInterface + Tabs
├── tests/
│   ├── unit/                       # pytest unit tests — one per module
│   └── integration/                # End-to-end tests
├── notebooks/                      # Scratch / exploration only — not production
├── data/
│   ├── eu_ai_act.pdf
│   ├── gdpr.pdf
│   ├── ai_act_timeline.csv
│   ├── risk_classification.csv
│   └── penalty_structure.csv
├── evals/
│   ├── qa_pairs.json               # 20 labelled Q&A pairs (Week 3)
│   └── promptfoo.yaml              # Regression test suite (Week 3)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                  # Modern Python packaging — not setup.py
└── README.md
```

---

## Python Clean Coding — The Rules for This Project

These are not optional. Apply them from Day 1. They become habit through repetition.

### Rule 1: Type hints everywhere
```python
# BAD
def chunk_text(text, size, overlap):
    ...

# GOOD
def chunk_text(text: str, size: int, overlap: int) -> list[dict[str, str]]:
    ...
```

### Rule 2: Pydantic for all data models
```python
from pydantic import BaseModel, Field
from typing import Literal

class RegulationChunk(BaseModel):
    content: str
    article_number: str
    article_title: str
    source_document: str
    page_number: int | None = None

class RegulationFinding(BaseModel):
    article: str
    risk_tier: Literal["Unacceptable", "High", "Limited", "Minimal"]
    obligation_type: str
    deadline: str
    confidence: float = Field(ge=0.0, le=1.0)
```

### Rule 3: Protocol for interfaces (Strategy pattern)
```python
from typing import Protocol

class Chunker(Protocol):
    def chunk(self, text: str) -> list[RegulationChunk]:
        ...

class EmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...

class VectorStore(Protocol):
    def add(self, chunks: list[RegulationChunk]) -> None:
        ...
    def search(self, query_embedding: list[float], k: int) -> list[RegulationChunk]:
        ...
```

This is why you can swap FAISS for ChromaDB in Week 4 with zero code changes outside `store.py`.

### Rule 4: Config via Pydantic BaseSettings (never hardcode)
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    llm_provider: str = "groq"
    llm_model: str = "qwen/qwen3-32b"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5

    # Vector store
    vector_store_backend: str = "faiss"       # swap to "chromadb" in Week 4
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    class Config:
        env_file = ".env"

settings = Settings()
```

### Rule 5: Proper logging (never print in production code)
```python
import logging

logger = logging.getLogger(__name__)

# BAD
print(f"Loaded {len(chunks)} chunks")

# GOOD
logger.info("Loaded %d chunks from %s", len(chunks), source_path)
```

### Rule 6: Context managers for resources
```python
# BAD — file left open if exception occurs
f = open("data.json")
data = json.load(f)
f.close()

# GOOD
with open("data.json") as f:
    data = json.load(f)
```

### Rule 7: One test file per module
```
tests/unit/test_chunkers.py      ← tests for ingestion/chunkers.py
tests/unit/test_retriever.py     ← tests for retrieval/retriever.py
tests/unit/test_guardrails.py    ← tests for evaluation/guardrails.py
```

Run with `pytest tests/ -v` after every significant change.

---

## Design Patterns — Where Each One Appears

| Pattern | Where in the Project | Why |
|---------|---------------------|-----|
| **Strategy** | `chunkers.py`, `embeddings.py`, LLM provider | Swap implementations without changing callers |
| **Factory** | `loaders.py` | `DocumentLoaderFactory.create("pdf")` → correct loader |
| **Repository** | `store.py` | `VectorStoreRepository` hides FAISS vs ChromaDB |
| **Chain of Responsibility** | `guardrails.py` | Each check is a handler; chain them together |
| **Observer** | Evaluation harness | Pipeline emits events; RAGAS subscribes |
| **Dependency Injection** | All constructors | Pass dependencies in — makes testing trivial |

You will encounter these patterns naturally as the project grows. The names are not important —
what matters is recognising when a pattern solves a specific problem.

---

## Week 1 — Foundation + First Working RAG

**Daily time:** 3–4 hours  
**Deliverable:** A working (rough) RAG pipeline that can answer "What are the prohibited AI practices?"
and return Article 5 with a citation.

### Day 1 — Environment + Project Structure

**Morning:**
- Create the project structure above exactly. Not a notebook. A real Python package.
- Set up `pyproject.toml` (modern Python packaging — learn this, not `setup.py`)
- Set up `.env` with your API keys (never commit this to git)
- Set up `pytest` with one empty test file

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[project]
name = "regulation-advisor"
requires-python = ">=3.11"
```

**Afternoon:**
- Create `src/regulation_advisor/config.py` with the `Settings` class above
- Write one test: `assert settings.llm_provider == "groq"` 
- Download the EU AI Act PDF and GDPR PDF into `data/`
- Create the three CSV files (timeline, risk classification, penalties) — this is 45 minutes of reading the Act, and it teaches you the regulation better than any tutorial

**Gate check:** `pytest tests/ -v` passes. You can import `from regulation_advisor.config import settings`.

---

### Day 2 — Document Loading (Factory Pattern)

**What you build:** `src/regulation_advisor/ingestion/loaders.py`

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol
import logging

logger = logging.getLogger(__name__)

class DocumentLoader(Protocol):
    def load(self, path: Path) -> list[str]:
        ...

class PDFLoader:
    def load(self, path: Path) -> list[str]:
        from llama_index.readers.file import PyMuPDFReader
        reader = PyMuPDFReader()
        documents = reader.load(file_path=str(path))
        logger.info("Loaded %d pages from %s", len(documents), path.name)
        return [doc.text for doc in documents]

class CSVLoader:
    def load(self, path: Path) -> list[str]:
        import csv
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
        return rows

class MarkdownLoader:
    def load(self, path: Path) -> list[str]:
        return [path.read_text(encoding="utf-8")]

class DocumentLoaderFactory:
    _loaders: dict[str, type[DocumentLoader]] = {
        ".pdf": PDFLoader,
        ".csv": CSVLoader,
        ".md": MarkdownLoader,
    }

    @classmethod
    def create(cls, path: Path) -> DocumentLoader:
        suffix = path.suffix.lower()
        loader_class = cls._loaders.get(suffix)
        if not loader_class:
            raise ValueError(f"No loader for file type: {suffix}")
        return loader_class()
```

**Afternoon:** Write `tests/unit/test_loaders.py` — load a small test PDF, assert you get back a non-empty list of strings.

**Gate check:** The factory correctly routes `.pdf`, `.csv`, `.md` to the right loader.

---

### Day 3 — Chunking (Strategy Pattern) — Most Important Day of Week 1

**What you build:** `src/regulation_advisor/ingestion/chunkers.py`

This is the most impactful engineering decision in a RAG system. Do all three approaches and measure.

```python
import re
from typing import Protocol
from regulation_advisor.models import RegulationChunk


class Chunker(Protocol):
    def chunk(self, text: str, source: str) -> list[RegulationChunk]:
        ...


class RecursiveCharacterChunker:
    """Baseline — splits by character count. Fast, dumb, breaks articles mid-sentence."""

    def __init__(self, size: int = 1000, overlap: int = 200):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> list[RegulationChunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.size, chunk_overlap=self.overlap
        )
        texts = splitter.split_text(text)
        return [
            RegulationChunk(
                content=t, article_number="unknown", article_title="", source_document=source
            )
            for t in texts
        ]


class ArticleAwareChunker:
    """
    Splits EU AI Act text by Article boundaries.
    Each chunk = one complete Article with metadata.
    This is the one that actually works for legal text.
    """

    ARTICLE_PATTERN = re.compile(
        r"(Article\s+(\d+[a-z]?)\s*\n([^\n]+)\n)", re.IGNORECASE
    )

    def chunk(self, text: str, source: str) -> list[RegulationChunk]:
        chunks: list[RegulationChunk] = []
        matches = list(self.ARTICLE_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            article_number = match.group(2)
            article_title = match.group(3).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            if len(content) > 50:  # skip near-empty matches
                chunks.append(
                    RegulationChunk(
                        content=content,
                        article_number=article_number,
                        article_title=article_title,
                        source_document=source,
                    )
                )

        return chunks
```

**Afternoon:** In a notebook (`notebooks/chunking_comparison.ipynb`), run both chunkers on the EU AI Act.
Ask "What AI practices are prohibited?" — does `ArticleAwareChunker` return Article 5?
Does `RecursiveCharacterChunker`? Write down the numbers. This becomes your LinkedIn article evidence.

**Gate check:** `ArticleAwareChunker` produces chunks where each has an article number in its metadata.

---

### Day 4 — Embeddings + FAISS (Repository Pattern)

**What you build:** `src/regulation_advisor/retrieval/`

```python
# embeddings.py
from typing import Protocol
import numpy as np

class EmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=True)
```

```python
# store.py
import pickle
from pathlib import Path
import numpy as np
import faiss
from regulation_advisor.models import RegulationChunk

class FAISSVectorStore:
    """In-memory vector store. Fast for development. Replaced by ChromaDB in Week 4."""

    def __init__(self, dimension: int = 384):
        self._index = faiss.IndexFlatL2(dimension)
        self._chunks: list[RegulationChunk] = []

    def add(self, chunks: list[RegulationChunk], embeddings: np.ndarray) -> None:
        self._index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[RegulationChunk]:
        distances, indices = self._index.search(
            query_embedding.reshape(1, -1).astype(np.float32), k
        )
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]

    def save(self, path: Path) -> None:
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, path: Path) -> None:
        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
```

**Gate check:** You can embed 50 EU AI Act chunks, save the index, reload it, and retrieve the top-3 chunks for any query using cosine similarity.

---

### Day 5 — Ingest Pipeline + First End-to-End Test

**What you build:** `src/regulation_advisor/ingestion/pipeline.py` + a `scripts/ingest.py`

```python
# ingestion/pipeline.py
import logging
from pathlib import Path
from regulation_advisor.ingestion.loaders import DocumentLoaderFactory
from regulation_advisor.ingestion.chunkers import ArticleAwareChunker
from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
from regulation_advisor.retrieval.store import FAISSVectorStore
from regulation_advisor.config import settings

logger = logging.getLogger(__name__)

def run_ingestion(data_dir: Path, index_dir: Path) -> FAISSVectorStore:
    store = FAISSVectorStore()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    chunker = ArticleAwareChunker()

    for file_path in sorted(data_dir.iterdir()):
        loader = DocumentLoaderFactory.create(file_path)
        texts = loader.load(file_path)
        full_text = "\n".join(texts)
        chunks = chunker.chunk(full_text, source=file_path.name)
        logger.info("Chunked %s into %d chunks", file_path.name, len(chunks))

        contents = [c.content for c in chunks]
        embeddings = embedder.encode(contents)
        store.add(chunks, embeddings)

    store.save(index_dir)
    logger.info("Saved index to %s", index_dir)
    return store
```

**Afternoon:** Run ingestion on your full corpus. Then write a retrieval test: ask "What are the prohibited AI practices?" and verify Article 5 is in the top-3 results.

**Gate check:** `python scripts/ingest.py` completes. `pytest tests/integration/test_retrieval.py` passes.

---

### Day 6 — First RAG Chain (Gradio v0)

**Resources to watch first (3 hours):**  
LangChain Academy "Introduction to LangChain" — do only the first 3 lessons (chains, prompts, structured output). Skip the rest — you'll hit them naturally.

**What you build:** Add `src/regulation_advisor/ui/gradio_app.py`

```python
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from regulation_advisor.retrieval.retriever import Retriever
from regulation_advisor.config import settings

def build_ui(retriever: Retriever) -> gr.Blocks:
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an EU AI Act compliance advisor. 
Answer based ONLY on the provided regulation text.
Always cite the specific Article number.
If the answer is not in the context, say so clearly.

Context:
{context}"""),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    def respond(message: str, history: list) -> str:
        chunks = retriever.search(message, k=5)
        context = "\n\n---\n\n".join(
            f"[{c.source_document} — Article {c.article_number}]\n{c.content}"
            for c in chunks
        )
        return chain.invoke({"context": context, "question": message})

    with gr.Blocks(title="RegulationAdvisor v0.1") as demo:
        gr.Markdown("## EU AI Act Compliance Advisor")
        chatbot = gr.ChatInterface(fn=respond, title="")
    
    return demo
```

Test the 5 benchmark queries. Note which ones fail and why.

**Push to HuggingFace Spaces.** You now have a live public URL.

**Gate check:** All 5 benchmark queries return an answer with a cited Article number.

---

### Week 1 Retrospective (30 minutes, Day 6 evening)

Write in `notebooks/week1_notes.md`:
- Which chunking approach gave better retrieval? By how much?
- What broke in the EU AI Act PDF extraction?
- Where did the Strategy pattern save you time?

---

## Week 2 — LangGraph Agent (RegulationAdvisor v0.2)

**Daily time:** 3–4 hours  
**Deliverable:** A LangGraph agent with 3 tools that correctly routes queries. Human-in-the-loop for critical findings.

### Days 1–2: Learn LangGraph Properly

**Watch (6 hours total — do not skip this):**  
LangChain Academy "Introduction to LangGraph" (free, official, current)  
URL: `academy.langchain.com/courses/intro-to-langgraph`

This is the only LangGraph resource you need. It covers:
- `StateGraph`, `TypedDict` state, nodes, edges
- Conditional routing
- Checkpointers (the modern answer to memory)
- Human-in-the-loop with `interrupt_before`

Do not touch the capstone code during these two days. Just build their course projects.

**After the course, make sure you can answer:**
1. What is a `TypedDict` state and why does LangGraph use it?
2. What is a checkpointer and how does it differ from conversation memory?
3. What does `interrupt_before` do and when would you use it?

---

### Day 3 — Define State + Tools

**What you build:** `src/regulation_advisor/agent/state.py` and `tools.py`

```python
# state.py
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from regulation_advisor.models import RegulationChunk

class RegAdvisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_chunks: list[RegulationChunk]
    tools_used: list[str]
    confidence_score: float
    is_critical_finding: bool
```

```python
# tools.py
from langchain_core.tools import tool
from regulation_advisor.retrieval.retriever import Retriever
from regulation_advisor.config import settings
import csv

_retriever: Retriever | None = None

def set_retriever(r: Retriever) -> None:
    global _retriever
    _retriever = r

@tool
def search_regulations(query: str) -> str:
    """Search EU AI Act and GDPR regulation documents for relevant articles and provisions."""
    if _retriever is None:
        return "Retriever not initialised."
    chunks = _retriever.search(query, k=5)
    return "\n\n---\n\n".join(
        f"[Article {c.article_number} — {c.source_document}]\n{c.content}"
        for c in chunks
    )

@tool
def query_structured_data(question: str) -> str:
    """Query structured regulation data: timelines, penalties, and risk classifications."""
    # Read CSVs and return relevant rows as formatted text
    results = []
    for csv_file in ["ai_act_timeline.csv", "risk_classification.csv", "penalty_structure.csv"]:
        with open(f"data/{csv_file}") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
                if any(word.lower() in row_text.lower() for word in question.split()):
                    results.append(row_text)
    return "\n".join(results) if results else "No matching structured data found."

@tool
def search_web(query: str) -> str:
    """Search the web for recent EU AI Act enforcement news and guidance."""
    from tavily import TavilyClient
    client = TavilyClient(api_key=settings.tavily_api_key)
    results = client.search(query, max_results=3)
    return "\n\n".join(r["content"] for r in results.get("results", []))
```

---

### Day 4 — Build the StateGraph

**What you build:** `src/regulation_advisor/agent/graph.py`

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from regulation_advisor.agent.state import RegAdvisorState
from regulation_advisor.agent.tools import search_regulations, query_structured_data, search_web
from regulation_advisor.config import settings

TOOLS = [search_regulations, query_structured_data, search_web]
CRITICAL_KEYWORDS = ["prohibited", "banned", "Article 5", "35,000,000", "7%", "illegal"]

def build_agent_graph():
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: RegAdvisorState) -> RegAdvisorState:
        response = llm_with_tools.invoke(state["messages"])
        # Check if this is a critical finding requiring human review
        is_critical = any(
            kw.lower() in response.content.lower() for kw in CRITICAL_KEYWORDS
        )
        return {
            "messages": [response],
            "is_critical_finding": is_critical,
        }

    def should_continue(state: RegAdvisorState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        if state.get("is_critical_finding"):
            return "human_review"
        return END

    graph = StateGraph(RegAdvisorState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("human_review", lambda s: s)  # interrupt_before handles the pause

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    graph.add_edge("human_review", END)

    checkpointer = MemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )
```

---

### Day 5 — Wire Agent into Gradio + Test

Update `gradio_app.py` to use the agent graph instead of the raw chain.
The conversation history is now handled by the LangGraph checkpointer — not a memory class.

**Test the multi-step query:**
> "My company deploys an AI system for employee emotion recognition in the workplace. What are our obligations?"

Expected: Agent calls `search_regulations` → finds Article 5(1)(f) →
calls `query_structured_data` → finds enforcement date → triggers `human_review` (CRITICAL).

**Gate check:** Agent correctly uses different tools for different queries. Human-in-the-loop fires for prohibited practice queries.

---

### Day 6 — smolagents Comparison (4 hours)

**Install:** `pip install smolagents`

Implement the same 3-tool agent in smolagents. It takes ~4 hours once you know what the agent needs to do:

```python
from smolagents import ToolCallingAgent, LiteLLMModel
from regulation_advisor.agent.tools import search_regulations, query_structured_data, search_web

model = LiteLLMModel(model_id="groq/qwen/qwen3-32b")

agent = ToolCallingAgent(
    tools=[search_regulations, query_structured_data, search_web],
    model=model,
    max_steps=5,
)

result = agent.run(
    "What are the prohibited AI practices and when did they become enforceable?"
)
```

**Write `docs/smolagents_comparison.md`** — compare:
- How you defined tools (schema vs docstring)
- How memory works (checkpointer vs in-run only)
- How much code it took
- What you lost and what you gained

This becomes LinkedIn Article #3. Interviewers at BMW and SAP ask about framework tradeoffs.

---

## Week 3 — Evaluation + Guardrails (RegulationAdvisor v0.3)

**Daily time:** 3–4 hours  
**Deliverable:** RAGAS scorecard, working guardrail layer, 30 regression test cases.

### Day 1 — Create Evaluation Dataset

**No code today.** Read the EU AI Act. Create `evals/qa_pairs.json`:

```json
[
  {
    "question": "What AI practices are completely prohibited under the EU AI Act?",
    "ground_truth_answer": "Article 5 prohibits: social scoring by public authorities, real-time remote biometric identification in public spaces (with narrow exceptions), AI that exploits vulnerabilities of persons, subliminal manipulation, emotion recognition in workplaces and education, untargeted facial image scraping, and biometric categorisation inferring protected characteristics.",
    "expected_article": "Article 5"
  }
]
```

Create 20 of these. You must know the correct answer for each — no guessing.
This dataset is permanent. It runs in Week 3, Week 4, and is extended to 30 pairs for promptfoo.

---

### Day 2 — RAGAS Evaluation Harness

**What you build:** `src/regulation_advisor/evaluation/harness.py`

```python
from dataclasses import dataclass
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import json
from pathlib import Path

@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def is_acceptable(self, threshold: float = 0.7) -> bool:
        return all([
            self.faithfulness >= threshold,
            self.answer_relevancy >= threshold,
        ])

class EvaluationHarness:
    def __init__(self, qa_pairs_path: Path):
        with open(qa_pairs_path) as f:
            self._qa_pairs = json.load(f)

    def run(self, pipeline_fn) -> RAGASResult:
        """
        pipeline_fn: takes a question string, returns (answer, list[context_strings])
        """
        data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        for pair in self._qa_pairs:
            answer, contexts = pipeline_fn(pair["question"])
            data["question"].append(pair["question"])
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(pair["ground_truth_answer"])

        result = evaluate(
            dataset=Dataset.from_dict(data),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return RAGASResult(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
        )
```

**Gate check:** You have a baseline RAGAS scorecard for your v0.1 pipeline. Write down the numbers.

---

### Day 3 — Guardrail Layer (Chain of Responsibility Pattern)

**What you build:** `src/regulation_advisor/evaluation/guardrails.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from regulation_advisor.models import RegulationChunk

LEGAL_CLAIM_PHRASES = ["you must", "it is illegal", "the fine is", "you are required to"]

@dataclass
class GuardrailResult:
    passed: bool
    warnings: list[str]
    confidence_score: float

class GuardrailHandler(ABC):
    def __init__(self, next_handler: "GuardrailHandler | None" = None):
        self._next = next_handler

    @abstractmethod
    def check(
        self, answer: str, chunks: list[RegulationChunk], faithfulness_score: float
    ) -> GuardrailResult:
        ...

    def _pass_to_next(
        self, answer: str, chunks: list[RegulationChunk], faithfulness_score: float
    ) -> GuardrailResult:
        if self._next:
            return self._next.check(answer, chunks, faithfulness_score)
        return GuardrailResult(passed=True, warnings=[], confidence_score=faithfulness_score)

class FaithfulnessCheck(GuardrailHandler):
    def __init__(self, threshold: float = 0.7, next_handler=None):
        super().__init__(next_handler)
        self._threshold = threshold

    def check(self, answer, chunks, faithfulness_score):
        if faithfulness_score < self._threshold:
            return GuardrailResult(
                passed=False,
                warnings=[f"Low confidence ({faithfulness_score:.2f}) — verify against original regulation"],
                confidence_score=faithfulness_score,
            )
        return self._pass_to_next(answer, chunks, faithfulness_score)

class CitationVerificationCheck(GuardrailHandler):
    def check(self, answer, chunks, faithfulness_score):
        import re
        cited_articles = re.findall(r"Article\s+(\d+)", answer, re.IGNORECASE)
        available_articles = {c.article_number for c in chunks}
        hallucinated = [a for a in cited_articles if a not in available_articles]
        if hallucinated:
            return GuardrailResult(
                passed=False,
                warnings=[f"Cited Articles not in retrieved context: {hallucinated}"],
                confidence_score=faithfulness_score,
            )
        return self._pass_to_next(answer, chunks, faithfulness_score)

class LegalClaimFlagCheck(GuardrailHandler):
    def check(self, answer, chunks, faithfulness_score):
        found = [p for p in LEGAL_CLAIM_PHRASES if p in answer.lower()]
        result = self._pass_to_next(answer, chunks, faithfulness_score)
        if found:
            result.warnings.append(
                "⚠️ This answer contains legal claims. This is AI-generated guidance, not legal advice."
            )
        return result

def build_guardrail_chain() -> GuardrailHandler:
    """Build the chain: faithfulness → citation → legal claim"""
    legal_check = LegalClaimFlagCheck()
    citation_check = CitationVerificationCheck(next_handler=legal_check)
    return FaithfulnessCheck(next_handler=citation_check)
```

---

### Days 4–5 — promptfoo Regression Suite

**Install:** `npm install -g promptfoo`

Extend your 20 Q&A pairs to 30. Create `evals/promptfoo.yaml`:

```yaml
prompts:
  - file://src/regulation_advisor/prompts/system_prompt.txt

providers:
  - id: python:scripts/eval_pipeline.py:run_query

tests:
  - vars:
      question: "What AI practices are completely prohibited?"
    assert:
      - type: contains
        value: "Article 5"
      - type: contains
        value: "social scoring"
      - type: llm-rubric
        value: "The answer names at least 3 of the prohibited practices from Article 5"

  - vars:
      question: "What is the fine for deploying a prohibited AI system?"
    assert:
      - type: contains
        value: "35,000,000"
      - type: contains
        value: "7%"
```

Create `.github/workflows/eval.yml` — runs `promptfoo eval` on every PR to `main`.

**Gate check:** A deliberate bad change to `system_prompt.txt` breaks at least 3 test cases and GitHub Actions catches it.

---

## Week 4 — FastAPI + Production Architecture

**Daily time:** 3–4 hours  
**Deliverable:** REST API alongside Gradio UI. Streaming responses. ChromaDB replacing FAISS.

### Days 1–2: FastAPI Backend

**Watch first (2 hours):**  
Fireship "FastAPI in 100 Seconds" (2 min) then FastAPI official tutorial — Chapters 1-5 only.  
URL: `fastapi.tiangolo.com/tutorial` — Focus: path operations, Pydantic models, streaming, background tasks.

**What you build:** `src/regulation_advisor/api/`

```python
# schemas.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence_score: float
    warnings: list[str]

class MetricsResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    last_evaluated_at: str
```

```python
# app.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import gradio as gr
from regulation_advisor.api.schemas import ChatRequest, ChatResponse, MetricsResponse
from regulation_advisor.agent.graph import build_agent_graph
from regulation_advisor.evaluation.harness import EvaluationHarness

app = FastAPI(title="RegulationAdvisor API", version="0.4.0")
agent = build_agent_graph()

@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def generate():
        config = {"configurable": {"thread_id": request.session_id}}
        async for event in agent.astream_events(
            {"messages": [("human", request.message)]}, config=config, version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data: {chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/metrics")
async def get_metrics() -> MetricsResponse:
    # Return cached metrics from last evaluation run
    ...

@app.post("/api/evaluate")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_evaluation)
    return {"status": "evaluation started"}

# Mount Gradio UI on top of FastAPI
from regulation_advisor.ui.gradio_app import build_ui
demo = build_ui(agent)
app = gr.mount_gradio_app(app, demo, path="/")
```

**Gate check:** `curl -X POST http://localhost:8000/api/chat -d '{"message":"What is Article 5?"}'` returns a streamed response.

---

### Day 3: ChromaDB Migration

```python
# retrieval/store.py — add ChromaDB implementation

class ChromaDBVectorStore:
    """Persistent vector store. Survives container restarts."""

    def __init__(self, host: str = "localhost", port: int = 8001, collection: str = "regulations"):
        import chromadb
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[RegulationChunk], embeddings: np.ndarray) -> None:
        self._collection.add(
            ids=[f"{c.source_document}_{c.article_number}_{i}" for i, c in enumerate(chunks)],
            embeddings=embeddings.tolist(),
            documents=[c.content for c in chunks],
            metadatas=[{"article": c.article_number, "source": c.source_document} for c in chunks],
        )

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[RegulationChunk]:
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=k
        )
        return [
            RegulationChunk(
                content=doc,
                article_number=meta["article"],
                article_title="",
                source_document=meta["source"],
            )
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
```

In `config.py`, change `vector_store_backend: str = "chromadb"`. The rest of the code does not change —  
because you used the Repository pattern.

**Gate check:** Restart the app. Data is still there.

---

### Days 4–5: Add Evaluation Tab to Gradio

```python
# In gradio_app.py — add a second tab

with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        gr.ChatInterface(fn=respond)

    with gr.Tab("Evaluation Dashboard"):
        with gr.Row():
            run_btn = gr.Button("Run RAGAS Evaluation", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            faithfulness_num = gr.Number(label="Faithfulness", precision=3)
            relevancy_num = gr.Number(label="Answer Relevancy", precision=3)
            precision_num = gr.Number(label="Context Precision", precision=3)
            recall_num = gr.Number(label="Context Recall", precision=3)
```

---

## Week 5 — Docker + Deployment

**Daily time:** 3–4 hours  
**Deliverable:** Containerised app deployed to HuggingFace Spaces AND AWS ECS.

### Day 1: Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e ".[all]"

# Copy source
COPY src/ src/
COPY data/ data/
COPY evals/ evals/

# Pre-build the FAISS index (or mount ChromaDB volume)
RUN python scripts/ingest.py

EXPOSE 8000
CMD ["uvicorn", "regulation_advisor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Day 2: Docker Compose

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      chromadb:
        condition: service_healthy
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - VECTOR_STORE_BACKEND=chromadb
      - CHROMA_HOST=chromadb
    env_file: .env

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  chroma_data:
```

**Gate check:** `docker compose up` → app works. `docker compose down && docker compose up` → vectors still there.

---

### Day 3: HuggingFace Spaces

```bash
huggingface-cli upload YOUR_USERNAME/regulation-advisor . --repo-type=space
```

Add your API keys as Space secrets (not in the repo). Live public URL in 10 minutes.

---

### Days 4–5: AWS ECS Deployment

This is where you learn AWS properly. Sequence:

1. **ECR** — Push your Docker image to AWS Container Registry
   ```bash
   aws ecr create-repository --repository-name regulation-advisor
   docker tag regulation-advisor:latest YOUR_ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/regulation-advisor
   docker push YOUR_ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/regulation-advisor
   ```

2. **ECS Task Definition** — Define CPU/memory (0.5 vCPU, 1GB RAM is enough for inference via API)

3. **ECS Service + Application Load Balancer** — Public URL, auto-restarts on failure

4. **Secrets Manager** — Store API keys securely (never in task definition as plaintext)

**What AWS skills you gain:** ECR, ECS, IAM roles, ALB, Secrets Manager — these appear in every ML engineering JD.

**Cost estimate:** ~$15–25/month for a small ECS service. Run it for the interview period, then tear down.

**Gate check:** `https://YOUR_ALB_DNS/api/metrics` returns your RAGAS scores JSON.

---

## Week 6 — Fine-Tuning (RegClassifier)

**Daily time:** 3–4 hours  
**Deliverable:** QLoRA fine-tuned classifier published on HuggingFace Hub. Integrated into the pipeline.

### Days 1–2: Dataset Creation + Concepts

**Read (not watch) — 2 hours:**
- Sebastian Raschka's LoRA insights article: `lightning.ai/pages/community/lora-insights`
- QLoRA paper abstract + Section 2: `arxiv.org/abs/2305.14314`

**Create training data** (~200 examples):

```json
{
  "instruction": "Classify this regulation finding by risk tier, obligation type, and urgency.\n\nText: An AI system that uses real-time remote biometric identification in publicly accessible spaces without exception.",
  "response": "{\n  \"risk_tier\": \"Unacceptable\",\n  \"obligation_type\": \"PROHIBITED\",\n  \"urgency\": \"IMMEDIATE\",\n  \"article_reference\": \"Article 5(1)(d)\",\n  \"reasoning\": \"Real-time remote biometric ID in public spaces is a prohibited practice under Article 5. Prohibition has been in force since 2 February 2025.\"\n}"
}
```

Use Claude to help you generate and validate these. You review and correct each one.
Split 80/10/10 (train/validation/test).

---

### Days 3–4: Fine-Tune with Unsloth

Use Colab or your local 16GB GPU. Qwen3-1.7B trains in approximately 45 minutes on your hardware.

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-1.7B-Instruct",  # or "google/gemma-3-4b-it"
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

# SFTTrainer from trl
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        output_dir="outputs/reg_classifier",
        logging_steps=10,
    ),
)

trainer.train()
```

**If you want a bigger model and more VRAM:** Rent `g5.xlarge` on AWS (A10G, 24GB) at ~$1.00/hour.
Train for 2 hours = $2. This is when AWS compute makes sense.

---

### Days 5–6: Evaluate + Publish

```python
# Compare base model vs fine-tuned on test set
from sklearn.metrics import classification_report

# Base model (prompted)
base_predictions = [classify_with_prompt(text) for text in test_texts]
# Fine-tuned model
finetuned_predictions = [reg_classifier.classify(text) for text in test_texts]

print("Base model:")
print(classification_report(test_labels, base_predictions))
print("\nFine-tuned:")
print(classification_report(test_labels, finetuned_predictions))
```

**Publish to HuggingFace Hub:**

```python
from huggingface_hub import HfApi
model.push_to_hub("YOUR_USERNAME/reg-classifier-qwen3-1.7b")
tokenizer.push_to_hub("YOUR_USERNAME/reg-classifier-qwen3-1.7b")
```

Write a proper model card — intended use, training data description, evaluation metrics,
known limitations, carbon footprint estimate. This is your HuggingFace ecosystem proof point.

---

### Day 7: Integration

Wire `RegClassifier` into the pipeline. When the agent retrieves findings, they pass through
the classifier before reaching the user. The Gradio UI shows:
- Risk tier badge (colour-coded)
- Obligation type
- Compliance deadline
- Classifier confidence

---

## Week 7 — Polish + Portfolio

**Daily time:** 3–4 hours  
**Deliverable:** A live, demo-ready system. Two articles published. Interview narrative polished.

### Day 1–2: Clean Up Code

Go through every file. Check:
- Every function has type hints
- Every public function has a docstring
- `pytest tests/ --cov=src` shows >70% coverage
- No `print()` in production code — only `logger`
- No hardcoded strings outside `config.py`

Run `ruff check src/` (modern Python linter). Fix everything it flags.

---

### Day 3: Write LinkedIn Article #1

**Title:** "Why chunking strategy matters more than your LLM choice for RAG"

**Content:** Use your actual numbers from Week 1 Day 3.
- `RecursiveCharacterChunker`: X% of queries returned the correct article in top-3
- `ArticleAwareChunker`: Y% of queries returned the correct article in top-3
- Show the actual code for both
- Explain why legal text is structurally different from prose

This is not a tutorial. It is evidence from something you actually built.

---

### Days 4–5: Write Comparison Document → Article #3

**Title:** "LangGraph vs smolagents: what I learned building the same agent twice"

Use your `docs/smolagents_comparison.md` from Week 2. Add concrete examples:
- The same query executed in both — show the different reasoning traces
- The one thing smolagents does better (less boilerplate for simple tasks)
- The one thing LangGraph does better (human-in-the-loop, audit trail)
- Which one you'd use in production and why

---

### Days 6–7: Interview Narrative + README

Write the final `README.md` for the repo. It should include:
- Architecture diagram
- Live demo link (HF Spaces)
- API documentation
- RAGAS baseline scores
- Before/after fine-tuning comparison
- One command to run locally (`docker compose up`)

**Polish your STAR+R narrative (under 3 minutes when spoken):**

> "The EU AI Act becomes fully enforceable in August 2026 — literally this month.
> Every company deploying AI in Germany needs to understand their obligations, but
> the regulation is 144 pages of dense legal text with deeply nested cross-references.
> I built RegulationAdvisor — a production-grade Agentic RAG system that answers
> compliance questions with cited answers. The hardest engineering problem was chunking:
> legal text has article-level semantic boundaries that naive character splitting destroys.
> I built a custom article-aware chunker that increased retrieval precision from X% to Y%.
> I then replaced the simple retrieval chain with a LangGraph agent with three tools —
> regulation search, structured data queries, and web search for enforcement news — and
> added a guardrail layer that catches hallucinated article citations before they reach users.
> The system runs on FastAPI with Gradio UI, deployed on HuggingFace Spaces and AWS ECS
> via Docker. I also fine-tuned a Qwen3-1.7B classifier for consistent risk tier labelling —
> it's published on HuggingFace Hub. RAGAS faithfulness score is above 0.85."

---

## Quick Reference: What to Use When

### LLM Provider
```python
# Development (free, fast)
llm = ChatGroq(model="qwen/qwen3-32b")

# Quality-critical (free, long context)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Swap with one config change — that's the point
```

### Embedding Model
```python
# Always local — no API cost, no latency
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")  # fast, 384d
embedder = SentenceTransformerEmbedder("all-mpnet-base-v2")  # slower, better
```

### Vector Store
```python
# Week 1-3: fast iteration
store = FAISSVectorStore()

# Week 4+: production
store = ChromaDBVectorStore(host=settings.chroma_host)
```

---

## Key Milestones

| End of Week | What Exists |
|-------------|------------|
| Week 1 | Working RAG pipeline + Gradio UI + HuggingFace Space |
| Week 2 | LangGraph agent with 3 tools + human-in-the-loop + smolagents comparison |
| Week 3 | RAGAS baseline score + guardrail layer + 30 regression tests |
| Week 4 | FastAPI REST API + ChromaDB + streaming responses |
| Week 5 | Docker Compose + HF Spaces + AWS ECS deployment |
| Week 6 | Fine-tuned classifier on HuggingFace Hub + integrated into pipeline |
| Week 7 | Clean code + 2 articles published + interview-ready |

---

## Things Deliberately Left Out

These were in the original READMEs. They are removed because they add no value for you:

| Removed | Why |
|---------|-----|
| Day 4–6 isolated notebooks (chains, parsers, memory) | You learn these inside the project |
| `ConversationBufferMemory` etc. | Deprecated in LangChain v1.0 — now LangGraph Checkpointers |
| Infrastructure monitoring fine-tuning task | Unrelated to the capstone |
| Prefect / DVC | Overkill for this project scope |
| Weights & Biases | Nice but adds a week; RAGAS dashboard is sufficient |
| 10-week timeline | Compressed to 7 weeks by removing isolation exercises |
