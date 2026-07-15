# Changelog

All notable changes to RegulationAdvisor are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.4.0] — 2026-07-15

**Week 4 complete: FastAPI REST API, streaming responses, ChromaDB, and Evaluation Dashboard.**

### Added

#### FastAPI Foundation (W4-D1)
- `src/regulation_advisor/api/app.py` — FastAPI app with `lifespan` context manager for
  startup/shutdown; Gradio UI lazy-mounted at `/`; health route at `/health`
- `src/regulation_advisor/api/routes.py` — all API routes in one module; imports cleanly
  from `app.py`
- `src/regulation_advisor/api/schemas.py` — `ChatRequest`, `ChatResponse`, `MetricsResponse`
  Pydantic models for request/response validation
- `docs/w4-d1-fastapi-foundation.md` — learning doc explaining lifespan, route separation,
  and Gradio mount pattern

#### Streaming Chat Endpoint (W4-D2)
- `POST /api/chat` — Server-Sent Events (SSE) streaming endpoint; streams LLM tokens via
  `astream_events` as `data: <chunk>\n\n`
- `POST /api/chat/sync` — synchronous fallback for clients that don't support SSE
- `tests/unit/test_api_chat.py` — 77-line unit test covering both endpoints
- `docs/w4-d2-streaming-chat.md` — SSE protocol explanation and async generator pattern

#### ChromaDB Wiring (W4-D3)
- `build_vector_store()` factory in `retrieval/store.py` — reads `VECTOR_STORE_BACKEND`
  from config and returns either `FAISSVectorStore` or `ChromaDBVectorStore`; all startup
  paths now call this factory instead of constructing stores directly
- `index_dir` config setting added for explicit FAISS index path control
- `src/regulation_advisor/config.py` — 3 new settings: `vector_store_backend`,
  `chroma_host`, `chroma_port`
- `tests/unit/test_store_factory.py` — 66-line unit test for factory routing
- `docs/w4-d3-chromadb.md` — Repository pattern recap and swap walkthrough

#### Metrics API (W4-D4)
- `GET /api/metrics` — returns cached RAGAS scores from last evaluation run
- `POST /api/evaluate` — triggers `EvaluationHarness.run()` as a `BackgroundTask`;
  returns `{"status": "evaluation started"}` immediately
- `src/regulation_advisor/api/metrics_store.py` — `MetricsStore` singleton that holds
  the latest `RAGASResult` in memory; shared between background task and GET handler
- `tests/unit/test_api_metrics.py` — 84-line unit test covering both endpoints
- `docs/w4-d4-metrics-api.md` — BackgroundTasks pattern and metrics store design

#### Evaluation Dashboard Tab (W4-D5)
- `src/regulation_advisor/ui/gradio_app.py` — second `gr.Tab("Evaluation Dashboard")`
  with Run RAGAS Evaluation button, live status textbox, and four `gr.Number` displays
  (faithfulness, answer_relevancy, context_precision, context_recall)
- Dashboard calls `POST /api/evaluate` and polls `GET /api/metrics` on completion
- `docs/w4-d5-eval-dashboard.md` — Gradio multi-tab pattern and API polling approach

#### Integration Test Suite (W4-D6)
- `tests/integration/test_api_integration.py` — 179-line end-to-end tests covering
  `/health`, `/api/metrics`, `/api/evaluate`, streaming `/api/chat`, Gradio mount
- `evals/baseline_scores.json` — placeholder baseline RAGAS scores committed for CI
  reference (faithfulness: 0.0, populated after first real evaluation run)
- `docs/w4-d6-integration.md` — integration test strategy and TestClient usage

### Changed
- `src/regulation_advisor/ui/app_runner.py` — startup now calls `build_vector_store()`
  factory; removed hardcoded `FAISSVectorStore` construction
- `pyproject.toml` version bumped `0.3.0 → 0.4.0`; description updated

---

## [0.3.0] — 2026-07-14

**Week 3 complete: RAGAS evaluation harness, guardrail layer, and promptfoo CI regression suite.**

### Added

#### Evaluation Dataset (W3-D1)
- `evals/qa_pairs.json` — expanded from 5 seed pairs to 20 verified ground-truth Q&A pairs
  covering prohibited practices (Article 5), high-risk obligations, GPAI rules, penalties,
  enforcement timeline, and GDPR overlap
- `tests/unit/test_eval_dataset.py` — validates schema of every pair (question, ground_truth_answer,
  expected_article fields present and non-empty)
- `learning/day1_eval_dataset.md` — notes on golden-set construction methodology

#### RAGAS Evaluation Harness (W3-D2)
- `src/regulation_advisor/evaluation/harness.py` — `EvaluationHarness` with `run()` method
  returning `RAGASResult` dataclass (faithfulness, answer_relevancy, context_precision,
  context_recall); `harness.save()` persists scores to JSON for CI comparison
- Faithfulness threshold tightened from 0.7 → 0.75 in `RAGASResult.is_acceptable()`
- `scripts/run_evaluation.py` — CLI runner: loads QA pairs, runs pipeline_fn, saves results
- `tests/unit/test_harness.py` — 37-line unit test with mock pipeline_fn
- `learning/day2_ragas_harness.md` — RAGAS metric definitions and observer pattern notes

#### Guardrail Layer (W3-D3)
- `src/regulation_advisor/evaluation/guardrails.py` — Chain of Responsibility:
  `FaithfulnessCheck` → `CitationVerificationCheck` → `LegalClaimFlagCheck`;
  `build_guardrail_chain()` factory wires all three
- `src/regulation_advisor/ui/gradio_app.py` — `respond()` now runs the guardrail chain
  after streaming; appends warning banners to the response when checks fail
- `tests/unit/test_guardrail_integration.py` — 47-line unit test covering all three handlers
  and the chain assembly
- `learning/day3_guardrail_integration.md` — Chain of Responsibility pattern walkthrough

#### promptfoo Regression Suite + CI (W3-D4)
- `evals/promptfoo.yaml` — 30-case regression suite expanded from seed; covers Article 5
  prohibitions, penalty amounts (35M / 7%), GPAI obligations, enforcement dates, risk
  tiers; assertions use `contains`, `not-contains`, and `llm-rubric` types
- `.github/workflows/eval.yml` — GitHub Actions workflow: runs `promptfoo eval` on every
  PR to `main`; fails build if any regression case breaks
- `scripts/eval_pipeline.py` — `run_query(prompt, options, ctx)` adapter used by promptfoo
  `python:` provider
- `learning/day4_promptfoo_suite.md` — promptfoo provider interface and CI integration guide

#### Week 3 Integration Tests (W3-D5)
- `tests/integration/test_week3_pipeline.py` — 110-line end-to-end tests: guardrail chain
  pass/fail scenarios, RAGAS harness smoke-run with mock pipeline, promptfoo config
  validation
- `evals/baseline_scores.json` — initial placeholder committed for CI reference
- `learning/day5_week3_integration.md` — integration test strategy notes

### Changed
- `src/regulation_advisor/evaluation/harness.py` — `RAGASResult.is_acceptable()` threshold
  raised from 0.70 → 0.75
- `pyproject.toml` version bumped `0.2.1 → 0.3.0`

---

## [0.2.1] — 2026-07-12

**F10: smolagents comparison agent added.**

### Added

#### smolagents Comparison (F10)
- `src/regulation_advisor/agent/smolagents_agent.py` — `build_smolagents_agent()` builds
  a `ToolCallingAgent` that reuses the same 3 LangChain tools via `LangChainTool` wrappers;
  model mapped from `LLM_PROVIDER`/`LLM_MODEL` settings to a LiteLLM model identifier
- `smolagents[litellm]` added to `pyproject.toml` and `requirements.txt`
- `docs/smolagents_comparison.md` — completed with real code, benchmark results for 5
  queries, and production decision guide (was placeholder stubs)

### Changed
- `pyproject.toml` version bumped `0.2.0 → 0.2.1`

---

## [0.2.0] — 2026-07-12

**Week 2 complete: LangGraph agent replaces the Week 1 RAG chain.**

### Added

#### Shared LLM Factory (F7)
- `src/regulation_advisor/llm.py` — `build_llm()` factory reads `LLM_PROVIDER` from `.env`
  and returns the correct LangChain chat model; eliminates duplicated provider logic
- `tests/unit/test_tools.py` — 3 unit tests: mock-retriever search, real CSV keyword match,
  graceful no-retriever error string

#### LangGraph Agent Graph (F7 / F8)
- `src/regulation_advisor/agent/tools.py` — fixed `query_structured_data` to use
  `Path(__file__)`-anchored absolute path (was cwd-relative, broke outside project root)
- `src/regulation_advisor/agent/graph.py` — `build_agent_graph()` now calls `build_llm()`
  instead of hardcoding `ChatGroq`; respects `LLM_PROVIDER` env var
- `tests/unit/test_agent_graph.py` — 3 unit tests: compile check, tool-call routing,
  END routing

#### Agent Wired into Gradio (F9)
- `src/regulation_advisor/ui/gradio_app.py` — rewritten around the LangGraph agent:
  `build_ui(agent)` replaces `build_ui(retriever)`; `respond()` calls
  `agent.invoke()` with a `thread_id` for multi-turn memory; appends a legal
  warning banner when `is_critical_finding` is `True`
- `src/regulation_advisor/ui/app_runner.py` — updated startup sequence:
  `set_retriever(retriever)` → `build_agent_graph()` → `build_ui(agent)`

### Changed
- `gradio_app.py` — removed `_build_chain()`, `_format_context()`, dead imports
  (`StrOutputParser`, `ChatPromptTemplate`, `Retriever`); file is now ~50 lines
- `pyproject.toml` version bumped `0.1.0 → 0.2.0`

---

## [0.1.0] — 2026-07-12

**Week 1 complete: full RAG pipeline from raw PDFs to a live Gradio chatbot.**

### Added

#### Ingestion (F1 – F3)
- `PDFLoader`, `CSVLoader`, `MarkdownLoader` with `DocumentLoaderFactory` (Strategy pattern)
- `ArticleAwareChunker` — regex-based chunker that splits legal text at `Article N` boundaries,
  extracting `article_number` and `article_title` as metadata on every chunk
- `RecursiveCharacterChunker` — configurable fallback chunker (size / overlap from `.env`)
- `SentenceTransformerEmbedder` — local `all-MiniLM-L6-v2` model, no API key required
- `FAISSVectorStore` — in-memory vector store with `save()` / `load()` round-trip
- `ChromaDBVectorStore` — stub for Week 4 migration (swap via `VECTOR_STORE_BACKEND` in `.env`)
- `Retriever` — wraps embedder + store; `search(query, k)` returns `RetrievalResult`
- `run_ingestion()` pipeline — reads all files in `data/`, chunks, embeds, saves FAISS index
- `scripts/ingest.py` — CLI entry point for building the index
- 213 chunks indexed: 114 from EU AI Act, 99 from GDPR
- 14 unit tests + 2 integration tests (skip gracefully if index not built)
- Config-driven chunker strategy: `CHUNKER_STRATEGY=article_aware` or `recursive`

#### RAG Chain (F4)
- `src/regulation_advisor/prompts/system_prompt.txt` — grounding prompt that forces
  Article citations and prohibits answers outside the provided context
- `_build_llm()` — provider factory supporting `openrouter`, `groq`, and `google`;
  switch provider and model with two lines in `.env`, no code changes needed
- `_build_chain()` — `ChatPromptTemplate | LLM | StrOutputParser` LCEL pipeline
- `_format_context()` — labels each retrieved chunk as `[source — Article N]` for citation
- `respond()` — closure that wires retrieval → context formatting → LLM generation

#### Gradio UI (F5)
- `build_ui(retriever)` — `gr.Blocks` with `gr.ChatInterface`; returns cited answers
- `src/regulation_advisor/ui/app_runner.py` — startup entry point:
  auto-ingests on cold start, resolves paths via `__file__`, launches with
  `server_name="0.0.0.0"` for container/HF Spaces compatibility

#### HuggingFace Spaces (F6)
- `README.md` — HF Spaces YAML front-matter (`sdk: gradio 6.20.0`, `app_file`)
- `requirements.txt` — pip-compatible dependency list for HF Spaces install
- `_ensure_index()` in `app_runner.py` — auto-runs ingestion on HF cold start

### Changed
- LLM default switched from `groq / qwen-qwen3-32b` (6k TPM free tier) to
  `openrouter / deepseek/deepseek-v4-flash` (1M token context, no TPM cap)
- `config.py` — added `openrouter_base_url` setting
- `.gitignore` — internal learning docs and `data/index/` excluded from git

### Infrastructure
- Package manager: `uv` with `pyproject.toml`
- Linting: `ruff` (E, F, I, UP, B rules)
- Type checking: `mypy --strict`
- Tests: `pytest` with `asyncio_mode = auto`

---

## [0.0.1] — 2026-07-11

**Day 1 scaffold.**

- Project structure: `src/regulation_advisor/` layout
- `config.py` — `pydantic-settings` with `.env` support
- `models.py` — `RegulationChunk`, `RegulationFinding`, `RetrievalResult`
- `loaders.py`, `chunkers.py`, `pipeline.py`, `embeddings.py`, `store.py`, `retriever.py`
- 14 unit tests passing on first commit
- GitHub repository initialised
