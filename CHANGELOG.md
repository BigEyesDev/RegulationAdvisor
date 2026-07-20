# Changelog

All notable changes to RegulationAdvisor are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.6.8] — 2026-07-20

### Security

- Audited the BYOK path end to end for key leakage: no code path logs a full
  request object or raw exception (`logger.exception` would print provider
  HTTP client errors that sometimes echo the key back — the BYOK handlers
  log only the exception type). No middleware or access log prints request
  bodies. The guardrail chain and RegClassifier only ever see the LLM's
  answer text, never the request. `ChatRequest.api_key` is marked
  `repr=False` so accidental `logger.info("%s", request)` calls can't leak
  it either. Regression tests assert the key never appears in the response
  body or in captured logs for a failing BYOK call.

## [0.6.7] — 2026-07-20

### Added

- BYOK (bring your own key) for `/api/chat` and `/api/chat/sync`: `ChatRequest`
  gains optional `api_key`, `provider`, and `model` fields. When `api_key` is
  set, the request is served by a throwaway agent built for that call only
  (`build_agent_graph()` now takes the same `provider`/`model`/`api_key`
  overrides as `build_llm()`) — the shared default agent used by everyone
  else is never touched. Nothing about the key or the throwaway agent is
  cached, stored, or attached to `session_id`; both are dropped when the
  request finishes.
- Both chat endpoints now catch LLM-provider failures (invalid key, rejected
  request) and return a clean 502 / SSE error event instead of an unhandled
  500 — the raw exception (which some providers echo the invalid key back
  into) is never included in the response or in logs, only the exception type.

## [0.6.6] — 2026-07-20

### Added

- `build_llm()` now accepts optional `provider`, `model`, and `api_key`
  overrides — falls back to `settings.*` when omitted, otherwise builds a
  one-off client with the override values. Global settings are never
  mutated. This is the foundation for bring-your-own-key (BYOK) support:
  a caller-supplied key builds a throwaway LLM client for a single request
  without touching the shared default agent used by everyone else.

## [0.6.5] — 2026-07-20

### Fixed

- `ruff check src/` now passes with zero errors: import ordering, an unused
  import (`metrics_store` in `ui/gradio_app.py`), an unused local
  (`result_holder`), a missing `strict=` on a `zip()` call in
  `retrieval/store.py`, and five lines over the 100-char limit.
- Two integration tests asserted stale hardcoded API versions (`0.4.0`,
  `0.6.3`) left over from earlier releases; updated to the current version.

## [0.6.4] — 2026-07-19

### Fixed

- `prompts/system_prompt.txt` was never actually loaded by any code path — `agent/graph.py`'s
  `agent_node` invoked the LLM directly on `state["messages"]` with no system message at all.
  Now loaded once at import time and prepended on every `agent_node` call (`agent/graph.py`).
  Also dropped the file's unused trailing `Context: {context}` block, a leftover from an
  earlier non-agentic prompt-template design — context now arrives via tool calls, not a
  pre-filled template slot.
- System prompt now (a) always closes an answer with "This is AI-generated guidance, not
  legal advice. Verify with a qualified lawyer," instead of only when the guardrail's
  phrase-matcher happens to catch legal-claim wording, and (b) added a scope gate: the model
  must first check whether the question actually concerns building/deploying/using an AI
  system or processing personal data before answering, and must clearly refuse — not search,
  not construct an answer — for questions that merely mention "AI Act" or "GDPR" without
  describing such an activity.

### Added

- `evals/regression_questions.json` + `scripts/run_regression_questions.py`: a 6-question
  regression set (2 legitimate compliance questions, 2 trick questions designed to bait the
  scope gate, 1 malformed/gibberish input, 1 gray-area question about a document-summarization
  app) run against the live agent graph after any change to the system prompt, agent graph,
  or tools. Checks for an "Article N" citation as the signal that a real answer was given
  (the prompt requires one), combined with rejection-phrase matching as a second signal,
  since the model phrases refusals differently run to run and single-signal matching produced
  false negatives during validation.

**Fine-tuning quality fixes, found by actually running the pipeline and checking raw model
output rather than trusting the aggregate accuracy number. Full write-up in
`docs/learning-by-doing-01-finetune-training-log.md` and
`docs/learning-by-doing-02-action-plan.md`.**

### Fixed

- `evals/finetune/examples.json` rebuilt from 200 templated examples (~40 genuinely unique,
  the rest near-duplicates padded with a `[Scenario variant N]` tag) to 102 examples, every
  one textually distinct. The fine-tuned model had scored *worse* than the un-tuned base
  model (35% vs 85% accuracy) because it was trained on repeated near-identical batches.
  Added `test_no_duplicate_instructions_in_train_set` as a regression test.
- `scripts/train_classifier.py`: LoRA now also targets the MLP layers (`gate_proj, up_proj,
  down_proj`, not just attention), rank raised `16 → 32`, effective batch size lowered
  `16 → 4` (more optimizer steps on a small dataset without raising epoch count past the
  small-dataset overfitting ceiling). The model had been inventing `obligation_type` values
  outside the schema entirely (`"REPUTATIONAL"`, `"CONFORMANCE"`) — a sign of too few
  optimizer steps and too little LoRA capacity to memorise a new rigid vocabulary, not a
  data problem this time.
- `torchao` removed from the `finetune` dependency group entirely. No installed version
  satisfies both `unsloth` (needs an old enough torchao to avoid a `torch.int1` crash against
  the `torch 2.5.1` already pinned by sentence-transformers/chromadb) and `peft`'s LoRA
  dispatch for MLP layers (hard-errors on an old-but-present torchao instead of skipping it).
  Fix: `uv sync --group dev --group finetune && uv pip uninstall torchao`.

---

## [0.6.2] — 2026-07-18

**Fix discovered while running the evaluation script for the first time.**

### Fixed

- `build_llm()` now passes an explicit `timeout` (default 45s, `LLM_REQUEST_TIMEOUT_SECONDS`
  in `.env`) to every provider client. Observed a single request to the configured
  OpenRouter model take 50s versus 3-5s for others, with no timeout previously configured
  anywhere — a slow or stuck provider response could hang a chat request or evaluation run
  indefinitely.

---

## [0.6.1] — 2026-07-17

**Fixes discovered while dry-running the fine-tuning pipeline for the first time.**

### Fixed

- `Qwen/Qwen3-1.7B-Instruct` does not exist on HuggingFace Hub — Qwen3's naming dropped the
  `-Instruct` suffix used by Qwen2.5. `scripts/train_classifier.py`'s default `--model` and
  the model card now point to `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` (a pre-quantized 4-bit
  build — smaller download and lower peak RAM/VRAM than quantizing the full-precision
  weights on the fly, which matters on constrained hardware)
- Pinned `torchao==0.9.0` in the `finetune` dependency group — newer `torchao` (>=0.14)
  references `torch.int1`, a sub-byte dtype that doesn't exist in the `torch 2.5.1` build
  already installed by `sentence-transformers`/`chromadb`. Pinning `torchao` down avoided a
  large, disk-risky `torch` reinstall while keeping `unsloth` importable
- Added `unsloth_compiled_cache/` to `.gitignore` — a build-time cache directory `unsloth`
  creates on first import

---

## [0.6.0] — 2026-07-17

**Fine-tuning: QLoRA-trained RegClassifier (Qwen3-1.7B), published to HuggingFace Hub and
wired into the REST API + Gradio UI.**

### Added

- `evals/finetune/examples.json` — 200 hand-reviewed EU AI Act classification examples
  across all four risk tiers plus edge cases; `scripts/build_finetune_dataset.py` validates
  schema and produces an 80/10/10 train/val/test split
- `scripts/train_classifier.py` — QLoRA fine-tuning of Qwen3-1.7B via Unsloth +
  `trl`'s `SFTTrainer`, saving a LoRA adapter checkpoint
- `scripts/evaluate_classifier.py` — `classification_report` comparison of the prompted
  base model vs. the fine-tuned checkpoint on the held-out test set
- `scripts/publish_to_hub.py` + `src/regulation_advisor/classifier/MODEL_CARD.md` —
  publishes the adapter to HuggingFace Hub with a documented model card
- `RegClassifier` rewritten from a stub to real inference: loads the fine-tuned checkpoint
  when `CLASSIFIER_CHECKPOINT` is set, otherwise falls back to a prompted LLM call
- `POST /api/chat/sync` now returns `risk_tier` and `classifier_confidence`; Gradio chat tab
  shows a colour-coded risk badge (🔴 Unacceptable / 🟠 High / 🟡 Limited / 🟢 Minimal)

### Changed

- `pyproject.toml` version `0.5.0 → 0.6.0`
- `/api/health` now reports `version: "0.6.0"`

---

## [0.5.0] — 2026-07-16

**Containerisation and cloud deployment: Docker, Docker Compose, HuggingFace Spaces (Docker SDK), AWS ECR + ECS Fargate + ALB.**

### Added

#### Docker
- `Dockerfile` — production-ready multi-stage build: `uv` builder stage + lean `python:3.11-slim`
  runtime; FAISS index pre-built at image build time via `RUN python scripts/ingest.py`
- `.dockerignore` — excludes `.venv`, `.env`, caches, `data/index/`, docs/, planning files from
  build context; reduces context size and prevents secret leakage
- `uv.lock` version field corrected to match `pyproject.toml` (was stale at 0.2.1)

#### Docker Compose (ChromaDB mode)
- `docker-compose.yml` rewritten: fixed `CHROMA_PORT=8000` (was 8001 — wrong internal port);
  removed broken `./data` volume mount; added `entrypoint` override to `scripts/entrypoint.sh`
- `scripts/entrypoint.sh` — idempotent startup: checks ChromaDB document count, ingests only
  if empty, then `exec uvicorn`; prevents duplicate ingestion on restarts

#### HuggingFace Spaces
- `README.md` front-matter: `sdk: gradio` → `sdk: docker`, `app_port: 8000`; removed Gradio-
  specific `sdk_version` and `app_file` fields
- Updated deploy section: Docker upload flow, secrets table with `VECTOR_STORE_BACKEND=faiss`

#### AWS Infrastructure
- `infra/task-definition.json` — ECS Fargate task: 0.5 vCPU / 1 GB, eu-central-1, API keys
  injected from Secrets Manager at runtime (no plaintext), CloudWatch log driver
- `infra/README.md` — deployment reference: ECR push, IAM role setup, service creation,
  CloudWatch tailing, cost-control commands (scale to 0, delete ALB)

#### ECS Service + ALB
- ECS cluster `regulation-advisor` with Fargate service (1 desired task)
- Application Load Balancer: internet-facing, health check on `/health`, HTTPS termination
- Public API endpoint documented in README

### Changed
- `pyproject.toml` version bumped `0.4.0 → 0.5.0`
- `src/regulation_advisor/api/app.py` version string updated to `0.5.0`
- `README.md` architecture diagram updated to v0.5

---

## [0.4.0] — 2026-07-15

**FastAPI REST API, streaming responses, ChromaDB persistence, and Evaluation Dashboard.**

### Added

#### REST API Layer
- `src/regulation_advisor/api/app.py` — FastAPI app with `lifespan` context manager;
  Gradio UI lazy-mounted at `/`; health endpoint at `/health`
- `src/regulation_advisor/api/routes.py` — all API route handlers in one module
- `src/regulation_advisor/api/schemas.py` — `ChatRequest`, `ChatResponse`,
  `MetricsResponse` Pydantic models

#### Streaming Chat
- `POST /api/chat` — Server-Sent Events streaming endpoint; LLM tokens streamed via
  `astream_events` as `data: <chunk>\n\n`
- `POST /api/chat/sync` — synchronous fallback for clients without SSE support
- `tests/unit/test_api_chat.py` — unit tests covering both endpoints

#### ChromaDB Vector Store
- `build_vector_store()` factory in `retrieval/store.py` — reads `VECTOR_STORE_BACKEND`
  from config and returns either `FAISSVectorStore` or `ChromaDBVectorStore`; all startup
  paths now call the factory instead of constructing stores directly
- `config.py` — three new settings: `vector_store_backend`, `chroma_host`, `chroma_port`
- `tests/unit/test_store_factory.py` — factory routing unit tests

#### Metrics and Evaluation API
- `GET /api/metrics` — returns cached RAGAS scores from the last evaluation run
- `POST /api/evaluate` — triggers `EvaluationHarness.run()` as a `BackgroundTask`;
  returns `{"status": "evaluation started"}` immediately
- `src/regulation_advisor/api/metrics_store.py` — `MetricsStore` singleton holding the
  latest `RAGASResult` in memory; shared between the background task and GET handler
- `tests/unit/test_api_metrics.py` — unit tests covering both endpoints

#### Evaluation Dashboard (Gradio)
- Second tab in `gradio_app.py` — "Evaluation Dashboard" with a Run RAGAS Evaluation
  button, live status textbox, and four score displays (faithfulness, answer_relevancy,
  context_precision, context_recall)
- Dashboard calls `POST /api/evaluate` and polls `GET /api/metrics` on completion

#### Integration Tests
- `tests/integration/test_api_integration.py` — end-to-end tests covering `/health`,
  `/api/metrics`, `/api/evaluate`, streaming `/api/chat`, and the Gradio mount
- `evals/baseline_scores.json` — baseline RAGAS scores committed for CI reference

### Changed
- `app_runner.py` — startup now calls `build_vector_store()` factory; removed hardcoded
  `FAISSVectorStore` construction
- `pyproject.toml` version bumped `0.3.0 → 0.4.0`

---

## [0.3.0] — 2026-07-14

**RAGAS evaluation harness, guardrail layer, and promptfoo CI regression suite.**

### Added

#### Evaluation Dataset
- `evals/qa_pairs.json` — 20 verified ground-truth Q&A pairs covering prohibited
  practices, high-risk obligations, GPAI rules, penalties, enforcement timeline,
  and GDPR overlap
- `tests/unit/test_eval_dataset.py` — validates schema of every pair

#### RAGAS Evaluation Harness
- `src/regulation_advisor/evaluation/harness.py` — `EvaluationHarness` with `run()`
  returning a `RAGASResult` dataclass (faithfulness, answer_relevancy,
  context_precision, context_recall); `harness.save()` persists scores to JSON
- `scripts/run_evaluation.py` — CLI runner: loads QA pairs, runs pipeline, saves results
- `tests/unit/test_harness.py` — unit test with mock pipeline function

#### Guardrail Layer
- `src/regulation_advisor/evaluation/guardrails.py` — Chain of Responsibility:
  `FaithfulnessCheck` → `CitationVerificationCheck` → `LegalClaimFlagCheck`;
  `build_guardrail_chain()` factory wires all three
- `gradio_app.py` — `respond()` now runs the guardrail chain after generation;
  appends warning banners to responses when checks fail
- `tests/unit/test_guardrail_integration.py` — covers all three handlers and the chain

#### promptfoo Regression Suite
- `evals/promptfoo.yaml` — 30-case regression suite covering Article 5 prohibitions,
  penalty amounts, GPAI obligations, enforcement dates, and risk tiers;
  assertions use `contains`, `not-contains`, and `llm-rubric`
- `.github/workflows/eval.yml` — GitHub Actions workflow: runs `promptfoo eval` on
  every PR to `main`; fails the build on any regression
- `scripts/eval_pipeline.py` — `run_query()` adapter for the promptfoo `python:` provider

#### Pipeline Integration Tests
- `tests/integration/test_week3_pipeline.py` — end-to-end tests: guardrail pass/fail
  scenarios, RAGAS smoke-run with mock pipeline, promptfoo config validation

### Changed
- `RAGASResult.is_acceptable()` faithfulness threshold raised `0.70 → 0.75`
- `pyproject.toml` version bumped `0.2.1 → 0.3.0`

---

## [0.2.1] — 2026-07-12

**smolagents comparison agent added.**

### Added

- `src/regulation_advisor/agent/smolagents_agent.py` — `build_smolagents_agent()` builds
  a `ToolCallingAgent` reusing the same 3 LangChain tools via `LangChainTool` wrappers;
  model mapped from `LLM_PROVIDER`/`LLM_MODEL` settings to a LiteLLM model identifier
- `smolagents[litellm]` added to `pyproject.toml` and `requirements.txt`
- `docs/smolagents_comparison.md` — completed with real code, benchmark results for 5
  queries, and production decision guide

### Changed
- `pyproject.toml` version bumped `0.2.0 → 0.2.1`

---

## [0.2.0] — 2026-07-12

**LangGraph agent replaces the RAG chain. Multi-turn memory via checkpointer.**

### Added

#### Shared LLM Factory
- `src/regulation_advisor/llm.py` — `build_llm()` factory reads `LLM_PROVIDER` from
  `.env` and returns the correct LangChain chat model; eliminates duplicated provider logic
- `tests/unit/test_tools.py` — mock-retriever search, real CSV keyword match,
  graceful no-retriever error

#### LangGraph Agent
- `src/regulation_advisor/agent/tools.py` — `query_structured_data` fixed to use
  `Path(__file__)`-anchored absolute path (was cwd-relative, broke outside project root)
- `src/regulation_advisor/agent/graph.py` — `build_agent_graph()` calls `build_llm()`
  instead of hardcoding `ChatGroq`; respects `LLM_PROVIDER` env var
- `tests/unit/test_agent_graph.py` — compile check, tool-call routing, END routing

#### Agent Wired into Gradio
- `src/regulation_advisor/ui/gradio_app.py` — rewritten around the LangGraph agent:
  `build_ui(agent)` replaces `build_ui(retriever)`; `respond()` uses `thread_id`
  for multi-turn memory via checkpointer; shows legal warning on critical findings
- `src/regulation_advisor/ui/app_runner.py` — updated startup:
  `set_retriever()` → `build_agent_graph()` → `build_ui(agent)`

### Changed
- `gradio_app.py` — removed `_build_chain()`, `_format_context()`, dead imports
- `pyproject.toml` version bumped `0.1.0 → 0.2.0`

---

## [0.1.0] — 2026-07-12

**Full RAG pipeline from raw PDFs to a live Gradio chatbot.**

### Added

#### Document Ingestion
- `PDFLoader`, `CSVLoader`, `MarkdownLoader` with `DocumentLoaderFactory` (Factory pattern)
- `ArticleAwareChunker` — splits legal text at `Article N` boundaries, extracting
  `article_number` and `article_title` as metadata on every chunk
- `RecursiveCharacterChunker` — configurable fallback chunker (size / overlap from `.env`)
- `SentenceTransformerEmbedder` — local `all-MiniLM-L6-v2` model, no API key required
- `FAISSVectorStore` — in-memory vector store with `save()` / `load()` round-trip
- `Retriever` — wraps embedder + store; `search(query, k)` returns `RetrievalResult`
- `run_ingestion()` pipeline — reads all files in `data/`, chunks, embeds, saves index
- `scripts/ingest.py` — CLI entry point for building the index
- 213 chunks indexed: 114 from EU AI Act, 99 from GDPR
- 14 unit tests + 2 integration tests

#### RAG Chain
- `src/regulation_advisor/prompts/system_prompt.txt` — grounding prompt that enforces
  Article citations and prohibits answers outside provided context
- `_build_llm()` — provider factory supporting `openrouter`, `groq`, and `google`;
  swap provider and model with two lines in `.env`
- `_build_chain()` — `ChatPromptTemplate | LLM | StrOutputParser` LCEL pipeline
- `_format_context()` — labels each retrieved chunk as `[source — Article N]`

#### Gradio UI
- `build_ui(retriever)` — `gr.Blocks` with `gr.ChatInterface`; returns cited answers
- `src/regulation_advisor/ui/app_runner.py` — startup entry point with auto-ingest on
  cold start; `server_name="0.0.0.0"` for container / HF Spaces compatibility

#### HuggingFace Spaces
- `README.md` — HF Spaces YAML front-matter (`sdk: gradio 6.20.0`, `app_file`)
- `requirements.txt` — pip-compatible dependency list for HF Spaces
- `_ensure_index()` in `app_runner.py` — auto-runs ingestion on HF cold start

### Changed
- LLM default switched to `openrouter / deepseek/deepseek-v4-flash` (1M token context)
- `config.py` — added `openrouter_base_url` setting

### Infrastructure
- Package manager: `uv` with `pyproject.toml`
- Linting: `ruff` (E, F, I, UP, B rules)
- Type checking: `mypy --strict`
- Tests: `pytest` with `asyncio_mode = auto`

---

## [0.0.1] — 2026-07-11

**Initial project scaffold.**

- Project structure: `src/regulation_advisor/` layout
- `config.py` — `pydantic-settings` with `.env` support
- `models.py` — `RegulationChunk`, `RegulationFinding`, `RetrievalResult`
- `loaders.py`, `chunkers.py`, `pipeline.py`, `embeddings.py`, `store.py`, `retriever.py`
- 14 unit tests passing
- GitHub repository initialised
