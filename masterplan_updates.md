# Master Plan — Two Sections That Need Updating

Replace these two sections in `RegulationAdvisor_MasterPlan.md` in your project knowledge files.

---

## REPLACEMENT 1
### Find this section (around line 257):
```
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
```

### Replace with:
```
### Day 1 — Environment + Project Structure

**The project structure is already created for you.** Extract the `regulation-advisor.zip`
from the project files and open it in your IDE. Then follow the steps below.

**Step 1 — Install uv (once, globally)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart terminal after this
uv --version  # verify it worked
```

**Why uv and not pip?** 10-100x faster. Manages the virtual environment automatically.
`uv run` activates it without you thinking about it. `uv.lock` guarantees reproducibility.
You will never type `pip install` or `source .venv/bin/activate` again.

**Step 2 — Set up the project**
```bash
cd regulation-advisor
uv python pin 3.11          # pins Python version for this project
uv sync --group dev         # creates .venv and installs everything
```

**Step 3 — Configure API keys**
```bash
cp .env.example .env
# Open .env and fill in GROQ_API_KEY at minimum (get free key at console.groq.com)
```

**Step 4 — Verify**
```bash
uv run pytest tests/unit/ -v
# Expected: 9 tests pass. No data files needed.
```

**Afternoon:**
- Verify `uv run python -c "from regulation_advisor.config import settings; print(settings.llm_model)"` prints `qwen/qwen3-32b`
- Download the EU AI Act PDF and GDPR PDF into `data/`
- Create the three CSV files (timeline, risk classification, penalties) — 45 minutes of reading the Act. This teaches you the regulation better than any tutorial.

**Gate check:** `uv run pytest tests/unit/ -v` shows 9 passed. You can import `from regulation_advisor.config import settings`.

**Daily command reminder — use these instead of plain python/pytest:**
```bash
uv run pytest tests/ -v              # run tests
uv run python scripts/ingest.py      # run scripts
uv run uvicorn ... --reload          # run the app (Week 4+)
uv add some-package                  # add a dependency (updates pyproject.toml automatically)
uv run ruff check src/               # lint
```
```

---

## REPLACEMENT 2
### Find this section (around line 1153):
```
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
```

### Replace with:
```
### Day 1: Dockerfile

```dockerfile
# Stage 1: install dependencies with uv
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first — Docker caches this layer if pyproject.toml unchanged
COPY pyproject.toml ./
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Stage 2: lean final image
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ src/
COPY data/ data/
COPY evals/ evals/
COPY scripts/ scripts/

# Make the venv the default Python
ENV PATH="/app/.venv/bin:$PATH"

# Pre-build the vector index
RUN python scripts/ingest.py

EXPOSE 8000
CMD ["uvicorn", "regulation_advisor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why two stages?** The builder stage has uv installed and compiles dependencies.
The final stage copies only the `.venv` — no build tools, no uv binary, smaller image.
This is the standard uv Docker pattern for production.
```

---

## Nothing else needs changing

Everything else in the master plan is current and correct:
- Model choices (Qwen3-32B, GLM-5.1, Qwen3-1.7B for fine-tuning) ✓
- API provider table (Groq, Google AI Studio, OpenRouter, NVIDIA NIM, Tavily) ✓
- 7-week timeline ✓
- All design patterns ✓
- LangGraph with Checkpointers (not deprecated memory classes) ✓
- RAGAS evaluation ✓
- AWS ECS deployment ✓
- Fine-tuning with Unsloth + regulation classifier (not infrastructure classifier) ✓
- "Things Deliberately Left Out" table ✓
