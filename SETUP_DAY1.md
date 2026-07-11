# Day 1 Setup Checklist

Work through this top to bottom. Each step has a clear success check.

---

## Step 1 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after this. Then verify:
```bash
uv --version
# Expected: uv 0.5.x or higher
```

**Why uv and not pip?**
- 10-100x faster than pip
- Automatically creates and manages the virtual environment
- `uv.lock` file guarantees exact same packages on every machine
- `uv run` activates the venv automatically — no `source .venv/bin/activate` needed

---

## Step 2 — Project Setup

```bash
cd regulation-advisor

# Pin Python 3.11 for this project
uv python pin 3.12

# Install all dependencies + dev tools
uv sync --group dev
```

Success check:
```bash
uv run python -c "import langchain; print('LangChain OK')"
uv run python -c "import sentence_transformers; print('sentence-transformers OK')"
uv run python -c "import gradio; print('Gradio OK')"
```

---

## Step 3 — API Keys

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:
- `GROQ_API_KEY` — from console.groq.com (free, no credit card)
- `TAVILY_API_KEY` — from tavily.com (free, 1000 req/month)

You can add the others later. Week 1 only needs Groq.

Success check:
```bash
uv run python -c "
from regulation_advisor.config import settings
print('Groq key set:', bool(settings.groq_api_key))
"
```

---

## Step 4 — Verify Tests Pass

```bash
uv run pytest tests/unit/ -v
```

Expected output:
```
tests/unit/test_chunkers.py::test_article_aware_chunker_finds_articles PASSED
tests/unit/test_chunkers.py::test_article_aware_chunker_sets_source     PASSED
tests/unit/test_chunkers.py::test_recursive_chunker_produces_output     PASSED
tests/unit/test_guardrails.py::test_low_confidence_fails                PASSED
tests/unit/test_guardrails.py::test_hallucinated_article_flagged        PASSED
tests/unit/test_guardrails.py::test_legal_claim_flagged                 PASSED
tests/unit/test_loaders.py::test_factory_raises_for_unsupported_extension PASSED
tests/unit/test_loaders.py::test_factory_supports_pdf                  PASSED
tests/unit/test_loaders.py::test_factory_does_not_support_json         PASSED

9 passed in X.XXs
```

If all 9 pass: Day 1 setup is complete. Start building.

---

## Step 5 — Download Data (do this while tests run)

Download into the `data/` folder:

| File | URL |
|------|-----|
| `eu_ai_act.pdf` | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689 |
| `gdpr.pdf` | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679 |

Then create the 3 CSV files manually (see `data/README.md` for columns).
This takes 45 minutes — it teaches you the regulation better than reading a summary.

---

## Common uv Commands (reference)

```bash
# Run anything in the project venv
uv run python script.py
uv run pytest
uv run uvicorn ...

# Add a new package (updates pyproject.toml + uv.lock automatically)
uv add some-package
uv add --group dev some-dev-tool

# Remove a package
uv remove some-package

# Update all packages
uv sync --upgrade

# See what's installed
uv pip list
```

**You never need to activate the virtual environment manually.**
Just prefix every command with `uv run`.
