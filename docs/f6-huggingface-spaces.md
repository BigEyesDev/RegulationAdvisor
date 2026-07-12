# F6 — HuggingFace Spaces Deployment

> **Audience:** Complete beginners. No cloud, Docker, or DevOps background required.
> **What you will understand after reading this:** How to take a Python app that runs
> on your laptop and make it accessible to anyone in the world via a public URL — for free.

---

## The Problem This Feature Solves

After F5, the chatbot works perfectly on your laptop. But:
- It only runs when your computer is on
- Nobody else can use it (they'd need to clone the repo, install everything, and run it themselves)
- There is no link you can share on LinkedIn or send to a recruiter

F6 solves this: **one public URL that anyone can open in a browser**.

```
Before F6:
  You   →  http://localhost:7860  (only you, only when your laptop is on)

After F6:
  You          ↘
  Recruiter    → https://huggingface.co/spaces/BigEyesDev/regulation-advisor
  Client       ↗
  (Anyone)
```

---

## Analogy: Renting Shop Space

Think of your laptop as a pop-up stall in your backyard. It works great for testing your
product, but only people who come to your house can buy from you.

HuggingFace Spaces is like renting a stall in a busy shopping centre:
- The mall (HuggingFace) provides the building, electricity, security, and foot traffic
- You bring your product (the Python code)
- Anyone walking through the mall can browse your stall (visit the URL)
- The mall is open 24/7 even when you're asleep

---

## What is HuggingFace Spaces?

[HuggingFace](https://huggingface.co) is the GitHub of AI. It hosts models, datasets, and
"Spaces" — small web applications that demonstrate AI capabilities.

**Key facts about Spaces:**
- Free tier available (CPU, 2 vCPU, 16 GB RAM, no GPU)
- Supports Gradio, Streamlit, and static HTML apps
- Each Space is backed by a **git repository** on HuggingFace's servers
- Your app is re-deployed automatically whenever you push a commit
- You can set "Secrets" — environment variables that are encrypted and injected at runtime

---

## How HF Spaces Works (The Mechanics)

When you push code to a HF Space, this is what happens automatically:

```
You push code → HF detects the push
                      │
                      ▼
              HF reads README.md YAML
              → finds sdk: gradio
              → finds app_file: src/regulation_advisor/ui/app_runner.py
                      │
                      ▼
              HF installs requirements.txt
              (pip install -r requirements.txt)
                      │
                      ▼
              HF runs the app file:
              python src/regulation_advisor/ui/app_runner.py
                      │
                      ▼
              HF exposes it at:
              https://huggingface.co/spaces/BigEyesDev/regulation-advisor
```

The whole process takes about 2–5 minutes on the first build.

---

## What Was Built in F6

### 1. `README.md` — HF Spaces YAML Front-Matter

HF Spaces reads a special YAML block at the very top of `README.md` to configure
the Space. This is called "front-matter" (a convention borrowed from static site generators).

```yaml
---
title: RegulationAdvisor
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.20.0"
app_file: src/regulation_advisor/ui/app_runner.py
pinned: false
---
```

| Field | Meaning |
|---|---|
| `title` | Displayed name in the HF Spaces gallery |
| `emoji` | The icon shown next to the title |
| `colorFrom` / `colorTo` | Gradient colour for the Space card in the gallery |
| `sdk` | Which framework powers the UI — `gradio` (or `streamlit`) |
| `sdk_version` | Must match the Gradio version in `requirements.txt` |
| `app_file` | The Python file HF runs to start the app |
| `pinned` | Whether your Space appears at the top of your profile |

**Why does `sdk_version` matter?** HF uses this to pre-install the exact Gradio version
before running your app. If it mismatches, the UI might look different or break.

---

### 2. `requirements.txt` — Dependencies for HF Spaces

HF Spaces uses `pip`, not `uv`. It reads `requirements.txt` to know what to install.

```txt
# Runtime dependencies for HuggingFace Spaces.
llama-index-readers-file
pymupdf
langchain>=1.0.0
langchain-groq
langchain-google-genai
langchain-openai
...
gradio
...
```

**Why a separate file if we already have `pyproject.toml`?**
- `pyproject.toml` is the modern Python standard, used by `uv`
- HF Spaces is a hosted server running plain `pip` — it doesn't understand `uv`'s `pyproject.toml` format
- So we maintain both: `pyproject.toml` for local dev, `requirements.txt` for HF

**Analogy:** `pyproject.toml` is your internal stock management system. `requirements.txt`
is the purchase order you send to a supplier who doesn't know your system — it's the
same information, just in the format they understand.

---

### 3. `app_runner.py` — Updated for HF Spaces

Three changes were made to the original `app_runner.py`:

#### Change 1: Auto-ingestion on cold start

```python
def _ensure_index() -> None:
    if (_INDEX_DIR / "index.faiss").exists():
        logger.info("Index already built — skipping ingestion.")
        return

    logger.info("Index not found — running ingestion (~20 s on first run)…")
    from regulation_advisor.ingestion.pipeline import run_ingestion
    run_ingestion(data_dir=_DATA_DIR, index_dir=_INDEX_DIR)
```

**Why this is needed:** The FAISS index (`data/index/`) is in `.gitignore` on GitHub —
binary files don't belong in version control (see F3 docs for why). But HF Spaces is a
different git repository, and we push the data files there.

On first run (HF cold start), the index doesn't exist yet. `_ensure_index()` detects
this and builds it automatically. On all subsequent restarts it finds the index already
on disk and skips the 20-second ingestion.

**This is a common pattern called "lazy initialization"** — do expensive work only once,
on the first time it's needed, then reuse the result forever.

#### Change 2: Absolute paths using `__file__`

```python
_ROOT = Path(__file__).parent.parent.parent.parent  # project root
_DATA_DIR = _ROOT / "data"
_INDEX_DIR = _ROOT / "data" / "index"
```

**Why:** `Path("data/index")` is relative to the *working directory* when the script runs.
On your laptop you run it from the project root so it works. On HF Spaces, the working
directory might be something else. Using `__file__` (the absolute path of *this* Python
file) as an anchor gives you the same correct path regardless of where Python was invoked.

**Analogy:** Instead of saying "turn right at the petrol station" (relative), you give
GPS coordinates (absolute). Anyone, anywhere, gets to the right place.

#### Change 3: HF-compatible `demo.launch()`

```python
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
```

- `server_name="0.0.0.0"` — Listen on all network interfaces (not just `localhost`).
  In a container or VM (which is what HF uses), "localhost" means the container itself —
  external traffic can't reach it. `0.0.0.0` means "accept connections from everywhere".
- `server_port=port` — HF Spaces injects a `PORT` environment variable. We read it and
  use it. On your laptop, `PORT` is not set, so it falls back to `7860`.

**Analogy:** `localhost` is like a lock on your front door that only opens from inside.
`0.0.0.0` removes that restriction — the postman (HF's reverse proxy) can now deliver
mail.

---

## How to Deploy to HF Spaces (Step-by-Step)

### Step 1: Create the Space on HuggingFace

1. Log in to [huggingface.co](https://huggingface.co)
2. Click your avatar → **New Space**
3. Fill in:
   - **Space name:** `regulation-advisor`
   - **License:** MIT (or your preference)
   - **SDK:** Gradio
4. Click **Create Space**

HF creates an empty git repo at `https://huggingface.co/spaces/BigEyesDev/regulation-advisor`

### Step 2: Set API Key Secrets

**Never put API keys in code.** Instead, use HF Space Secrets:

1. Go to your Space → **Settings** tab → **Variables and secrets**
2. Add each key as a **Secret** (encrypted, not visible to anyone):
   - `OPENROUTER_API_KEY` → your OpenRouter key

HF injects secrets as environment variables at runtime. Our `config.py` already reads
them via `pydantic-settings` (which reads environment variables automatically).

### Step 3: Push the Code

```bash
# Install the HF CLI (once)
pip install huggingface-hub

# Log in with your token (get it at huggingface.co/settings/tokens)
huggingface-cli login

# Push the whole project to your Space
# This includes data files and the pre-built index — even though they're gitignored
# from GitHub, they're on your disk and the HF CLI will include them.
huggingface-cli upload BigEyesDev/regulation-advisor . --repo-type=space

# Or if you prefer git:
git remote add hf https://huggingface.co/spaces/BigEyesDev/regulation-advisor
git push hf dev:main
```

**Note on data files:** The PDFs and CSVs in `data/` are gitignored from GitHub (they
can be large, and they're downloaded separately). For HF Spaces, you have two options:

**Option A — Push data + pre-built index (recommended for Week 1):**
The `huggingface-cli upload` command uploads everything on disk, including gitignored files.
This means the HF Space starts instantly (no 20-second ingestion on cold start).

**Option B — Push only code, let `_ensure_index()` build on first start:**
Add data files to the HF Space repo separately, then push code. HF will build the index
on first start (~20 seconds). All subsequent restarts load from the cached index.

### Step 4: Verify

Open `https://huggingface.co/spaces/BigEyesDev/regulation-advisor` in a browser.

Test with the 5 benchmark queries:
1. "What AI practices are completely prohibited?" → must cite Article 5
2. "What are the penalties for deploying a prohibited AI system?" → must cite Article 99
3. "Is emotion recognition in the workplace allowed?" → must cite Article 6 or 9
4. "What is a high-risk AI system?" → must cite Article 6
5. "When does the EU AI Act become fully enforceable?" → must cite Article 113

---

## Understanding Environment Variables and Secrets

This is one of the most important concepts in deployment.

**On your laptop:**
- You have a `.env` file with all your API keys
- `pydantic-settings` reads it automatically: `GROQ_API_KEY=xxx` → `settings.groq_api_key`
- The `.env` file is gitignored — it never goes to GitHub

**On HF Spaces:**
- There is no `.env` file (it's gitignored and never pushed)
- Instead, you set "Secrets" in the HF Space UI
- HF injects them as real environment variables when the container starts
- `pydantic-settings` reads them the same way — it doesn't care whether they came from a
  file or from an environment variable

**Analogy:** Your `.env` file is like a notepad on your desk. When you go to an office
(HF Spaces), you can't bring the notepad — but the office has a secure safe (Secrets)
where the same information is stored. Your code reads from "the environment" and doesn't
care where it came from.

```
Local:                         HF Spaces:
.env file                      Space Secrets UI
    │                              │
    ▼                              ▼
Environment variables  ←── same interface ──→  Environment variables
    │                                               │
    ▼                                               ▼
pydantic-settings                           pydantic-settings
reads settings.openrouter_api_key           reads settings.openrouter_api_key
```

---

## What "Cold Start" Means

A "cold start" is when a service starts from zero — no cached data, no running processes.

In HF Spaces:
- If your Space has been idle for a while, HF may shut it down to save resources
- The next time someone visits, HF boots it from scratch (cold start)
- This is when `_ensure_index()` does its job: if the index wasn't committed to the Space
  repo, it rebuilds it (takes ~20 seconds)
- After that, the app serves requests in milliseconds

**Why does this matter?** The first user after a cold start might wait 20+ seconds instead
of 2 seconds. This is acceptable for a free tier. On the paid tier (or with a committed
index), cold starts are ~2 seconds.

---

## What Changed in This Feature

### Files Created
- `requirements.txt` — pip-compatible dependency list for HF Spaces

### Files Modified
- `README.md` — HF Spaces YAML front-matter added at the top
- `src/regulation_advisor/ui/app_runner.py` — three upgrades:
  1. `_ensure_index()` auto-ingestion on first run
  2. Absolute path resolution using `__file__`
  3. `demo.launch(server_name="0.0.0.0", server_port=PORT)`

### Files NOT Changed
- `gradio_app.py` — UI logic is unchanged
- `config.py` — secrets already read from env vars (works for both local and HF)
- `.gitignore` — unchanged (data files are still gitignored from GitHub)

---

## Key Vocabulary

| Term | Plain English |
|---|---|
| **HuggingFace Spaces** | Free cloud hosting for AI demo apps (Gradio, Streamlit) |
| **Front-matter** | YAML config block at the top of README.md that HF reads |
| **requirements.txt** | A file listing Python packages for `pip install` |
| **Secret** | An encrypted environment variable set in the HF UI, never in code |
| **Cold start** | When an app boots from zero (no cached state) |
| **`server_name="0.0.0.0"`** | Listen on all network interfaces (needed for containers) |
| **Lazy initialization** | Build something expensive once on first use, cache it forever |
| **`__file__`** | Python built-in: the absolute path of the currently running script |
| **`os.environ.get("PORT", 7860)`** | Read env var `PORT`, fall back to `7860` if not set |
| **Reverse proxy** | HF's internal server that routes public traffic to your container |

---

## Summary

F6 is the "opening ceremony" of the project — the moment the chatbot stops being a local
experiment and becomes a public product anyone can use.

The three technical changes are small but important:
1. **README front-matter** tells HF Spaces how to run the app
2. **requirements.txt** gives HF the package list in the format it understands
3. **app_runner.py upgrades** handle the three differences between laptop and cloud:
   missing index → build it; relative paths → make them absolute; localhost → 0.0.0.0

After F6, you have a live URL you can add to your LinkedIn profile, send to a client,
or share in a conference talk. The Week 1 milestone is complete.
