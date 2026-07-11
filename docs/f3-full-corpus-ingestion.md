# F3 — Full Corpus Ingestion

> **Audience:** Complete beginners. No databases or machine learning background required.
> **What you will understand after reading this:** How we turn 5 raw documents (PDFs and CSVs)
> into a searchable database that a chatbot can query in milliseconds.

---

## The Problem This Feature Solves

In F2 we proved that `ArticleAwareChunker` is the best strategy for legal text.
In F3, we actually **run it on everything** — all 5 documents in `data/` — and **save the result
to disk** so the Gradio chatbot can load it instantly when it starts up.

Without F3, every time the app starts it would have to re-read 144 pages, re-run the regex,
and re-compute 213 embeddings from scratch. That takes ~20 seconds. With F3, startup takes ~2 seconds
(just loading pre-computed numbers from disk).

---

## Analogy: Building a Library Card Catalogue

Imagine you're setting up a new library with 5 books:

1. You read every book page by page
2. For each chapter, you fill out an index card: title, keywords, location
3. You file all 3,000 index cards alphabetically
4. From now on, any patron who asks "do you have something about prohibited AI?" gets an answer in seconds

`run_ingestion()` is the librarian who reads and indexes. FAISS is the filing cabinet.
When the chatbot starts, it opens that filing cabinet — it never re-reads the books.

---

## What Gets Ingested

| File | Type | Chunks produced | Why |
|---|---|---|---|
| `eu_ai_act.pdf` | PDF | **114** | 114 articles in the EU AI Act |
| `gdpr.pdf` | PDF | **99** | 99 articles in GDPR |
| `ai_act_timeline.csv` | CSV | **0** | CSV rows become flat strings — no "Article" header → ArticleAwareChunker skips them |
| `penalty_structure.csv` | CSV | **0** | Same |
| `risk_classification.csv` | CSV | **0** | Same |
| `README.md` | Markdown | **0** | No article headers |
| **Total** | | **213 chunks** | |

> **Why do CSVs produce 0 chunks?** The `ArticleAwareChunker` looks for patterns like
> `\nArticle 5\nProhibited AI practices\n`. A CSV row like
> `"risk_level: High, category: Facial recognition"` has no such pattern.
> For Week 1 we're fine with this — the penalty and timeline data is in the articles themselves.
> In a future sprint, a `CSVChunker` that turns each row into a metadata-rich chunk could be added.

---

## The Pipeline: Step by Step

```python
# scripts/ingest.py calls:
run_ingestion(data_dir=Path("data"), index_dir=Path("data/index"))
```

Here's what happens inside `run_ingestion()`:

```
data/
  eu_ai_act.pdf  ──► PDFLoader.load()  ──► ["page 1 text", "page 2 text", ...]
  gdpr.pdf       ──► PDFLoader.load()  ──►  ...
  *.csv          ──► CSVLoader.load()  ──► ["col1: val1, col2: val2", ...]
                          │
                          ▼
              ArticleAwareChunker.chunk()
                          │
                  [RegulationChunk, ...]    ← 213 total
                          │
                          ▼
           SentenceTransformerEmbedder.encode()
                          │
                  np.ndarray shape(213, 384) ← every chunk → 384 numbers
                          │
                          ▼
               FAISSVectorStore.add()
                          │
               FAISSVectorStore.save()
                          │
                   data/index/
                     index.faiss   ← 320 KB (the 213×384 matrix)
                     chunks.pkl    ← 562 KB (the original chunk objects)
```

Let's unpack each step.

### Step 1: Document Loading

```python
# loaders.py — Factory pattern
for path in sorted(data_dir.iterdir()):
    if not DocumentLoaderFactory.supports(path):
        continue
    texts = DocumentLoaderFactory.create(path).load(path)
```

**What's happening:** We loop over every file in `data/`. The `DocumentLoaderFactory` checks
the file extension (`.pdf` → `PDFLoader`, `.csv` → `CSVLoader`, `.md` → `MarkdownLoader`).
Files like `.gitkeep` or `.DS_Store` are skipped automatically.

**Analogy:** You hire a different specialist for each book format — a PDF expert reads PDFs,
a spreadsheet analyst reads CSVs.

### Step 2: Chunking

```python
all_chunks.extend(chunker.chunk("\n".join(texts), source=path.name))
```

For the EU AI Act, this produces 114 `RegulationChunk` objects, each containing:
- `content` — the full article text (e.g. "Article 5\nProhibited AI practices\n1. The following AI practices shall be prohibited…")
- `article_number` — "5"
- `article_title` — "Prohibited AI practices"
- `source_document` — "eu_ai_act.pdf"

### Step 3: Embedding

```python
embeddings = embedder.encode([c.content for c in all_chunks])
# Shape: (213, 384)  ←  213 chunks, each encoded as 384 numbers
```

**What's happening:** The `SentenceTransformerEmbedder` uses the `all-MiniLM-L6-v2` model
(22 MB, runs locally on CPU in seconds) to convert each chunk's text into a 384-dimensional vector.

**Analogy:** Imagine each document chunk is a voice recording. The embedding model is
a music identification app (like Shazam) that converts every recording into a unique
"fingerprint" — 384 numbers. Songs that sound similar have similar fingerprints.

This is the core trick of semantic search: instead of matching keywords, we match *meaning*.
A query about "banned AI" will match chunks about "prohibited practices" even if the exact
word "banned" never appears.

### Step 4: FAISS Store

```python
store.add(all_chunks, embeddings)
```

FAISS (Facebook AI Similarity Search) is a library designed for one purpose:
**"Given a new vector, find the K most similar vectors in a large collection — fast"**.

Think of it as a sorted address book for vectors. When you search, it doesn't compare
your query to all 213 chunks one by one — it uses a spatial index to jump straight to
the nearest neighbours.

### Step 5: Saving to Disk

```python
store.save(index_dir)
# Creates:
#   data/index/index.faiss  ← the FAISS spatial index (320 KB)
#   data/index/chunks.pkl   ← the original RegulationChunk objects (562 KB)
```

**Why two files?** FAISS stores only the 384-dimensional vectors (pure math). It doesn't
remember what text those vectors came from. So we save the `RegulationChunk` objects
separately in `chunks.pkl` (Python's pickle format). When loading, we reload both and
re-link them by position index.

**Analogy:** The index card catalogue has two parts: the index box (FAISS) which maps topics
to shelf numbers, and the actual books (pickle) which are on those shelves. You need both
to give someone the full answer.

---

## What Changed in This Feature

### 1. `config.py` — New `chunker_strategy` setting

```python
# Before: no way to switch chunkers without changing code
# After:
chunker_strategy: str = "article_aware"  # or "recursive"
```

To switch chunking strategy, change `CHUNKER_STRATEGY=recursive` in your `.env`.
No code changes needed.

### 2. `pipeline.py` — Config-driven chunker factory

```python
# Before:
chunker = chunker or ArticleAwareChunker()

# After:
chunker = chunker or _build_default_chunker()
```

`_build_default_chunker()` reads from settings:

```python
def _build_default_chunker() -> Chunker:
    if settings.chunker_strategy == "recursive":
        return RecursiveCharacterChunker(
            size=settings.chunk_size,    # default: 1000 chars
            overlap=settings.chunk_overlap,  # default: 200 chars
        )
    return ArticleAwareChunker()
```

**Why this matters:** Before, the `chunk_size` and `chunk_overlap` config values existed
but were never actually used — `RecursiveCharacterChunker` always used its own defaults.
Now they're wired through. If you set `CHUNK_SIZE=500` in `.env`, the recursive chunker
will produce 500-character chunks.

### 3. `.gitignore` — Added `data/index/`

```gitignore
# Before: *.faiss and *.pkl were ignored but not the directory
# After:
data/index/
```

**Why:** The FAISS index is a large binary file (~320 KB) that can be regenerated anytime
by running `scripts/ingest.py`. Binary files in git are bad practice:
- They can't be diff'd (you can't see what changed line by line)
- They inflate the repository size permanently (even after deletion)
- They cause merge conflicts that can't be resolved textually

### 4. `tests/integration/test_retrieval.py` — Two integration tests

```python
def test_article_5_retrieved_for_prohibited_query():
    """Article 5 must appear in top-5 for a 'prohibited AI' query."""
    ...

def test_faiss_save_load_round_trip():
    """Load persisted index → search → get non-empty results."""
    ...
```

Both tests are marked with `@_skip_if_no_index` — they skip automatically when the index
hasn't been built yet (e.g. on a fresh clone). This is better than the original `skipif`
that checked for the PDF: the PDF could exist but the index might not.

---

## How to Run Ingestion

```bash
# Build the index (first time or after adding new documents)
uv run python scripts/ingest.py

# Run integration tests (after index is built)
uv run pytest tests/integration/ -v
```

Expected output:
```
INFO — Using ArticleAwareChunker (strategy=article_aware)
INFO — ArticleAwareChunker: 114 chunks from eu_ai_act.pdf
INFO — ArticleAwareChunker: 99 chunks from gdpr.pdf
INFO — Added 213 chunks to FAISS index
INFO — Saved 213 chunks to data/index
```

---

## Searching the Index (What Happens at Query Time)

When a user asks "What is a high-risk AI system?" through the Gradio UI:

```python
# retriever.py
def search(self, query: str, k: int = 5) -> RetrievalResult:
    embedding = self._embedder.encode([query])[0]  # 1. Embed the question
    chunks = self._store.search(embedding, k=k)    # 2. Find 5 nearest chunks
    return RetrievalResult(chunks=chunks, query=query)
```

1. The question `"What is a high-risk AI system?"` is embedded → 384 numbers
2. FAISS finds the 5 chunks whose embedding is most similar (nearest in 384-D space)
3. The LLM receives those 5 chunks as context and generates a cited answer

**The full journey in one picture:**

```
User types question
        │
        ▼
  Embedder encodes question → [0.21, -0.14, 0.87, ...]  (384 numbers)
        │
        ▼
  FAISS searches index.faiss
        │
        ▼
  Returns 5 chunk positions: [42, 6, 71, 55, 8]
        │
        ▼
  Loads those chunks from chunks.pkl:
    - Article 6: "Classification rules for high-risk AI systems"
    - Article 7: "Amendments to Annex III"
    - Article 8: "Compliance with the requirements"
    - ...
        │
        ▼
  LLM generates: "According to Article 6, a high-risk AI system is..."
```

---

## Key Vocabulary

| Term | Plain English |
|---|---|
| **Ingestion** | The process of reading documents, chunking them, embedding them, and saving to disk |
| **FAISS** | Facebook AI Similarity Search — a library for fast nearest-neighbour search in high-dimensional space |
| **Pickle** | Python's native format for saving any Python object to disk (like JSON, but binary and Python-specific) |
| **Vector** | A list of numbers that represents the meaning of a piece of text |
| **Dimension** | The length of the vector — our model produces 384-dimensional vectors |
| **Nearest neighbour** | The vector that is closest in meaning to a given query vector |
| **Index** | A pre-built data structure (like a sorted phone book) that speeds up nearest-neighbour search |
| **`.gitignore`** | A file that tells git which files/folders NOT to track (used for binary outputs) |

---

## Actual Results from the First Run

```
Files processed:
  README.md          →   0 chunks  (no article headers)
  ai_act_timeline.csv →  0 chunks  (CSV rows, no article pattern)
  eu_ai_act.pdf      → 114 chunks  (Articles 1–113 + metadata articles)
  gdpr.pdf           →  99 chunks  (Articles 1–99)
  penalty_structure.csv → 0 chunks
  risk_classification.csv → 0 chunks

Total: 213 chunks
Embedding time: ~1.5 seconds (GPU) / ~8 seconds (CPU)
Index size: index.faiss = 320 KB, chunks.pkl = 562 KB
```

---

## Summary

F3 is the "compilation step" of the RAG system. The output — two small files totalling ~880 KB —
encodes all the legal knowledge the chatbot will ever draw from. Building it takes ~20 seconds.
Loading it at chatbot startup takes ~2 seconds. Every subsequent query takes ~50 ms.

The key design principle here is **separation of concerns**:
- `ingestion` (this feature) runs once and writes to disk
- `retrieval` reads from disk and answers queries
- Neither cares about the other's internals — they communicate only through the FAISS files
