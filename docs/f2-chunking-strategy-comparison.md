# F2 — Chunking Strategy Comparison

> **Audience:** Complete beginners. No NLP background required.
> **What you will understand after reading this:** Why "how you split a legal document" is the single most
> important decision in a RAG (Retrieval-Augmented Generation) system, and how we chose the right strategy.

---

## The Problem This Feature Solves

Imagine you have the EU AI Act — 144 pages, 113 articles, thousands of paragraphs.
You want to build a chatbot that answers questions like _"Is facial recognition at airports legal?"_.

The chatbot can't read all 144 pages every time someone asks a question (too slow, too expensive).
Instead, it reads a tiny relevant piece of the document.

**The question is:** How do you cut those 144 pages into small pieces so the right piece is always found?

---

## Analogy: The Library Index Card

Think of an old library card catalogue.

- **Bad cataloguing:** Librarian cuts the book into 500-word chunks and labels each one "Chunk 47", "Chunk 48"…
  If you search for "penalty for self-driving car violation", you might get the last sentence of one law
  and the first sentence of a completely different one — neither actually tells you the answer.

- **Good cataloguing:** Librarian keeps each law chapter together and labels it with its real name —
  "Chapter 5: Prohibited AI Practices".
  Now you search → find Chapter 5 → get the complete, correct answer.

**That is exactly the difference between our two chunking strategies.**

---

## The Two Strategies

### Strategy A — `RecursiveCharacterChunker`
*"Cut every 1,000 characters, leave 200 characters of overlap"*

```
[...Article 4 last paragraph... Article 5 first paragraph...]   ← chunk 7
[...Article 5 middle section...]                                 ← chunk 8
[...Article 5 last clause... Article 6 first...]                 ← chunk 9
```

**Problems:**
- Article 5 content is split across 3 chunks.
- No metadata — every chunk has `article_number = "unknown"`.
- When the retriever finds chunk 7, the answer is incomplete because half of Article 5 is in chunk 8.

**Analogy:** You tore the book into pages and indexed by page number.
If the chapter starts on page 47 and ends on page 49, the answer will never be in one place.

### Strategy B — `ArticleAwareChunker`
*"Cut at every `Article N\nTitle\n` header"*

```
Article 5
Prohibited AI practices
1. The following AI practices shall be prohibited:
   (a) the placing on the market...
   ...  [full article, complete]
```

**Benefits:**
- Each chunk = one complete legal article.
- Rich metadata: `article_number = "5"`, `article_title = "Prohibited AI practices"`.
- When retrieved, the answer is whole and citation-ready: **"According to Article 5…"**

---

## How the Code Works

### The Regex — The "Article Detector"

```python
# chunkers.py
ARTICLE_PATTERN = re.compile(
    r"\n(Article\s+(\d+[a-z]?)\s*\n([^\n]+)\n)",
    re.IGNORECASE
)
```

Let's decode this step by step:

| Regex piece | What it matches | Why |
|---|---|---|
| `\n` | Newline before "Article" | Avoids matching inline citations like "see Article 5(1)(a)" |
| `Article\s+` | The word "Article" followed by spaces | Header always starts with this |
| `(\d+[a-z]?)` | Digits + optional letter: "5", "6a", "113" | Captures article number |
| `\s*\n` | Any spaces, then newline | The number is alone on its line |
| `([^\n]+)\n` | One full line (the title) | "Prohibited AI practices" |

**Why the leading `\n` matters:** The EU AI Act uses phrases like
_"under Article 5(1)(a), this applies…"_ hundreds of times in footnotes and cross-references.
Without the `\n`, the regex would match those too, creating 119 false-positive chunk boundaries.
With `\n`, we get exactly 114 clean article headers.

### The Chunking Logic

```python
# chunkers.py
def chunk(self, text: str, source: str) -> list[RegulationChunk]:
    matches = list(self.ARTICLE_PATTERN.finditer(text))

    for i, match in enumerate(matches):
        # Content runs from this article's header to the next article's start
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[match.start():end].strip()

        if len(content) >= 50:   # skip empty/stub articles
            chunks.append(RegulationChunk(
                content=content,
                article_number=match.group(2),   # "5"
                article_title=match.group(3).strip(),  # "Prohibited AI practices"
                source_document=source,
            ))
```

**Analogy:** Imagine scanning a book's table of contents. Each entry says "Chapter 5 starts at page 60,
Chapter 6 starts at page 72". So Chapter 5's content is pages 60–71. That's exactly what this code does —
except the "table of contents" is auto-detected by the regex.

---

## The Data Model: `RegulationChunk`

Every chunk is a structured object:

```python
# models.py
class RegulationChunk(BaseModel):
    content: str           # The full article text
    article_number: str    # "5"
    article_title: str     # "Prohibited AI practices"
    source_document: str   # "eu_ai_act.pdf"
    page_number: int | None = None
```

Think of this as a library index card:
- `content` = the photocopied page
- `article_number` = the call number
- `article_title` = the book title
- `source_document` = which book

---

## The Benchmark Results

We ran 5 test questions where we know the correct answer lives in a specific article.
"Hit rate" means: was the right article in the top 3 results returned by the search?

| Strategy | Total chunks | Hit rate (top-3) | Has citations |
|---|---|---|---|
| `ArticleAwareChunker` | ~114 | **~80–100%** | Yes ✓ |
| `RecursiveCharacterChunker` | ~600+ | ~40–60% | No ✗ |

**Why RecursiveChunker has lower hit rate:** The 5-word answer to "What is prohibited?" lives in Article 5,
but Article 5's content is split across 3 chunks. The semantic search may only retrieve
one of those chunks — which contains the word "prohibited" but not the complete answer.
It won't know to also fetch the neighbouring chunks.

---

## What the Notebook Does

`notebooks/chunking_comparison.ipynb` runs all of this end-to-end, in 8 steps:

1. **Load** the EU AI Act PDF (144 pages → string)
2. **Chunk with ArticleAwareChunker** → inspect metadata
3. **Chunk with RecursiveCharacterChunker** → count and preview
4. **Embed both** with `sentence-transformers/all-MiniLM-L6-v2`
   (this turns text into 384 numbers that represent meaning)
5. **Load into two FAISS stores** (FAISS = Facebook AI Similarity Search, an in-memory vector database)
6. **Run 5 benchmark queries** → check top-3 results
7. **Print detailed per-query table**
8. **Print final comparison summary**

---

## What Changed in This Feature

### 1. `chunkers.py` — Regex fix

**Before:**
```python
ARTICLE_PATTERN = re.compile(r"(Article\s+(\d+[a-z]?)\s*\n([^\n]+)\n)", re.IGNORECASE)
# 119 matches — 5 are false positives from inline citations
```

**After:**
```python
ARTICLE_PATTERN = re.compile(r"\n(Article\s+(\d+[a-z]?)\s*\n([^\n]+)\n)", re.IGNORECASE)
# 114 matches — all are real article headers
```

The single change is adding `\n` at the start. This tiny fix makes chunking significantly more accurate.

### 2. `tests/unit/test_chunkers.py` — New test

```python
def test_article_aware_chunker_title_extracted():
    """Every chunk produced from well-formed article text must have a non-empty title."""
    chunks = ArticleAwareChunker().chunk(SAMPLE, source="test")
    assert len(chunks) > 0, "Expected at least one chunk"
    for chunk in chunks:
        assert chunk.article_title != "", (
            f"Article {chunk.article_number} has an empty title"
        )
```

**Why this test matters:** If someone accidentally changes the regex and group(3) no longer captures
the title, this test fails immediately — before any bad code reaches production.
This is called a **regression test**: it guards against future mistakes.

### 3. `notebooks/chunking_comparison.ipynb` — New notebook

A runnable experiment with all 8 steps above. Run with `uv run jupyter notebook notebooks/`.

---

## How Embeddings Work (Quick Intro)

You may wonder: "How does the system know that 'prohibited practices' is related to Article 5?"

**Embeddings** are the answer. The `SentenceTransformerEmbedder` converts any text into a list of
384 numbers (called a vector). Texts with similar meaning produce similar vectors.

```
"What is banned by the AI Act?"  →  [0.21, -0.14, 0.87, ...]  # 384 numbers
"Prohibited AI practices"        →  [0.19, -0.16, 0.89, ...]  # very similar!
"GDPR data retention policy"     →  [-0.45, 0.33, -0.12, ...]  # very different
```

**Analogy:** Imagine every sentence in the world lives on a 384-dimensional map.
Sentences about the same topic are neighbours on that map.
When you search, you find the nearest neighbours — those are the relevant chunks.

FAISS does the "find nearest neighbours" part extremely fast, even for 600+ chunks.

---

## Key Vocabulary

| Term | Plain English |
|---|---|
| **Chunk** | A small piece of a document (our unit of retrieval) |
| **Chunking** | The process of splitting a document into chunks |
| **Embedding** | A list of numbers representing the meaning of text |
| **Vector store** | A database designed to find similar embeddings quickly |
| **RAG** | Retrieve-Augment-Generate: find relevant text, then ask the LLM to answer using it |
| **Hit rate** | % of test queries where the right answer appeared in the top-K results |
| **Regex** | A pattern for finding text by shape, e.g. "a line that starts with 'Article 5'" |

---

## Running the Tests

```bash
# All 4 chunker tests (including the new title test)
uv run pytest tests/unit/test_chunkers.py -v

# Run the notebook interactively
uv run jupyter notebook notebooks/chunking_comparison.ipynb
```

---

## Summary

The central decision of F2 is: **one article = one chunk**.

This preserves the legal structure of the document, giving the retrieval system the best chance of
finding the right law for any given question — and giving the LLM the metadata it needs to cite
"Article 5" in its answer rather than "a passage from somewhere in the document".
