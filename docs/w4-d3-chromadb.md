# W4-D3 — ChromaDB Migration

**Branch:** `feat/w4-d3-chromadb`  
**Files changed:** `config.py`, `ui/app_runner.py`, `api/app.py`  
**Tests:** `tests/unit/test_store_factory.py`

---

## What we built

Wired `build_vector_store()` (the factory function that already existed in `store.py`) into every startup path. Added `index_dir` to config. Now you switch from FAISS to ChromaDB with one environment variable.

---

## Why do we need to migrate at all?

**FAISS** stores vectors in RAM. When the server stops, they're gone. Every restart requires re-ingesting all 400+ document chunks (20-30 seconds). On HuggingFace Spaces, that means every cold start is slow.

**ChromaDB** stores vectors on disk (or on a separate server). When the server stops, the data stays. Restart = instant startup.

**Analogy:** FAISS is a whiteboard — you write everything at the start, it disappears when you erase it. ChromaDB is a filing cabinet — data lives there permanently.

---

## What the Repository pattern gives us

The Repository pattern (used since Week 1) is why today's change is small. The rest of the codebase only talks to the `VectorStore` Protocol:

```python
class VectorStore(Protocol):
    def add(self, chunks: list[RegulationChunk], embeddings: np.ndarray) -> None: ...
    def search(self, query_embedding: np.ndarray, k: int) -> list[RegulationChunk]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

`FAISSVectorStore` and `ChromaDBVectorStore` both implement this protocol. The `Retriever`, `pipeline.py`, and all agent tools never know which backend they're using — they just call `.search()`.

**Analogy:** A Universal Serial Bus (USB). Your laptop doesn't care if you plug in a keyboard, a mouse, or a hard drive — as long as it follows the USB protocol, everything works. `VectorStore` is the USB protocol. FAISS and ChromaDB are the devices.

---

## The factory function

Already existed in `store.py` since it was written with Week 4 in mind:

```python
def build_vector_store() -> VectorStore:
    from regulation_advisor.config import settings
    if settings.vector_store_backend == "chromadb":
        return ChromaDBVectorStore(host=settings.chroma_host, port=settings.chroma_port)
    return FAISSVectorStore()
```

This is a **Factory pattern** — instead of calling `FAISSVectorStore()` or `ChromaDBVectorStore()` directly, you call `build_vector_store()`. The factory decides which one to create based on config.

**Why a factory?** So you never have to find every place in the code that creates a vector store and change it. There's only one place. Change config, change behaviour everywhere.

---

## What changed today

### `config.py`
Added `index_dir` setting:
```python
index_dir: str = "data/index"
```

This replaces the hardcoded path `"data/index"` in `app.py` and `app_runner.py`. Now you can change the index location via `INDEX_DIR=/my/path` in `.env` without touching code.

### `ui/app_runner.py`
Before:
```python
from regulation_advisor.retrieval.store import FAISSVectorStore
store = FAISSVectorStore()
store.load(_INDEX_DIR)
```

After:
```python
from regulation_advisor.retrieval.store import build_vector_store
store = build_vector_store()
store.load(_INDEX_DIR)
```

One-line change. Same behaviour with FAISS (default), different behaviour with ChromaDB.

### `api/app.py`
Same change: `build_vector_store()` instead of hardcoded `FAISSVectorStore()`. Also uses `settings.index_dir` instead of `"data/index"` literal.

---

## How ChromaDB works differently

### FAISS: save/load is explicit
```python
store.save(index_dir)   # writes index.faiss + chunks.pkl to disk
store.load(index_dir)   # reads them back into RAM
```

### ChromaDB: save/load is a no-op
```python
def save(self, path: Path) -> None:
    pass  # ChromaDB persists automatically on write

def load(self, path: Path) -> None:
    pass  # ChromaDB loads automatically on query
```

ChromaDB is a separate server process. When you call `self._col.add(...)`, it writes to disk immediately. When you restart your app and call `self._col.query(...)`, the data is already there.

**Analogy:** FAISS is like saving a Word document manually (Ctrl+S). ChromaDB is like Google Docs — every keystroke is already saved, no explicit save needed.

---

## How to run ChromaDB locally

For development, ChromaDB can run as a local server:

```bash
# Install the server component (already in pyproject.toml)
pip install "chromadb[server]"

# Start the ChromaDB server
chroma run --host localhost --port 8001 --path ./data/chroma_db
```

Then in `.env`:
```
VECTOR_STORE_BACKEND=chromadb
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

Then re-ingest:
```bash
python scripts/ingest.py
```

Then restart your app — vectors are already in ChromaDB, startup is instant.

**Week 5** will add ChromaDB to Docker Compose so you don't have to start it manually.

---

## ChromaDB uses cosine similarity, FAISS uses L2

This is a retrieval quality difference:

- **L2 (FAISS default):** measures the straight-line distance between two vectors. Works fine but can deprioritise relevant short chunks.
- **Cosine similarity (ChromaDB):** measures the angle between vectors, ignoring their length. Better for text similarity — it captures "how similar in meaning" not "how similar in magnitude".

```python
self._col = self._client.get_or_create_collection(
    "regulations",
    metadata={"hnsw:space": "cosine"}   # this sets the distance metric
)
```

You may see `context_precision` improve in Week 4's RAGAS re-run because cosine similarity retrieves more semantically relevant chunks.

---

## The `save` / `load` protocol and why ChromaDB no-ops them

The `VectorStore` Protocol requires `.save()` and `.load()`. FAISS needs them — it's all in RAM. ChromaDB doesn't need them — it's a server. But because the Protocol requires them, `ChromaDBVectorStore` still implements them as empty functions (`pass`). This keeps the interface consistent. Callers don't need to know whether save/load does something or nothing.

**This is the Liskov Substitution Principle:** any code that works with a `VectorStore` should work correctly whether it gets a `FAISSVectorStore` or a `ChromaDBVectorStore`. The caller never checks which one it has.

---

## Tests

```python
def test_factory_returns_faiss_by_default():
    store = build_vector_store()
    assert isinstance(store, FAISSVectorStore)
```

Tests the factory with the default config — no FAISS index needed, we're just checking the type returned.

```python
def test_faiss_save_and_load_roundtrip(tmp_path: Path):
    """FAISS index survives a save/load cycle."""
    store = FAISSVectorStore(dimension=4)
    chunk = RegulationChunk(content="...", article_number="5", ...)
    embeddings = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    store.add([chunk], embeddings)
    store.save(tmp_path)

    store2 = FAISSVectorStore(dimension=4)
    store2.load(tmp_path)
    results = store2.search(embeddings[0], k=1)
    assert results[0].article_number == "5"
```

This test proves data actually persists through save/load. `tmp_path` is a pytest built-in fixture that creates a temporary directory and cleans it up after the test.

---

## Gate check

```bash
pytest tests/unit/test_store_factory.py -v   # 4/4 pass

# To verify ChromaDB manually (requires chroma server running):
VECTOR_STORE_BACKEND=chromadb python scripts/ingest.py
# Restart app — data already there, no re-ingestion
uvicorn regulation_advisor.api.app:app --port 8000
curl http://localhost:8000/api/health
# {"vector_store_backend": "chromadb", ...}
```
