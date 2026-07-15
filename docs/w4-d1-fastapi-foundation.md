# W4-D1 — FastAPI Foundation

**Branch:** `feat/w4-d1-fastapi-foundation`  
**Files changed:** `api/schemas.py`, `api/routes.py` (new), `api/app.py`, `ui/gradio_app.py`, `ui/app_runner.py`  
**Tests:** `tests/unit/test_api_health.py`

---

## What we built

A proper FastAPI application skeleton with one live endpoint (`GET /api/health`), automatic API documentation, and the Gradio UI mounted on top of it. The whole thing starts with one command.

---

## Why FastAPI and not just Gradio?

Think of Gradio as a food truck — it serves food (answers) to one type of customer (humans with a browser). FastAPI is a restaurant kitchen that also has a food truck window out front. The kitchen can serve food trucks, delivery apps, and corporate catering all at once.

Right now RegulationAdvisor only has the food truck. Starting from Week 4:
- **Humans** use the Gradio chat window (mounted at `/`)
- **Other programs** (CI, other services, your own scripts) call the REST API (at `/api/...`)
- **You** read the auto-generated documentation at `/docs`

---

## The three files

### `api/schemas.py` — what data looks like

This file only defines data shapes. No logic lives here. Every request and response has a corresponding class.

```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)
```

**Analogy:** Schemas are like order forms at a restaurant. The `Field(min_length=1)` is like the waiter refusing to accept a blank order.

**Why Pydantic?** FastAPI reads these classes and automatically:
1. Validates the incoming JSON (returns a 422 error if the shape is wrong)
2. Generates the API documentation at `/docs`
3. Serialises the Python object back to JSON for the response

You write the shape once. FastAPI does everything else.

```python
class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_backend: str
```

This is the response shape for `GET /api/health`. When the route returns a `HealthResponse` object, FastAPI converts it to:
```json
{"status": "ok", "version": "0.4.0", "vector_store_backend": "faiss"}
```

---

### `api/routes.py` — the endpoints

```python
router = APIRouter()

_agent = None

def set_agent(agent: object) -> None:
    global _agent
    _agent = agent

@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.4.0",
        vector_store_backend=settings.vector_store_backend,
    )
```

**Three things to notice:**

1. **`APIRouter` not `FastAPI`** — A router is a collection of routes. You attach it to the main `FastAPI` app with `app.include_router(router)`. This lets you split routes across files without circular imports.

2. **`_agent = None` with `set_agent()`** — This is the same pattern already used in `tools.py` for the retriever. The agent is expensive to build (loads FAISS index, creates LLM client). We build it once at startup and store a reference here. Routes read it when a request arrives.

3. **`async def health()`** — FastAPI is async-native. `async def` tells Python: "this function can be paused while waiting for I/O (network, disk)". For the health endpoint it doesn't matter much, but for the chat endpoint (Day 2) it means hundreds of users can wait for LLM responses without the server freezing.

---

### `api/app.py` — the main application

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... load everything once at startup ...
    store = build_vector_store()
    store.load(_ROOT / "data" / "index")
    agent = build_agent_graph()
    routes.set_agent(agent)
    yield
    # ... cleanup at shutdown ...
```

**What is `@asynccontextmanager`?**

It's a way to say "run some code at startup, then run some other code at shutdown". The `yield` is the dividing line — everything above `yield` runs when the server starts, everything below runs when it stops.

**Analogy:** Like a restaurant opening and closing procedure. Before the first customer: unlock doors, preheat ovens, seat staff (startup). After the last customer: wash dishes, lock doors, turn off lights (shutdown).

Why not just build the agent at module level (outside any function)? Because that would run when Python imports the file — including during tests. Loading the FAISS index takes a few seconds and requires disk access. By putting it in `lifespan`, it only runs when uvicorn actually starts a real server.

```python
app = gr.mount_gradio_app(_fastapi_app, demo, path="/")
```

`gr.mount_gradio_app` returns a new ASGI app that includes both FastAPI routes (at `/api/...`) and the Gradio UI (at `/`). It's like putting two restaurants under one roof — they share the building but serve different menus.

---

### The lazy agent pattern in `gradio_app.py`

Before this change, `build_ui(agent)` took the agent as a parameter — it had to exist before Gradio could start. Now:

```python
def _get_agent():
    from regulation_advisor.api.routes import _agent
    return _agent

def build_ui() -> gr.Blocks:
    def respond(message, history):
        agent = _get_agent()   # read from routes at request time, not at build time
        if agent is None:
            yield "Service not ready yet."
            return
        ...
```

**Why?** `api/app.py` calls `build_ui()` at module import time (to pass it to `gr.mount_gradio_app`). But `lifespan` hasn't run yet when imports happen — the agent doesn't exist yet. The lazy getter defers the lookup to when a real user sends a message, at which point lifespan has already run.

**Analogy:** A waiter who doesn't check if the chef is in until a customer places an order, not when the restaurant opens. If the customer arrives before the chef — the waiter says "not ready yet".

---

## Understanding `async def` vs `def`

This confuses beginners. Here is the plain-English version:

```python
# Regular function — runs top to bottom, blocks everything else while running
def get_data():
    result = some_slow_operation()  # server frozen for 5 seconds
    return result

# Async function — can be paused while waiting, other requests run during the pause
async def get_data():
    result = await some_slow_operation()  # paused here, other requests processed
    return result
```

For a web server: if 100 users ask a question at the same time and each question takes 3 seconds, a non-async server takes 300 seconds to answer everyone. An async server answers everyone in ~3 seconds because while one request waits for the LLM to respond, the server processes other requests.

---

## How to run

```bash
uvicorn regulation_advisor.api.app:app --reload --port 8000
```

- `regulation_advisor.api.app` — Python module path (dots = folders)
- `app` — the variable name in that module
- `--reload` — restart on file changes (development only)

Then open:
- `http://localhost:8000/` — Gradio chat UI
- `http://localhost:8000/docs` — Swagger UI (interactive API documentation)
- `http://localhost:8000/api/health` — health endpoint directly

---

## Tests

```python
# We build a minimal app with just the router — no lifespan, no index loading
_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)

def test_health_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

**Why a separate `_app` instead of importing the real `app`?** Importing `regulation_advisor.api.app` triggers the lifespan (and loads the FAISS index). Unit tests should be fast and isolated — no disk, no network. By building a minimal `FastAPI()` with just the router, the test runs in milliseconds.

**HTTP status codes to know:**
- `200 OK` — request succeeded
- `422 Unprocessable Entity` — validation failed (wrong request shape)
- `404 Not Found` — route doesn't exist
- `500 Internal Server Error` — unhandled exception in your code

---

## Gate check

```bash
pytest tests/unit/test_api_health.py -v   # 4/4 pass
uvicorn regulation_advisor.api.app:app --port 8000
curl http://localhost:8000/api/health
# {"status":"ok","version":"0.4.0","vector_store_backend":"faiss"}
```
