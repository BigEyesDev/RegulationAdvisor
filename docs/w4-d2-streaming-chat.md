# W4-D2 — Streaming Chat API

**Branch:** `feat/w4-d2-streaming-chat`  
**Files changed:** `api/routes.py`  
**Tests:** `tests/unit/test_api_chat.py`

---

## What we built

Two new endpoints in `routes.py`:

| Endpoint | Type | Use case |
|---|---|---|
| `POST /api/chat` | Streaming (SSE) | Browser — tokens appear as they're generated |
| `POST /api/chat/sync` | Synchronous (JSON) | Eval harness, scripts, testing |

---

## What is streaming and why does it matter?

Without streaming, a user asks a question and sees nothing for 4–8 seconds, then the full answer appears at once.

With streaming (SSE), the user sees tokens appearing one at a time — like watching someone type. This is how ChatGPT works. Users feel a big speed difference even though the total time is the same.

**SSE = Server-Sent Events.** It's a simple protocol built on top of HTTP:
- The server keeps the HTTP connection open
- It sends small chunks of data, each prefixed with `data: `
- Each chunk ends with two newlines (`\n\n`)
- The client reads chunks as they arrive

```
data: {"type": "token", "content": "Article"}

data: {"type": "token", "content": " 5"}

data: {"type": "token", "content": " prohibits"}

data: {"type": "done"}
```

The client assembles these into the full answer in real time.

---

## The streaming endpoint

```python
@router.post("/api/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    async def generate():
        config = {"configurable": {"thread_id": request.session_id}}
        async for event in _agent.astream_events(
            {"messages": [("human", request.message)]}, config=config, version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**Breaking this down piece by piece:**

### `async def generate()` — a generator function

`generate()` is an **async generator**. Instead of `return`ing one big thing, it `yield`s many small things over time.

**Regular function analogy:** Fill a bucket with water, carry it to the garden, pour it all at once.

**Generator analogy:** Connect a hose to the tap, water flows to the garden continuously.

Each time the LLM produces a token, we `yield` it immediately. The HTTP connection stays open and the client receives each chunk as it arrives.

### `_agent.astream_events(...)` — LangGraph async event stream

This is LangGraph's async streaming API. It produces a stream of events as the agent runs. There are many event types — node entering, tool starting, tool ending, LLM streaming a token. We only care about one:

```python
if event["event"] == "on_chat_model_stream":
    token = event["data"]["chunk"].content
```

`on_chat_model_stream` fires every time the LLM generates a token. `chunk.content` is that token (often 1-4 characters). We wrap it in JSON and `yield` it.

### `version="v2"` — LangGraph events API version

LangGraph has two event stream formats. `v2` is the current one and the one the master plan specifies. Always use `v2`.

### `StreamingResponse` — FastAPI's streaming response

```python
return StreamingResponse(
    generate(),               # the async generator
    media_type="text/event-stream",   # SSE content type
    headers={
        "Cache-Control": "no-cache",     # don't cache streaming data
        "X-Accel-Buffering": "no",       # tell nginx NOT to buffer (send immediately)
    },
)
```

`X-Accel-Buffering: no` is specifically for nginx (common reverse proxy). Without it, nginx buffers the whole response before sending — defeating streaming.

---

## The sync endpoint

```python
@router.post("/api/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest) -> ChatResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    config = {"configurable": {"thread_id": request.session_id}}
    result = await _agent.ainvoke(
        {"messages": [("human", request.message)]}, config=config
    )

    answer = result["messages"][-1].content
    retrieved = result.get("retrieved_chunks", [])
    sources = [
        SourceReference(article_number=c.article_number, source_document=c.source_document)
        for c in retrieved
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence_score=result.get("confidence_score", 1.0),
        warnings=[],
        session_id=request.session_id,
    )
```

`ainvoke()` is the async version of `invoke()` — it runs the whole agent and returns the final state. No streaming.

**Why do we need both?**

| Caller | Needs |
|---|---|
| Browser user | Streaming — wants to see words appear immediately |
| `scripts/run_evaluation.py` | Sync — needs the complete answer, processes it programmatically |
| Tests | Sync — easier to assert on a complete JSON response |
| Other services | Either, depending on their needs |

Having both means every caller gets what it needs without workarounds.

---

## Test approach — mock the agent

The agent requires a loaded FAISS index and a valid API key. Unit tests must not need either of those.

```python
@pytest.fixture()
def client_with_agent():
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Article 5 prohibits social scoring.")],
        "retrieved_chunks": [],
        "confidence_score": 0.95,
    })

    app = FastAPI()
    app.include_router(router)

    original = routes._agent     # save original
    routes.set_agent(mock_agent)  # inject mock
    with TestClient(app) as c:
        yield c, mock_agent
    routes.set_agent(original)   # restore original
```

`MagicMock()` creates a fake object that pretends to be anything. `AsyncMock` is a fake that returns a coroutine — needed because our route uses `await agent.ainvoke(...)`. If we used a plain `MagicMock`, the `await` would fail.

**Pattern:** Save the original `_agent`, inject the mock, test, restore the original. This is called **test isolation** — each test leaves the system exactly as it found it.

---

## How to test manually

With the server running (`uvicorn regulation_advisor.api.app:app --port 8000`):

```bash
# Streaming — you see tokens arrive one by one
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What practices does Article 5 prohibit?"}'

# Sync — full JSON response
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Article 5?", "session_id": "test-1"}' | python -m json.tool
```

The `-N` flag on `curl` disables buffering so you see the SSE stream live.

---

## Understanding `await`

```python
result = await _agent.ainvoke(...)
```

Without `await`, `ainvoke()` returns a coroutine object — essentially a "promise to compute something". The `await` tells Python: "pause this function here, let other requests run, and resume when `ainvoke()` finishes."

**Analogy:** You're at a restaurant. You place your order (`ainvoke()`) and the waiter says "I'll bring it when it's ready" (returns a coroutine). `await` means you sit down and let other customers order while you wait, instead of standing at the counter blocking everyone else.

---

## Gate check

```bash
pytest tests/unit/test_api_chat.py -v   # 5/5 pass
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Article 5?"}' | python -m json.tool
# Returns: {"answer": "...", "sources": [...], ...}
```
