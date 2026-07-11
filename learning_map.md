# Learning Map — What to Study Before Each Week

This is your study guide. It tells you exactly what to learn, where, and for how long
before you write any code that week. Nothing more, nothing less.

---

## Before Week 1 — Two Days of Study First

You cannot write Week 1 code without this. Do both on consecutive days before starting.

---

### Study Block 1 — LangChain (6–8 hours)

**Where:** `academy.langchain.com/courses/intro-to-langchain`
Free. Official. Built on LangChain v1.0 (current).

**What to do:** Work through the course in order. Build every project they give you.
Do not skip the coding parts — watching without coding teaches nothing.

**Stop when you can answer these without looking it up:**
- What does `prompt | llm | parser` mean and why does it work?
- What is a `ChatPromptTemplate` and how do you add variables to it?
- What is `StrOutputParser` doing and when would you use `JsonOutputParser` instead?
- What is a retriever and how does it connect to a chain?

**What you can skip:** Anything about agents, LCEL advanced patterns, or memory classes
like `ConversationBufferMemory` — those are deprecated in v1.0.

---

### Study Block 2 — LlamaIndex + Gradio (1 hour total)

**LlamaIndex (30 minutes):**
Go to `docs.llamaindex.ai` → Getting Started → Starter Tutorial.
Run their example locally. That's it. You only need to understand document loading
and how it produces `Document` objects. Everything else you'll learn as you need it.

**Gradio (30 minutes):**
Go to `gradio.app/guides/quickstart`.
Run their `gr.ChatInterface` example locally. Change one thing. Run it again.
That's all you need before Week 1 Day 6.

---

## Before Week 2, Days 1–2 — LangGraph (6 hours)

**Where:** `academy.langchain.com/courses/intro-to-langgraph`
Free. Official. Current.

**Do this before writing any Week 2 agent code. Do not skip this.**
LangGraph's state machine model is genuinely confusing if you try to learn it
from code alone. The course explains the mental model first.

**Stop when you can answer these:**
- What is a `TypedDict` state and why does LangGraph require it?
- What is the difference between a node and an edge?
- What is a conditional edge and when do you use one?
- What is a checkpointer and how does it replace conversation memory classes?
- What does `interrupt_before` do?

**What you can skip:** Multi-agent coordination and subgraphs — those are Week 3+ concepts.

---

## Before Week 3, Day 1 — RAGAS (20 minutes)

**Where:** `docs.ragas.io` → Quickstart
Run their example. Understand what `faithfulness`, `answer_relevancy`,
`context_precision`, and `context_recall` mean conceptually.
That's the entire pre-study for this week. Everything else is implementation.

---

## Before Week 4, Days 1–2 — FastAPI (3 hours)

**Where:** `fastapi.tiangolo.com/tutorial`
Work through Chapters 1–5 only:
1. First Steps
2. Path Parameters
3. Query Parameters
4. Request Body
5. Response Model

Run every example yourself. FastAPI is easy to read but tricky to write until you've
seen the patterns a few times.

**Stop when you can write a simple POST endpoint that:**
- Takes a Pydantic request body
- Returns a Pydantic response model
- Has a background task

**What you can skip for now:** WebSockets, security, database integration, testing.

---

## Before Week 5, Day 1 — Docker (1 hour)

**Watch:** Search YouTube for "Docker Compose in 12 minutes TechWorld with Nana"
That single video is enough to understand `docker-compose.yml`, services, volumes,
and health checks — everything you need for Week 5.

If you already know Docker basics, skip this entirely.

---

## Before Week 6, Days 1–2 — Fine-tuning Concepts (2 hours reading)

**Read these two things, in this order:**

1. Sebastian Raschka's LoRA insights article
   URL: `lightning.ai/pages/community/lora-insights`
   Time: ~1 hour. Read it carefully. He explains rank, alpha, target modules,
   and dropout with actual intuition — not just formulas.

2. Unsloth quickstart
   URL: `docs.unsloth.ai`
   Time: ~30 minutes. Run their basic fine-tuning Colab notebook.
   Don't modify it yet — just run it end to end and understand each step.

**You do not need to read the LoRA or QLoRA papers.** The Raschka article covers
everything practically relevant. The papers are useful later if you want depth.

---

## Everything Else — Learn In 30 Minutes When You Need It

These tools have tiny APIs. Don't pre-study them. Just read their quickstart
when you first use them in the project.

| Tool | When you need it | Where to look |
|------|-----------------|---------------|
| `sentence-transformers` | Week 1 Day 4 | `sbert.net/docs` → Usage → Computing Embeddings |
| FAISS | Week 1 Day 4 | Their GitHub README — the first 10 lines of code |
| ChromaDB | Week 4 Day 3 | `docs.trychroma.com` → Getting Started |
| promptfoo | Week 3 Day 4 | `promptfoo.dev/docs` → Getting Started |
| Tavily | Week 2 Day 3 | `docs.tavily.com` → Python SDK quickstart |
| HuggingFace Hub push | Week 6 Day 5 | `huggingface.co/docs/huggingface_hub` → Upload a model |
| AWS ECS | Week 5 Day 4 | Follow the plan step by step — it guides you |
| smolagents | Week 2 Day 6 | `huggingface.co/docs/smolagents` → Guided Tour — 1 hour |

---

## The Rule for All of This

**Study until you can read the code. Then start building.**

You don't need to understand every edge case before you write your first line.
You need to understand the mental model — what the tool is doing and why —
so that when you hit an error, you know where to look.

If you're spending more than the time listed above on pre-study for any week,
you're over-preparing. Start building and learn the rest in context.
