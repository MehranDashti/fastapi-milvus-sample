# 05 — Putting It All Together

> **Goal of this document:** Understand how RAG, Milvus, LangChain, and
> LangGraph fit together into one production service — the full data flow from
> HTTP request to response, the design decisions behind each layer, and how to
> evolve the system further.

---

## 1. The Full Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HTTP Client                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ REST API
┌──────────────────────────────▼──────────────────────────────────────────┐
│                        FastAPI Application                              │
│                         (app/main.py)                                   │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  /ingest/*  │  │   /query    │  │  /agent/*    │  │   /chat     │  │
│  │  /lc/ingest │  │  /lc/query  │  │              │  │             │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  │
└─────────┼────────────────┼────────────────┼─────────────────┼──────────┘
          │                │                │                 │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼───────┐  ┌────▼────────┐
   │  app/core/  │  │  app/core/  │  │ app/agents/ │  │ app/memory/ │
   │  ingester   │  │  searcher   │  │  rag_agent  │  │ conversation│
   │  (chunk +   │  │  (ANN query)│  │  (LangGraph)│  │  (history + │
   │   embed +   │  │             │  │             │  │   search)   │
   │   store)    │  │  llm.py     │  │  tool_agent │  │             │
   └──────┬──────┘  └──────┬──────┘  └─────┬───────┘  └────┬────────┘
          │                │                │               │
   ┌──────▼──────┐  ┌──────▼──────────────────────────────▼────────┐
   │ app/core/   │  │             app/core/embedder                 │
   │ milvus_     │  │        (OpenAI text-embedding-3-small)        │
   │ client.py   │  └──────────────────────────────────────────────┘
   │ (singleton) │
   └──────┬──────┘
          │
   ┌──────▼───────────────────────────────────────────────────────┐
   │                     Milvus Vector DB                         │
   │                    (localhost:19530)                          │
   │              Collection: "documents" (1536-dim HNSW)         │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │              app/langchain/ (parallel path)                  │
   │   lc_components.py → Milvus LC wrapper → "lc_documents"      │
   └──────────────────────────────────────────────────────────────┘
```

---

## 2. Layer Responsibilities

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api/routes/` | HTTP request/response, validation, duration measurement |
| Models | `app/api/models/` | Pydantic schema for every endpoint's I/O |
| Core | `app/core/` | Pure RAG logic: chunk, embed, store, search, answer |
| Agents | `app/agents/` | LangGraph agentic RAG + OpenAI tool-use loop |
| LangChain | `app/langchain/` | Alternative RAG pipeline via LangChain LCEL |
| Memory | `app/memory/` | Conversation history management + multi-turn chat |
| Config | `app/config.py` | Single settings object, loaded once from `.env` |
| Logger | `app/logger.py` | Structured logging, one logger per module |

Each layer depends only on layers *below* it. Routes depend on core.
Core depends on config. Nothing depends on routes.

---

## 3. Startup Sequence

```
uvicorn app.main:app --reload
          │
          ↓
    FastAPI creates app instance
          │
          ↓
    @asynccontextmanager lifespan() begins
          │
          ├── get_client()
          │     └── MilvusClient(uri="http://localhost:19530")
          │           → opens TCP connection
          │           → authenticates
          │
          ├── ensure_collection(client)
          │     ├── has_collection("documents") → True/False
          │     ├── if False: create_collection(schema, index_params)
          │     └── load_collection("documents")
          │           → segments loaded into RAM
          │           → HNSW graph ready for search
          │
          ├── get_agent()
          │     └── build_agent()
          │           ├── StateGraph(AgentState)
          │           ├── add_node × 4
          │           ├── add_edge × 3
          │           ├── add_conditional_edges × 1
          │           └── graph.compile()
          │                 → validates graph topology
          │                 → JIT-compiles routing functions
          │
          └── yield   ← "Service ready." server starts accepting requests
```

The lifespan pattern ensures the Milvus connection and LangGraph compilation
happen once at startup. If either fails (Milvus unreachable, graph invalid),
the server fails fast rather than on the first request.

---

## 4. Request Flows — End to End

### 4.1 POST /ingest/text

```
Request: {"text": "FastAPI is...", "source": "docs.txt"}
  │
  ↓ app/api/routes/ingest.py → ingest_text_route()
  │   validate: text.strip() non-empty
  │   validate: source.strip() non-empty
  │   start = time.time()
  │
  ↓ app/core/ingester.py → ingest_text(text, source)
  │
  ├─ get_client()              [Milvus singleton]
  ├─ ensure_collection()       [idempotent — no-op if loaded]
  │
  ├─ client.query(filter='source == "docs.txt"')
  │     → [check for existing chunks]
  ├─ if exists: client.delete(filter='source == "docs.txt"')
  │     → delete old chunks before re-ingest (idempotent)
  │
  ├─ chunk_text(text)
  │     → tiktoken.encode()
  │     → sliding window (SIZE=256, OVERLAP=30)
  │     → ["FastAPI is a modern...", "...modern web framework built on..."]
  │
  ├─ embed_batch(chunks)
  │     → OpenAI API: text-embedding-3-small
  │     → [[0.023, -0.114, ...], [0.045, 0.221, ...]]   # 1536 floats each
  │
  ├─ client.insert(entities)
  │     → each entity: {embedding, content, source, chunk_index, created_at}
  │     → Milvus: buffer in growing segment → eventually seal + index
  │
  └─ return {"source": "docs.txt", "chunks": 2, "inserted": 2}
  │
  ↓ IngestResponse(source=..., chunks=..., inserted=..., duration_ms=...)
  ↓ JSON response: 200 OK
```

### 4.2 POST /query

```
Request: {"question": "What is dependency injection?", "top_k": 5, "min_score": 0.45}
  │
  ↓ app/api/routes/query.py → query_route()
  │
  ↓ app/core/searcher.py → search(query, top_k=5)
  │   embed_text("What is dependency injection?")
  │     → [0.087, -0.203, 0.441, ...]   (1536 floats)
  │
  │   client.search(
  │     data=[query_vector],
  │     limit=5,
  │     search_params={"ef": 64},   # HNSW ef >= top_k
  │     output_fields=["content", "source", "chunk_index"]
  │   )
  │     → [
  │         {score: 0.91, entity: {content: "...", source: "fastapi_docs.pdf"}},
  │         {score: 0.84, entity: {...}},
  │         {score: 0.71, entity: {...}},
  │         {score: 0.55, entity: {...}},
  │         {score: 0.31, entity: {...}},
  │       ]
  │
  ↓ app/core/llm.py → ask_with_score_filter(query, chunks, min_score=0.45)
  │   filter chunks: [0.91, 0.84, 0.71, 0.55] pass; [0.31] rejected
  │   build_context(filtered_chunks)
  │     → "[1] (source: fastapi_docs.pdf, chunk: 14)\nFastAPI handles DI via..."
  │   OpenAI API: gpt-4o-mini, temperature=0.1
  │     → "FastAPI uses a Depends() function for dependency injection..."
  │
  ↓ QueryResponse(question=..., answer=..., sources=..., chunks_used=4, tokens=..., duration_ms=...)
  ↓ JSON response: 200 OK
```

### 4.3 POST /agent/query

```
Request: {"question": "Who invented FastAPI?"}
  │
  ↓ app/api/routes/agent.py → agent_query_route()
  │
  ↓ app/agents/rag_agent.py → agent_query(question)
  │
  │   initial_state = {question: "Who invented FastAPI?", attempts: 0, ...}
  │   agent.invoke(initial_state)
  │
  │   ┌─ Node: retrieve ─────────────────────────────────────────────┐
  │   │  search("Who invented FastAPI?", top_k=5)                    │
  │   │  best_score = 0.38  (poor — question uses "invented")        │
  │   │  attempts = 1                                                 │
  │   └──────────────────────────────────────────────────────────────┘
  │        ↓
  │   ┌─ Node: evaluate ─────────────────────────────────────────────┐
  │   │  0.38 < 0.50 AND 1 < 2 → quality_ok = False                 │
  │   └──────────────────────────────────────────────────────────────┘
  │        ↓ (conditional edge: route to "rephrase")
  │   ┌─ Node: rephrase ─────────────────────────────────────────────┐
  │   │  GPT: "Who invented FastAPI?" →                              │
  │   │       "What is the creator and origin of FastAPI framework?"  │
  │   └──────────────────────────────────────────────────────────────┘
  │        ↓ (fixed edge: back to retrieve)
  │   ┌─ Node: retrieve ─────────────────────────────────────────────┐
  │   │  search("What is the creator and origin of FastAPI...", top_k=5)│
  │   │  best_score = 0.89  (good — matches indexed content)          │
  │   │  attempts = 2                                                 │
  │   └──────────────────────────────────────────────────────────────┘
  │        ↓
  │   ┌─ Node: evaluate ─────────────────────────────────────────────┐
  │   │  0.89 >= 0.50 → quality_ok = True                            │
  │   └──────────────────────────────────────────────────────────────┘
  │        ↓ (conditional edge: route to "generate")
  │   ┌─ Node: generate ─────────────────────────────────────────────┐
  │   │  ask_with_score_filter(question, chunks, min_score=0.35)      │
  │   │  answer = "FastAPI was created by Sebastián Ramírez..."       │
  │   └──────────────────────────────────────────────────────────────┘
  │        ↓ (fixed edge: END)
  │
  ↓ AgentResponse(question=..., answer=..., attempts=2, best_score=0.89,
  │               reasoning=[...], chunks_used=5, duration_ms=...)
  ↓ JSON response: 200 OK
```

### 4.4 POST /chat

```
Request: {
  "question": "Who created it?",
  "history": [
    {"role": "user",      "content": "What is FastAPI?"},
    {"role": "assistant", "content": "FastAPI is a modern Python web framework..."}
  ]
}
  │
  ↓ app/api/routes/chat.py → chat_route()
  │   convert ChatMessage objects → plain dicts
  │
  ↓ app/memory/conversation.py → chat()
  │
  │   truncate_history(history, max_turns=10)  [no-op here, only 2 messages]
  │
  │   search_query = "What is FastAPI? Who created it?"
  │     (enriched: last user message + current question)
  │
  │   search(search_query, top_k=5)
  │     → chunks about FastAPI and Sebastián Ramírez
  │
  │   relevant = [chunks where score >= 0.40]
  │
  │   build_context(relevant) → formatted string
  │
  │   build_messages_with_history(question, history, context):
  │     [
  │       SystemMessage("Answer based ONLY on context..."),
  │       HumanMessage("Here is the relevant context:\n[1]...\nUse this..."),
  │       AIMessage("Understood. I'll answer from the context."),
  │       HumanMessage("What is FastAPI?"),           ← from history
  │       AIMessage("FastAPI is a modern..."),        ← from history
  │       HumanMessage("Who created it?"),            ← current question
  │     ]
  │
  │   ChatOpenAI.invoke(messages)
  │     GPT resolves "it" via history → answers about Sebastián Ramírez
  │
  │   updated_history = history + [
  │     {"role": "user",      "content": "Who created it?"},
  │     {"role": "assistant", "content": "FastAPI was created by Sebastián Ramírez..."},
  │   ]
  │
  ↓ ChatResponse(question=..., answer=..., history=updated_history, ...)
  ↓ JSON response: 200 OK
```

---

## 5. Two RAG Pipelines, Side by Side

This project implements the same RAG logic twice — once with raw SDK and once
with LangChain — to make the comparison concrete.

```
Endpoint              Implementation           Collection
─────────────────────────────────────────────────────────
POST /ingest/text     app/core/ingester.py     documents
POST /query           app/core/searcher.py     documents
                      app/core/llm.py
─────────────────────────────────────────────────────────
POST /lc/ingest       app/langchain/           lc_documents
POST /lc/query        lc_components.py
─────────────────────────────────────────────────────────
POST /agent/query     app/agents/rag_agent.py  documents
                      (uses app/core/searcher)
─────────────────────────────────────────────────────────
POST /agent/tool      app/agents/tool_agent.py documents
                      app/agents/tools.py
─────────────────────────────────────────────────────────
POST /chat            app/memory/conversation  documents
                      app/memory/memory.py
                      (uses app/core/searcher)
```

The LangChain pipeline uses a **separate collection** (`lc_documents`) because
LangChain's Milvus wrapper uses its own schema (fields named `text` and
`vector` instead of `content` and `embedding`). Documents ingested via
`/lc/ingest` are only searchable via `/lc/query`. Documents ingested via
`/ingest/text` are only searchable via `/query`, `/agent/query`, `/chat`.

In production, you would pick one pipeline and standardize on it. This project
keeps both to show both approaches.

---

## 6. Configuration Flow

```
.env file
  │  OPENAI_API_KEY=sk-...
  │  MILVUS_HOST=localhost
  │  MILVUS_COLLECTION=documents
  │  CHUNK_SIZE=256
  │
  ↓ python-dotenv: load_dotenv() in app/config.py
  │
  ↓ class Settings:
  │     milvus = MilvusConfig()    (reads MILVUS_* env vars)
  │     openai = OpenAIConfig()    (reads OPENAI_*, EMBEDDING_*, LLM_*)
  │     chunk  = ChunkConfig()     (reads CHUNK_*)
  │
  ↓ settings = Settings()   ← module-level singleton
  │
  ↓ Imported by every module that needs configuration:
      app/core/embedder.py   → settings.openai.API_KEY
      app/core/ingester.py   → settings.chunk.SIZE, settings.milvus.COLLECTION
      app/core/searcher.py   → settings.milvus.COLLECTION
      app/core/llm.py        → settings.openai.LLM_MODEL
      app/core/milvus_client → settings.milvus.URI
```

There is one source of truth for settings. No environment variable is read
outside `app/config.py`. Changing a setting requires only changing `.env` and
restarting — no code changes.

---

## 7. Error Handling Architecture

```
HTTP Request
  │
  ↓ FastAPI route handler
  │   ├── Pydantic validation (auto) → 422 Unprocessable Entity
  │   ├── HTTPException → 4xx response (explicit error)
  │   └── Uncaught exception → global_exception_handler
  │
  ↓ global_exception_handler (app/main.py)
      logger.error(f"Unhandled error on {method} {url}: {exc}")
      return JSONResponse(500, {"error": "Internal server error", "detail": str(exc)})
```

**Retry logic** lives in the embedding layer:

```
embed_text() / embed_batch()
  ↓
_with_retry(fn, retries=3, backoff=2.0)
  ├── RateLimitError → wait 2s, 4s, 8s → retry
  ├── APIStatusError >= 500 → retry
  └── APIStatusError < 500 → raise immediately (client error, don't retry)
```

Retries are appropriate for transient failures (rate limiting, server errors).
They are not appropriate for permanent failures (invalid API key, wrong model
name) — those are raised immediately.

---

## 8. Component Interaction Map

```
app/main.py
  imports → app/agents/rag_agent.py (get_agent)
  imports → app/api/routes/ingest, query, agent, chat

app/api/routes/ingest.py
  imports → app/core/ingester (ingest_text, delete_by_source)
  imports → app/core/milvus_client (get_client, get_collection_stats)
  imports → app/core/pdf_extractor (extract_text_from_pdf_bytes)
  imports → app/langchain/lc_components (lc_ingest)

app/api/routes/query.py
  imports → app/core/searcher (search)
  imports → app/core/llm (ask_with_score_filter)
  imports → app/langchain/lc_components (lc_query)

app/api/routes/agent.py
  imports → app/agents/rag_agent (agent_query)
  imports → app/agents/tool_agent (run_tool_agent)

app/api/routes/chat.py
  imports → app/memory/conversation (chat)

app/agents/rag_agent.py
  imports → app/core/searcher (search)
  imports → app/core/llm (ask_with_score_filter)
  imports → app/config (settings)

app/agents/tool_agent.py
  imports → app/agents/tools (TOOL_SCHEMAS, execute_tool)
  imports → app/config (settings)

app/agents/tools.py
  imports → app/core/searcher (search)   ← tool_milvus_search uses this

app/memory/conversation.py
  imports → app/core/searcher (search)
  imports → app/core/llm (build_context)
  imports → app/memory/memory (build_messages_with_history, truncate_history)

app/core/ingester.py
  imports → app/core/milvus_client (get_client, ensure_collection)
  imports → app/core/embedder (embed_batch)
  imports → app/core/pdf_extractor (extract_text_from_pdf)
  imports → app/config (settings)

app/core/searcher.py
  imports → app/core/milvus_client (get_client, ensure_collection)
  imports → app/core/embedder (embed_text)
  imports → app/config (settings)

app/core/embedder.py
  imports → app/config (settings)

app/core/milvus_client.py
  imports → app/config (settings)

app/langchain/lc_components.py
  imports → app/config (settings)
```

No circular imports. The dependency graph is a DAG rooted at `app/config.py`.

---

## 9. Why Each Technology Was Chosen

### FastAPI over Flask / Django

- Async-native: file uploads, streaming responses, concurrent requests
- Pydantic validation built in: request bodies are typed and validated automatically
- OpenAPI docs auto-generated from route definitions
- `lifespan` context manager for clean startup/shutdown

### Milvus over Pinecone / Weaviate / ChromaDB

- Self-hosted: data never leaves your infrastructure (compliance)
- HNSW index: high recall at production scale
- Hybrid search: scalar filter + vector ANN in one query
- Pymilvus client: full control over schema, index, and collection management

### OpenAI `text-embedding-3-small` over alternatives

- 1536 dimensions: high quality without the cost of `text-embedding-3-large` (3072 dim)
- 62.3% MTEB score: best cost/quality ratio for English text
- Consistent: same model always produces the same vector for the same text
  (deterministic, unlike LLMs)

### LangGraph over CrewAI / AutoGen

- Explicit state machine: agent logic is inspectable and auditable
- Checkpointing built in: pause/resume for long-running agents
- Human-in-the-loop built in: approval gates without custom code
- Fine-grained streaming: every node's output streams separately
- LangChain ecosystem: works with all LangChain components directly

### tiktoken over character-based chunking

- Token-accurate: chunk sizes respect the actual tokenization used by OpenAI
- Overlap is in tokens, not characters: consistent semantic coverage
- `cl100k_base` encoding matches both the embedding model and GPT-4o

---

## 10. Extension Points

The architecture is designed to be extended without breaking existing
functionality.

### Add a new document type (e.g., DOCX)

1. Create `app/core/docx_extractor.py` with `extract_text_from_docx(filepath)`
2. Add `POST /ingest/docx` route in `app/api/routes/ingest.py`
3. Add `DocxIngestResponse` model in `app/api/models/ingest.py` (or reuse `IngestResponse`)

No other files change.

### Add a new tool to the tool agent

1. Add the function in `app/agents/tools.py`
2. Add it to `TOOL_REGISTRY` and `TOOL_SCHEMAS`

The agent loop in `tool_agent.py` automatically picks it up.

### Add a new agentic node to the RAG agent

1. Define the node function in `app/agents/rag_agent.py`
2. Add fields to `AgentState` if needed
3. Register with `graph.add_node()` and wire with `graph.add_edge()` or `add_conditional_edges()`

### Add a new LLM provider

1. Change `settings.openai.LLM_MODEL` to any OpenAI-compatible model name, or
2. Replace `ChatOpenAI` with `ChatAnthropic`, `ChatCohere`, etc. in `app/core/llm.py`
   and `app/langchain/lc_components.py` — the interface is identical

### Add multi-tenancy

1. Add `tenant_id` to the Milvus schema as a dynamic field
2. Thread `tenant_id` through all ingest/search calls
3. All queries must include `filter=f'tenant_id == "{tenant_id}"'`

### Add conversation state server-side (stateful sessions)

Replace client-side history with server-side LangGraph checkpointing:

```python
# Replace the stateless chat() call with a checkpointed agent
agent = build_conversational_agent().compile(checkpointer=RedisSaver(...))
config = {"configurable": {"thread_id": session_id}}
result = agent.invoke({"question": question}, config=config)
```

The client no longer needs to send history — the server manages it via the
checkpoint store keyed by `session_id`.

---

## 11. Production Checklist

### Security

- [ ] `OPENAI_API_KEY` is in `.env`, never committed to git
- [ ] `.env` is in `.gitignore`
- [ ] File upload endpoints validate file extension before reading content
- [ ] `tool_calculator` uses sandboxed `eval` with `__builtins__: {}`
- [ ] Source names from user input are parameterized in Milvus filter expressions
      (already done: `f'source == "{source}"'` — validate source is not SQL-injectable)

### Performance

- [ ] Milvus client is a singleton (no per-request reconnect)
- [ ] LangGraph agent is compiled once at startup
- [ ] `embed_batch` is used for bulk ingestion (one API call per 100 chunks)
- [ ] Collection is loaded at startup (not per-request)
- [ ] `ef` parameter in search is tuned (`max(top_k * 2, 64)`)

### Observability

- [ ] Structured logging in every module (`get_logger(__name__)`)
- [ ] `duration_ms` in every response (latency tracking)
- [ ] Reasoning trace returned in agent responses (decision audit)
- [ ] Global exception handler logs every unhandled error

### Reliability

- [ ] Retry on OpenAI rate limits (exponential backoff: 2s, 4s, 8s)
- [ ] `min_score` filter prevents garbage chunks reaching the LLM
- [ ] Idempotent ingestion (delete-before-insert)
- [ ] `ensure_collection` is idempotent (safe to call on every startup)
- [ ] `MAX_ATTEMPTS` in the agent prevents infinite retry loops

### Testing

- [ ] Unit tests run without Milvus or OpenAI (`-m "not integration"`)
- [ ] Integration tests create a separate `test_documents` collection
- [ ] Test collection is dropped after the test session
- [ ] `ruff check` passes with zero errors (`make lint`)

---

## 12. How the Four Technologies Form a Complete System

```
Technology     Role in the System
───────────────────────────────────────────────────────────────────────
RAG            The *strategy*: retrieve context, inject into prompt.
               Without RAG, the LLM answers from training memory only.
               RAG makes the LLM answer from *your* documents.

Milvus         The *retrieval engine*: stores 1536-dimensional chunk
               vectors and finds the K nearest ones to a query vector
               in milliseconds. Without Milvus, retrieval would require
               scanning every chunk in a flat list — unusable at scale.

LangChain      The *integration layer*: standard interfaces for
               embeddings, vector stores, LLMs, and chains. LCEL lets
               you compose them with |. Without LangChain, every
               component combination needs custom glue code.

LangGraph      The *agent engine*: turns the static RAG chain into a
               dynamic agent that evaluates its own output quality,
               retries with improved queries, and maintains state across
               multiple steps. Without LangGraph, agents are manual
               while loops — hard to audit, test, and extend.
```

Together:
- **Milvus** provides fast semantic search over your documents
- **RAG** defines how to use that search to ground LLM answers
- **LangChain** provides the composable building blocks for the pipeline
- **LangGraph** gives the pipeline agency — the ability to reason about
  its own quality and adapt

The result is a system that can answer questions from private documents, admit
uncertainty when evidence is insufficient, retry intelligently when retrieval
quality is poor, hold multi-turn conversations, and use external tools when
the document store does not have the answer — all with a traceable,
auditable decision trail at every step.
