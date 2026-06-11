# 01 — Retrieval-Augmented Generation (RAG)

> **Goal of this document:** Understand what RAG is, why it exists, how every
> component works internally, and how the patterns scale from the simplest
> one-shot lookup to a full agentic retry loop.

---

## 1. The Problem RAG Solves

Large language models are frozen in time. When training ends, the model's
knowledge ends. More importantly, LLMs have no access to:

- Your private documents, contracts, or internal wikis
- Data that did not exist when training was performed
- Information that changes after the training cutoff

The naive fix is fine-tuning — retrain the model on your data. Fine-tuning is
expensive, slow to iterate, and teaches the model *facts*, not *reasoning* over
live data. It also does not scale: you cannot retrain every time a document
changes.

**RAG decouples knowledge from the model.** The model stays general-purpose and
cheap to run. Knowledge lives in a separate, searchable store that can be
updated instantly without touching the model at all.

---

## 2. What RAG Is — One Sentence

> RAG = find the most relevant chunks of text for a question, then give those
> chunks to the LLM as context so it can answer from them instead of from its
> training memory.

That is the entire idea. Everything else is engineering detail around making
that process fast, accurate, and scalable.

---

## 3. The Three Phases

```
┌──────────────────────────────────────────────────────────────┐
│                       RAG Pipeline                           │
│                                                              │
│  PHASE 1: INDEXING (offline, run once per document)          │
│  ─────────────────────────────────────────────────────────   │
│  Document → Chunk → Embed → Store in Vector DB               │
│                                                              │
│  PHASE 2: RETRIEVAL (online, per query)                      │
│  ─────────────────────────────────────────────────────────   │
│  Question → Embed → ANN Search → Top-K chunks                │
│                                                              │
│  PHASE 3: GENERATION (online, per query)                     │
│  ─────────────────────────────────────────────────────────   │
│  Top-K chunks + Question → LLM Prompt → Answer               │
└──────────────────────────────────────────────────────────────┘
```

### Phase 1: Indexing

Before any question can be answered, every document must be preprocessed and
stored. This is offline work — you run it once when a document is added or
updated.

**Step 1.1 — Chunking**

A document may be thousands of tokens long. LLM context windows are finite.
More importantly, long documents introduce noise: if the answer is on page 2,
sending pages 1–50 to the LLM dilutes the signal.

The solution is to split the document into overlapping chunks. Each chunk is
a self-contained passage small enough to be useful in isolation.

```python
# app/core/ingester.py
def chunk_text(text: str) -> list[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")  # same tokenizer as OpenAI
    tokens = tokenizer.encode(text)

    chunk_size = settings.chunk.SIZE      # e.g. 512 tokens
    overlap    = settings.chunk.OVERLAP   # e.g. 50 tokens
    step       = chunk_size - overlap     # advance 462 tokens each iteration

    chunks = []
    start  = 0
    while start < len(tokens):
        chunk_tokens = tokens[start : start + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += step
    return chunks
```

**Why overlap?**  
If a sentence that answers the question begins at token 510 and the first chunk
ends at token 512, that sentence is split across two chunks and neither chunk
contains a complete answer. Overlap of 50 tokens means the sentence appears
fully in at least one chunk.

**Chunk size trade-offs:**

| Chunk size | Pros | Cons |
|---|---|---|
| Small (128–256 tokens) | High precision, less noise per chunk | May lack enough context to form an answer |
| Medium (512–1024 tokens) | Good balance | Standard choice for most use cases |
| Large (2048+ tokens) | Full paragraphs preserved | High noise, expensive to embed |

**Step 1.2 — Embedding**

Each chunk is converted to a high-dimensional vector by an embedding model.
Semantically similar text produces vectors that are geometrically close. This
is what makes similarity search possible.

```python
# app/core/embedder.py
def embed_text(text: str) -> list[float]:
    text = text.strip().replace("\n", " ")
    response = _with_retry(
        lambda: _client.embeddings.create(
            model=settings.openai.EMBEDDING_MODEL,   # text-embedding-3-small
            input=text,
        )
    )
    return response.data[0].embedding  # list of 1536 floats
```

`text-embedding-3-small` produces 1536-dimensional vectors. The distance
between two vectors in this 1536-dimensional space directly measures semantic
similarity.

The embedding model is *not* the LLM. It is a smaller, specialized model whose
only job is to map text → vector. It does not generate text.

**Step 1.3 — Storage**

The chunk text, its vector, and its metadata (source file, chunk index,
timestamp) are stored together in a vector database.

```python
# app/core/ingester.py
entities = [
    {
        "embedding":   vectors[i],   # the 1536-float vector
        "content":     chunks[i],    # the raw text — sent to GPT later
        "source":      source,       # filename or URL
        "chunk_index": i,            # position in original document
        "created_at":  timestamp,    # for auditing and cache invalidation
    }
    for i in range(len(chunks))
]
client.insert(collection_name=settings.milvus.COLLECTION, data=entities)
```

---

### Phase 2: Retrieval

At query time, the user's question is embedded using the *same* embedding
model. The resulting vector is compared against every stored chunk vector using
Approximate Nearest Neighbor (ANN) search. The top-K closest chunks are
returned.

```
Question: "How does FastAPI handle dependency injection?"
    ↓
Embed question → [0.023, -0.114, 0.887, ...]  (1536 floats)
    ↓
ANN Search in Milvus
    ↓
Top 5 chunks by cosine similarity:
  [score=0.91] chunk from "fastapi_docs.pdf", chunk 14
  [score=0.87] chunk from "fastapi_docs.pdf", chunk 15
  [score=0.74] chunk from "python_frameworks.txt", chunk 3
  [score=0.61] chunk from "fastapi_docs.pdf", chunk 22
  [score=0.55] chunk from "webdev_intro.txt", chunk 8
```

**Cosine similarity** measures the angle between two vectors. A score of 1.0
means the vectors point in exactly the same direction (identical meaning).
A score of 0.0 means they are orthogonal (completely unrelated).

In practice:
- Score > 0.85 → highly relevant
- Score 0.6–0.85 → probably relevant
- Score < 0.45 → likely irrelevant noise

```python
# app/core/searcher.py
def search(query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict]:
    query_vector = embed_text(query)

    results = client.search(
        collection_name=settings.milvus.COLLECTION,
        data=[query_vector],
        limit=top_k,
        filter=f'source == "{source_filter}"' if source_filter else None,
        search_params={"ef": max(top_k * 2, 64)},  # HNSW quality parameter
        output_fields=["content", "source", "chunk_index", "created_at"],
    )

    return [
        {
            "content":     hit["entity"]["content"],
            "source":      hit["entity"]["source"],
            "chunk_index": hit["entity"]["chunk_index"],
            "score":       round(hit["distance"], 4),
        }
        for hit in results[0]
    ]
```

**Source filtering** is scalar pre-filtering — restrict the vector search to
only chunks from a specific document before the ANN search runs. This is
equivalent to `WHERE source = ?` in SQL combined with a semantic search.

---

### Phase 3: Generation

The retrieved chunks are formatted into a context block and injected into the
LLM prompt. The LLM is instructed to answer *only* from the provided context.

```python
# app/core/llm.py
def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[{i+1}] (source: {chunk['source']}, chunk: {chunk['chunk_index']})\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)

def ask(query: str, chunks: list[dict]) -> dict:
    context = build_context(chunks)

    system_prompt = """You are a precise question-answering assistant.
Answer based ONLY on the provided context. Do not use outside knowledge.
If the context does not contain the answer, say so explicitly."""

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = _client.chat.completions.create(
        model=settings.openai.LLM_MODEL,
        max_tokens=settings.openai.LLM_MAX_TOKENS,
        temperature=0.1,   # low temperature = deterministic, factual answers
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return {
        "answer":  response.choices[0].message.content.strip(),
        "sources": list({c["source"] for c in chunks}),
        "tokens":  {...},
    }
```

**Why temperature=0.1?**  
RAG is a factual retrieval task. High temperature introduces randomness, which
causes hallucinations. The model should extract and paraphrase what the context
says, not creatively riff on it.

---

## 4. Score Filtering — Preventing Garbage In

Raw ANN search always returns K results, even if none of them are actually
relevant. If the user asks "What is the speed of light?" and your database only
contains FastAPI documentation, Milvus will still return the 5 "least unrelated"
chunks. Sending those to GPT causes hallucination.

The fix is a minimum score threshold. Only send chunks above the threshold:

```python
# app/core/llm.py
def ask_with_score_filter(query: str, chunks: list[dict], min_score: float = 0.45) -> dict:
    filtered = [c for c in chunks if c["score"] >= min_score]

    if not filtered:
        return {
            "answer":  "I could not find sufficiently relevant information to answer your question.",
            "sources": [],
            "tokens":  0,
        }

    return ask(query, filtered)
```

This is a critical production pattern. Without it, RAG answers are unreliable
for out-of-domain questions.

---

## 5. Idempotent Ingestion

Re-ingesting the same document should not create duplicate chunks. Before
inserting new chunks, delete any existing chunks for that source:

```python
# app/core/ingester.py
existing = client.query(
    collection_name=settings.milvus.COLLECTION,
    filter=f'source == "{source}"',
    output_fields=["id"],
)
if existing:
    client.delete(
        collection_name=settings.milvus.COLLECTION,
        filter=f'source == "{source}"',
    )
```

This makes `ingest_text` safe to call multiple times on the same file —
like database migrations that are idempotent.

---

## 6. Conversational RAG

Standard RAG is stateless — each question is independent. Real users ask
follow-up questions that reference previous turns:

- "What is FastAPI?"
- "Who created *it*?" ← "it" refers to FastAPI
- "When was *that* released?" ← "that" refers to FastAPI

The model needs conversation history to resolve pronouns. The pattern is
client-side history: the server returns the updated history and the client
sends it back with the next question.

```
┌────────┐  question + history  ┌────────┐
│ Client │ ─────────────────→  │ Server │
│        │ ←─────────────────  │        │
│        │  answer + new history│        │
└────────┘                      └────────┘
```

The server is stateless — no session state, no user context stored server-side.
This is essential for horizontal scaling.

**Query enrichment:**  
When history exists, combine the last user message with the current question
before running the vector search. This gives better retrieval when the current
question is underspecified.

```python
# app/memory/conversation.py
search_query = question
if history:
    last_user_msg = next(
        (h["content"] for h in reversed(history) if h["role"] == "user"), None
    )
    if last_user_msg:
        search_query = f"{last_user_msg} {question}"
```

---

## 7. Agentic RAG

Standard RAG retrieves once and answers. Agentic RAG adds a quality
evaluation step and a retry loop:

```
Question
  ↓
Retrieve (attempt 1)
  ↓
Evaluate quality (best_score >= threshold?)
  ├─ YES → Generate answer
  └─ NO  → Rephrase question with LLM
              ↓
            Retrieve (attempt 2)
              ↓
            Evaluate quality
              ├─ YES → Generate answer
              └─ NO  → Generate anyway (max_attempts reached)
```

The key insight: if the initial retrieval scores are poor, the question is
probably phrased in a way that does not match the embedding space of the stored
documents. An LLM can rephrase it to use terminology that better aligns with
the indexed content.

This is implemented as a LangGraph state machine — see `doc/04-langgraph.md`.

---

## 8. Common RAG Failure Modes

| Failure | Cause | Fix |
|---|---|---|
| Hallucination | Low-score chunks sent to LLM | Add `min_score` filter |
| "Not found" for valid questions | Chunk size too small, answer split | Increase chunk size or overlap |
| Stale answers | Old chunks not deleted before re-ingest | Idempotent ingestion (delete-before-insert) |
| Wrong document in results | No source filter | Use `source_filter` parameter |
| Pronoun resolution fails | No history in conversational RAG | Client-side history pattern |
| Good retrieval, bad answer | Temperature too high | Set `temperature=0.1` |
| Slow startup | Embedding model loaded per request | Singleton `_client` pattern |

---

## 9. RAG vs Fine-tuning — When to Use Which

| Scenario | Use RAG | Use Fine-tuning |
|---|---|---|
| Private documents that change frequently | ✅ | ❌ expensive to retrain |
| Need cited, verifiable sources | ✅ | ❌ model can't cite |
| Domain-specific *style* or *tone* | ❌ | ✅ |
| Very large knowledge base | ✅ scales to millions of chunks | ❌ training cost |
| Real-time updated data | ✅ just re-ingest | ❌ |
| Specific output format (e.g. JSON) | Both | ✅ |

In production, RAG and fine-tuning are not mutually exclusive. Fine-tune the
LLM for tone and format, then use RAG to inject current knowledge.

---

## 10. Summary

RAG works by converting documents and questions into the same vector space,
finding the nearest document chunks to the question vector, and injecting those
chunks into the LLM context. The LLM does not need to memorize facts — it
reads them from the context at inference time. This makes the knowledge store
updatable without retraining, auditable (you can see which chunks were used),
and scalable to billions of documents.

**Next:** `02-milvus.md` — how the vector database stores, indexes, and searches
those 1536-dimensional vectors at production scale.
