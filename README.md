# RAG Service

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, Milvus, and OpenAI. Supports standard RAG, LangChain pipelines, LangGraph agentic RAG, conversational RAG with history, and tool-use agents.

## Learn the Stack

New to RAG, Milvus, LangChain, or LangGraph? The `doc/` folder contains deep-dive guides for each technology, written in a step-by-step style using the actual code in this project as examples.

| # | Document | What you will learn |
|---|---|---|
| 1 | [`doc/01-rag.md`](doc/01-rag.md) | What RAG is, the three phases (chunk → embed → store → retrieve → generate), score filtering, conversational RAG, agentic retry loops |
| 2 | [`doc/02-milvus.md`](doc/02-milvus.md) | Vector databases, HNSW indexing, cosine similarity, CRUD operations, scalar filtering, multi-tenancy patterns |
| 3 | [`doc/03-langchain.md`](doc/03-langchain.md) | LangChain abstractions, LCEL pipe composition, embeddings, retrievers, chat models, streaming, when to use vs raw SDK |
| 4 | [`doc/04-langgraph.md`](doc/04-langgraph.md) | Stateful agent graphs, nodes, conditional edges, retry loops, checkpointing, human-in-the-loop, streaming |
| 5 | [`doc/05-putting-it-together.md`](doc/05-putting-it-together.md) | Full architecture, end-to-end request traces for every endpoint, component dependency map, extension points, production checklist |

**Recommended reading order:** 01 → 02 → 03 → 04 → 05.
Each document ends with a pointer to the next one.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for linting and tests
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY
```

### 4. Start Milvus with Docker Compose

```bash
# Download the Milvus standalone docker-compose file
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

Or use the official docker-compose:

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.9/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

## Running

```bash
make run
# or: uvicorn app.main:app --reload --host 0.0.0.0 --port 9090
```

## Linting

```bash
make lint      # check for issues
make format    # auto-format code
make fix       # auto-fix fixable lint issues
```

## Testing

```bash
make test
# or: pytest tests/ -v

# Skip integration tests (no Milvus/OpenAI needed):
pytest tests/ -v -m "not integration"
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Collection stats (row count, fields) |
| `POST` | `/ingest/text` | Ingest raw text into vector store |
| `POST` | `/ingest/file` | Upload and ingest a `.txt` file |
| `POST` | `/ingest/pdf` | Upload and ingest a `.pdf` file |
| `DELETE` | `/source/{source_name}` | Delete all chunks for a source |
| `POST` | `/query` | Semantic Q&A (retrieve + generate) |
| `POST` | `/lc/ingest` | Ingest text via LangChain pipeline |
| `POST` | `/lc/query` | Query via LangChain RAG chain |
| `POST` | `/agent/query` | Agentic RAG with auto-rephrasing |
| `POST` | `/agent/tool` | Tool-use agent (search + web + calculator + date) |
| `POST` | `/chat` | Conversational RAG with history |

### Example curl commands

```bash
# Health check
curl http://localhost:9090/health

# Ingest text
curl -X POST http://localhost:9090/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern Python web framework.", "source": "intro.txt"}'

# Upload file
curl -X POST http://localhost:9090/ingest/file \
  -F "file=@sample.txt"

# Upload PDF
curl -X POST http://localhost:9090/ingest/pdf \
  -F "file=@document.pdf"

# Query
curl -X POST http://localhost:9090/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?", "top_k": 5, "min_score": 0.45}'

# LangChain query
curl -X POST http://localhost:9090/lc/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'

# Agentic RAG
curl -X POST http://localhost:9090/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'

# Tool agent
curl -X POST http://localhost:9090/agent/tool \
  -H "Content-Type: application/json" \
  -d '{"question": "What is todays date and what is 144 * 25?"}'

# Conversational RAG
curl -X POST http://localhost:9090/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?", "history": []}'

# Delete source
curl -X DELETE http://localhost:9090/source/intro.txt
```

## Project Structure

```
app/
├── main.py              # FastAPI app, lifespan, router registration
├── config.py            # Settings from .env
├── logger.py            # Structured logging
├── api/
│   ├── routes/          # Route handlers (ingest, query, agent, chat)
│   └── models/          # Pydantic request/response models
├── core/                # Core RAG components (milvus, embedder, ingester, searcher, llm)
├── agents/              # LangGraph agent + tool-use agent + tool registry
├── langchain/           # LangChain pipeline components
└── memory/              # Conversation history + multi-turn chat
tests/                   # pytest test suite
```
