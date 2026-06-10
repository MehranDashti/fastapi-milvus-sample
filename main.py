import time
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import Request
from fastapi.responses import JSONResponse

from milvus_client import get_client, ensure_collection, get_collection_stats
from ingester import ingest_text, delete_by_source
from searcher import search
from llm import ask_with_score_filter
from logger import get_logger
from lc_components import lc_ingest, lc_query
from rag_agent import agent_query, get_agent
from pdf_extractor import extract_text_from_pdf_bytes
from conversation import chat
from tool_agent import run_tool_agent

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Milvus...")
    client = get_client()
    ensure_collection(client)
    get_agent()                # ← compile agent graph at startup
    logger.info("Service ready.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="RAG Service",
    description="Document ingestion and semantic Q&A powered by Milvus + OpenAI",
    version="1.0.0",
    lifespan=lifespan,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

class IngestTextRequest(BaseModel):
    text: str
    source: str                

class IngestResponse(BaseModel):
    source: str
    chunks: int
    inserted: int
    duration_ms: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5               
    min_score: float = 0.45      
    source_filter: str = None    

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    tokens: dict | int
    duration_ms: int

class DeleteResponse(BaseModel):
    source: str
    deleted: int

class StatsResponse(BaseModel):
    collection: str
    row_count: int
    fields: list[str]

class LCQueryRequest(BaseModel):
    question: str

class LCQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    duration_ms: int

class LCIngestRequest(BaseModel):
    text: str
    source: str

class AgentQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    attempts: int
    best_score: float
    reasoning: list[str]
    chunks_used: int
    duration_ms: int

class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []    # empty list = new conversation
    top_k: int = 5
    min_score: float = 0.40
    source_filter: str = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    history: list[ChatMessage]         # updated history — client stores this
    duration_ms: int

class ToolAgentRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    max_iterations: int = 5

class ToolResult(BaseModel):
    tool: str
    args: dict
    result: str

class ToolAgentResponse(BaseModel):
    question: str
    answer: str
    tools_used: list[str]
    tool_results: list[ToolResult]
    iterations: int
    duration_ms: int

@app.post("/chat", response_model=ChatResponse)
def chat_route(request: ChatRequest):
    """
    Conversational RAG with memory.

    The client sends conversation history with each request.
    The server returns updated history which the client stores and sends back.

    This keeps the server stateless while maintaining conversation context.
    """
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Convert Pydantic models to plain dicts for conversation.py
    history = [{"role": m.role, "content": m.content} for m in request.history]

    result = chat(
        question=request.question,
        history=history,
        top_k=request.top_k,
        min_score=request.min_score,
        source_filter=request.source_filter,
    )

    # Convert history back to ChatMessage objects for response
    updated_history = [
        ChatMessage(role=h["role"], content=h["content"])
        for h in result["history"]
    ]

    return ChatResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        history=updated_history,
        duration_ms=int((time.time() - start) * 1000),
    )

class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []    # empty list = new conversation
    top_k: int = 5
    min_score: float = 0.40
    source_filter: str = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    history: list[ChatMessage]         # updated history — client stores this
    duration_ms: int

@app.get("/health")
def health():
    """Simple health check — confirms API is running."""
    return {"status": "ok", "timestamp": int(time.time())}


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Returns collection info — row count, field names."""
    client = get_client()
    return get_collection_stats(client)


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text_route(request: IngestTextRequest):
    """
    Ingest raw text into the vector store.
    Automatically chunks, embeds, and stores.
    Re-ingesting the same source replaces old chunks.
    """
    start = time.time()

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if not request.source.strip():
        raise HTTPException(status_code=400, detail="Source cannot be empty.")

    result = ingest_text(request.text, request.source)
    duration = int((time.time() - start) * 1000)

    return IngestResponse(
        source=result["source"],
        chunks=result["chunks"],
        inserted=result["inserted"],
        duration_ms=duration,
    )

@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file_route(file: UploadFile = File(...)):
    """
    Upload a .txt file and ingest it.
    The filename is used as the source identifier.
    """
    start = time.time()

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    content = await file.read()
    text = content.decode("utf-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    result = ingest_text(text, source=file.filename)
    duration = int((time.time() - start) * 1000)

    return IngestResponse(
        source=result["source"],
        chunks=result["chunks"],
        inserted=result["inserted"],
        duration_ms=duration,
    )


@app.post("/query", response_model=QueryResponse)
def query_route(request: QueryRequest):
    """
    Ask a question. Retrieves relevant chunks from Milvus
    and generates a grounded answer via GPT.
    """
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1: retrieve
    chunks = search(
        query=request.question,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )

    # Step 2: generate
    result = ask_with_score_filter(
        query=request.question,
        chunks=chunks,
        min_score=request.min_score,
    )

    duration = int((time.time() - start) * 1000)

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=len([c for c in chunks if c["score"] >= request.min_score]),
        tokens=result["tokens"],
        duration_ms=duration,
    )


@app.delete("/source/{source_name}", response_model=DeleteResponse)
def delete_source(source_name: str):
    """
    Delete all chunks belonging to a source.
    Use when a document is outdated and needs to be removed.
    """
    result = delete_by_source(source_name)
    return DeleteResponse(
        source=result["source"],
        deleted=result["deleted"],
    )

@app.post("/lc/ingest", response_model=IngestResponse)
def lc_ingest_route(request: LCIngestRequest):
    """Ingest text using LangChain pipeline."""
    start = time.time()
    result = lc_ingest(request.text, request.source)
    return IngestResponse(
        source=result["source"],
        chunks=result["chunks"],
        inserted=result["inserted"],
        duration_ms=int((time.time() - start) * 1000),
    )


@app.post("/lc/query", response_model=LCQueryResponse)
def lc_query_route(request: LCQueryRequest):
    """Query using LangChain RAG chain."""
    start = time.time()
    result = lc_query(request.question)
    return LCQueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        duration_ms=int((time.time() - start) * 1000),
    )

@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query_route(request: QueryRequest):
    """
    Agentic RAG: automatically retries with rephrased question
    if initial retrieval quality is poor.
    """
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = agent_query(request.question)

    return AgentQueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        attempts=result["attempts"],
        best_score=result["best_score"],
        reasoning=result["reasoning"],
        chunks_used=result["chunks_used"],
        duration_ms=int((time.time() - start) * 1000),
    )
    
@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf_route(file: UploadFile = File(...)):
    """
    Upload a PDF file and ingest it into the vector store.
    Extracts text page by page, chunks it, embeds, and stores.

    Supports: text-based PDFs
    Does NOT support: scanned/image PDFs (no OCR)
    """
    start = time.time()

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are supported on this endpoint."
        )

    # Read file bytes
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Extract text from PDF bytes
    try:
        text = extract_text_from_pdf_bytes(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Ingest extracted text
    result = ingest_text(text, source=file.filename)
    duration = int((time.time() - start) * 1000)

    return IngestResponse(
        source=result["source"],
        chunks=result["chunks"],
        inserted=result["inserted"],
        duration_ms=duration,
    )

@app.post("/chat", response_model=ChatResponse)
def chat_route(request: ChatRequest):
    """
    Conversational RAG with memory.

    The client sends conversation history with each request.
    The server returns updated history which the client stores and sends back.

    This keeps the server stateless while maintaining conversation context.
    """
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Convert Pydantic models to plain dicts for conversation.py
    history = [{"role": m.role, "content": m.content} for m in request.history]

    result = chat(
        question=request.question,
        history=history,
        top_k=request.top_k,
        min_score=request.min_score,
        source_filter=request.source_filter,
    )

    # Convert history back to ChatMessage objects for response
    updated_history = [
        ChatMessage(role=h["role"], content=h["content"])
        for h in result["history"]
    ]

    return ChatResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        history=updated_history,
        duration_ms=int((time.time() - start) * 1000),
    )

@app.post("/agent/tool", response_model=ToolAgentResponse)
def tool_agent_route(request: ToolAgentRequest):
    """
    True agentic RAG with tool use.
    The agent decides which tools to call based on the question.
    Can combine: document search + web search + calculator + date.
    """
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    result = run_tool_agent(
        question=request.question,
        history=history,
        max_iterations=request.max_iterations,
    )

    return ToolAgentResponse(
        question=request.question,
        answer=result["answer"],
        tools_used=result["tools_used"],
        tool_results=[ToolResult(**tr) for tr in result["tool_results"]],
        iterations=result["iterations"],
        duration_ms=int((time.time() - start) * 1000),
    )