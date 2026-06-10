import time
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import Request


from milvus_client import get_client, ensure_collection, get_collection_stats
from ingester import ingest_text, delete_by_source
from searcher import search
from llm import ask_with_score_filter
from logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Startup] Connecting to Milvus...")
    client = get_client()
    ensure_collection(client)
    logger.info("[Startup] Ready.")
    yield
    logger.info("[Shutdown] Bye.")

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