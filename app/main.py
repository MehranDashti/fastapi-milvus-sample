import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.rag_agent import get_agent
from app.api.routes import agent, chat, ingest, query
from app.core.milvus_client import ensure_collection, get_client
from app.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Milvus...")
    client = get_client()
    ensure_collection(client)
    get_agent()
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


@app.get("/health")
def health():
    """Simple health check — confirms API is running."""
    return {"status": "ok", "timestamp": int(time.time())}


app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(agent.router)
app.include_router(chat.router)
