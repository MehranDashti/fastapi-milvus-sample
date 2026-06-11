import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.models.ingest import (
    DeleteResponse,
    IngestRequest,
    IngestResponse,
    LCIngestRequest,
    StatsResponse,
)
from app.core.ingester import delete_by_source, ingest_text
from app.core.milvus_client import get_client, get_collection_stats
from app.core.pdf_extractor import extract_text_from_pdf_bytes
from app.langchain.lc_components import lc_ingest

router = APIRouter()


@router.post("/ingest/text", response_model=IngestResponse)
def ingest_text_route(request: IngestRequest):
    """Ingest raw text into the vector store."""
    start = time.time()

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if not request.source.strip():
        raise HTTPException(status_code=400, detail="Source cannot be empty.")

    result = ingest_text(request.text, request.source)
    return IngestResponse(
        source=result["source"],
        chunks=result["chunks"],
        inserted=result["inserted"],
        duration_ms=int((time.time() - start) * 1000),
    )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file_route(file: UploadFile = File(...)):
    """Upload a .txt file and ingest it."""
    start = time.time()

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    content = await file.read()
    text = content.decode("utf-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    result = ingest_text(text, source=file.filename)
    return IngestResponse(
        source=result["source"],
        chunks=result["chunks"],
        inserted=result["inserted"],
        duration_ms=int((time.time() - start) * 1000),
    )


@router.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf_route(file: UploadFile = File(...)):
    """Upload a PDF file and ingest it into the vector store."""
    start = time.time()

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are supported on this endpoint.",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = extract_text_from_pdf_bytes(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    result = ingest_text(text, source=file.filename)
    return IngestResponse(
        source=result["source"],
        chunks=result["chunks"],
        inserted=result["inserted"],
        duration_ms=int((time.time() - start) * 1000),
    )


@router.post("/lc/ingest", response_model=IngestResponse)
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


@router.delete("/source/{source_name}", response_model=DeleteResponse)
def delete_source(source_name: str):
    """Delete all chunks belonging to a source."""
    result = delete_by_source(source_name)
    return DeleteResponse(source=result["source"], deleted=result["deleted"])


@router.get("/stats", response_model=StatsResponse)
def stats():
    """Returns collection info — row count, field names."""
    client = get_client()
    return get_collection_stats(client)
