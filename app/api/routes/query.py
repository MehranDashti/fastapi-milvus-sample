import time

from fastapi import APIRouter, HTTPException

from app.api.models.query import LCQueryRequest, LCQueryResponse, QueryRequest, QueryResponse
from app.core.llm import ask_with_score_filter
from app.core.searcher import search
from app.langchain.lc_components import lc_query

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_route(request: QueryRequest):
    """Ask a question — retrieves relevant chunks and generates a grounded answer."""
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = search(query=request.question, top_k=request.top_k, source_filter=request.source_filter)
    result = ask_with_score_filter(
        query=request.question,
        chunks=chunks,
        min_score=request.min_score,
    )

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=len([c for c in chunks if c["score"] >= request.min_score]),
        tokens=result["tokens"],
        duration_ms=int((time.time() - start) * 1000),
    )


@router.post("/lc/query", response_model=LCQueryResponse)
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
