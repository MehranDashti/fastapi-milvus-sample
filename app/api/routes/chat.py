import time

from fastapi import APIRouter, HTTPException

from app.api.models.chat import ChatMessage, ChatRequest, ChatResponse
from app.memory.conversation import chat

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat_route(request: ChatRequest):
    """Conversational RAG with memory — client sends history, server returns updated history."""
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    result = chat(
        question=request.question,
        history=history,
        top_k=request.top_k,
        min_score=request.min_score,
        source_filter=request.source_filter,
    )

    updated_history = [
        ChatMessage(role=h["role"], content=h["content"]) for h in result["history"]
    ]

    return ChatResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        history=updated_history,
        duration_ms=int((time.time() - start) * 1000),
    )
