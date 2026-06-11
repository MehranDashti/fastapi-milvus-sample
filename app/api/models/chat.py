from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    top_k: int = 5
    min_score: float = 0.40
    source_filter: str | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    history: list[ChatMessage]
    duration_ms: int
