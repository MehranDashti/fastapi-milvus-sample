from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    min_score: float = 0.45
    source_filter: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    tokens: dict | int
    duration_ms: int


class LCQueryRequest(BaseModel):
    question: str


class LCQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    duration_ms: int
