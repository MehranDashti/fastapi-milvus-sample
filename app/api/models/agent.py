from pydantic import BaseModel

from app.api.models.chat import ChatMessage


class AgentRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    attempts: int
    best_score: float
    reasoning: list[str]
    chunks_used: int
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
