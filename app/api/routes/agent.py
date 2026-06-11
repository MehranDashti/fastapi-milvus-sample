import time

from fastapi import APIRouter, HTTPException

from app.agents.rag_agent import agent_query
from app.agents.tool_agent import run_tool_agent
from app.api.models.agent import (
    AgentRequest,
    AgentResponse,
    ToolAgentRequest,
    ToolAgentResponse,
    ToolResult,
)

router = APIRouter()


@router.post("/agent/query", response_model=AgentResponse)
def agent_query_route(request: AgentRequest):
    """Agentic RAG: automatically retries with rephrased question if retrieval quality is poor."""
    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = agent_query(request.question)

    return AgentResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        attempts=result["attempts"],
        best_score=result["best_score"],
        reasoning=result["reasoning"],
        chunks_used=result["chunks_used"],
        duration_ms=int((time.time() - start) * 1000),
    )


@router.post("/agent/tool", response_model=ToolAgentResponse)
def tool_agent_route(request: ToolAgentRequest):
    """True agentic RAG with tool use — can combine document search, web search, calculator, date."""
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
