from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config import settings
from app.core.llm import ask_with_score_filter
from app.core.searcher import search
from app.logger import get_logger

logger = get_logger(__name__)

SCORE_THRESHOLD = 0.50
MAX_ATTEMPTS = 2

_llm = ChatOpenAI(
    api_key=settings.openai.API_KEY,
    model=settings.openai.LLM_MODEL,
    temperature=0.1,
)


class AgentState(TypedDict):
    question: str
    rephrased_question: str | None
    chunks: list
    best_score: float
    answer: str | None
    attempts: int
    quality_ok: bool
    reasoning: list[str]


def retrieve_node(state: AgentState) -> dict:
    query = state.get("rephrased_question") or state["question"]
    attempts = state.get("attempts", 0) + 1

    logger.info(f"[Agent] Retrieve attempt {attempts} | query: '{query}'")

    chunks = search(query, top_k=5)
    best_score = max((c["score"] for c in chunks), default=0.0)

    logger.info(f"[Agent] Got {len(chunks)} chunks | best score: {best_score:.4f}")

    reasoning = state.get("reasoning", [])
    reasoning.append(
        f"Attempt {attempts}: retrieved {len(chunks)} chunks, best score={best_score:.4f}"
    )

    return {"chunks": chunks, "best_score": best_score, "attempts": attempts, "reasoning": reasoning}


def evaluate_node(state: AgentState) -> dict:
    best_score = state["best_score"]
    attempts = state["attempts"]
    quality_ok = best_score >= SCORE_THRESHOLD or attempts >= MAX_ATTEMPTS

    reasoning = state.get("reasoning", [])
    if quality_ok:
        reasoning.append(
            f"Quality check PASSED (score={best_score:.4f} >= {SCORE_THRESHOLD} "
            f"OR attempts={attempts} >= {MAX_ATTEMPTS}) → generating answer"
        )
    else:
        reasoning.append(
            f"Quality check FAILED (score={best_score:.4f} < {SCORE_THRESHOLD}) "
            "→ rephrasing question"
        )

    logger.info(f"[Agent] Evaluate: quality_ok={quality_ok}")
    return {"quality_ok": quality_ok, "reasoning": reasoning}


def rephrase_node(state: AgentState) -> dict:
    original = state["question"]
    logger.info(f"[Agent] Rephrasing question: '{original}'")

    messages = [
        SystemMessage(
            content=(
                "You are a search query optimizer. "
                "Rephrase the given question to improve document retrieval. "
                "Make it more specific and use alternative terminology. "
                "Return ONLY the rephrased question, nothing else."
            )
        ),
        HumanMessage(content=f"Original question: {original}\nRephrased question:"),
    ]

    response = _llm.invoke(messages)
    rephrased = response.content.strip()

    logger.info(f"[Agent] Rephrased to: '{rephrased}'")

    reasoning = state.get("reasoning", [])
    reasoning.append(f"Rephrased question: '{rephrased}'")

    return {"rephrased_question": rephrased, "reasoning": reasoning}


def generate_node(state: AgentState) -> dict:
    question = state["question"]
    chunks = state["chunks"]

    logger.info(f"[Agent] Generating answer from {len(chunks)} chunks")

    result = ask_with_score_filter(query=question, chunks=chunks, min_score=0.35)

    reasoning = state.get("reasoning", [])
    reasoning.append(f"Generated answer using {len(chunks)} chunks")

    return {"answer": result["answer"], "reasoning": reasoning}


def route_after_evaluate(state: AgentState) -> str:
    if state["quality_ok"]:
        return "generate"
    return "rephrase"


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("rephrase", rephrase_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("rephrase", "retrieve")
    graph.add_edge("generate", END)

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"generate": "generate", "rephrase": "rephrase"},
    )

    return graph.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
        logger.info("[Agent] LangGraph agent compiled and ready.")
    return _agent


def agent_query(question: str) -> dict:
    agent = get_agent()

    initial_state: AgentState = {
        "question": question,
        "rephrased_question": None,
        "chunks": [],
        "best_score": 0.0,
        "answer": None,
        "attempts": 0,
        "quality_ok": False,
        "reasoning": [],
    }

    final_state = agent.invoke(initial_state)
    sources = list({c["source"] for c in final_state["chunks"]})

    return {
        "answer": final_state["answer"],
        "sources": sources,
        "attempts": final_state["attempts"],
        "best_score": round(final_state["best_score"], 4),
        "reasoning": final_state["reasoning"],
        "chunks_used": len(final_state["chunks"]),
    }
