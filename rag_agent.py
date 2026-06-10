from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from searcher import search
from llm import ask_with_score_filter
from config import settings
from logger import get_logger

logger = get_logger(__name__)

# ─── State ────────────────────────────────────────────────────────────────────
# The state object travels through every node in the graph.
# Each node reads from it and returns a partial update.
# Think of it like a request object that accumulates data

class AgentState(TypedDict):
    question: str                    # original user question — never changes
    rephrased_question: str | None   # GPT-rephrased version for better retrieval
    chunks: list                     # retrieved chunks from Milvus
    best_score: float                # highest similarity score in chunks
    answer: str | None               # final generated answer
    attempts: int                    # how many retrieval attempts so far
    quality_ok: bool                 # did retrieval meet quality threshold
    reasoning: list[str]             # log of what the agent did (for debugging)


# ─── Config ───────────────────────────────────────────────────────────────────

SCORE_THRESHOLD = 0.50   # minimum acceptable top chunk score
MAX_ATTEMPTS = 2         # max retrieval retries before giving up

_llm = ChatOpenAI(
    api_key=settings.openai.API_KEY,
    model=settings.openai.LLM_MODEL,
    temperature=0.1,
)


# ─── Nodes ────────────────────────────────────────────────────────────────────
# Each node is a plain Python function.
# It receives the full state and returns a PARTIAL update (dict).
# LangGraph merges the returned dict into the state automatically.

def retrieve_node(state: AgentState) -> dict:
    """
    Node 1: Retrieve chunks from Milvus.
    Uses rephrased_question if available, otherwise original question.
    """
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

    return {
        "chunks": chunks,
        "best_score": best_score,
        "attempts": attempts,
        "reasoning": reasoning,
    }


def evaluate_node(state: AgentState) -> dict:
    """
    Node 2: Evaluate retrieval quality.
    Decides if chunks are good enough to generate an answer.

    This is the brain of the agent — makes the branching decision.
    """
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
            f"→ rephrasing question"
        )

    logger.info(f"[Agent] Evaluate: quality_ok={quality_ok}")

    return {"quality_ok": quality_ok, "reasoning": reasoning}


def rephrase_node(state: AgentState) -> dict:
    """
    Node 3: Ask GPT to rephrase the question for better retrieval.
    Only reached when evaluate_node decides quality is not good enough.

    Example:
        "Who made FastAPI?" → "What is the creator and origin of FastAPI framework?"
    """
    original = state["question"]
    logger.info(f"[Agent] Rephrasing question: '{original}'")

    messages = [
        SystemMessage(content=(
            "You are a search query optimizer. "
            "Rephrase the given question to improve document retrieval. "
            "Make it more specific and use alternative terminology. "
            "Return ONLY the rephrased question, nothing else."
        )),
        HumanMessage(content=f"Original question: {original}\nRephrased question:"),
    ]

    response = _llm.invoke(messages)
    rephrased = response.content.strip()

    logger.info(f"[Agent] Rephrased to: '{rephrased}'")

    reasoning = state.get("reasoning", [])
    reasoning.append(f"Rephrased question: '{rephrased}'")

    return {"rephrased_question": rephrased, "reasoning": reasoning}


def generate_node(state: AgentState) -> dict:
    """
    Node 4: Generate the final answer using retrieved chunks.
    Always the last node before END.
    """
    question = state["question"]
    chunks = state["chunks"]

    logger.info(f"[Agent] Generating answer from {len(chunks)} chunks")

    result = ask_with_score_filter(
        query=question,
        chunks=chunks,
        min_score=0.35,    # lower threshold here since agent already evaluated quality
    )

    reasoning = state.get("reasoning", [])
    reasoning.append(f"Generated answer using {len(chunks)} chunks")

    return {"answer": result["answer"], "reasoning": reasoning}


# ─── Routing ──────────────────────────────────────────────────────────────────
# Conditional edge function — called after evaluate_node.
# Returns the NAME of the next node to execute.
# This is what makes LangGraph different from a simple chain —
# the graph branches based on state.

def route_after_evaluate(state: AgentState) -> str:
    """
    Decision point: after evaluating chunk quality, where do we go?

    Returns node name as string — LangGraph uses this to route.
    """
    if state["quality_ok"]:
        return "generate"   # good enough → generate answer
    return "rephrase"        # not good enough → rephrase and retry


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_agent():
    """
    Assembles the LangGraph agent.
    Call once at startup, reuse the compiled graph for all requests.
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("rephrase", rephrase_node)
    graph.add_node("generate", generate_node)

    # Entry point — where the graph starts
    graph.set_entry_point("retrieve")

    # Fixed edges — always go from A to B
    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("rephrase", "retrieve")    # ← this creates the retry LOOP
    graph.add_edge("generate", END)

    # Conditional edge — evaluate decides: rephrase or generate?
    graph.add_conditional_edges(
        "evaluate",              # from this node
        route_after_evaluate,    # call this function to decide
        {
            "generate": "generate",   # if returns "generate" → go to generate
            "rephrase": "rephrase",   # if returns "rephrase" → go to rephrase
        }
    )

    return graph.compile()


# Singleton — compiled once, reused for all requests
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
        logger.info("[Agent] LangGraph agent compiled and ready.")
    return _agent


# ─── Public Interface ─────────────────────────────────────────────────────────

def agent_query(question: str) -> dict:
    """
    Run a question through the agentic RAG pipeline.
    Returns answer, sources, attempts made, and full reasoning trace.
    """
    agent = get_agent()

    # Initial state — only question is set, everything else starts empty/default
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