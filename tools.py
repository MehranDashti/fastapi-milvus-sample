import math
import datetime
import json
from ddgs import DDGS
from searcher import search as milvus_search_raw
from logger import get_logger

logger = get_logger(__name__)


# ─── Tool Functions ───────────────────────────────────────────────────────────
# Each function is a real tool the agent can call.
# They take simple string/int arguments and return strings.
# GPT receives the return value as a string and reasons about it.

def tool_milvus_search(query: str, top_k: int = 5) -> str:
    """
    Search the internal document database (Milvus).
    Use this for questions about documents that have been ingested:
    FastAPI, Milvus, RAG, Python frameworks, etc.
    Returns the most relevant text chunks with their sources.
    """
    logger.info(f"[Tool] milvus_search: '{query}'")
    chunks = milvus_search_raw(query, top_k=top_k)

    if not chunks:
        return "No relevant documents found in the database."

    results = []
    for i, chunk in enumerate(chunks):
        if chunk["score"] >= 0.40:
            results.append(
                f"[{i+1}] Source: {chunk['source']} (score: {chunk['score']:.2f})\n"
                f"{chunk['content'][:500]}"
            )

    if not results:
        return "No sufficiently relevant documents found (all scores below threshold)."

    return "\n\n".join(results)

def tool_web_search(query: str, max_results: int = 4) -> str:
    logger.info(f"[Tool] web_search: '{query}'")
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                max_results=max_results,
                region="wt-wt",
                safesearch="off",
            ):
                results.append(
                    f"Title: {r['title']}\n"
                    f"URL: {r['href']}\n"
                    f"Snippet: {r['body']}"
                )

        if not results:
            return (
                "Web search returned no results. "
                "Please use your training knowledge to answer, "
                "and note that the information may not be current."
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error(f"[Tool] web_search failed: {e}")
        return (
            f"Web search failed ({str(e)}). "
            "Please use your training knowledge to answer, "
            "and note that the information may not be current."
        )

def tool_calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, abs, round, floor, ceil, log, sin, cos, tan.
    Example: "2847 * 391", "sqrt(144)", "log(100)", "round(3.14159, 2)"
    Returns the result as a string.
    """
    logger.info(f"[Tool] calculator: '{expression}'")
    try:
        # Safe evaluation — only allow math operations
        # Never use eval() on raw user input — this sandboxed version is safe
        allowed_names = {
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def tool_get_current_date() -> str:
    """
    Returns the current date and time.
    Use this when the question involves today's date, current time,
    or any time-relative calculations.
    """
    logger.info("[Tool] get_current_date called")
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# ─── Tool Registry ────────────────────────────────────────────────────────────
# Maps tool names to their functions.
# The agent calls tools by name — this registry executes them.

TOOL_REGISTRY = {
    "milvus_search": tool_milvus_search,
    "web_search": tool_web_search,
    "calculator": tool_calculator,
    "get_current_date": tool_get_current_date,
}


# ─── Tool Schemas ─────────────────────────────────────────────────────────────
# JSON schemas describing each tool to GPT.
# GPT reads these to understand what each tool does and what arguments it takes.
# This is what OpenAI calls "function calling" or "tool use".

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "milvus_search",
            "description": (
                "Search the internal document database. Use this for questions about "
                "FastAPI, Milvus, vector databases, RAG, Python frameworks, Redis, "
                "MongoDB, Docker, microservices, DDD, RabbitMQ, or any topic that "
                "might be covered in ingested documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return. Default is 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this for: "
                "latest software versions, recent news, current events, "
                "real-time data, or anything not likely in internal documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results. Default is 4.",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate mathematical expressions. Use for any calculations: "
                "arithmetic, square roots, logarithms, trigonometry, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate. Example: '2847 * 391'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time. Use when the question involves today's date or time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """
    Execute a tool by name with given arguments.
    Called by the agent when GPT requests a tool call.
    """
    if name not in TOOL_REGISTRY:
        return f"Unknown tool: {name}"

    tool_fn = TOOL_REGISTRY[name]
    logger.info(f"[Tool] Executing '{name}' with args: {args}")

    try:
        result = tool_fn(**args)
        logger.info(f"[Tool] '{name}' completed. Result length: {len(str(result))}")
        return str(result)
    except Exception as e:
        logger.error(f"[Tool] '{name}' failed: {e}")
        return f"Tool '{name}' failed: {str(e)}"