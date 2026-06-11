import datetime
import math

from ddgs import DDGS

from app.core.searcher import search as milvus_search_raw
from app.logger import get_logger

logger = get_logger(__name__)


def tool_milvus_search(query: str, top_k: int = 5) -> str:
    logger.info(f"[Tool] milvus_search: '{query}'")
    chunks = milvus_search_raw(query, top_k=top_k)

    if not chunks:
        return "No relevant documents found in the database."

    results = []
    for i, chunk in enumerate(chunks):
        if chunk["score"] >= 0.40:
            results.append(
                f"[{i + 1}] Source: {chunk['source']} (score: {chunk['score']:.2f})\n"
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
            f"Web search failed ({e!s}). "
            "Please use your training knowledge to answer, "
            "and note that the information may not be current."
        )


def tool_calculator(expression: str) -> str:
    logger.info(f"[Tool] calculator: '{expression}'")
    try:
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
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e!s}"


def tool_get_current_date() -> str:
    logger.info("[Tool] get_current_date called")
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


TOOL_REGISTRY = {
    "milvus_search": tool_milvus_search,
    "web_search": tool_web_search,
    "calculator": tool_calculator,
    "get_current_date": tool_get_current_date,
}

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
        return f"Tool '{name}' failed: {e!s}"
