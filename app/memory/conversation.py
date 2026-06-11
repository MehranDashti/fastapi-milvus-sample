from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import settings
from app.core.llm import build_context
from app.core.searcher import search
from app.logger import get_logger
from app.memory.memory import build_messages_with_history, truncate_history

logger = get_logger(__name__)

_llm = ChatOpenAI(
    api_key=settings.openai.API_KEY,
    model=settings.openai.LLM_MODEL,
    max_tokens=settings.openai.LLM_MAX_TOKENS,
    temperature=0.1,
)

_parser = StrOutputParser()


def chat(
    question: str,
    history: list[dict],
    top_k: int = 5,
    min_score: float = 0.40,
    source_filter: str | None = None,
) -> dict:
    history = truncate_history(history, max_turns=10)

    search_query = question
    if history:
        last_user_msg = next(
            (h["content"] for h in reversed(history) if h["role"] == "user"),
            None,
        )
        if last_user_msg:
            search_query = f"{last_user_msg} {question}"
            logger.info(f"[Conversation] Enriched search query: '{search_query}'")

    chunks = search(search_query, top_k=top_k, source_filter=source_filter)
    relevant = [c for c in chunks if c["score"] >= min_score]

    if not relevant:
        chunks = search(question, top_k=top_k, source_filter=source_filter)
        relevant = [c for c in chunks if c["score"] >= min_score]

    context = build_context(relevant) if relevant else "No relevant documents found."

    messages = build_messages_with_history(
        question=question,
        history=history,
        context=context,
    )

    logger.info(f"[Conversation] Sending {len(messages)} messages to GPT")
    response = _llm.invoke(messages)
    answer = _parser.invoke(response)

    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    sources = list({c["source"] for c in relevant})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(relevant),
        "history": updated_history,
    }
