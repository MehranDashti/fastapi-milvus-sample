from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from searcher import search
from memory import build_messages_with_history, truncate_history
from llm import build_context
from config import settings
from logger import get_logger

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
    source_filter: str = None,
) -> dict:
    """
    Conversational RAG — answers using both retrieved context AND chat history.

    Args:
        question:      current user question
        history:       previous turns [{"role": "user"|"assistant", "content": "..."}]
        top_k:         chunks to retrieve
        min_score:     minimum relevance score
        source_filter: optional source restriction

    Returns:
        dict with answer, sources, chunks_used, and updated history
    """
    # Step 1: truncate history to avoid token overflow
    history = truncate_history(history, max_turns=10)

    # Step 2: build a search query
    # If history exists, enrich query with last user message for better retrieval
    # Example: "Who created it?" + history → search for "Who created FastAPI?"
    search_query = question
    if history:
        last_user_msg = next(
            (h["content"] for h in reversed(history) if h["role"] == "user"),
            None
        )
        if last_user_msg:
            # Combine last question + current for better context in search
            search_query = f"{last_user_msg} {question}"
            logger.info(f"[Conversation] Enriched search query: '{search_query}'")

    # Step 3: retrieve relevant chunks
    chunks = search(search_query, top_k=top_k, source_filter=source_filter)
    relevant = [c for c in chunks if c["score"] >= min_score]

    if not relevant:
        # Fall back to raw question if enriched query found nothing
        chunks = search(question, top_k=top_k, source_filter=source_filter)
        relevant = [c for c in chunks if c["score"] >= min_score]

    # Step 4: format context from retrieved chunks
    context = build_context(relevant) if relevant else "No relevant documents found."

    # Step 5: build full message list with history
    messages = build_messages_with_history(
        question=question,
        history=history,
        context=context,
    )

    # Step 6: call GPT with full conversation
    logger.info(f"[Conversation] Sending {len(messages)} messages to GPT")
    response = _llm.invoke(messages)
    answer = _parser.invoke(response)

    # Step 7: update history with this turn
    updated_history = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    sources = list({c["source"] for c in relevant})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(relevant),
        "history": updated_history,
    }