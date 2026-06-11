from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.logger import get_logger

logger = get_logger(__name__)


def build_messages_with_history(
    question: str,
    history: list[dict],
    context: str,
) -> list:
    messages = []

    messages.append(
        SystemMessage(
            content="""You are a precise question-answering assistant.
Answer based ONLY on the provided context and conversation history.
If the context doesn't contain enough information, say so clearly.
When the user uses pronouns like "it", "they", "this" — use conversation history to resolve them.
Be concise and cite sources when relevant."""
        )
    )

    messages.append(
        HumanMessage(
            content=f"""Here is the relevant context from our documents:

{context}

Use this context to answer my questions in this conversation."""
        )
    )

    messages.append(
        AIMessage(content="Understood. I'll answer your questions based on the provided context.")
    )

    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=question))

    return messages


def truncate_history(history: list[dict], max_turns: int = 10) -> list[dict]:
    max_messages = max_turns * 2
    if len(history) > max_messages:
        logger.info(f"[Memory] Truncating history from {len(history)} to {max_messages} messages")
        return history[-max_messages:]
    return history
