from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from logger import get_logger

logger = get_logger(__name__)


def build_messages_with_history(
    question: str,
    history: list[dict],
    context: str,
) -> list:
    """
    Build the full message list for GPT including conversation history.

    Args:
        question: current user question
        history:  list of previous turns [{"role": "user"|"assistant", "content": "..."}]
        context:  retrieved chunks formatted as a string

    Returns:
        list of LangChain message objects ready for ChatOpenAI.invoke()

    Example history:
        [
            {"role": "user",      "content": "What is FastAPI?"},
            {"role": "assistant", "content": "FastAPI is a modern..."},
        ]
    """
    messages = []

    # System message — sets behavior for the entire conversation
    messages.append(SystemMessage(content="""You are a precise question-answering assistant.
Answer based ONLY on the provided context and conversation history.
If the context doesn't contain enough information, say so clearly.
When the user uses pronouns like "it", "they", "this" — use conversation history to resolve them.
Be concise and cite sources when relevant."""))

    # Inject retrieved context as the first user message
    # This gives GPT the documents to answer from
    messages.append(HumanMessage(content=f"""Here is the relevant context from our documents:

{context}

Use this context to answer my questions in this conversation."""))

    # Acknowledge context (required to maintain alternating user/assistant pattern)
    messages.append(AIMessage(content="Understood. I'll answer your questions based on the provided context."))

    # Add conversation history — previous turns
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    # Add current question
    messages.append(HumanMessage(content=question))

    return messages


def truncate_history(history: list[dict], max_turns: int = 10) -> list[dict]:
    """
    Keep only the last N turns to avoid exceeding GPT context limits.
    Each turn = one user message + one assistant message = 2 items.

    max_turns=10 means keep last 10 pairs = 20 messages max.
    """
    max_messages = max_turns * 2
    if len(history) > max_messages:
        logger.info(f"[Memory] Truncating history from {len(history)} to {max_messages} messages")
        return history[-max_messages:]
    return history