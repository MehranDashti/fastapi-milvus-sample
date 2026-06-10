from openai import OpenAI
from config import settings

_client = OpenAI(api_key=settings.openai.API_KEY)


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block for GPT.
    Each chunk is numbered and labeled with its source.

    Example output:
        [1] (source: manual.pdf, chunk: 0)
        FastAPI is a modern web framework...

        [2] (source: manual.pdf, chunk: 1)
        FastAPI uses Python type hints...
    """
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[{i+1}] (source: {chunk['source']}, chunk: {chunk['chunk_index']})\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


def ask(query: str, chunks: list[dict]) -> dict:
    """
    Send retrieved chunks + user question to GPT and get a grounded answer.

    Args:
        query:  the user's original question
        chunks: list of dicts from searcher.search() — each has content, source, score

    Returns:
        dict with: answer, sources used, token usage
    """
    if not chunks:
        return {
            "answer": "I could not find any relevant information to answer your question.",
            "sources": [],
            "tokens": 0,
        }

    context = build_context(chunks)

    # The system prompt defines GPT's behavior:
    # - answer only from the provided context
    # - cite which chunk the answer came from
    # - admit when the context doesn't contain the answer
    system_prompt = """You are a precise question-answering assistant.
You are given a set of context chunks retrieved from a document database.
Your job is to answer the user's question based ONLY on the provided context.

Rules:
- Answer based strictly on the context. Do not use outside knowledge.
- If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this."
- Be concise and direct.
- When relevant, mention which source the information came from.
"""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = _client.chat.completions.create(
        model=settings.openai.LLM_MODEL,
        max_tokens=settings.openai.LLM_MAX_TOKENS,
        temperature=0.1,      # low temperature = factual, deterministic answers
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()

    # Collect unique sources used
    sources = list({chunk["source"] for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "tokens": {
            "prompt":     response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total":      response.usage.total_tokens,
        },
    }


def ask_with_score_filter(
    query: str,
    chunks: list[dict],
    min_score: float = 0.45,
) -> dict:
    """
    Same as ask() but filters out low-relevance chunks before sending to GPT.
    Prevents GPT from hallucinating based on loosely related context.

    min_score=0.45 means: only use chunks with COSINE similarity >= 0.45
    """
    filtered = [c for c in chunks if c["score"] >= min_score]

    if not filtered:
        return {
            "answer": "I could not find sufficiently relevant information to answer your question.",
            "sources": [],
            "tokens": 0,
        }

    print(f"[LLM] Using {len(filtered)}/{len(chunks)} chunks after score filter (min={min_score})")
    return ask(query, filtered)