from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai.API_KEY)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[{i + 1}] (source: {chunk['source']}, chunk: {chunk['chunk_index']})\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


def ask(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "I could not find any relevant information to answer your question.",
            "sources": [],
            "tokens": 0,
        }

    context = build_context(chunks)

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
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()
    sources = list({chunk["source"] for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
    }


def ask_with_score_filter(
    query: str,
    chunks: list[dict],
    min_score: float = 0.45,
) -> dict:
    filtered = [c for c in chunks if c["score"] >= min_score]

    if not filtered:
        return {
            "answer": "I could not find sufficiently relevant information to answer your question.",
            "sources": [],
            "tokens": 0,
        }

    print(f"[LLM] Using {len(filtered)}/{len(chunks)} chunks after score filter (min={min_score})")
    return ask(query, filtered)
