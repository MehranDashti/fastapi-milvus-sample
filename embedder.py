from openai import OpenAI
from config import settings


# Single client instance — like Laravel's singleton binding
# Created once, reused across all calls
_client = OpenAI(api_key=settings.openai.API_KEY)


def embed_text(text: str) -> list[float]:
    """
    Embed a single string → returns a vector (list of 1536 floats).
    Used at query time: embed the user's question.

    Example:
        vector = embed_text("How do I reset my password?")
        # → [0.012, -0.543, 0.871, ...]  (1536 numbers)
    """
    text = text.strip().replace("\n", " ")  # clean whitespace

    response = _client.embeddings.create(
        model=settings.openai.EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple strings in ONE API call → returns list of vectors.
    Used at ingest time: embed all chunks of a document together.

    Example:
        vectors = embed_batch(["chunk one...", "chunk two...", "chunk three..."])
        # → [[0.012, ...], [0.543, ...], [0.871, ...]]

    OpenAI limit: max 2048 inputs per call.
    We handle large batches by splitting into pages of 100.
    """
    if not texts:
        return []

    # Clean all texts
    texts = [t.strip().replace("\n", " ") for t in texts]

    # Split into pages of 100 to stay well within OpenAI limits
    page_size = 100
    all_vectors = []

    for i in range(0, len(texts), page_size):
        page = texts[i : i + page_size]

        response = _client.embeddings.create(
            model=settings.openai.EMBEDDING_MODEL,
            input=page,
        )

        # response.data is ordered same as input — safe to extend directly
        page_vectors = [item.embedding for item in response.data]
        all_vectors.extend(page_vectors)

        print(f"[Embedder] Embedded {min(i + page_size, len(texts))}/{len(texts)} texts")

    return all_vectors


def get_embedding_dimension() -> int:
    """
    Returns the dimension of the embedding model.
    Useful for sanity checks — must match collection schema.
    """
    return settings.openai.EMBEDDING_DIMENSION