import time
from openai import OpenAI, RateLimitError, APIStatusError
from config import settings

_client = OpenAI(api_key=settings.openai.API_KEY)


def _with_retry(fn, retries: int = 3, backoff: float = 2.0):
    """
    Retry wrapper for OpenAI calls.
    On RateLimitError waits exponentially: 2s, 4s, 8s before giving up.

    backoff=2.0 means:
        attempt 1 fails → wait 2s
        attempt 2 fails → wait 4s
        attempt 3 fails → raise
    """
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except RateLimitError as e:
            last_error = e
            wait = backoff ** (attempt + 1)
            print(f"[Embedder] Rate limited. Retrying in {wait}s... (attempt {attempt+1}/{retries})")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code >= 500:  # server error — retry
                last_error = e
                wait = backoff ** (attempt + 1)
                print(f"[Embedder] OpenAI server error {e.status_code}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise  # client error (400, 401) — don't retry
    raise last_error


def embed_text(text: str) -> list[float]:
    text = text.strip().replace("\n", " ")
    response = _with_retry(
        lambda: _client.embeddings.create(
            model=settings.openai.EMBEDDING_MODEL,
            input=text,
        )
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    texts = [t.strip().replace("\n", " ") for t in texts]
    page_size = 100
    all_vectors = []

    for i in range(0, len(texts), page_size):
        page = texts[i: i + page_size]
        response = _with_retry(
            lambda: _client.embeddings.create(
                model=settings.openai.EMBEDDING_MODEL,
                input=page,
            )
        )
        all_vectors.extend([item.embedding for item in response.data])
        print(f"[Embedder] Embedded {min(i + page_size, len(texts))}/{len(texts)} texts")

    return all_vectors


def get_embedding_dimension() -> int:
    return settings.openai.EMBEDDING_DIMENSION