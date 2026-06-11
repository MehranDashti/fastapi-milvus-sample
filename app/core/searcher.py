from app.config import settings
from app.core.embedder import embed_text
from app.core.milvus_client import ensure_collection, get_client


def search(query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict]:
    client = get_client()
    ensure_collection(client)

    print(f"[Searcher] Embedding query: '{query}'")
    query_vector = embed_text(query)

    filter_expr = None
    if source_filter:
        filter_expr = f'source == "{source_filter}"'

    results = client.search(
        collection_name=settings.milvus.COLLECTION,
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        search_params={"ef": max(top_k * 2, 64)},
        output_fields=["content", "source", "chunk_index", "created_at"],
    )

    hits = results[0]
    return [
        {
            "content": hit["entity"]["content"],
            "source": hit["entity"]["source"],
            "chunk_index": hit["entity"]["chunk_index"],
            "score": round(hit["distance"], 4),
        }
        for hit in hits
    ]


def search_and_print(query: str, top_k: int = 3) -> list[dict]:
    results = search(query, top_k=top_k)

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}")

    for i, result in enumerate(results):
        print(
            f"\n[Result {i + 1}] Score: {result['score']} | "
            f"Source: {result['source']} | Chunk: {result['chunk_index']}"
        )
        print(f"{result['content'][:300]}...")

    return results
