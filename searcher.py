from milvus_client import get_client, ensure_collection
from embedder import embed_text
from config import settings


def search(query: str, top_k: int = 5, source_filter: str = None) -> list[dict]:
    """
    Semantic search: find the most relevant chunks for a query.

    Steps:
        1. Embed the query string into a vector
        2. Search Milvus for the top_k closest vectors
        3. Return chunks with their metadata and similarity scores

    Args:
        query:         the user's question as plain text
        top_k:         how many chunks to return (default 5)
        source_filter: optional — restrict search to a specific source file

    Returns:
        list of dicts, each with: content, source, chunk_index, score
    """
    client = get_client()
    ensure_collection(client)

    # Step 1: embed the query
    print(f"[Searcher] Embedding query: '{query}'")
    query_vector = embed_text(query)

    # Step 2: build optional scalar filter
    # This is like adding WHERE source = ? to a SQL query
    filter_expr = None
    if source_filter:
        filter_expr = f'source == "{source_filter}"'

    # Step 3: search Milvus
    # search_params ef must be >= top_k (HNSW requirement)
    results = client.search(
        collection_name=settings.milvus.COLLECTION,
        data=[query_vector],            # list of query vectors (we send one)
        limit=top_k,                    # top K results
        filter=filter_expr,             # optional scalar pre-filter
        search_params={"ef": max(top_k * 2, 64)},  # HNSW search quality
        output_fields=[                 # which scalar fields to return
            "content",
            "source",
            "chunk_index",
            "created_at",
        ],
    )

    # Step 4: format results
    # results[0] because we sent one query vector
    # each hit has: id, distance (similarity score), entity (fields)
    hits = results[0]
    formatted = []

    for hit in hits:
        formatted.append({
            "content": hit["entity"]["content"],
            "source": hit["entity"]["source"],
            "chunk_index": hit["entity"]["chunk_index"],
            "score": round(hit["distance"], 4),  # COSINE: 1.0 = identical, 0.0 = unrelated
        })

    return formatted


def search_and_print(query: str, top_k: int = 3) -> list[dict]:
    """
    Helper for development — search and print results in a readable format.
    """
    results = search(query, top_k=top_k)

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    for i, result in enumerate(results):
        print(f"\n[Result {i+1}] Score: {result['score']} | Source: {result['source']} | Chunk: {result['chunk_index']}")
        print(f"{result['content'][:300]}...")

    return results