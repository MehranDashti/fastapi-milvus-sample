import time
import tiktoken
from milvus_client import get_client, ensure_collection
from embedder import embed_batch
from config import settings


def get_tokenizer():
    """
    Returns tiktoken encoder for the embedding model.
    cl100k_base is the encoding used by text-embedding-3-small.
    """
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> list[str]:
    """
    Split a long text into overlapping chunks based on token count.

    Why overlapping? If an answer sits at the boundary of two chunks,
    overlap ensures it fully appears in at least one chunk.

    Example with SIZE=10, OVERLAP=3:
    tokens: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    Chunk 1: [1,2,3,4,5,6,7,8,9,10]
    Chunk 2: [8,9,10,11,12,13,14,15,16,17]  ← starts 3 tokens back
    Chunk 3: [15,16,17,18,19,20,21,22,23,24] ← starts 3 tokens back
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    chunk_size = settings.chunk.SIZE
    overlap = settings.chunk.OVERLAP
    step = chunk_size - overlap  # how far to advance each iteration

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += step

    return chunks

def ingest_text(text: str, source: str) -> dict:
    client = get_client()
    ensure_collection(client)

    # Auto-delete existing chunks for this source before re-ingesting
    # This makes ingest idempotent — safe to call multiple times on same file
    existing = client.query(
        collection_name=settings.milvus.COLLECTION,
        filter=f'source == "{source}"',
        output_fields=["id"],
    )
    if existing:
        client.delete(
            collection_name=settings.milvus.COLLECTION,
            filter=f'source == "{source}"',
        )
        print(f"[Ingester] Deleted {len(existing)} old chunks for '{source}'.")

    # Step 1: chunk
    print(f"[Ingester] Chunking '{source}'...")
    chunks = chunk_text(text)
    print(f"[Ingester] {len(chunks)} chunks created.")

    # Step 2: embed
    print(f"[Ingester] Embedding {len(chunks)} chunks...")
    vectors = embed_batch(chunks)

    # Step 3: build entities
    timestamp = int(time.time())
    entities = [
        {
            "embedding": vectors[i],
            "content": chunks[i],
            "source": source,
            "chunk_index": i,
            "created_at": timestamp,
        }
        for i in range(len(chunks))
    ]

    # Step 4: insert + flush
    print(f"[Ingester] Inserting {len(entities)} entities into Milvus...")
    result = client.insert(
        collection_name=settings.milvus.COLLECTION,
        data=entities,
    )
    client.flush(collection_name=settings.milvus.COLLECTION)

    print(f"[Ingester] Done. Inserted IDs count: {result['insert_count']}")

    return {
        "source": source,
        "chunks": len(chunks),
        "inserted": result["insert_count"],
    }

def ingest_file(filepath: str) -> dict:
    """
    Read a .txt file and ingest it.
    Easy to extend for PDF, DOCX, etc. later.
    """
    print(f"[Ingester] Reading file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    return ingest_text(text, source=filepath)


def delete_by_source(source: str) -> dict:
    """
    Delete all chunks from a specific source.
    Useful when re-ingesting an updated document.
    Like DELETE FROM documents WHERE source = ?
    """
    client = get_client()

    result = client.delete(
        collection_name=settings.milvus.COLLECTION,
        filter=f'source == "{source}"',
    )

    print(f"[Ingester] Deleted chunks from source: {source}")
    return {"source": source, "deleted": result["delete_count"]}