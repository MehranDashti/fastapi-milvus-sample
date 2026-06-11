import time

import tiktoken

from app.config import settings
from app.core.embedder import embed_batch
from app.core.milvus_client import ensure_collection, get_client
from app.core.pdf_extractor import extract_text_from_pdf
from app.logger import get_logger

logger = get_logger(__name__)


def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> list[str]:
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    chunk_size = settings.chunk.SIZE
    overlap = settings.chunk.OVERLAP
    step = chunk_size - overlap

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_str = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_str)
        start += step

    return chunks


def ingest_text(text: str, source: str) -> dict:
    client = get_client()
    ensure_collection(client)

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

    print(f"[Ingester] Chunking '{source}'...")
    chunks = chunk_text(text)
    print(f"[Ingester] {len(chunks)} chunks created.")

    print(f"[Ingester] Embedding {len(chunks)} chunks...")
    vectors = embed_batch(chunks)

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

    print(f"[Ingester] Inserting {len(entities)} entities into Milvus...")
    result = client.insert(
        collection_name=settings.milvus.COLLECTION,
        data=entities,
    )
    print(f"[Ingester] Done. Inserted IDs count: {result['insert_count']}")

    return {
        "source": source,
        "chunks": len(chunks),
        "inserted": result["insert_count"],
    }


def ingest_file(filepath: str) -> dict:
    print(f"[Ingester] Reading file: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    return ingest_text(text, source=filepath)


def delete_by_source(source: str) -> dict:
    client = get_client()
    result = client.delete(
        collection_name=settings.milvus.COLLECTION,
        filter=f'source == "{source}"',
    )
    print(f"[Ingester] Deleted chunks from source: {source}")
    return {"source": source, "deleted": result["delete_count"]}


def ingest_pdf(filepath: str) -> dict:
    logger.info(f"[Ingester] Reading PDF: {filepath}")
    text = extract_text_from_pdf(filepath)
    return ingest_text(text, source=filepath)
