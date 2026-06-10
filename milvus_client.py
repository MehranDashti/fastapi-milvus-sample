from pymilvus import MilvusClient, DataType
from config import settings


def get_client() -> MilvusClient:
    """
    Create and return a Milvus client.
    Like Laravel's DB::connection() — single entry point.
    """
    client = MilvusClient(uri=settings.milvus.URI)
    return client


def build_schema(client: MilvusClient):
    """
    Define the full collection schema.
    Like a Laravel migration — defines all fields and their types.
    """
    schema = client.create_schema(
        auto_id=True,               # Milvus generates IDs (like AUTO_INCREMENT)
        enable_dynamic_field=True,  # allow extra fields not in schema (like MongoDB)
    )

    # Primary key — auto generated, don't insert this manually
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )

    # The vector field — MUST exist in every collection
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=settings.openai.EMBEDDING_DIMENSION,  # 1536 for text-embedding-3-small
    )

    # The original text chunk — what we show to GPT as context
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )

    # Where the chunk came from (filename, URL, etc.)
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=512,
    )

    # Position of this chunk in the original document (0, 1, 2, ...)
    schema.add_field(
        field_name="chunk_index",
        datatype=DataType.INT32,
    )

    # Unix timestamp — when this chunk was ingested
    schema.add_field(
        field_name="created_at",
        datatype=DataType.INT64,
    )

    return schema


def build_index(client: MilvusClient):
    """
    Define HNSW index on the embedding field.
    This is what makes vector search fast.
    Without an index, Milvus would scan every vector (like a full table scan).
    """
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",   # best for OpenAI embeddings
        params={
            "M": 16,            # connections per node — higher = better recall, more RAM
            "efConstruction": 200,  # build quality — higher = better index, slower build
        }
    )

    return index_params


def ensure_collection(client: MilvusClient) -> None:
    """
    Create collection if it doesn't exist, then load it into memory.
    Safe to call on every app startup — like 'php artisan migrate' (idempotent).
    """
    collection_name = settings.milvus.COLLECTION

    if not client.has_collection(collection_name):
        print(f"[Milvus] Collection '{collection_name}' not found. Creating...")

        schema = build_schema(client)
        index_params = build_index(client)

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"[Milvus] Collection '{collection_name}' created.")
    else:
        print(f"[Milvus] Collection '{collection_name}' already exists. Skipping creation.")

    # Always load into memory — idempotent, safe to call multiple times
    client.load_collection(collection_name)
    print(f"[Milvus] Collection '{collection_name}' loaded into memory. Ready.")


def get_collection_stats(client: MilvusClient) -> dict:
    """
    Returns basic info about the collection.
    Useful for debugging and health checks.
    """
    collection_name = settings.milvus.COLLECTION

    stats = client.get_collection_stats(collection_name)
    description = client.describe_collection(collection_name)

    return {
        "collection": collection_name,
        "row_count": stats.get("row_count", 0),
        "fields": [f["name"] for f in description.get("fields", [])],
    }