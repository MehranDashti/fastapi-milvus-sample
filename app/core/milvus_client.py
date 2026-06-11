from pymilvus import DataType, MilvusClient

from app.config import settings

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=settings.milvus.URI)
        print(f"[Milvus] New connection → {settings.milvus.URI}")
    return _client


def reset_client() -> None:
    global _client
    _client = None


def build_schema(client: MilvusClient):
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=settings.openai.EMBEDDING_DIMENSION,
    )
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
    schema.add_field(field_name="created_at", datatype=DataType.INT64)
    return schema


def build_index(client: MilvusClient):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    return index_params


def ensure_collection(client: MilvusClient, collection_name: str | None = None) -> None:
    name = collection_name or settings.milvus.COLLECTION

    if not client.has_collection(name):
        print(f"[Milvus] Collection '{name}' not found. Creating...")
        schema = build_schema(client)
        index_params = build_index(client)
        client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        print(f"[Milvus] Collection '{name}' created.")
    else:
        print(f"[Milvus] Collection '{name}' already exists. Skipping creation.")

    client.load_collection(name)
    print(f"[Milvus] Collection '{name}' loaded into memory. Ready.")


def get_collection_stats(client: MilvusClient) -> dict:
    collection_name = settings.milvus.COLLECTION
    stats = client.get_collection_stats(collection_name)
    description = client.describe_collection(collection_name)
    return {
        "collection": collection_name,
        "row_count": stats.get("row_count", 0),
        "fields": [f["name"] for f in description.get("fields", [])],
    }
