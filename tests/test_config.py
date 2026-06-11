from app.config import Settings


def test_milvus_defaults():
    s = Settings()
    assert s.milvus.HOST == "localhost"
    assert s.milvus.PORT == 19530
    assert s.milvus.URI == "http://localhost:19530"


def test_milvus_collection_set():
    s = Settings()
    assert isinstance(s.milvus.COLLECTION, str)
    assert len(s.milvus.COLLECTION) > 0


def test_openai_embedding_model():
    s = Settings()
    assert s.openai.EMBEDDING_MODEL == "text-embedding-3-small"


def test_openai_embedding_dimension():
    s = Settings()
    assert s.openai.EMBEDDING_DIMENSION == 1536


def test_chunk_config():
    s = Settings()
    assert s.chunk.SIZE > 0
    assert s.chunk.OVERLAP >= 0
    assert s.chunk.OVERLAP < s.chunk.SIZE
