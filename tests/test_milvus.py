import pytest

from app.core.milvus_client import ensure_collection, get_client, get_collection_stats


@pytest.mark.integration
def test_get_client_returns_client(milvus_client):
    client = get_client()
    assert client is not None


@pytest.mark.integration
def test_ensure_collection_idempotent(milvus_client):
    client = get_client()
    ensure_collection(client)
    ensure_collection(client)


@pytest.mark.integration
def test_get_collection_stats_returns_dict(milvus_client):
    client = get_client()
    stats = get_collection_stats(client)
    assert "collection" in stats
    assert "row_count" in stats
    assert "fields" in stats
    assert isinstance(stats["fields"], list)


@pytest.mark.integration
def test_collection_has_expected_fields(milvus_client):
    client = get_client()
    stats = get_collection_stats(client)
    expected = {"id", "embedding", "content", "source", "chunk_index", "created_at"}
    actual = set(stats["fields"])
    assert expected.issubset(actual)
