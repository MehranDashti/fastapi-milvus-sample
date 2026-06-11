import pytest

from app.core.ingester import chunk_text, delete_by_source, ingest_text


def test_chunk_text_basic(sample_text):
    chunks = chunk_text(sample_text)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_text_single_chunk_for_short_text():
    short = "Hello world."
    chunks = chunk_text(short)
    assert len(chunks) == 1
    assert "Hello" in chunks[0]


def test_chunk_text_multiple_chunks_for_long_text():
    long_text = " ".join(["word"] * 1000)
    chunks = chunk_text(long_text)
    assert len(chunks) > 1


@pytest.mark.integration
def test_ingest_text_returns_correct_structure(milvus_client, sample_text):
    result = ingest_text(sample_text, source="test_source.txt")
    assert result["source"] == "test_source.txt"
    assert result["chunks"] >= 1
    assert result["inserted"] >= 1


@pytest.mark.integration
def test_ingest_text_idempotent(milvus_client, sample_text):
    result1 = ingest_text(sample_text, source="idempotent_test.txt")
    result2 = ingest_text(sample_text, source="idempotent_test.txt")
    assert result1["chunks"] == result2["chunks"]
    assert result2["inserted"] >= 1


@pytest.mark.integration
def test_delete_by_source(milvus_client, sample_text):
    ingest_text(sample_text, source="delete_me.txt")
    result = delete_by_source("delete_me.txt")
    assert result["source"] == "delete_me.txt"
    assert result["deleted"] >= 0
