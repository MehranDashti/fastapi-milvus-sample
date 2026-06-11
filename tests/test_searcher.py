import pytest

from app.core.ingester import ingest_text
from app.core.searcher import search


@pytest.mark.integration
def test_search_returns_list(milvus_client, sample_text):
    ingest_text(sample_text, source="searcher_test.txt")
    results = search("FastAPI web framework", top_k=3)
    assert isinstance(results, list)


@pytest.mark.integration
def test_search_result_has_expected_fields(milvus_client, sample_text):
    ingest_text(sample_text, source="searcher_fields_test.txt")
    results = search("FastAPI", top_k=1)
    if results:
        result = results[0]
        assert "content" in result
        assert "source" in result
        assert "chunk_index" in result
        assert "score" in result


@pytest.mark.integration
def test_search_score_is_between_zero_and_one(milvus_client, sample_text):
    ingest_text(sample_text, source="searcher_score_test.txt")
    results = search("FastAPI", top_k=3)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


@pytest.mark.integration
def test_search_with_source_filter(milvus_client, sample_text):
    source = "filter_source_test.txt"
    ingest_text(sample_text, source=source)
    results = search("FastAPI", top_k=5, source_filter=source)
    for r in results:
        assert r["source"] == source


@pytest.mark.integration
def test_search_top_k_respected(milvus_client, sample_text):
    ingest_text(sample_text, source="topk_test.txt")
    results = search("FastAPI", top_k=2)
    assert len(results) <= 2
