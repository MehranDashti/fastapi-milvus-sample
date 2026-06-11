import pytest

from app.core.llm import ask, ask_with_score_filter, build_context


def test_build_context_empty():
    result = build_context([])
    assert result == ""


def test_build_context_single_chunk():
    chunks = [{"source": "doc.txt", "chunk_index": 0, "content": "Hello world"}]
    result = build_context(chunks)
    assert "doc.txt" in result
    assert "Hello world" in result
    assert "[1]" in result


def test_build_context_multiple_chunks():
    chunks = [
        {"source": "a.txt", "chunk_index": 0, "content": "First"},
        {"source": "b.txt", "chunk_index": 1, "content": "Second"},
    ]
    result = build_context(chunks)
    assert "[1]" in result
    assert "[2]" in result
    assert "First" in result
    assert "Second" in result


def test_ask_no_chunks():
    result = ask("What is FastAPI?", [])
    assert "could not find" in result["answer"].lower()
    assert result["sources"] == []
    assert result["tokens"] == 0


def test_ask_with_score_filter_all_below_threshold():
    chunks = [
        {"source": "doc.txt", "chunk_index": 0, "content": "Some text", "score": 0.1},
    ]
    result = ask_with_score_filter("question", chunks, min_score=0.5)
    assert "could not find" in result["answer"].lower()
    assert result["sources"] == []


@pytest.mark.integration
def test_ask_with_score_filter_passes_good_chunks():
    chunks = [
        {
            "source": "doc.txt",
            "chunk_index": 0,
            "content": "FastAPI is a modern Python web framework built by Sebastián Ramírez.",
            "score": 0.9,
        }
    ]
    result = ask_with_score_filter("What is FastAPI?", chunks, min_score=0.5)
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert isinstance(result["tokens"], dict)
