import pytest

from app.core.embedder import embed_batch, embed_text, get_embedding_dimension


@pytest.mark.integration
def test_embed_text_returns_vector():
    vector = embed_text("How do I reset my password?")
    assert isinstance(vector, list)
    assert len(vector) == get_embedding_dimension()


@pytest.mark.integration
def test_embed_text_all_floats():
    vector = embed_text("test sentence")
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.integration
def test_embed_batch_returns_correct_count():
    texts = [
        "FastAPI is a modern Python web framework.",
        "Milvus is a vector database for AI applications.",
        "OpenAI provides embedding and LLM APIs.",
    ]
    vectors = embed_batch(texts)
    assert len(vectors) == len(texts)


@pytest.mark.integration
def test_embed_batch_dimension_matches():
    texts = ["first", "second"]
    vectors = embed_batch(texts)
    dim = get_embedding_dimension()
    for vec in vectors:
        assert len(vec) == dim


@pytest.mark.integration
def test_embed_batch_empty_returns_empty():
    result = embed_batch([])
    assert result == []


def test_get_embedding_dimension():
    dim = get_embedding_dimension()
    assert dim == 1536
