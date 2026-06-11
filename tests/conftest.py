import pytest

from app.config import Settings
from app.core.milvus_client import build_index, build_schema, get_client, reset_client

TEST_COLLECTION = "test_documents"


@pytest.fixture(scope="session")
def settings():
    return Settings()


@pytest.fixture(scope="session")
def milvus_client(settings):
    reset_client()
    client = get_client()

    if client.has_collection(TEST_COLLECTION):
        client.drop_collection(TEST_COLLECTION)

    schema = build_schema(client)
    index_params = build_index(client)
    client.create_collection(
        collection_name=TEST_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(TEST_COLLECTION)

    yield client

    client.drop_collection(TEST_COLLECTION)
    reset_client()


@pytest.fixture
def sample_text():
    return (
        "FastAPI is a modern, fast (high-performance) web framework for building APIs "
        "with Python based on standard Python type hints. It was created by Sebastián "
        "Ramírez and first released in 2018. FastAPI is built on top of Starlette for "
        "the web parts and Pydantic for the data parts."
    )
