from dotenv import load_dotenv
import os

load_dotenv()

class MilvusConfig:
    HOST: str = os.getenv("MILVUS_HOST", "localhost")
    PORT: int = int(os.getenv("MILVUS_PORT", 19530))
    COLLECTION: str = os.getenv("MILVUS_COLLECTION", "documents")
    URI: str = f"http://{HOST}:{PORT}"

class OpenAIConfig:
    API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 1536))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1000))

class ChunkConfig:
    SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
    OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

class Settings:
    milvus = MilvusConfig()
    openai = OpenAIConfig()
    chunk = ChunkConfig()

settings = Settings()