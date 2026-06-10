from config import settings

print("Milvus URI:", settings.milvus.URI)
print("Collection:", settings.milvus.COLLECTION)
print("OpenAI model:", settings.openai.EMBEDDING_MODEL)
print("Embedding dim:", settings.openai.EMBEDDING_DIMENSION)
print("Chunk size:", settings.chunk.SIZE)
print("Chunk overlap:", settings.chunk.OVERLAP)