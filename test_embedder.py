from embedder import embed_text, embed_batch, get_embedding_dimension

# Test 1: single embed
print("--- Single Embed ---")
vector = embed_text("How do I reset my password?")
print("Type       :", type(vector))
print("Dimension  :", len(vector))
print("First 5    :", vector[:5])
print("All floats :", all(isinstance(v, float) for v in vector))

# Test 2: batch embed
print("\n--- Batch Embed ---")
texts = [
    "FastAPI is a modern Python web framework.",
    "Milvus is a vector database for AI applications.",
    "OpenAI provides embedding and LLM APIs.",
]
vectors = embed_batch(texts)
print("Batch size     :", len(vectors))
print("Each dimension :", len(vectors[0]))

# Test 3: dimension check
print("\n--- Dimension Check ---")
print("Config dimension :", get_embedding_dimension())
print("Actual dimension :", len(vector))
print("Match            :", get_embedding_dimension() == len(vector))