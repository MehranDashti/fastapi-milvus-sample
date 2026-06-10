from searcher import search
from llm import ask_with_score_filter

def rag_query(question: str):
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print('='*60)

    # Step 1: retrieve relevant chunks
    chunks = search(question, top_k=5)

    # Step 2: generate answer
    result = ask_with_score_filter(question, chunks, min_score=0.45)

    print(f"A: {result['answer']}")
    print(f"\nSources : {result['sources']}")
    print(f"Tokens  : {result['tokens']}")

# Test questions
rag_query("What is FastAPI and who created it?")
rag_query("What index types does Milvus support and which is best?")
rag_query("How does the RAG pipeline work step by step?")
rag_query("What is the capital of France?")  # should say "not in documents"