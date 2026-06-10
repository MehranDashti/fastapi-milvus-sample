from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import settings
from logger import get_logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = get_logger(__name__)


# ─── Embeddings ───────────────────────────────────────────────────────────────
# Replaces your embedder.py
# Same OpenAI model, same 1536 dimensions — just LC's interface

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai.API_KEY,
        model=settings.openai.EMBEDDING_MODEL,
    )


# ─── Vector Store ─────────────────────────────────────────────────────────────
# Replaces your milvus_client.py + searcher.py
# LC's Milvus wrapper handles connection, collection, insert, and search
def get_vectorstore() -> Milvus:
    return Milvus(
        embedding_function=get_embeddings(),
        collection_name="lc_documents",    # ← separate collection for LC
        connection_args={
            "host": settings.milvus.HOST,
            "port": str(settings.milvus.PORT),
        },
        auto_id=True,
        # Don't specify text_field/vector_field — let LC use its defaults
        # LC uses: "text" for content, "vector" for embedding
    )

# ─── Text Splitter ────────────────────────────────────────────────────────────
# Replaces your chunk_text() in ingester.py
# RecursiveCharacterTextSplitter is smarter than pure token splitting:
# it tries to split on paragraph breaks first (\n\n),
# then sentences (\n), then words (" "), then characters ("")
# This means chunks are more likely to end at natural boundaries

def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk.SIZE,
        chunk_overlap=settings.chunk.OVERLAP,
        length_function=len,        # character-based (faster than token-based)
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ─── LLM ──────────────────────────────────────────────────────────────────────
# Replaces your llm.py OpenAI call

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai.API_KEY,
        model=settings.openai.LLM_MODEL,
        max_tokens=settings.openai.LLM_MAX_TOKENS,
        temperature=0.1,
    )


def lc_ingest(text: str, source: str) -> dict:
    splitter = get_splitter()

    chunks = splitter.split_text(text)
    logger.info(f"[LC Ingester] {len(chunks)} chunks from '{source}'")

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
    logger.info(f"[LC Ingester] Inserted {len(documents)} chunks")

    return {"source": source, "chunks": len(chunks), "inserted": len(documents)}

# ─── RAG Chain ────────────────────────────────────────────────────────────────
# This is where LangChain really shines — the full RAG pipeline
# expressed as a composable chain using the | pipe operator
# Like Unix pipes: input | step1 | step2 | step3 | output

def build_rag_chain():
    """
    Builds and returns a LangChain RAG chain.

    The chain:
      question
        → retriever (search Milvus, get top 4 chunks)
        → prompt (format context + question into a prompt)
        → llm (send to GPT)
        → output parser (extract text from response)

    The | operator is LangChain's LCEL (LangChain Expression Language)
    """
    vectorstore = get_vectorstore()
    llm = get_llm()

    # Retriever = vectorstore with search config
    # This is equivalent to your searcher.search()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},     # top 4 chunks
    )

    # Prompt template
    # {context} and {question} are filled in automatically by the chain
    prompt = ChatPromptTemplate.from_template("""
You are a precise question-answering assistant.
Answer based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

    # Helper to format retrieved documents into a string
    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[{i+1}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    # The full chain using LCEL pipe syntax:
    # RunnablePassthrough() passes the question through unchanged to the prompt
    # retriever fetches relevant docs and formats them as context
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()     # extracts .content string from ChatMessage
    )

    return chain, retriever


def lc_query(question: str) -> dict:
    """
    Run a question through the LC RAG chain.
    Returns answer + sources.
    """
    chain, retriever = build_rag_chain()

    # Get answer
    answer = chain.invoke(question)

    # Get sources separately (chain only returns the final string)
    docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(docs),
    }