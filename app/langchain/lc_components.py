from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings
from app.logger import get_logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = get_logger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai.API_KEY,
        model=settings.openai.EMBEDDING_MODEL,
    )


def get_vectorstore() -> Milvus:
    return Milvus(
        embedding_function=get_embeddings(),
        collection_name="lc_documents",
        connection_args={
            "host": settings.milvus.HOST,
            "port": str(settings.milvus.PORT),
        },
        auto_id=True,
    )


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk.SIZE,
        chunk_overlap=settings.chunk.OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


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
        Document(page_content=chunk, metadata={"source": source, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
    logger.info(f"[LC Ingester] Inserted {len(documents)} chunks")

    return {"source": source, "chunks": len(chunks), "inserted": len(documents)}


def build_rag_chain():
    vectorstore = get_vectorstore()
    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    prompt = ChatPromptTemplate.from_template("""
You are a precise question-answering assistant.
Answer based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[{i + 1}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def lc_query(question: str) -> dict:
    chain, retriever = build_rag_chain()
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(docs),
    }
