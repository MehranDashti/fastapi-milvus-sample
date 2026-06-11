# 03 — LangChain: Composable AI Pipelines

> **Goal of this document:** Understand LangChain's core abstractions, how
> LCEL (LangChain Expression Language) lets you compose components with the
> pipe operator, and how the LangChain version of the RAG pipeline in this
> project compares to the hand-rolled version.

---

## 1. What Problem LangChain Solves

Without LangChain, building a RAG pipeline requires writing integration code
for every combination of components:

- OpenAI embeddings + Milvus + GPT-4 answer
- Cohere embeddings + Pinecone + Claude answer
- Local Llama embeddings + ChromaDB + Mistral answer

Each combination needs custom glue code: embedding calls, vector store
queries, prompt formatting, LLM invocation, output parsing.

LangChain defines **standard interfaces** for each component type. An
`Embeddings` object always has `.embed_query()`. A `VectorStore` always has
`.as_retriever()`. A retriever always returns `list[Document]`. A chain
always has `.invoke()`.

Swap the implementation (OpenAI → Cohere, Milvus → Pinecone) without
changing the pipeline code.

---

## 2. Core Abstractions

### 2.1 Documents

The universal data container in LangChain. Everything — chunked text,
web pages, PDF pages, database rows — is converted to `Document` before
entering the pipeline.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="FastAPI is a modern Python web framework...",
    metadata={"source": "fastapi_docs.pdf", "chunk_index": 14}
)

print(doc.page_content)    # the text
print(doc.metadata)        # any key-value metadata
```

Documents are immutable value objects. They carry metadata through the
pipeline without the pipeline stages needing to know about it.

### 2.2 Text Splitters

Split a long string or `Document` into chunks. LangChain provides many
splitter strategies; `RecursiveCharacterTextSplitter` is the default
for unstructured text.

```python
# app/langchain/lc_components.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,          # max characters per chunk
    chunk_overlap=50,        # overlap between consecutive chunks
    length_function=len,     # character-based (faster than token-based)
    separators=["\n\n", "\n", ". ", " ", ""],
    # ^ try to split at paragraph → sentence → word → character
    # The first separator that fits is used, preserving natural boundaries
)

chunks: list[str] = splitter.split_text("Long document text here...")
docs:   list[Document] = splitter.split_documents([doc1, doc2])
```

**RecursiveCharacterTextSplitter vs token-based splitting (this project's
hand-rolled chunker):**

| | RecursiveCharacterTextSplitter | Token-based (tiktoken) |
|---|---|---|
| Split unit | Characters | Tokens |
| Natural boundaries | Tries paragraph/sentence first | Purely positional |
| Embedding alignment | May not align with model tokenization | Perfect alignment |
| Speed | Fast | Slightly slower (tokenize + decode) |

For most RAG use cases, both produce similar quality. Token-based is
strictly more aligned with the embedding model's actual input.

### 2.3 Embeddings

Standard interface for any embedding model. The embedding model transforms
text into vectors.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key="sk-...",
    model="text-embedding-3-small",   # 1536 dimensions
)

# Single text
vector: list[float] = embeddings.embed_query("How does FastAPI handle auth?")

# Batch (for indexing)
vectors: list[list[float]] = embeddings.embed_documents(["chunk 1", "chunk 2"])
```

The interface is identical for `CohereEmbeddings`, `HuggingFaceEmbeddings`,
`OllamaEmbeddings`, etc. The pipeline code does not change.

### 2.4 Vector Stores

Standard interface for storing and searching vectors. LangChain has
integrations for 40+ vector databases.

```python
from langchain_milvus import Milvus

vectorstore = Milvus(
    embedding_function=embeddings,    # used for both indexing and search
    collection_name="lc_documents",   # separate collection from the raw client
    connection_args={
        "host": "localhost",
        "port": "19530",
    },
    auto_id=True,
)

# Insert documents (embed + store in one call)
vectorstore.add_documents([doc1, doc2, doc3])

# Similarity search
docs: list[Document] = vectorstore.similarity_search(
    query="What is dependency injection?",
    k=4,
)

# With scores
docs_with_scores: list[tuple[Document, float]] = \
    vectorstore.similarity_search_with_score("...", k=4)
```

Note: `langchain_milvus` creates its own schema internally (fields named
`text` and `vector`). This is a separate collection from the one managed by
`app/core/milvus_client.py` — they coexist without conflict.

### 2.5 Retrievers

A retriever is a callable that takes a string query and returns
`list[Document]`. It wraps a vector store (or any other source) and adds
configuration for how to search.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",       # or "mmr" (max marginal relevance)
    search_kwargs={"k": 4},
)

# Invoke (the standard way in LCEL)
docs: list[Document] = retriever.invoke("What is FastAPI?")
```

`MMR` (Maximal Marginal Relevance) is an alternative to pure similarity
search. It balances relevance with diversity — avoids returning 4 chunks
from the same paragraph.

### 2.6 Chat Models (LLMs)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=1000,
    temperature=0.1,
)

from langchain_core.messages import HumanMessage, SystemMessage

response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is FastAPI?"),
])
print(response.content)   # "FastAPI is a modern web framework..."
print(type(response))     # AIMessage
```

LangChain wraps the OpenAI response in a `BaseMessage` object. The raw
string is in `.content`. This is why output parsers exist.

### 2.7 Output Parsers

Convert the `AIMessage` (or any LLM output) into a Python type.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

parser = StrOutputParser()
text: str = parser.invoke(response)   # extracts .content as plain string

# JSON parser extracts structured output
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

json_parser = JsonOutputParser(pydantic_object=Answer)
structured = json_parser.invoke(response)
```

### 2.8 Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a precise assistant. Answer ONLY from the context below.

Context:
{context}

Question: {question}

Answer:""")

# Fill in variables
messages = prompt.invoke({"context": "...", "question": "What is FastAPI?"})
# Returns: list[BaseMessage] ready for the LLM
```

---

## 3. LCEL — LangChain Expression Language

LCEL is the composition system that lets you connect components with the `|`
pipe operator. Every LCEL component is a `Runnable` with a standard `.invoke()`
interface.

```
chain = component_a | component_b | component_c

chain.invoke(input)
# equivalent to: component_c.invoke(component_b.invoke(component_a.invoke(input)))
```

This is the Unix pipe philosophy applied to AI pipelines. Each stage has one
responsibility. Stages are composable, swappable, and independently testable.

### The Full RAG Chain

```python
# app/langchain/lc_components.py
def build_rag_chain():
    vectorstore = get_vectorstore()
    llm         = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    prompt = ChatPromptTemplate.from_template("""
You are a precise question-answering assistant.
Answer based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[{i+1}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    chain = (
        {
            "context":  retriever | format_docs,  # question → docs → formatted string
            "question": RunnablePassthrough(),     # question passes through unchanged
        }
        | prompt           # dict → list[BaseMessage]
        | llm              # list[BaseMessage] → AIMessage
        | StrOutputParser() # AIMessage → str
    )

    return chain, retriever
```

**Data flow through the chain:**

```
Input: "What is FastAPI?"
  ↓
{ "context": retriever | format_docs, "question": RunnablePassthrough() }
  ↓
{
  "context":  "[1] (source: docs.pdf)\nFastAPI is...\n\n[2] ...",
  "question": "What is FastAPI?"
}
  ↓
prompt
  ↓
[SystemMessage(...), HumanMessage("Context:\n[1]...\n\nQuestion: What is FastAPI?\n\nAnswer:")]
  ↓
llm
  ↓
AIMessage(content="FastAPI is a modern Python web framework...")
  ↓
StrOutputParser()
  ↓
"FastAPI is a modern Python web framework..."
```

### RunnablePassthrough

`RunnablePassthrough()` is a no-op runnable — it passes its input to the
output unchanged. It is used when you need to route the same input to
multiple downstream components.

```python
# Both branches receive the same question string:
{
    "context":  retriever | format_docs,  # question → search → format
    "question": RunnablePassthrough(),    # question → question (unchanged)
}
```

### RunnableParallel

When you use `{...}` in a chain, LangChain wraps it in a `RunnableParallel`.
Both branches run concurrently (async) or sequentially (sync) depending on the
chain's execution context.

---

## 4. LCEL Batch and Streaming

Every LCEL chain supports batch processing and streaming out of the box:

```python
# Batch: process multiple questions in parallel
answers = chain.batch([
    "What is FastAPI?",
    "What is Milvus?",
    "What is RAG?",
])
# Returns list of answers, executed concurrently

# Stream: receive answer token by token
for token in chain.stream("What is FastAPI?"):
    print(token, end="", flush=True)

# Async batch (for async contexts)
answers = await chain.abatch(["question 1", "question 2"])
```

This is free — you do not need to write any async or threading code. The
runnable protocol handles it.

---

## 5. Ingestion with LangChain

```python
# app/langchain/lc_components.py
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
    vectorstore.add_documents(documents)   # embeds + stores in one call

    return {"source": source, "chunks": len(chunks), "inserted": len(documents)}
```

`add_documents` handles everything: calling the embedding model on each
`page_content`, building the insert payload, and writing to Milvus. The
equivalent hand-rolled code in `app/core/ingester.py` is ~50 lines;
LangChain reduces it to ~5.

---

## 6. Query with LangChain

```python
# app/langchain/lc_components.py
def lc_query(question: str) -> dict:
    chain, retriever = build_rag_chain()

    # Run the full chain — retrieval + generation in one call
    answer = chain.invoke(question)

    # Get retrieved docs separately to extract sources
    docs = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    return {
        "answer":     answer,
        "sources":    sources,
        "chunks_used": len(docs),
    }
```

One limitation of the pure chain approach: the chain only returns the
final string (after `StrOutputParser`). To get the intermediate retrieved
documents (for source attribution), you need a separate retriever call.
The `RunnableParallel` pattern can fix this:

```python
# Advanced: return both answer and source docs from one chain call
from langchain_core.runnables import RunnableParallel

chain_with_sources = RunnableParallel(
    answer=chain,
    docs=retriever,
)

result = chain_with_sources.invoke("What is FastAPI?")
print(result["answer"])  # "FastAPI is..."
print(result["docs"])    # [Document(...), ...]
```

---

## 7. LangChain Message Types

LangChain wraps all LLM messages in typed objects:

| Type | Role | Usage |
|---|---|---|
| `SystemMessage` | `system` | Sets model behavior for entire conversation |
| `HumanMessage` | `user` | User input |
| `AIMessage` | `assistant` | Model output (also used to inject history) |
| `ToolMessage` | `tool` | Tool call result (for function calling) |
| `FunctionMessage` | `function` | Legacy (pre-tool-call API) |

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# These are equivalent:
llm.invoke("What is FastAPI?")
llm.invoke([HumanMessage(content="What is FastAPI?")])

# Multi-turn conversation
llm.invoke([
    SystemMessage(content="Answer in bullet points."),
    HumanMessage(content="What is FastAPI?"),
    AIMessage(content="- FastAPI is a web framework\n- Built on Starlette..."),
    HumanMessage(content="Who created it?"),  # "it" resolves via history
])
```

This is exactly what the conversational RAG does — inject retrieved context
as a HumanMessage, acknowledge it as an AIMessage, then replay conversation
history before the current question.

---

## 8. When to Use LangChain vs Raw SDK

**Use LangChain when:**
- You want to swap embedding/LLM/vector store providers without rewriting logic
- You need streaming output
- You want to use LangChain ecosystem tools (LangSmith observability, Hub prompts)
- Building agentic systems with many connected steps (especially with LangGraph)
- You want batching, retries, and fallbacks via `with_retry()` and `with_fallbacks()`

**Use raw SDK when:**
- You need precise control over API parameters not exposed by LangChain
- You want minimal dependencies
- Performance is critical (LangChain adds abstraction overhead)
- The pipeline is simple and unlikely to change providers

In this project, both exist side-by-side — the `/query` and `/ingest` routes
use the raw SDK, `/lc/query` and `/lc/ingest` use LangChain. The LangChain
version is shorter but uses a separate Milvus collection.

---

## 9. Observability with LangSmith

LangChain integrates with LangSmith for tracing every chain invocation:

```python
# Set environment variables to enable tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=my-rag-service
```

After that, every `chain.invoke()` is automatically traced — you can see
the input/output of every node, latency, token counts, and errors in the
LangSmith UI. Essential for debugging production RAG quality issues.

---

## 10. Summary

LangChain provides a standard interface (`Runnable`) for every AI component:
embeddings, vector stores, retrievers, LLMs, and parsers. LCEL's `|` operator
chains them into pipelines where output flows automatically from one stage to
the next. The result is concise, readable, and swappable pipeline code.
Compared to the hand-rolled RAG pipeline, the LangChain version is shorter
but less transparent — you trade visibility into internals for composability
and ecosystem integration.

**Next:** `04-langgraph.md` — how LangGraph extends these composable components
into stateful graphs that can loop, branch, and retry, enabling true agentic
behavior.
