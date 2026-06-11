# 02 — Milvus: The Vector Database

> **Goal of this document:** Understand what a vector database is, how Milvus
> stores and indexes high-dimensional vectors, why HNSW makes search fast, and
> how to design collections, run searches, and manage data at scale.

---

## 1. What Is a Vector Database?

A relational database stores rows and finds them by exact match or range:
`WHERE price BETWEEN 10 AND 50`. This breaks down for similarity questions:
"find me products *similar to* this one."

A vector database stores dense numeric vectors and finds them by geometric
proximity. Given a query vector, it returns the K database vectors that are
closest in high-dimensional space. This is called **Approximate Nearest
Neighbor (ANN) search**.

The "approximate" is intentional. Exact nearest neighbor in 1536 dimensions
requires comparing the query against every stored vector — O(n) per query. At
millions of vectors, this is too slow. ANN indexes trade a small amount of
recall accuracy for a large speedup, returning results in milliseconds instead
of seconds.

---

## 2. Milvus Architecture

Milvus Standalone (the mode used in this project) is a single-process
deployment suitable for development and moderate production loads.

```
┌─────────────────────────────────────────────────┐
│                  Milvus Standalone               │
│                                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │  Proxy Layer │    │   Query Coordinator  │   │
│  │  (REST/gRPC) │    │   (routing, planning)│   │
│  └──────────────┘    └──────────────────────┘   │
│                                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │  Index Node  │    │     Data Node        │   │
│  │  (HNSW build)│    │  (insert, segment)   │   │
│  └──────────────┘    └──────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │           Object Storage (MinIO)         │   │
│  │         Segments, Index Files            │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │        etcd (metadata store)             │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

For high-scale deployments, each node type runs as a separate pod in
Kubernetes, but the API surface is identical.

---

## 3. Core Concepts

### 3.1 Collection

A collection is Milvus's equivalent of a database table. It has a fixed schema
that defines which fields each entity (row) has. Unlike a SQL table, every
collection must have exactly one vector field.

```python
# app/core/milvus_client.py
client.create_collection(
    collection_name="documents",
    schema=schema,
    index_params=index_params,
)
```

### 3.2 Schema

The schema defines the shape of every entity stored in the collection.

```python
def build_schema(client: MilvusClient):
    schema = client.create_schema(
        auto_id=True,             # Milvus generates primary keys (like SERIAL in Postgres)
        enable_dynamic_field=True # allow extra fields not declared in schema (like MongoDB)
    )

    # Primary key — INT64, auto-generated, never inserted manually
    schema.add_field(field_name="id",          datatype=DataType.INT64,         is_primary=True)

    # THE vector field — this is what makes it a vector database
    schema.add_field(field_name="embedding",   datatype=DataType.FLOAT_VECTOR,  dim=1536)

    # Scalar fields — stored alongside the vector, returned with search results
    schema.add_field(field_name="content",     datatype=DataType.VARCHAR,       max_length=65535)
    schema.add_field(field_name="source",      datatype=DataType.VARCHAR,       max_length=512)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
    schema.add_field(field_name="created_at",  datatype=DataType.INT64)

    return schema
```

`enable_dynamic_field=True` allows storing fields not declared in the schema
(they go into a hidden `$meta` JSON field). Useful for adding metadata later
without schema migrations.

### 3.3 Entity

An entity is one row — one chunk of text with its vector and metadata.

```python
entity = {
    "embedding":   [0.023, -0.114, 0.887, ...],  # 1536 floats
    "content":     "FastAPI is a modern Python web framework...",
    "source":      "fastapi_docs.pdf",
    "chunk_index": 14,
    "created_at":  1718000000,
}
```

### 3.4 Segments

Milvus stores entities in segments (immutable files). When you insert entities,
they first go into a growing segment (in memory/WAL). When the segment is full
or flushed, it becomes a sealed segment written to object storage. The index is
built on sealed segments.

This is why a `flush()` call is sometimes needed before querying fresh inserts —
entities in the growing segment may not yet be indexed and therefore not
searchable via ANN.

---

## 4. Vector Indexes — How Search Becomes Fast

### 4.1 The Brute-Force Baseline (FLAT)

FLAT index compares the query vector against every stored vector. It gives
perfect recall (finds the true nearest neighbors) but is O(n) per query. At
1 million vectors: ~1 million cosine computations per query.

Use FLAT only for datasets under ~10,000 vectors or for testing index quality.

### 4.2 IVF_FLAT (Inverted File Index)

IVF clusters vectors into `nlist` Voronoi cells using k-means. At search time,
only the `nprobe` closest cells are searched.

```
nlist = 128 cells, each containing ~7,800 vectors (for 1M total)
nprobe = 16 → search 16 cells → check ~125,000 vectors instead of 1,000,000
Speedup: ~8x
Recall: ~95% (misses 5% of true nearest neighbors)
```

### 4.3 HNSW (Hierarchical Navigable Small World) — Used in This Project

HNSW is a graph-based index. During construction, each vector is connected to
its M nearest neighbors. Vectors are organized in layers — upper layers are
sparse (few vectors, long-range connections), lower layers are dense (all
vectors, short-range connections).

```
Layer 2 (sparse):    • ─────────────────────────── •
                           (long jumps)
Layer 1 (medium):    • ── • ─────── • ────── • ─── •
                        (medium jumps)
Layer 0 (dense):     • ─ • ─ • ─ • ─ • ─ • ─ • ─ •
                       (all vectors, short jumps)
```

Search starts at Layer 2, greedily moves toward the query, descends to Layer 0,
and explores local neighbors. This achieves near-exact results with sub-linear
complexity.

```python
# app/core/milvus_client.py
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,              # connections per node
                              # Higher M → better recall, more RAM, slower build
                              # Typical range: 4–64, default 16 is good

        "efConstruction": 200 # search width during index build
                              # Higher → better index quality, slower build
                              # Typical range: 64–512
    }
)
```

At search time, `ef` controls the search width:

```python
search_params={"ef": max(top_k * 2, 64)}
# ef must be >= top_k
# Higher ef → better recall, slower search
```

**HNSW vs IVF trade-offs:**

| | HNSW | IVF_FLAT |
|---|---|---|
| Recall at same speed | Higher | Lower |
| Memory usage | Higher (graph edges stored) | Lower |
| Build time | Slower | Faster |
| Good for | High-recall production | Large datasets, memory-constrained |

---

## 5. Similarity Metrics

The metric type defines how distance between vectors is measured.

### COSINE (used in this project)

Measures the angle between two vectors, ignoring magnitude.

```
COSINE(a, b) = (a · b) / (|a| × |b|)

Result: -1.0 (opposite) to 1.0 (identical)
In Milvus: returned as distance, higher = more similar
```

Cosine similarity is the standard for text embeddings because embedding
magnitude does not carry semantic meaning — only direction does.

### L2 (Euclidean Distance)

Measures the straight-line distance between two points.

```
L2(a, b) = sqrt(sum((a_i - b_i)²))

Lower = more similar. Milvus returns it as distance.
```

L2 is better for image embeddings where magnitude carries information
(e.g., pixel intensity).

### IP (Inner Product)

Raw dot product, no normalization. Only meaningful when vectors are
already normalized to unit length (in which case IP == COSINE).

---

## 6. CRUD Operations

### Insert

```python
result = client.insert(
    collection_name="documents",
    data=[
        {"embedding": [...], "content": "...", "source": "doc.txt", "chunk_index": 0, "created_at": 1718000000},
        {"embedding": [...], "content": "...", "source": "doc.txt", "chunk_index": 1, "created_at": 1718000000},
    ]
)
print(result["insert_count"])  # 2
```

Milvus does not enforce uniqueness on scalar fields. If you insert the same
source twice, you get duplicate chunks. The idempotent pattern (delete-before-
insert) in this project prevents that.

### Search (ANN)

```python
results = client.search(
    collection_name="documents",
    data=[[0.023, -0.114, 0.887, ...]],  # list of query vectors (batch supported)
    limit=5,                              # top-K
    filter='source == "fastapi_docs.pdf"',# optional scalar pre-filter
    search_params={"ef": 64},            # HNSW-specific
    output_fields=["content", "source", "chunk_index"],
)

hits = results[0]   # results[i] = hits for query i (we sent 1 query)
for hit in hits:
    print(hit["distance"])        # cosine score
    print(hit["entity"]["content"])
```

### Scalar Query (non-vector filter)

For exact lookups without a vector — like "give me all chunks from source X":

```python
rows = client.query(
    collection_name="documents",
    filter='source == "fastapi_docs.pdf"',
    output_fields=["id", "content", "chunk_index"],
    limit=1000,
)
```

### Delete

```python
result = client.delete(
    collection_name="documents",
    filter='source == "fastapi_docs.pdf"',
)
print(result["delete_count"])
```

Delete in Milvus is soft-delete (marks entities as deleted). They are
physically removed during segment compaction. Compaction is triggered
automatically but can also be called manually.

---

## 7. Collection Lifecycle

```python
# Check existence
client.has_collection("documents")  # True / False

# Create (with schema + index — Milvus 2.4 creates both together)
client.create_collection(
    collection_name="documents",
    schema=schema,
    index_params=index_params,
)

# Load into memory — REQUIRED before search/query
# Milvus does not automatically load on startup
client.load_collection("documents")

# Release from memory (free RAM when not in use)
client.release_collection("documents")

# Drop (irreversible)
client.drop_collection("documents")

# Stats
client.get_collection_stats("documents")   # row_count etc.
client.describe_collection("documents")    # fields, indexes
```

**Why load?**  
Milvus stores segments on disk (object storage). Loading moves segment data
into memory for fast search. Without loading, queries return errors. Loading
is idempotent — calling it on an already-loaded collection is a no-op.

In this project, `ensure_collection` loads on every startup. Loading an already-
loaded collection does not reload it — Milvus tracks load state.

---

## 8. Singleton Connection Pattern

Creating a `MilvusClient` is expensive (TCP handshake, auth, capability
negotiation). Every request must not create a new connection.

```python
# app/core/milvus_client.py
_client: MilvusClient | None = None

def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=settings.milvus.URI)
    return _client
```

This is a module-level singleton. In a multi-process deployment (multiple
uvicorn workers), each process gets its own connection — this is correct
because `MilvusClient` is not fork-safe.

`reset_client()` is provided for testing — it forces a reconnect. Tests can
use it to get a clean state.

---

## 9. Scalar Filtering — Hybrid Search

Milvus supports pre-filtering the vector search with a boolean expression
evaluated on scalar fields. This happens *before* the ANN search — only
entities matching the filter are candidates.

```python
# Restrict to one source
filter='source == "legal_contracts.pdf"'

# Restrict by time range
filter='created_at > 1718000000'

# Combine conditions
filter='source == "docs.txt" AND chunk_index < 10'
```

This is hybrid search: scalar filter + vector similarity. The filter
runs on the inverted index for scalar fields, then ANN search runs on the
remaining candidates. This is much faster than post-filtering.

**Important:** If the filter is very selective (e.g., only 5 entities match),
the search pool is tiny and ANN quality degrades. In extreme cases, Milvus
falls back to brute-force over the filtered set.

---

## 10. Segments and Consistency

When entities are inserted, there is a brief window before they appear in
search results. This is because:

1. Entities are buffered in a growing (unsealed) segment
2. The growing segment is not indexed
3. Milvus searches sealed segments by default

To force immediate searchability:

```python
client.flush(collection_name="documents")
```

Flush seals the growing segment and triggers index build. For high-ingest
workloads, flush frequently adds overhead. For RAG use cases (ingest then
query), a single flush after ingest is sufficient.

---

## 11. Multi-tenancy Pattern

For a multi-tenant RAG service (multiple customers, isolated data):

**Option A: One collection per tenant**
- Complete isolation
- High overhead if many tenants (Milvus has collection limits)
- Use for < 100 tenants with high data volume each

**Option B: Shared collection with `tenant_id` scalar field**
- All tenants share one collection
- Queries always filter on `tenant_id`
- Scales to millions of tenants
- Requires careful query construction — missing a filter exposes other tenants' data

```python
# Insert with tenant isolation
entity = {
    "embedding":   vector,
    "content":     chunk,
    "source":      source,
    "tenant_id":   "customer_abc",  # dynamic field (enable_dynamic_field=True)
    "chunk_index": i,
    "created_at":  timestamp,
}

# Search with mandatory tenant filter
results = client.search(
    filter='tenant_id == "customer_abc"',
    ...
)
```

This project uses a single collection without tenancy — adequate for a
single-tenant service. Add `tenant_id` as a dynamic field to extend it.

---

## 12. Production Considerations

| Topic | Recommendation |
|---|---|
| Index type | HNSW for high recall; IVF_SQ8 for memory-constrained |
| `M` parameter | 16 for most cases; increase to 32 for higher recall |
| `efConstruction` | 200 for balanced build; 400+ for maximum recall |
| Chunk embedding dimension | Match the embedding model exactly (1536 for text-embedding-3-small) |
| Flush strategy | Flush after batch ingest, not per-entity |
| Collection naming | Use environment prefix (`prod_documents`, `staging_documents`) |
| Backup | Export segments regularly; Milvus Backup tool for full snapshots |
| Monitoring | Expose collection stats as metrics; alert on growing segment age |

---

## 13. Summary

Milvus solves the core engineering problem of RAG: storing millions of
1536-dimensional vectors and finding the K closest ones to a query vector in
milliseconds. HNSW makes this possible by organizing vectors into a navigable
graph instead of scanning every vector. Scalar fields stored alongside vectors
enable hybrid search — combining semantic similarity with exact metadata
matching. The singleton connection pattern, idempotent collection creation, and
consistent load-before-query lifecycle are the production patterns that make the
service reliable across restarts and deploys.

**Next:** `03-langchain.md` — how LangChain provides composable abstractions
over embeddings, vector stores, LLMs, and chains so you can build the same RAG
pipeline with less boilerplate.
