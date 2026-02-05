"""
VECTOR DATABASES ITERATIONS - Deep Dive into Vector Storage
============================================================
Qdrant, Pinecone, Weaviate, Milvus, Chroma optimization patterns

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


VECTOR_TOPICS = [
    # Qdrant Deep Dive
    {"topic": "Qdrant: HNSW indexing, quantization, payload filtering", "area": "qdrant"},
    {"topic": "Qdrant collections: sharding, replication, on-disk storage", "area": "qdrant"},
    {"topic": "Qdrant search: dense, sparse, hybrid, multi-vector", "area": "qdrant"},
    {"topic": "Qdrant optimization: batch upsert, prefetch, scroll", "area": "qdrant"},

    # Pinecone
    {"topic": "Pinecone: serverless, pods, namespaces, metadata filtering", "area": "pinecone"},
    {"topic": "Pinecone hybrid search: sparse-dense vectors, BM25", "area": "pinecone"},
    {"topic": "Pinecone Assistant: RAG-as-a-service, file upload", "area": "pinecone"},
    {"topic": "Pinecone inference: embedding API, reranking", "area": "pinecone"},

    # Weaviate
    {"topic": "Weaviate: schema, classes, properties, vectorizers", "area": "weaviate"},
    {"topic": "Weaviate modules: text2vec, img2vec, multi2vec", "area": "weaviate"},
    {"topic": "Weaviate GraphQL: Get, Aggregate, Explore queries", "area": "weaviate"},
    {"topic": "Weaviate generative: RAG queries, grouped tasks", "area": "weaviate"},

    # Milvus
    {"topic": "Milvus: collections, partitions, segments, indexes", "area": "milvus"},
    {"topic": "Milvus indexing: IVF_FLAT, HNSW, DISKANN, GPU indexes", "area": "milvus"},
    {"topic": "Milvus Lite: embedded, serverless, edge deployment", "area": "milvus"},
    {"topic": "Milvus search: ANN, range, hybrid, multi-vector", "area": "milvus"},

    # Optimization
    {"topic": "Vector index tuning: HNSW ef, M parameters, recall vs speed", "area": "optimization"},
    {"topic": "Quantization: scalar, product, binary quantization trade-offs", "area": "optimization"},
    {"topic": "Multi-tenancy: namespace isolation, metadata partitioning", "area": "optimization"},
    {"topic": "Vector database benchmarking: ANN-benchmarks, latency profiles", "area": "optimization"},
]


class VectorDatabasesExecutor(BaseResearchExecutor):
    """Custom executor with vector DB-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Vector database implementation and optimization: {topic}"


if __name__ == "__main__":
    run_research(
        "vectordb",
        "VECTOR DATABASES ITERATIONS",
        VECTOR_TOPICS,
        executor_class=VectorDatabasesExecutor,
    )
