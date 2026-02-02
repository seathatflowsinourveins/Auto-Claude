# Voyage AI Embedding Layer V37.0 Architecture

## Version History
- V35.0: LRU cache with TTL, RateLimiter, semantic_search(), cosine_similarity()
- V36.0: Model mixing (_get_effective_model), dynamic batch sizing, create_model_mixing_layer()
- V37.0: Reranking integration, two-stage search, all production models

## Key V37.0 Additions

### 1. Reranking Models (Cross-Encoder)
```python
class RerankModel(str, Enum):
    RERANK_2_5 = "rerank-2.5"           # Best quality, 32K tokens
    RERANK_2_5_LITE = "rerank-2.5-lite" # Fast, 32K tokens
    RERANK_2 = "rerank-2"               # Quality, 16K tokens
    RERANK_2_LITE = "rerank-2-lite"     # Balanced, 8K tokens
```

### 2. Reranking API
```python
result = await layer.rerank(
    query="What is machine learning?",
    documents=documents,
    model="rerank-2.5",
    top_k=5,
)
# Returns RerankResult with (index, score, document) tuples
```

### 3. Two-Stage Search (Embeddings + Reranking)
```python
result = await layer.semantic_search_with_rerank(
    query="How do neural networks optimize learning?",
    documents=documents,
    initial_k=50,  # Embedding candidates
    final_k=5,     # After reranking
    rerank_model="rerank-2.5",
)
```

### 4. All Production Embedding Models
- **Voyage 4 Series**: voyage-4-large, voyage-4, voyage-4-lite (shared embedding space)
- **Specialized**: voyage-code-3 (2048d), voyage-finance-2, voyage-law-2
- **Voyage 3.5**: voyage-3.5, voyage-3.5-lite
- **Multimodal**: voyage-multimodal-3.5, voyage-multimodal-3

## Best Practices

### Model Mixing (Cost Optimization)
```python
layer = create_model_mixing_layer(
    document_model="voyage-4-large",  # High quality for storage
    query_model="voyage-4-lite",       # Fast for queries
)
# ~40% query cost reduction, same embedding space
```

### Two-Stage Retrieval Pipeline
1. **Stage 1 (Embeddings)**: O(n) with indexing, scales to millions
2. **Stage 2 (Reranking)**: O(k) cross-encoder, k=50-100 candidates
3. Result: Fast + Accurate semantic search

### SDK Limitations (voyageai 0.2.3)
- `output_dimension` not supported yet (API supports 2048/1024/512/256)
- `quantization` not supported yet (API supports int8/uint8/binary/ubinary)
- Future: Need HTTP API fallback for these advanced features

## REAL API Validation (2026-01-25)
- Document Embeddings: PASS (voyage-4-large, 1024d)
- Query Embeddings: PASS (voyage-4-lite via model mixing)
- Semantic Search: PASS
- Reranking: PASS (rerank-2.5, top score 0.8750)
- Two-Stage Search: PASS
- Code Embeddings: PASS (voyage-code-3, 1024d)
- Cache: 50% hit rate

## File Location
`Z:\insider\AUTO CLAUDE\unleash\core\orchestration\embedding_layer.py`
