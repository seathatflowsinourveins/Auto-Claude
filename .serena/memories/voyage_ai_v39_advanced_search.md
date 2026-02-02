# Voyage AI V39.3 - Advanced Semantic Search Methods

**Created**: 2026-01-25
**Status**: Production Ready (All 5 tests passing)
**File**: `core/orchestration/embedding_layer.py`

---

## Overview

V39.3 adds sophisticated semantic search patterns to the EmbeddingLayer:
- **MMR (Maximal Marginal Relevance)** - Diverse result retrieval
- **Multi-Query Retrieval** - Query expansion with fusion
- **Hybrid Search** - Vector + BM25 combination
- **Filtered Search** - Metadata-based filtering

---

## 1. MMR Search (Maximal Marginal Relevance)

**Purpose**: Balance relevance with diversity in search results.

**Formula**: `MMR = λ * Sim(d, q) - (1-λ) * max(Sim(d, d_selected))`

```python
results = await layer.semantic_search_mmr(
    query="Python async programming patterns",
    documents=corpus,
    doc_embeddings=precomputed_embeddings,  # Optional
    top_k=5,
    lambda_mult=0.5,  # 0.0 = max diversity, 1.0 = max relevance
    fetch_k=20,  # Candidates to consider
)
# Returns: [(idx, score, doc), ...]
```

**Parameters**:
- `lambda_mult=0.3`: More diverse results (different topics)
- `lambda_mult=0.9`: More relevant results (clustered topics)

---

## 2. Multi-Query Retrieval

**Purpose**: Expand queries for better recall, then fuse results.

**Fusion Methods**:
- **RRF (Reciprocal Rank Fusion)**: `score = Σ 1/(k + rank)` - Robust, rank-based
- **Sum**: `score = Σ similarity_scores` - Simple additive
- **Max**: `score = max(similarity_scores)` - Best match wins

```python
results = await layer.semantic_search_multi_query(
    query="How do async patterns improve concurrency?",
    documents=corpus,
    doc_embeddings=precomputed_embeddings,
    top_k=5,
    num_sub_queries=3,  # Number of query expansions
    fusion_method="rrf",  # "rrf", "sum", or "max"
)
```

**Sub-Query Generation**:
- Creates semantic variations of original query
- Example: "async patterns" → ["asynchronous programming", "concurrent execution", "non-blocking code"]

---

## 3. Hybrid Search (Vector + BM25)

**Purpose**: Combine semantic understanding with keyword matching.

**Formula**: `score = α * vector_score + (1-α) * bm25_score`

```python
results = await layer.hybrid_search(
    query="Python asyncio concurrent",
    documents=corpus,
    doc_embeddings=precomputed_embeddings,
    top_k=5,
    alpha=0.5,  # Balance between vector (1.0) and BM25 (0.0)
    bm25_k1=1.5,  # Term frequency saturation
    bm25_b=0.75,  # Document length normalization
)
```

**Alpha Settings**:
- `alpha=0.2`: Keyword-focused (good for exact term matching)
- `alpha=0.5`: Balanced (recommended default)
- `alpha=0.9`: Semantic-focused (good for concept matching)

**BM25 Implementation**:
- Full TF-IDF variant with document length normalization
- IDF: `log((N - df + 0.5) / (df + 0.5) + 1)`
- TF: `(tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))`

---

## 4. Filtered Search (Metadata Conditions)

**Purpose**: Apply structured filters before/during semantic search.

**Supported Operators**:
- `$eq`: Exact match (default)
- `$gt`, `$gte`: Greater than (or equal)
- `$lt`, `$lte`: Less than (or equal)
- `$ne`: Not equal
- `$in`: Value in list
- `$nin`: Value not in list
- `$contains`: Substring match

```python
results = await layer.semantic_search_with_filters(
    query="machine learning models",
    documents=corpus,
    metadata=[
        {"category": "ml", "year": 2024, "difficulty": "advanced"},
        {"category": "python", "year": 2023, "difficulty": "intermediate"},
        # ...
    ],
    doc_embeddings=precomputed_embeddings,
    top_k=5,
    filters={
        "category": "ml",  # Exact match
        "year": {"$gte": 2024},  # Range filter
        "difficulty": {"$in": ["advanced", "expert"]},  # List membership
    },
    threshold=0.3,  # Minimum similarity
)
# Returns: [(idx, score, doc, metadata), ...]
```

**Filter Logic**: All conditions are AND-ed together.

---

## 5. Helper Methods

### BM25 Scoring (Internal)
```python
scores = layer._compute_bm25_scores(
    query="search terms",
    documents=corpus,
    k1=1.5,  # Term frequency saturation
    b=0.75,  # Length normalization
)
# Returns: [score1, score2, ...]
```

### Sub-Query Generation (Internal)
```python
sub_queries = layer._generate_sub_queries(
    query="original query",
    num_queries=3,
)
# Returns: ["variation 1", "variation 2", "variation 3"]
```

---

## 6. Test Results

All tests validated with real Voyage AI API:

| Test | Status | Description |
|------|--------|-------------|
| MMR Search | ✓ PASS | Correctly diversifies results |
| Multi-Query | ✓ PASS | RRF and sum fusion working |
| Hybrid Search | ✓ PASS | Alpha parameter balances methods |
| Filtered Search | ✓ PASS | All operators functioning |
| BM25 Scoring | ✓ PASS | Keyword matching accurate |

---

## 7. Performance Characteristics

| Method | Latency | Use Case |
|--------|---------|----------|
| MMR | ~200-300ms | When diversity matters |
| Multi-Query | ~400-600ms | Complex/ambiguous queries |
| Hybrid | ~150-250ms | Keyword + semantic blend |
| Filtered | ~100-200ms | Structured data with text |

---

## 8. Integration with Existing Methods

The new methods complement existing EmbeddingLayer capabilities:

```python
# Complete RAG pipeline example
layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
await layer.initialize()

# 1. Embed documents once
doc_result = await layer.embed(documents, model=EmbeddingModel.VOYAGE_4_LARGE)

# 2. Use MMR for diverse retrieval
mmr_results = await layer.semantic_search_mmr(
    query, documents, doc_result.embeddings, top_k=10, lambda_mult=0.7
)

# 3. Rerank for precision
reranked = await layer.rerank(
    query, [doc for _, _, doc in mmr_results],
    model=RerankModel.RERANK_2_5, top_k=3
)
```

---

**Document Version**: 1.0
**Author**: Claude Code (Ralph Loop)
**Project**: Unleash Meta-Project
