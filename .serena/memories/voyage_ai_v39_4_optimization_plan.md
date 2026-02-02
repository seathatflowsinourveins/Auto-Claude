# Voyage AI V39.4 - Optimization Plan

**Created**: 2026-01-25
**Status**: Planning
**Dependencies**: V39.3 (completed)

---

## Current State (V39.3 Complete)

### Implemented Features
- ✅ 11 embedding models (voyage-4-large, code-3, finance-2, etc.)
- ✅ 6 reranking models
- ✅ 30+ public methods
- ✅ Cache system with warming, persistence, efficiency reports
- ✅ Advanced search: MMR, multi-query, hybrid, filtered
- ✅ Project adapters: Witness, Trading, Unleash

### Test Suite Status
- ✅ 5 cache tests passing
- ✅ 5 advanced search tests passing
- ✅ Real API validation (no mocks)

---

## V39.4 Optimization Targets

### 1. Cross-Project Search Enhancements

**WitnessVectorAdapter Improvements**:
- Add `find_similar_poses_mmr()` for diverse archetype discovery
- Add `hybrid_shader_search()` combining code semantics + keywords
- Add `search_particles_with_filters()` for physics parameter filtering

**TradingVectorAdapter Improvements**:
- Add `find_similar_signals_mmr()` for diverse market pattern discovery
- Add `hybrid_strategy_search()` for strategy description + keyword matching
- Add temporal filtering for time-series data

**UnleashVectorAdapter Improvements**:
- Add cross-collection search capability
- Add skill recommendation with MMR diversity

### 2. Performance Optimizations

**Streaming Embeddings**:
```python
async def embed_streaming(
    self,
    texts: AsyncIterator[str],
    batch_size: int = 128,
) -> AsyncIterator[list[float]]:
    """Yield embeddings as they complete for memory efficiency."""
```

**Prefetch Pipeline**:
```python
async def semantic_search_prefetch(
    self,
    queries: list[str],
    documents: list[str],
) -> list[tuple[str, list[tuple[int, float, str]]]]:
    """Pre-fetch embeddings while preparing queries for parallel search."""
```

### 3. Adaptive Search Strategies

**Auto-tuning Alpha for Hybrid Search**:
- Analyze query characteristics (keywords vs concepts)
- Dynamically adjust alpha based on query type
- Learn optimal alpha from search feedback

**Query Complexity Detection**:
```python
def detect_query_complexity(self, query: str) -> QueryComplexity:
    """
    Detect whether query needs:
    - Simple semantic search
    - Multi-query expansion
    - Hybrid with keywords
    """
```

### 4. Memory-Efficient Operations

**Chunked Document Processing**:
- Process large document collections in memory-efficient chunks
- Support streaming from file/database sources

**Lazy Embedding Loading**:
- Load pre-computed embeddings on demand
- Memory-mapped embedding storage for large corpora

### 5. Project Integration Tests

Create comprehensive cross-project test suite:
- `tests/voyage_v39_witness_integration_test.py`
- `tests/voyage_v39_trading_integration_test.py`
- `tests/voyage_v39_cross_project_test.py`

---

## Implementation Priority

| Feature | Priority | Effort | Impact |
|---------|----------|--------|--------|
| Adapter MMR methods | HIGH | Medium | High |
| Hybrid shader search | HIGH | Low | High |
| Streaming embeddings | MEDIUM | High | Medium |
| Adaptive alpha | LOW | High | Medium |
| Cross-project tests | HIGH | Low | High |

---

## Next Steps

1. Implement `find_similar_poses_mmr()` in WitnessVectorAdapter
2. Add `hybrid_shader_search()` with code+keyword matching
3. Create integration tests for Witness adapter
4. Validate with TouchDesigner real-world data
5. Document V39.4 enhancements

---

**Document Version**: 1.0
**Author**: Claude Code (Ralph Loop)
**Project**: Unleash Meta-Project
