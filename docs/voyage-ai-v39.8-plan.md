# Voyage AI V39.8 Implementation Plan
## Batch-Cache Integration & Optimization

**Status**: ✅ FULLY IMPLEMENTED (2026-01-26)
**Target**: Full batch integration with caching system
**Dependencies**: V39.7 Batch API implementation
**Priority**: MEDIUM (optimization for production workloads)
**Phase 3**: ✅ COMPLETE (batch contextualized/reranking)

---

## Gap Analysis

| Current Limitation | V39.8 Solution |
|-------------------|----------------|
| Batch results not cached | Auto-populate cache from download |
| GestureEmbeddingLibrary uses real-time API | Optional batch initialization |
| No batch contextualized embeddings | Add batch contextualized method |
| No batch reranking | Add batch rerank method |
| No batch progress streaming | WebSocket/SSE progress updates |

---

## V39.8 Features

### 1. Cache Integration for Batch Results

```python
async def download_batch_results(
    self,
    batch_id: str,
    output_path: Optional[str] = None,
    populate_cache: bool = False,  # V39.8: Auto-cache results
    original_texts: Optional[list[str]] = None,  # V39.8: Required for cache
    input_type: InputType = InputType.DOCUMENT,  # V39.8: For cache key
) -> list[EmbeddingResult]:
    """
    Download batch results and optionally populate cache.

    V39.8 Enhancement: Automatic cache integration.

    When populate_cache=True:
    - Each embedding is added to the cache
    - Cache keys use original input text hash
    - Enables instant retrieval for repeated queries

    Example:
        # Download and cache all gesture embeddings
        results = await layer.download_batch_results(
            batch_id="batch-abc123",
            populate_cache=True,
            cache_key_prefix="gesture_library_v2",
        )
        # Future requests for same texts hit cache instantly
    """
```

**Implementation Notes (COMPLETED):**
- Pass `original_texts` parameter when downloading results
- Cache keys generated via `_get_cache_key(text, model, input_type)`
- Embeddings added directly to `self._cache` dict with `CacheEntry`
- Validation: `ValueError` raised if `populate_cache=True` without `original_texts`

### 2. GestureEmbeddingLibrary Batch Mode

```python
class GestureEmbeddingLibrary:
    async def initialize(
        self,
        use_batch: bool = False,  # V39.8: Use batch API
        batch_wait: bool = True,  # V39.8: Wait for completion
    ) -> Optional[str]:  # Returns batch_id if fire-and-forget
        """
        Initialize gesture embeddings.

        V39.8 Enhancement: Optional batch mode for 33% cost savings.

        Args:
            use_batch: If True, use batch API (async, cheaper)
            batch_wait: If True, wait for batch completion

        Example (real-time):
            await library.initialize()  # Immediate, full price

        Example (batch, 33% savings):
            await library.initialize(use_batch=True)  # Wait up to 12h

        Example (batch, fire-and-forget):
            await library.initialize(use_batch=True, batch_wait=False)
            # Returns batch_id, poll separately
    """

    async def initialize_from_batch_job(self, batch_id: str) -> None:
        """
        Initialize from a previously created batch job.

        Useful for loading pre-computed gesture embeddings.

        Example:
            # Previously created batch job
            job = await layer.create_batch_embedding_job(gesture_texts)
            # ... hours later ...
            await library.initialize_from_batch_job(job.id)
        """
```

### 3. Batch Contextualized Embeddings

```python
async def create_batch_contextualized_job(
    self,
    documents: list[list[str]],  # Each document as list of chunks
    output_dimension: Optional[int] = None,
    metadata: Optional[dict[str, str]] = None,
) -> BatchJob:
    """
    Create batch job for contextualized chunk embeddings.

    V39.8 Feature: Batch processing for large document corpora.

    Perfect for:
    - Large document corpus embedding
    - State of Witness session recordings
    - Archetype training datasets

    Example:
        # Pre-compute session recordings
        job = await layer.create_batch_contextualized_job(
            documents=[
                ["Frame 1: Warrior stance...", "Frame 2: Transition..."],
                ["Frame 1: Sage meditation...", "Frame 2: Stillness..."],
            ],
            metadata={"session": "performance_2026_01_26"}
        )
    """
```

### 4. Batch Reranking

```python
async def create_batch_rerank_job(
    self,
    queries: list[str],
    documents_per_query: list[list[str]],
    model: str = "rerank-2.5",
    metadata: Optional[dict[str, str]] = None,
) -> BatchJob:
    """
    Create batch reranking job for large-scale retrieval refinement.

    V39.8 Feature: Cost-effective reranking for large query sets.

    Example:
        # Rerank archetype descriptions for each session
        job = await layer.create_batch_rerank_job(
            queries=["warrior pose", "sage meditation", "jester dance"],
            documents_per_query=[
                archetype_descriptions,  # Same docs for each query
                archetype_descriptions,
                archetype_descriptions,
            ],
        )
    """
```

### 5. Batch Progress Streaming (Optional)

```python
async def stream_batch_progress(
    self,
    batch_id: str,
    on_progress: Callable[[BatchJob], Awaitable[None]],
) -> AsyncGenerator[BatchJob, None]:
    """
    Stream batch progress updates using async generator.

    V39.8 Feature: Real-time progress visibility.

    Example:
        async for update in layer.stream_batch_progress(job.id):
            print(f"Progress: {update.request_counts.completed}/{update.request_counts.total}")
            if update.is_successful:
                break
    """
```

---

## State of Witness Use Cases

### 1. Session Recording Pre-computation
```python
# After a performance session, batch embed all poses
session_poses = [pose_to_text(p) for p in recorded_poses]
job = await layer.create_batch_embedding_job(
    texts=session_poses,
    metadata={"session": "performance_2026_01_26"},
)
# Wait for completion and cache
await layer.wait_for_batch_completion(job.id)
results = await layer.download_batch_results(
    batch_id=job.id,
    populate_cache=True,  # V39.8: Auto-cache
)
```

### 2. Gesture Library Pre-warming
```python
# Initialize gestures with batch (33% cheaper)
library = GestureEmbeddingLibrary(layer)
await library.initialize(use_batch=True, batch_wait=True)  # V39.8
```

### 3. Archetype Training Pipeline
```python
# Batch embed training data
training_job = await layer.create_batch_embedding_job(
    texts=training_pose_descriptions,
    output_dimension=512,
    metadata={"purpose": "archetype_training_v2"},
)
# Continue with other work while batch processes
# Check later...
status = await layer.get_batch_status(training_job.id)
```

---

## Implementation Order

### Phase 1: Cache Integration (Priority: HIGH)
1. Add `populate_cache` parameter to `download_batch_results()`
2. Track input texts in batch job metadata
3. Implement cache population on download
4. Add tests for cache integration

### Phase 2: Gesture Library Batch (Priority: MEDIUM)
5. Add `use_batch` and `batch_wait` parameters to `initialize()`
6. Implement `initialize_from_batch_job()` method
7. Add tests for batch initialization

### Phase 3: Extended Batch Features (Priority: LOW) ✅ COMPLETE
8. ✅ Implement `create_batch_contextualized_job()`
9. ✅ Implement `create_batch_rerank_job()`
10. Add progress streaming with async generator (Optional - deferred to V39.9)

---

## Cost Analysis

### Current Real-Time Costs (V39.6)
- Gesture library init: 15 texts × $0.00002 = $0.0003
- Session recording: 3600 poses/min × $0.00002 = $0.072/minute

### With V39.8 Batch Integration
- Gesture library batch: 15 texts × $0.00002 × 0.67 = **$0.0002** (33% savings)
- Session recording batch: 3600 poses × $0.00002 × 0.67 = **$0.048/minute** (33% savings)

### For 8-hour Performance Day
- V39.6 (real-time): ~$34.56
- V39.8 (batch): ~$23.04
- **Daily savings: $11.52**

---

## Success Criteria

- [x] Cache integration for batch results working ✅ `populate_cache` + `original_texts` params
- [x] GestureEmbeddingLibrary supports batch initialization ✅ `use_batch`, `batch_wait`, `initialize_from_batch_job()`
- [x] Batch contextualized embeddings working ✅ `create_batch_contextualized_job()`
- [x] Batch reranking working ✅ `create_batch_rerank_job()`
- [x] Tests with real Voyage AI API ✅ 10/10 V39.8 tests passing (Phase 1-2: 6, Phase 3: 4)
- [ ] Cost savings validated in production

---

## Implementation Details

### Phase 1-2: Cache Integration & Batch Gesture Mode

**Methods modified in EmbeddingLayer:**
- `download_batch_results()` - Added `populate_cache`, `original_texts`, `input_type` params (lines 4620-4780)

**Methods modified in GestureEmbeddingLibrary:**
- `__init__()` - Added `_batch_job_id: Optional[str] = None` attribute
- `initialize()` - Added `use_batch` and `batch_wait` parameters (lines 6585-6672)
- `initialize_from_batch_job()` - NEW method for loading from batch (lines 6674-6714)

### Phase 3: Batch Contextualized & Reranking

**Methods added to EmbeddingLayer:**
- `create_batch_contextualized_job(documents, output_dimension, metadata)` - Batch contextualized chunk embeddings
  - Accepts `list[list[str]]` where each document is a list of chunks
  - Uses `/v1/contextualizedembeddings` endpoint with batch processing
  - Returns `BatchJob` for async tracking
- `create_batch_rerank_job(queries, documents_per_query, model, top_k, metadata)` - Batch reranking
  - Accepts parallel lists of queries and their candidate documents
  - Uses `/v1/rerank` endpoint with batch processing
  - Supports `RerankModel.RERANK_2_5` (default) or `RERANK_2_LITE`
  - Returns `BatchJob` for async tracking

**Tests:**
- `tests/voyage_v39_8_cache_test.py` - 10 tests all passing (6 Phase 1-2 + 4 Phase 3)

---

Document Version: 1.2
Created: 2026-01-26
Updated: 2026-01-26
Author: Claude (Ralph Loop V39.8 Planning)
Phase 1-2: Claude (Ralph Loop V39.8 Implementation)
Phase 3: Claude (Ralph Loop V39.8 Continuation)
