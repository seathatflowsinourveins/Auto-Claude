# V120 Optimization: Embedding Cache with TTL

> **Date**: 2026-01-30
> **Priority**: P2 (Performance)
> **Status**: IMPLEMENTED ✅

---

## Problem Statement

Redundant embedding API calls for the same text:

```python
# ❌ INEFFICIENT - Same text embedded multiple times
async def search_similar(query: str):
    embedding = await provider.embed(query)  # API call 1
    # ...

async def index_document(text: str):
    embedding = await provider.embed(text)  # Might be same as previous query!
    # ...
```

**Impact**:
- ~50-200ms latency per redundant API call
- Unnecessary API costs ($0.0001 per 1K tokens)
- Rate limiting risk with high-volume usage
- Wasted bandwidth

---

## Solution

### 1. EmbeddingCache Class

```python
# ✅ OPTIMIZED - LRU cache with TTL (V120)
class EmbeddingCache:
    """LRU cache for embeddings with TTL support."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,  # 1 hour default
    ):
        self._cache: OrderedDict[str, Tuple[List[float], float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if exists and not expired."""
        key = self._make_key(text, model)
        if key not in self._cache:
            self._misses += 1
            return None

        embedding, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return embedding

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache an embedding with LRU eviction."""
        key = self._make_key(text, model)

        # Evict LRU items if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (embedding, time.time())

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

# Global cache instance
_embedding_cache = EmbeddingCache(max_size=2000, ttl_seconds=3600.0)
```

### 2. Integration with OpenAIEmbeddingProvider

```python
# ✅ Cache integration in embed method
async def embed(self, text: str) -> EmbeddingResult:
    """V120: Generate embedding via OpenAI API with caching."""
    # Check cache first
    cached = _embedding_cache.get(text, self._model)
    if cached is not None:
        return EmbeddingResult(
            text=text,
            embedding=cached,
            model=self._model,
            dimensions=len(cached),
            tokens_used=0,  # No tokens used for cache hit
        )

    # Cache miss - call API
    client = self._get_client()
    response = await client.post(...)

    embedding = data["data"][0]["embedding"]

    # Cache the result
    _embedding_cache.set(text, self._model, embedding)

    return EmbeddingResult(text=text, embedding=embedding, ...)
```

### 3. Batch Method Optimization

```python
# ✅ Smart batch caching - only call API for uncached texts
async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
    """V120: Generate embeddings for batch with caching."""
    # Check cache for all texts first
    results: List[Optional[EmbeddingResult]] = [None] * len(texts)
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []

    for i, text in enumerate(texts):
        cached = _embedding_cache.get(text, self._model)
        if cached is not None:
            results[i] = EmbeddingResult(...)  # Cache hit
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    # If all cached, return early
    if not uncached_texts:
        return cast(List[EmbeddingResult], results)

    # Make API call only for uncached texts
    response = await client.post(json={"input": uncached_texts, ...})

    # Process and cache new results
    for j, item in enumerate(data["data"]):
        original_idx = uncached_indices[j]
        _embedding_cache.set(uncached_texts[j], self._model, item["embedding"])
        results[original_idx] = EmbeddingResult(...)

    return cast(List[EmbeddingResult], results)
```

---

## Files Modified (1 Python File)

| File | Class | Change | Status |
|------|-------|--------|--------|
| `platform/core/advanced_memory.py` | `EmbeddingCache` | New class with LRU + TTL | ✅ Added |
| `platform/core/advanced_memory.py` | `OpenAIEmbeddingProvider` | Cache integration in embed/embed_batch | ✅ Fixed |

---

## Quantified Expected Gains

### Latency Reduction (Repeated Queries)

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| Same query twice | 2 × 100ms = 200ms | 100ms + 0ms = 100ms | **~50%** |
| 10 repeated queries | 10 × 100ms = 1000ms | 100ms + 0ms × 9 = 100ms | **~90%** |
| Batch with 50% duplicates | Full API call | 50% reduction | **~50%** |

### Cost Reduction

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| API calls for repeated text | Every time | Once per TTL | **Variable** |
| Tokens per repeated query | Full tokens | Zero | **100%** |
| Monthly cost (high usage) | $X | $X × (1 - hit_rate) | **Up to 80%** |

### Memory Usage

| Setting | Value | Rationale |
|---------|-------|-----------|
| max_size | 2000 entries | ~10MB for 384-dim embeddings |
| ttl_seconds | 3600 (1 hour) | Balance freshness vs performance |

---

## Cache Key Design

```python
def _make_key(self, text: str, model: str) -> str:
    """Create cache key from text and model."""
    content = f"{model}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]
```

**Key Properties**:
- Model-specific: Different models produce different embeddings
- Collision-resistant: SHA-256 truncated to 32 chars
- Case-sensitive: "Hello" ≠ "hello"
- Whitespace-sensitive: "a b" ≠ "a  b"

---

## Observability

```python
# Get cache statistics
stats = _embedding_cache.stats
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

**Returned Stats**:
```python
{
    "hits": 150,
    "misses": 50,
    "hit_rate": 0.75,  # 75% hit rate
    "size": 200,
    "max_size": 2000,
}
```

---

## TTL Considerations

**Why 1 Hour Default?**
1. Embeddings are deterministic (same text → same embedding)
2. Model updates are infrequent (but do happen)
3. 1 hour covers typical session length
4. Prevents indefinite stale data

**When to Adjust?**
- Development: `ttl_seconds=300` (5 min) for faster iteration
- Production: `ttl_seconds=3600` (1 hour) for performance
- Stable models: `ttl_seconds=86400` (24 hours) for max caching

---

## Verification

### Test Command
```bash
cd "Z:\insider\AUTO CLAUDE\unleash"
pytest platform/tests/test_v120_embedding_cache.py -v
```

### Performance Benchmark
```python
import asyncio
import time
from platform.core.advanced_memory import OpenAIEmbeddingProvider, _embedding_cache

async def benchmark():
    provider = OpenAIEmbeddingProvider(api_key="sk-...")
    text = "Sample text for embedding"

    # First call - cache miss
    _embedding_cache.clear()
    start = time.perf_counter()
    result1 = await provider.embed(text)
    cold_time = time.perf_counter() - start

    # Second call - cache hit
    start = time.perf_counter()
    result2 = await provider.embed(text)
    warm_time = time.perf_counter() - start

    print(f"Cold (API call): {cold_time*1000:.1f}ms")
    print(f"Warm (cache hit): {warm_time*1000:.1f}ms")
    print(f"Speedup: {cold_time/warm_time:.0f}x")
    print(f"Cache stats: {_embedding_cache.stats}")

asyncio.run(benchmark())
```

---

## Related Optimizations

- V115: Letta Cloud initialization fix
- V116: Sleep-time agent configuration fix
- V117: Deprecated token= parameter fix
- V118: Connection pooling for HTTP clients
- V119: Async batch processing
- **V120**: Embedding cache with TTL (this document)

---

## Future Improvements

1. **V121**: Circuit breaker for API failures
2. **Persistent Cache**: Save to disk for cross-process caching
3. **Distributed Cache**: Redis backend for multi-instance
4. **Semantic Deduplication**: Cache similar (not identical) texts

---

*Optimization completed 2026-01-30 as part of autonomous system optimization iteration.*
