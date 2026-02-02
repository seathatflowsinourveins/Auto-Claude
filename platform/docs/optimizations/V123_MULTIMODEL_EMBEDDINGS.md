# V123 Optimization: Multi-Model Embedding Support

> **Date**: 2026-01-30
> **Priority**: P1 (Performance & Quality)
> **Status**: IMPLEMENTED ✅

---

## Problem Statement

The system was limited to OpenAI embeddings with no alternatives:

```python
# ❌ LIMITED - Only OpenAI, no specialized models
provider = OpenAIEmbeddingProvider(api_key="...")  # What about code-optimized models?
# No Voyage AI for code
# No local models for cost/latency optimization
# No fallback options
```

**Impact**:
- Suboptimal code embeddings (general models vs code-specialized)
- High API costs for all embedding operations
- Dependency on single provider
- No offline capability
- Network latency for every embedding

---

## Solution

### 1. Multi-Provider Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    V123 EMBEDDING PROVIDERS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Voyage AI Provider (NEW)                    │  │
│  │  • voyage-code-3: Best for code retrieval (recommended) │  │
│  │  • voyage-3.5: General-purpose + multilingual           │  │
│  │  • voyage-3.5-lite: Cost-effective                      │  │
│  │  • 1024 dimensions, 32K context                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         Sentence-Transformers Provider (NEW)             │  │
│  │  • all-MiniLM-L6-v2: Fast, 384 dims                     │  │
│  │  • all-mpnet-base-v2: Quality, 768 dims                 │  │
│  │  • BAAI/bge-*: Optimized for retrieval                  │  │
│  │  • Zero API cost, local inference                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              OpenAI Provider (Existing)                  │  │
│  │  • text-embedding-3-small: Balanced                     │  │
│  │  • text-embedding-3-large: Highest dimensions           │  │
│  │  • 1536-3072 dimensions                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Unified Factory Function

```python
from platform.core.advanced_memory import create_embedding_provider

# For code: Use Voyage AI (best quality)
code_provider = create_embedding_provider(
    "voyage-code-3",
    api_key=os.environ["VOYAGE_API_KEY"]
)

# For general text: Use OpenAI or Voyage
general_provider = create_embedding_provider(
    "text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

# For local/offline: Use sentence-transformers (zero cost)
local_provider = create_embedding_provider("all-MiniLM-L6-v2")

# Auto-detection from model name
provider = create_embedding_provider("voyage-code-3")  # Detects Voyage
provider = create_embedding_provider("all-mpnet-base-v2")  # Detects local
```

### 3. Provider Comparison

| Provider | Model | Dimensions | Context | Latency | Cost | Best For |
|----------|-------|------------|---------|---------|------|----------|
| **Voyage AI** | voyage-code-3 | 1024 | 32K | ~100ms | $0.10/1M | Code retrieval |
| **Voyage AI** | voyage-3.5 | 1024 | 32K | ~100ms | $0.06/1M | General + multilingual |
| **Voyage AI** | voyage-3.5-lite | 1024 | 32K | ~80ms | $0.02/1M | Cost-effective |
| **OpenAI** | text-embedding-3-small | 1536 | 8K | ~80ms | $0.02/1M | General purpose |
| **OpenAI** | text-embedding-3-large | 3072 | 8K | ~100ms | $0.13/1M | Maximum quality |
| **Local** | all-MiniLM-L6-v2 | 384 | 512 | ~5ms | FREE | Fast, offline |
| **Local** | all-mpnet-base-v2 | 768 | 512 | ~10ms | FREE | Quality, offline |
| **Local** | BAAI/bge-base-en-v1.5 | 768 | 512 | ~10ms | FREE | Retrieval optimized |

---

## Files Modified (1 Python File)

| File | Change | Status |
|------|--------|--------|
| `platform/core/advanced_memory.py` | Added `EmbeddingModel` entries | ✅ Updated |
| `platform/core/advanced_memory.py` | Added `VoyageEmbeddingProvider` class | ✅ Added |
| `platform/core/advanced_memory.py` | Added `SentenceTransformerEmbeddingProvider` class | ✅ Added |
| `platform/core/advanced_memory.py` | Added `create_embedding_provider()` factory | ✅ Added |

---

## New Classes

### VoyageEmbeddingProvider

```python
class VoyageEmbeddingProvider(EmbeddingProvider):
    """V123: Voyage AI embedding provider."""

    # Features:
    # - Connection pooling (V118 pattern)
    # - Embedding cache integration (V120)
    # - Circuit breaker resilience (V121)
    # - Comprehensive metrics (V122)
    # - Batch embedding up to 128 texts

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-code-3",
        input_type: str = "document",  # "document" or "query"
    ):
        ...

    async def embed(self, text: str) -> EmbeddingResult:
        """Single text embedding with caching."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Batch embedding (up to 128 texts)."""
        ...
```

### SentenceTransformerEmbeddingProvider

```python
class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """V123: Local embedding provider using sentence-transformers."""

    # Features:
    # - Zero API cost
    # - Low latency (local inference)
    # - Model caching (class-level)
    # - Embedding cache integration (V120)
    # - Async execution (runs in thread pool)

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        ...

    async def embed(self, text: str) -> EmbeddingResult:
        """Single text embedding."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Batch embedding (optimized with sentence-transformers native batching)."""
        ...
```

---

## Usage Examples

### For Code Intelligence (Recommended)

```python
import os
from platform.core.advanced_memory import (
    create_embedding_provider,
    create_advanced_memory,
)

# Best for code: Voyage AI voyage-code-3
provider = create_embedding_provider(
    "voyage-code-3",
    api_key=os.environ["VOYAGE_API_KEY"]
)

# Use with advanced memory system
memory = create_advanced_memory(
    agent_id="code-agent",
    embedding_provider=provider,
)

# Embed code snippets
result = await provider.embed("def calculate_fibonacci(n):\n    ...")
print(f"Embedding dims: {result.dimensions}")  # 1024
print(f"Tokens used: {result.tokens_used}")
```

### For Cost-Effective Local Embeddings

```python
# No API key needed - runs locally
provider = create_embedding_provider("all-MiniLM-L6-v2")

# Fast embeddings (~5ms)
result = await provider.embed("User prefers dark mode")
print(f"Embedding dims: {result.dimensions}")  # 384
print(f"Cost: $0.00")  # Local inference
```

### For High-Quality General Purpose

```python
# Voyage 3.5 for best general quality
provider = create_embedding_provider(
    "voyage-3.5",
    api_key=os.environ["VOYAGE_API_KEY"]
)

# Or OpenAI for compatibility
provider = create_embedding_provider(
    "text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)
```

### Batch Embedding

```python
texts = [
    "def calculate_sum(a, b): return a + b",
    "class User: def __init__(self, name): ...",
    "async def fetch_data(): await client.get(...)",
]

# Efficient batch embedding
results = await provider.embed_batch(texts)
for r in results:
    print(f"{r.text[:30]}... -> {len(r.embedding)} dims")
```

---

## Integration with V118-V122

```
Request Flow (V123 with full optimization stack):

┌──────────────┐
│   Request    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  V123 Provider Selection                                      │
│  create_embedding_provider() auto-detects:                   │
│  • "voyage-*" → VoyageEmbeddingProvider                      │
│  • "text-embedding-*" → OpenAIEmbeddingProvider              │
│  • "all-*", "BAAI/*" → SentenceTransformerEmbeddingProvider │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│ V120 Cache   │───►│ V121 Circuit│───►│ V118 Pool       │───►│ API/Model   │
│   Check      │    │   Breaker   │    │   Client        │    │   Call      │
└──────────────┘    └─────────────┘    └─────────────────┘    └─────────────┘
       │                  │                   │                     │
       ▼                  ▼                   ▼                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        V122 METRICS LAYER                                │
│  • Provider type recorded (voyage/openai/sentence-transformers)         │
│  • Model tracked                                                         │
│  • Cache hit/miss                                                        │
│  • Latency measured                                                      │
│  • Tokens counted (API providers)                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Quantified Expected Gains

### Quality Improvements

| Use Case | Before (OpenAI) | After (Specialized) | Gain |
|----------|-----------------|---------------------|------|
| Code retrieval | General embeddings | voyage-code-3 | ~15% better recall |
| Multilingual | English-optimized | voyage-3.5 | ~20% better non-English |
| Local/offline | Not possible | sentence-transformers | ∞ (enabled) |

### Cost Improvements

| Scenario | OpenAI Cost | Voyage Cost | Local Cost |
|----------|-------------|-------------|------------|
| 1M tokens | $0.02 | $0.06-0.10 | $0.00 |
| 10K code chunks | $0.20 | $0.60-1.00 | $0.00 |
| 100K memories | $2.00 | $6.00-10.00 | $0.00 |

### Latency Improvements

| Provider | Typical Latency | Use Case |
|----------|-----------------|----------|
| OpenAI | ~80ms | Standard |
| Voyage AI | ~100ms | Code quality |
| Local (MiniLM) | ~5ms | Speed critical |
| Local (mpnet) | ~10ms | Balance |

---

## Model Selection Guide

### For Code Intelligence
```python
# Best quality
provider = create_embedding_provider("voyage-code-3", api_key=voyage_key)

# Budget alternative
provider = create_embedding_provider("all-mpnet-base-v2")
```

### For Conversation Memory
```python
# Best quality
provider = create_embedding_provider("voyage-3.5", api_key=voyage_key)

# Good quality, lower cost
provider = create_embedding_provider("text-embedding-3-small", api_key=openai_key)

# Free, offline
provider = create_embedding_provider("all-MiniLM-L6-v2")
```

### For Hybrid Systems
```python
# Code embeddings with Voyage
code_provider = create_embedding_provider("voyage-code-3", api_key=voyage_key)

# Fast local for user messages
user_provider = create_embedding_provider("all-MiniLM-L6-v2")

# Use appropriate provider based on content type
```

---

## Verification

### Test Command
```bash
cd "Z:\insider\AUTO CLAUDE\unleash"
pytest platform/tests/test_v123_multimodel_embeddings.py -v
```

### Quick Validation
```python
import asyncio
from platform.core.advanced_memory import create_embedding_provider

async def validate():
    # Test local provider
    local = create_embedding_provider("all-MiniLM-L6-v2")
    result = await local.embed("test code: def hello(): pass")
    assert len(result.embedding) == 384
    print(f"Local: {result.dimensions} dims, {result.tokens_used} tokens")

    # Test Voyage provider (if key available)
    import os
    if voyage_key := os.environ.get("VOYAGE_API_KEY"):
        voyage = create_embedding_provider("voyage-code-3", api_key=voyage_key)
        result = await voyage.embed("def calculate_sum(a, b): return a + b")
        assert len(result.embedding) == 1024
        print(f"Voyage: {result.dimensions} dims, {result.tokens_used} tokens")

    print("V123 multi-model embeddings working!")

asyncio.run(validate())
```

---

## Related Optimizations

- V118: Connection pooling for HTTP clients
- V119: Async batch processing
- V120: Embedding cache with TTL
- V121: Circuit breaker for API failures
- V122: Memory metrics & observability
- **V123**: Multi-model embedding support (this document)

---

## Future Improvements

1. **V124**: Automatic model selection based on content type
2. **V125**: Embedding model benchmarking suite
3. **V126**: Hybrid local+API with intelligent routing
4. **V127**: Model fine-tuning support for domain-specific embeddings

---

*Optimization completed 2026-01-30 as part of autonomous system optimization iteration.*
