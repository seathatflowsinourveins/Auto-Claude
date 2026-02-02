# Cross-Session Bootstrap V39.0 - COMPLETE Voyage AI Integration

## FEATURE COMPLETENESS AUDIT ✓

### All Voyage AI Features Implemented

## Complete Method Reference (EmbeddingLayer)

| Method | Purpose | Model Used |
|--------|---------|------------|
| `embed()` | Core embedding with caching | voyage-4-large (default) |
| `embed_documents()` | Document embedding | voyage-4-large |
| `embed_query()` | Query embedding | voyage-4-large |
| `embed_queries()` | Batch query embedding | voyage-4-large |
| `embed_code()` | Code-specific | voyage-code-3 |
| `embed_with_options()` | Full control (dim, dtype) | Any |
| `embed_quantized()` | int8/binary shortcut | Any |
| `embed_multimodal()` | Text + Images | voyage-multimodal-3.5 |
| `embed_contextualized()` | Context-aware chunks | voyage-context-3 |
| `embed_with_chunking()` | Long doc aggregation | Any |
| `embed_batch_optimized()` | Concurrent batching | Any |
| `rerank()` | Rerank results | rerank-2.5 |
| `semantic_search()` | Similarity search | Any |
| `semantic_search_with_rerank()` | Search + rerank | voyage-4-large + rerank-2.5 |

### Static Utilities
| Method | Purpose |
|--------|---------|
| `truncate_matryoshka()` | Post-hoc dimension reduction |
| `normalize_embeddings()` | L2 normalize (after truncation) |

### Aggregation Strategies (embed_with_chunking)
- **mean**: Average of chunk embeddings (stable, default)
- **first**: Use first chunk only (fast, title-heavy docs)
- **max**: Max-pool across chunks (strongest signals)
- **weighted**: More weight to early chunks (structured docs)

## Version History
- **V38.0**: HTTP API, output dimensions, quantization (int8/binary), reranking
- **V39.0**: Multimodal embeddings, intelligent chunking, batch optimization, Qdrant integration, project adapters
- **V39.1** (current): Fixed contextualized/multimodal endpoints via direct REST API

## V39.1 Bug Fixes (2026-01-25)

### Contextualized Embeddings (`embed_contextualized`)
**Endpoint**: POST `/v1/contextualizedembeddings`

**Issues Fixed**:
1. ❌ `truncation` parameter NOT supported by this endpoint (causes 400 error)
2. ❌ `input_type` parameter NOT supported by this endpoint (causes 400 error)
3. Response structure is nested: `data[0].data[i].embedding` (not `data[0].embeddings[i]`)

**Working Payload**:
```json
{
  "inputs": [["chunk1", "chunk2", "chunk3"]],
  "model": "voyage-context-3"
}
```

### Multimodal Embeddings (`embed_multimodal`)
**Endpoint**: POST `/v1/multimodalembeddings`

**Issue Fixed**:
- Each input must be wrapped in `{"content": [...]}` format

**Working Payload**:
```json
{
  "inputs": [
    {"content": [{"type": "text", "text": "description"}]},
    {"content": [{"type": "image_base64", "image_base64": "..."}]}
  ],
  "model": "voyage-multimodal-3.5"
}

## V39.0 New Features

### 1. Multimodal Embeddings (voyage-multimodal-3.5)
```python
# Text + Image embeddings through unified backbone
result = await layer.embed_multimodal(
    inputs=[["Red leather handbag", pil_image], "Blue denim jacket"],
    input_type=InputType.DOCUMENT,
)
```
**Constraints**: Max 1000 inputs, 16M pixels/image, 320K total tokens

### 2. Intelligent Document Chunking
```python
# Long document → overlapping chunks → aggregated embedding
result = await layer.embed_with_chunking(
    texts=long_documents,
    chunk_size=8000,       # chars per chunk (~2000 tokens)
    chunk_overlap=200,     # overlap for continuity
    aggregation="weighted", # mean, first, max, weighted
)
```
**Strategies**: mean (stable), first (fast), max (signal capture), weighted (structured docs)

### 3. Batch Optimization
```python
# Token-aware batching with controlled concurrency
result = await layer.embed_batch_optimized(
    texts=million_docs,
    target_batch_tokens=50000,
    max_concurrent=5,
    progress_callback=lambda done, total: print(f"{done}/{total}"),
)
```

### 4. Qdrant Vector Database Integration
```python
from core.orchestration.embedding_layer import (
    QdrantConfig, QdrantVectorStore, EmbeddingLayer
)

# Initialize
config = QdrantConfig(host="localhost", port=6333)
store = QdrantVectorStore(config, embedding_layer)
await store.initialize()

# Create collection
await store.create_collection("my_vectors", dimension=1024)

# Store embeddings
await store.upsert(
    collection="my_vectors",
    embeddings=result.embeddings,
    payloads=[{"type": "document", "name": "foo"}],
)

# Search
matches = await store.search(
    collection="my_vectors",
    query_vector=query_embedding,
    filter={"type": "document"},
    limit=10,
)
```

### 5. Project-Specific Adapters

#### WitnessVectorAdapter (State of Witness - Creative AI)
Collections: `witness_poses`, `witness_shaders`, `witness_particles`, `witness_archetypes`
```python
adapter = WitnessVectorAdapter(embedding_layer, qdrant_store)
await adapter.initialize_collections()

# Index shaders with code-3 model
shader_id = await adapter.index_shader(
    shader_code=glsl_code,
    name="noise_field",
    shader_type="fragment",
    tags=["noise", "procedural"],
)

# Search shaders semantically
matches = await adapter.search_shaders("flowing water effect")
```

#### TradingVectorAdapter (AlphaForge - Trading)
Collections: `trading_signals`, `trading_strategies`, `trading_risk`, `trading_sentiment`
```python
adapter = TradingVectorAdapter(embedding_layer, qdrant_store)
await adapter.initialize_collections()

# Index strategy with weighted chunking (prioritize abstract)
strategy_id = await adapter.index_strategy(
    strategy_doc=long_strategy_document,
    name="momentum_crossover",
    category="trend_following",
)

# Find similar market signals
matches = await adapter.find_similar_signals("bullish divergence RSI")
```

#### UnleashVectorAdapter (Meta-Project)
Collections: `unleash_skills`, `unleash_memory`, `unleash_research`, `unleash_patterns`
```python
adapter = UnleashVectorAdapter(embedding_layer, qdrant_store)
await adapter.initialize_collections()

# Index Claude skill
skill_id = await adapter.index_skill(
    skill_content=skill_text,
    name="tdd-workflow",
    category="quality",
    tags=["testing", "development"],
)

# Find relevant skills for task
matches = await adapter.find_relevant_skills("write unit tests for API")

# Store conversation memory
await adapter.store_conversation_memory(
    conversation_summary="Implemented V39.0 embedding layer...",
    session_id="session-123",
    topics=["voyage-ai", "qdrant", "embeddings"],
    importance=0.8,
)
```

## Complete Model Reference (V39.0)

| Model | Dimension | Use Case |
|-------|-----------|----------|
| voyage-4-large | 1024 (256-2048) | General-purpose, highest quality |
| voyage-4-lite | 512 (256-1024) | Fast, cost-effective |
| voyage-code-3 | 1024 (256-2048) | Code, shaders, patterns |
| voyage-finance-2 | 1024 | Financial, trading signals |
| voyage-multimodal-3.5 | 1024 | Text + images together |
| rerank-2.5 | - | Reranking search results |

## API Key
```
pa-KCpYL_zzmvoPK1dM6tN5kdCD8e6qnAndC-dSTlCuzK4
```
Status: Paid tier, all models available

## Test Validation (13/13 PASSED)
- All 5 embedding models working
- Reranking with top_k
- Semantic search >0.4 threshold
- Dimension reduction (256d, 512d)
- Quantization (int8: 4x, binary: 32x savings)
- Ultra compact (256d + int8: 16x savings)
- Caching with 50% hit rate

## File Location
`Z:\insider\AUTO CLAUDE\unleash\core\orchestration\embedding_layer.py`

## Dependencies
```
pip install voyageai qdrant-client
```
