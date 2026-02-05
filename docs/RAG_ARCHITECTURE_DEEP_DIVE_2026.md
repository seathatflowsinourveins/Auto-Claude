# RAG Architecture Deep Dive - February 2026

**Research Date**: 2026-02-04 | **Version**: V65

## Critical Patterns (Ranked by Impact)

### 1. Corrective RAG (CRAG) - 67% Failure Reduction
- **Paper**: arxiv.org/abs/2401.15884
- Evaluator grades retrieval: Correct (>0.7), Ambiguous (0.3-0.7), Incorrect (<0.3)
- Incorrect triggers web search fallback (Tavily/Serper)
- Ambiguous combines local + web results

### 2. RAPTOR - +20% Multi-hop Accuracy
- **Paper**: arxiv.org/abs/2401.18059 | **GitHub**: parthsarthi03/raptor (2.5k stars)
- Recursive: chunk -> cluster (GMM) -> summarize -> repeat until root
- Store ALL levels in vector DB ("collapsed tree")
- QuALITY benchmark: 35.7% -> 55.7%

### 3. Contextual Retrieval (Anthropic) - 67% Fewer Failures
- **Source**: anthropic.com/news/contextual-retrieval
- Prepend document-level context to each chunk before embedding
- Cost: ~$1.02/M tokens with prompt caching
- Stack: Contextual Embeddings + BM25 + Reranking = 1.9% failure rate

### 4. Self-RAG - ICLR 2024 Oral (Top 1%)
- **Paper**: arxiv.org/abs/2310.11511 | **GitHub**: AkariAsai/self-rag
- Reflection tokens: [Retrieve], [IsREL], [IsSUP], [IsUSE]
- Model decides WHEN to retrieve (saves cost on simple queries)
- Better factuality than ChatGPT + Llama2-RAG

### 5. Adaptive RAG - Cost/Latency Optimization
- **Paper**: arxiv.org/abs/2403.14403
- Route by complexity: Simple (no retrieval) | Moderate (single-step) | Complex (multi-hop)
- 40-60% cost reduction on mixed workloads

### 6. Mem0 Memory Layer - 91% Faster, +26% Accuracy
- **GitHub**: mem0ai/mem0 (22k stars)
- Dual storage: Vector DB + Graph DB
- Multi-level: User/Session/Agent memory
- LOCOMO benchmark: 93% accuracy

## Embeddings & Reranking (2026 SOTA)

| Model | MTEB Score | Context | Notes |
|-------|-----------|---------|-------|
| Voyage-3-large | 68.2% | 32K | #1 overall, +9.74% vs OpenAI |
| Jina v3 | 65.5% | 8K | Task-specific adapters, 89 languages |
| OpenAI 3-large | 64.6% | 8K | Baseline comparison |

**Reranking**: Jina Reranker v3 (+4.88% vs v2), ColBERT v2 (32-128x faster than cross-encoder)
**Binary quantization**: 96% storage reduction, 92-96% accuracy preserved

## Production Stack

```
BM25 (top-100) + Vector (top-100) -> RRF Fusion -> Reranker (top-10) -> LLM
```

| Method | Precision@5 | Improvement |
|--------|-------------|-------------|
| BM25 only | 0.62 | baseline |
| Vector only | 0.68 | +10% |
| Hybrid (BM25+Vec) | 0.75 | +21% |
| + Reranker | 0.84 | +35% |

## UNLEASH Integration Priority

1. **Corrective RAG** (2-3 days) - web fallback for failures
2. **RAPTOR** (3-5 days) - recursive tree for multi-hop
3. **Contextual Retrieval** (2-3 days) - Anthropic proven pattern
4. **Self-RAG** (2-3 days) - adaptive retrieval with reflection
5. **Adaptive RAG** (2-3 days) - cost optimization routing

## References

- RAPTOR: arxiv.org/abs/2401.18059
- Self-RAG: arxiv.org/abs/2310.11511 (ICLR 2024)
- CRAG: arxiv.org/abs/2401.15884
- Adaptive RAG: arxiv.org/abs/2403.14403
- Modular RAG: arxiv.org/abs/2407.21059
- Contextual Retrieval: anthropic.com/news/contextual-retrieval
- Letta Sleep-Time: docs.letta.com/guides/agents/architectures/sleeptime/
- Mem0: github.com/mem0ai/mem0
- Graphiti: github.com/getzep/graphiti
