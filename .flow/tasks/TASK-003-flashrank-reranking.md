# TASK-003: FlashRank Reranking Layer

**Task ID**: TASK-003
**Epic**: EPIC-001 (Advanced Research Architecture v3.0)
**Status**: Ready
**Priority**: P1
**Complexity**: Medium
**Estimated Hours**: 3
**Depends On**: TASK-002
**Blocks**: TASK-004

---

## Objective

Implement a reranking layer using FlashRank to improve relevance ordering of retrieved documents before synthesis.

---

## Background

FlashRank is an ultra-lite reranking library (~4MB model) that runs on CPU without Torch dependencies. It provides significant relevance improvements with minimal overhead.

### Key Features
- No Torch/Transformers dependencies
- ~4MB model footprint
- CPU-only execution
- Cross-encoder based scoring
- Supports pairwise and listwise reranking

---

## Implementation Plan

### 1. Install FlashRank

```bash
# Note: Requires Python <3.14
pip install flashrank>=3.0.0
```

### 2. Create Reranker Class

```python
# ~/.claude/integrations/reranker.py

from flashrank import Ranker, RerankRequest

class FlashReranker:
    def __init__(self, model: str = "ms-marco-MultiBERT-L-12"):
        """
        Models:
        - ms-marco-MultiBERT-L-12 (default, balanced)
        - ms-marco-MiniLM-L-12-v2 (faster, smaller)
        - rank-T5-flan (listwise, larger context)
        """
        self.ranker = Ranker(model_name=model, cache_dir=".cache/flashrank")
        
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10
    ) -> list[RankedDocument]:
        """
        Rerank documents by relevance to query.
        
        Returns documents sorted by relevance score (highest first).
        """
        request = RerankRequest(
            query=query,
            passages=[{"text": doc} for doc in documents]
        )
        
        results = self.ranker.rerank(request)
        
        return [
            RankedDocument(
                content=r["text"],
                score=r["score"],
                original_rank=documents.index(r["text"])
            )
            for r in results[:top_k]
        ]
```

### 3. Integration with Research Orchestrator

```python
# In _synthesize_results()

# After collecting all contents
contents = [r.content for r in successful]

# Rerank by relevance to query
if len(contents) > 1 and self._reranker:
    ranked = await self._reranker.rerank(query, contents, top_k=5)
    contents = [r.content for r in ranked]
    
# Continue with synthesis using reranked contents
```

### 4. Fallback Strategy

```python
# Use rerankers library as fallback
try:
    from flashrank import Ranker
    FLASHRANK_AVAILABLE = True
except ImportError:
    try:
        from rerankers import Reranker
        # rerankers supports: flashrank, cohere, jina, etc.
        FLASHRANK_AVAILABLE = False
        RERANKERS_AVAILABLE = True
    except ImportError:
        RERANKERS_AVAILABLE = False
```

---

## Acceptance Criteria

### Functional
- [ ] FlashReranker class with model selection
- [ ] Async reranking support
- [ ] Top-k filtering
- [ ] Original rank preservation
- [ ] Graceful fallback if unavailable

### Non-Functional
- [ ] Reranking adds < 200ms latency
- [ ] Works on CPU only
- [ ] Model cached for fast loading
- [ ] Memory footprint < 100MB

### Testing
- [ ] Unit tests for FlashReranker
- [ ] Test with 20+ documents
- [ ] Measure relevance improvement
- [ ] Integration test with research pipeline

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `reranker.py` | Create | Reranking module |
| `research_orchestrator.py` | Modify | Add reranking integration |
| `tests/test_reranker.py` | Create | Unit tests |

---

## Technical Notes

### FlashRank SDK (from GitHub)
```python
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-MultiBERT-L-12")
request = RerankRequest(query="...", passages=[{"text": "..."}])
results = ranker.rerank(request)  # Returns list with score, text
```

### Python Version Constraint
FlashRank requires Python >=3.10, <3.14. If using Python 3.14, use the rerankers library with ONNX backend:
```python
from rerankers import Reranker
ranker = Reranker("flashrank", model_type="FlashRankRanker")
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| Python 3.14 incompatibility | Use rerankers library or dedicated venv |
| Model download latency | Pre-download and cache model |
| Low-quality reranking | Test multiple models, use ensemble |

---

**Assignee**: TBD
**Started**: -
**Completed**: -
