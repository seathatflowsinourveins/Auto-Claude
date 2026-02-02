# TASK-004: CRAG Self-Corrective Retrieval

**Task ID**: TASK-004
**Epic**: EPIC-001 (Advanced Research Architecture v3.0)
**Status**: Ready
**Priority**: P0
**Complexity**: High
**Estimated Hours**: 6
**Depends On**: TASK-003
**Blocks**: TASK-008

---

## Objective

Implement Corrective RAG (CRAG) pattern to automatically detect and correct low-quality retrievals, reducing hallucination risk.

---

## Background

CRAG (Corrective Retrieval Augmented Generation) evaluates retrieval quality BEFORE generation and takes corrective action:
- **Correct**: Refine and use retrieved docs
- **Incorrect**: Discard and trigger web search
- **Ambiguous**: Combine refined retrieval with web search

### Research Paper
- [arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884)
- Improves RAG by 19% accuracy on PopQA
- 36.6% improvement on PubHealth

---

## Implementation Plan

### 1. Create Retrieval Evaluator

```python
# ~/.claude/integrations/crag_evaluator.py

from enum import Enum
from dataclasses import dataclass

class RetrievalQuality(Enum):
    CORRECT = "correct"      # High confidence, use as-is
    INCORRECT = "incorrect"  # Low confidence, trigger web search
    AMBIGUOUS = "ambiguous"  # Medium confidence, combine sources

@dataclass
class EvaluationResult:
    quality: RetrievalQuality
    confidence: float
    reasoning: str
    suggested_action: str

class CRAGEvaluator:
    def __init__(
        self,
        correct_threshold: float = 0.7,
        incorrect_threshold: float = 0.3
    ):
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold
        
    async def evaluate_retrieval(
        self,
        query: str,
        documents: list[str],
        rerank_scores: list[float] | None = None
    ) -> EvaluationResult:
        """
        Evaluate retrieval quality using multiple signals:
        1. Reranking scores (if available)
        2. Query-document similarity
        3. Document coherence
        4. Coverage analysis
        """
        scores = []
        
        # Signal 1: Reranking scores
        if rerank_scores:
            avg_score = sum(rerank_scores[:3]) / min(3, len(rerank_scores))
            scores.append(avg_score)
            
        # Signal 2: Keyword coverage
        query_terms = set(query.lower().split())
        doc_text = " ".join(documents).lower()
        coverage = sum(1 for t in query_terms if t in doc_text) / len(query_terms)
        scores.append(coverage)
        
        # Signal 3: Document agreement (using existing discrepancy detector)
        # Lower discrepancy = higher agreement = higher quality
        
        # Combine signals
        final_score = sum(scores) / len(scores) if scores else 0.5
        
        # Classify
        if final_score >= self.correct_threshold:
            return EvaluationResult(
                quality=RetrievalQuality.CORRECT,
                confidence=final_score,
                reasoning="High-quality retrieval, documents are relevant",
                suggested_action="refine_and_use"
            )
        elif final_score <= self.incorrect_threshold:
            return EvaluationResult(
                quality=RetrievalQuality.INCORRECT,
                confidence=1 - final_score,
                reasoning="Low-quality retrieval, documents not relevant",
                suggested_action="trigger_web_search"
            )
        else:
            return EvaluationResult(
                quality=RetrievalQuality.AMBIGUOUS,
                confidence=final_score,
                reasoning="Uncertain retrieval quality",
                suggested_action="combine_with_web_search"
            )
```

### 2. Decompose-then-Recompose Algorithm

```python
async def decompose_then_recompose(
    self,
    documents: list[str],
    query: str
) -> list[str]:
    """
    Decompose documents into knowledge strips, filter irrelevant,
    then recompose into focused context.
    """
    # Step 1: Decompose into knowledge strips
    strips = []
    for doc in documents:
        chunks = await self._chunk_document(doc)
        for chunk in chunks:
            relevance = await self._score_relevance(chunk, query)
            if relevance > 0.5:
                strips.append({
                    "content": chunk,
                    "relevance": relevance
                })
    
    # Step 2: Sort by relevance
    strips.sort(key=lambda x: x["relevance"], reverse=True)
    
    # Step 3: Recompose top strips
    recomposed = []
    token_budget = 4096
    current_tokens = 0
    
    for strip in strips:
        strip_tokens = len(strip["content"].split()) * 1.3  # Estimate
        if current_tokens + strip_tokens <= token_budget:
            recomposed.append(strip["content"])
            current_tokens += strip_tokens
            
    return recomposed
```

### 3. Corrective Actions

```python
async def take_corrective_action(
    self,
    evaluation: EvaluationResult,
    query: str,
    documents: list[str],
    mcp_executor: Callable
) -> list[str]:
    """Execute corrective action based on evaluation."""
    
    if evaluation.quality == RetrievalQuality.CORRECT:
        # Refine documents using decompose-then-recompose
        return await self.decompose_then_recompose(documents, query)
        
    elif evaluation.quality == RetrievalQuality.INCORRECT:
        # Discard and trigger web search
        web_results = await mcp_executor("tavily", query)
        return [web_results.get("content", "")]
        
    else:  # AMBIGUOUS
        # Combine refined retrieval with web search
        refined = await self.decompose_then_recompose(documents, query)
        web_results = await mcp_executor("tavily", query)
        return refined + [web_results.get("content", "")]
```

### 4. Integration with Research Orchestrator

Add CRAG evaluation after reranking, before synthesis:

```python
# In _synthesize_results()
if self._crag_evaluator:
    evaluation = await self._crag_evaluator.evaluate_retrieval(
        query, contents, rerank_scores
    )
    
    if evaluation.quality != RetrievalQuality.CORRECT:
        contents = await self._crag_evaluator.take_corrective_action(
            evaluation, query, contents, self.mcp_executor
        )
```

---

## Acceptance Criteria

### Functional
- [ ] CRAGEvaluator with 3-class classification
- [ ] Multiple evaluation signals (rerank, coverage, agreement)
- [ ] Decompose-then-recompose algorithm
- [ ] Web search fallback integration
- [ ] Configurable thresholds

### Non-Functional
- [ ] Evaluation adds < 300ms latency
- [ ] Web search triggered only when needed
- [ ] Reduces hallucination by >20%

### Testing
- [ ] Unit tests for CRAGEvaluator
- [ ] Test each corrective action path
- [ ] Integration test with intentionally bad retrieval
- [ ] Measure hallucination reduction

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `crag_evaluator.py` | Create | CRAG implementation |
| `research_orchestrator.py` | Modify | Add CRAG integration |
| `tests/test_crag_evaluator.py` | Create | Unit tests |

---

## Technical Notes

### Signal Combination
Use weighted combination of signals:
- Reranking score: 0.4 weight
- Keyword coverage: 0.3 weight
- Document agreement: 0.3 weight

### Web Search Integration
Uses existing Tavily integration from mcp_executor.py

---

## Risks

| Risk | Mitigation |
|------|------------|
| Over-correction (too many web searches) | Tune thresholds, add cooldown |
| Web search latency | Parallel execution with timeout |
| Complex evaluation model | Start simple, iterate |

---

**Assignee**: TBD
**Started**: -
**Completed**: -
