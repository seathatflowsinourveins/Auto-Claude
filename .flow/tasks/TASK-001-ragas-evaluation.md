# TASK-001: RAGAS Evaluation Layer

**Task ID**: TASK-001
**Epic**: EPIC-001 (Advanced Research Architecture v3.0)
**Status**: Ready
**Priority**: P0
**Complexity**: Medium
**Estimated Hours**: 4
**Depends On**: None
**Blocks**: TASK-009

---

## Objective

Implement a RAGAS-based evaluation layer to measure RAG quality metrics (Faithfulness, Context Recall, Context Precision, Answer Relevancy) and establish quality baselines.

---

## Background

RAGAS (Retrieval Augmented Generation Assessment) provides objective metrics for evaluating RAG pipelines. Currently, research_orchestrator.py has no quality measurement - we use source_agreement as a proxy but this does not measure actual answer quality.

### RAGAS Metrics
- **Faithfulness**: Factual accuracy of generated answer (0-1)
- **Context Recall**: Whether retrieved docs contain ground truth (0-1)
- **Context Precision**: Signal-to-noise ratio of retrieved context (0-1)
- **Answer Relevancy**: How relevant is the answer to the question (0-1)

---

## Implementation Plan

### 1. Create RAGASEvaluator Class

```python
# ~/.claude/integrations/ragas_evaluator.py

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextRecall,
    LLMContextPrecisionWithoutReference
)
from datasets import Dataset

class RAGASEvaluator:
    def __init__(self, llm_provider: str = "openai"):
        self.metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            LLMContextPrecisionWithoutReference()
        ]
        
    async def evaluate_result(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None
    ) -> dict[str, float]:
        # Build dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        results = evaluate(dataset, metrics=self.metrics)
        
        return {
            "faithfulness": results["faithfulness"],
            "relevancy": results["answer_relevancy"],
            "context_recall": results.get("context_recall", 0.0),
            "context_precision": results.get("context_precision", 0.0),
            "overall_score": self._compute_overall(results)
        }
```

### 2. Integration Points

- Add `evaluate_response()` method to `SynthesizedResult`
- Call RAGAS evaluation in `_synthesize_results()` 
- Store metrics in `SynthesizedResult.quality_metrics`
- Add quality thresholds to trigger re-research

### 3. Quality Thresholds

```python
QUALITY_THRESHOLDS = {
    "faithfulness": 0.7,    # Re-research if below
    "relevancy": 0.6,       # Warning if below
    "context_recall": 0.5,  # Trigger CRAG if below
}
```

---

## Acceptance Criteria

### Functional
- [ ] RAGASEvaluator class created with all 4 metrics
- [ ] Integration with SynthesizedResult dataclass
- [ ] Async evaluation support
- [ ] Quality thresholds configurable
- [ ] Evaluation results stored in result object

### Non-Functional
- [ ] Evaluation adds < 500ms latency
- [ ] Works without ground_truth (reference-free)
- [ ] Graceful fallback if RAGAS unavailable

### Testing
- [ ] Unit tests for RAGASEvaluator
- [ ] Integration test with real research query
- [ ] Test quality threshold triggering
- [ ] Test without ground truth

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `ragas_evaluator.py` | Create | New evaluator module |
| `research_orchestrator.py` | Modify | Add evaluation integration |
| `tests/test_ragas_evaluator.py` | Create | Unit tests |

---

## Technical Notes

### SDK Signatures (verified from ragas==0.2.8)
```python
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from datasets import Dataset

# evaluate(dataset, metrics=[...]) returns dict with metric names as keys
```

### Environment Requirements
- OPENAI_API_KEY (for LLM-based metrics)
- datasets library (already installed with ragas)

---

## Risks

| Risk | Mitigation |
|------|------------|
| RAGAS API call latency | Cache evaluations, async execution |
| OpenAI API costs | Use lightweight metrics first |
| False positives | Calibrate thresholds with real data |

---

**Assignee**: TBD
**Started**: -
**Completed**: -
