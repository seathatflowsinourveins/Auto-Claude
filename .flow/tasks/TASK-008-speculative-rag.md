# TASK-008: Speculative RAG

**Task ID**: TASK-008
**Epic**: EPIC-001 (Advanced Research Architecture v3.0)
**Status**: Ready
**Priority**: P2
**Complexity**: High
**Estimated Hours**: 5
**Depends On**: TASK-004, TASK-006
**Blocks**: TASK-009

---

## Objective

Implement Speculative RAG pattern for latency reduction through parallel draft generation and verification.

---

## Background

Speculative RAG uses a two-phase approach:
1. **Drafting**: Smaller specialized LM generates multiple answer drafts in parallel, each from different document subsets
2. **Verification**: Larger generalist LM evaluates drafts and selects best one

### Key Results (from paper)
- Up to 51% latency reduction
- Up to 12.97% accuracy improvement
- Handles diverse document perspectives

Reference: [arxiv.org/abs/2407.08223](https://arxiv.org/abs/2407.08223)

---

## Implementation Plan

### 1. Document Subset Creation

```python
def create_diverse_subsets(documents: list, num_subsets: int = 3):
    # Cluster documents by content similarity
    # Sample one document per cluster for each subset
    # Return diverse subsets for parallel processing
```

### 2. Parallel Draft Generation

```python
async def generate_drafts(
    query: str,
    document_subsets: list[list[str]],
    drafter_model: str
) -> list[Draft]:
    # Generate drafts in parallel using smaller model
    drafts = await asyncio.gather(*[
        generate_single_draft(query, subset, drafter_model)
        for subset in document_subsets
    ])
    return drafts
```

### 3. Draft Verification

```python
async def verify_drafts(
    query: str,
    drafts: list[Draft],
    verifier_model: str
) -> Draft:
    # Score each draft using larger model
    # Consider: factuality, relevance, completeness
    # Return best draft
```

### 4. Integration with Research Orchestrator

- Activate for complex queries (via adaptive router)
- Use alongside CRAG for quality assurance
- Cache verified drafts

---

## Acceptance Criteria

### Functional
- [ ] Document clustering for diverse subsets
- [ ] Parallel draft generation
- [ ] Draft verification with scoring
- [ ] Best draft selection

### Non-Functional
- [ ] Latency reduction > 30% vs sequential
- [ ] Quality maintained or improved
- [ ] Parallel execution efficient

### Testing
- [ ] Unit tests for each component
- [ ] Latency benchmarks
- [ ] Quality comparison tests

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| speculative_rag.py | Create | Speculative RAG implementation |
| research_orchestrator.py | Modify | Add speculative mode |
| tests/test_speculative_rag.py | Create | Unit tests |

---

## Technical Notes

### Model Selection
- Drafter: Claude Haiku or GPT-4o-mini (fast, cheap)
- Verifier: Claude Sonnet or GPT-4o (accurate)

### Parallelism
Use asyncio.gather for concurrent draft generation.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Draft quality variance | More drafts, better clustering |
| Verification overhead | Cache, batch when possible |
| Model cost | Use smaller drafter |

---

**Assignee**: TBD
**Started**: -
**Completed**: -
