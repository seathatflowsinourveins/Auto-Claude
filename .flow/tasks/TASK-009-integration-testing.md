# TASK-009: Integration Testing

**Task ID**: TASK-009
**Epic**: EPIC-001 (Advanced Research Architecture v3.0)
**Status**: Blocked
**Priority**: P0
**Complexity**: Medium
**Estimated Hours**: 4
**Depends On**: TASK-001 through TASK-008
**Blocks**: TASK-010

---

## Objective

Create comprehensive integration tests for Advanced Research Architecture v3.0 with REAL API calls (no mocks).

---

## Background

Per UNLEASH protocol: NO MOCKS. All tests must hit real services to verify actual behavior. This catches API signature mismatches that mock tests miss.

---

## Test Scenarios

### 1. Full Pipeline Test
- Query -> Router -> HyDE -> Retrieval -> Rerank -> CRAG -> Synthesis -> RAGAS
- Verify each component activates correctly
- Measure end-to-end latency

### 2. Complexity Routing Tests
- Simple query: verify DIRECT_LLM route taken
- Moderate query: verify single-source retrieval
- Complex query: verify full parallel research
- Time-sensitive: verify web search priority

### 3. CRAG Correction Tests
- Test with intentionally poor retrieval
- Verify INCORRECT classification triggers web search
- Verify AMBIGUOUS triggers hybrid

### 4. RAGAS Quality Tests
- Run on known-good queries with ground truth
- Verify metrics > thresholds
- Test reference-free evaluation mode

### 5. Large Response Tests
- Test with 50k+ token responses
- Verify chunking activates
- Verify map-reduce synthesis quality

### 6. GraphRAG Tests (if available)
- Test entity extraction
- Test relationship queries
- Compare with/without GraphRAG

### 7. Speculative RAG Tests
- Test parallel draft generation
- Measure latency improvement
- Verify quality maintenance

---

## Acceptance Criteria

### Functional
- [ ] All 7 test scenarios passing
- [ ] Real API calls (no mocks)
- [ ] Error handling tested
- [ ] Fallback paths tested

### Non-Functional
- [ ] Test suite runs in < 5 minutes
- [ ] >80% code coverage
- [ ] CI/CD integration ready

### Documentation
- [ ] Test cases documented
- [ ] Expected results documented
- [ ] Known limitations noted

---

## Files to Create

| File | Purpose |
|------|---------|
| tests/test_research_v3_integration.py | Main integration tests |
| tests/test_ragas_real.py | RAGAS with real data |
| tests/test_crag_real.py | CRAG with real retrieval |
| tests/conftest.py | Shared fixtures |

---

## Test Commands

```bash
# Run all integration tests
pytest tests/test_research_v3_integration.py -v

# Run with coverage
pytest tests/ --cov=integrations --cov-report=term-missing

# Run specific scenario
pytest tests/test_research_v3_integration.py::test_full_pipeline -v
```

---

**Assignee**: TBD
**Started**: -
**Completed**: -
