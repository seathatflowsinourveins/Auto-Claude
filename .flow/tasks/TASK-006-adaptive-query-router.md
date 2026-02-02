# TASK-006: Adaptive Query Router

**Task ID**: TASK-006
**Epic**: EPIC-001 (Advanced Research Architecture v3.0)
**Status**: Ready
**Priority**: P1
**Complexity**: Medium
**Estimated Hours**: 4
**Depends On**: TASK-005
**Blocks**: TASK-008

---

## Objective

Implement complexity-based adaptive query routing to optimize resource usage.

---

## Background

Adaptive RAG routes queries dynamically based on complexity:
- **Simple queries**: Direct LLM response (skip retrieval)
- **Moderate queries**: Vector search only
- **Complex queries**: Full multi-source research
- **Time-sensitive**: Web search priority

---

## Implementation Plan

### Query Complexity Classifier

QueryComplexity enum: SIMPLE, MODERATE, COMPLEX, TIME_SENSITIVE

RouteDecision enum: DIRECT_LLM, VECTOR_SEARCH, WEB_SEARCH, FULL_RESEARCH, HYBRID

### Routing Logic

- SIMPLE -> Direct LLM (skip retrieval)
- MODERATE -> Vector search (single source)
- COMPLEX -> Full research (all tools + HyDE + CRAG)
- TIME_SENSITIVE -> Web search first

---

## Acceptance Criteria

### Functional
- [ ] QueryComplexity enum with 4 levels
- [ ] Heuristic classifier (fast, no LLM)
- [ ] Optional LLM classifier (accurate)
- [ ] Route decision with tool selection
- [ ] Integration with HyDE and CRAG decisions

### Non-Functional
- [ ] Heuristic routing < 10ms
- [ ] LLM routing < 500ms
- [ ] Reduces unnecessary tool calls by >30%

### Testing
- [ ] Unit tests for each complexity level
- [ ] Test heuristic patterns
- [ ] Integration test with full pipeline

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| query_router.py | Create | Router implementation |
| research_orchestrator.py | Modify | Add routing integration |
| tests/test_query_router.py | Create | Unit tests |

---

**Assignee**: TBD
**Started**: -
**Completed**: -
