# Flow-Next Status: Advanced Research Architecture v3.0

**Epic**: EPIC-001
**Last Updated**: 2026-02-01
**Overall Progress**: 0/10 tasks (0%)

---

## Current Sprint

### Ready to Start
| Task | Name | Depends On | Hours |
|------|------|------------|-------|
| TASK-001 | RAGAS Evaluation Layer | None | 4 |
| TASK-002 | Chonkie Response Chunking | None | 2 |
| TASK-005 | HyDE Query Expansion | None | 3 |
| TASK-007 | GraphRAG Integration | None | 6 |

### Blocked
| Task | Name | Blocked By | Hours |
|------|------|------------|-------|
| TASK-003 | FlashRank Reranking | TASK-002 | 3 |
| TASK-004 | CRAG Self-Correction | TASK-003 | 6 |
| TASK-006 | Adaptive Query Router | TASK-005 | 4 |
| TASK-008 | Speculative RAG | TASK-004, TASK-006 | 5 |
| TASK-009 | Integration Testing | All above | 4 |
| TASK-010 | Documentation | TASK-009 | 2 |

---

## Dependency Graph

```
Week 1 (Parallel):
  TASK-001 (RAGAS)           [4h] ─────────────┐
  TASK-002 (Chonkie)         [2h] ─> TASK-003 │
  TASK-005 (HyDE)            [3h] ─> TASK-006 │
  TASK-007 (GraphRAG)        [6h] ─────────────┤
                                               │
Week 2 (Dependent):                            │
  TASK-003 (FlashRank)       [3h] ─> TASK-004 │
  TASK-006 (Router)          [4h] ─────────────┤
                                               │
Week 3 (Convergent):                           │
  TASK-004 (CRAG)            [6h] ─────────────┤
  TASK-008 (Speculative)     [5h] <────────────┤
                                               │
Week 4 (Final):                                │
  TASK-009 (Testing)         [4h] <────────────┘
  TASK-010 (Documentation)   [2h]
```

---

## Estimated Timeline

| Week | Tasks | Total Hours |
|------|-------|-------------|
| Week 1 | TASK-001, TASK-002, TASK-005, TASK-007 | 15 |
| Week 2 | TASK-003, TASK-006 | 7 |
| Week 3 | TASK-004, TASK-008 | 11 |
| Week 4 | TASK-009, TASK-010 | 6 |
| **Total** | **10 tasks** | **39 hours** |

---

## Recommended Start Order

1. **TASK-002** (Chonkie) - Lowest effort, unblocks TASK-003
2. **TASK-005** (HyDE) - Medium effort, unblocks TASK-006
3. **TASK-001** (RAGAS) - Parallel with above, provides quality baseline
4. **TASK-007** (GraphRAG) - Can run in parallel, no blockers

---

## Commands

```bash
# Start a task
/flow-next:work TASK-001

# Mark task complete
# (Edit task file, set Status: Complete)

# View dependencies
cat .flow/epics/EPIC-001-advanced-research-v3.md
```

---

## Notes

- All tasks have detailed specs in .flow/tasks/
- SDK versions verified: ragas 0.2.8, chonkie 1.5.4, langgraph 1.0.7
- FlashRank requires Python <3.14 (use dedicated venv if on 3.14)
- GraphRAG package not installed (install separately)
