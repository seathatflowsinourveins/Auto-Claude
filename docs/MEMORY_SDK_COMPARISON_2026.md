# Memory SDK & Persistent Memory Architecture Research 2026

**Research Date**: 2026-02-04
**Researcher**: Claude (UNLEASH Research Specialist)
**Purpose**: Deep investigation into memory SDKs for cross-session memory and self-improvement

## Executive Summary

This research examines four leading memory systems for AI agents in 2026:

### Quick Comparison

| System | GitHub Stars | Latest Version | Key Innovation | Performance |
|--------|-------------|----------------|----------------|-------------|
| **Letta** | 21,000 | v0.16.4 | Sleeptime compute, 3-tier memory | 93.4% DMR |
| **Mem0** | 46,600 | v1.0.3 | Hybrid vector+graph+KV | +26% acc, 91% faster, 90% tokens |
| **Zep/Graphiti** | 22,500 | Active | Bi-temporal knowledge graph | 94.8% DMR, 90% latency cut |
| **HippoRAG** | N/A | NeurIPS'24 | Neurobiological PageRank | +20% multi-hop, 10-30x cheaper |

### UNLEASH Integration Recommendations

1. **Immediate**: Enhance sleeptime_compute.py with Letta-style background consolidation
2. **High Priority**: Add hybrid vector + graph storage (Neo4j + Qdrant)
3. **Medium Priority**: Implement bi-temporal tracking (event time vs ingestion time)
4. **Standard**: Use importance scoring weights: recency (0.3), usage (0.4), confidence (0.3)

---

## 1. Letta (formerly MemGPT)

**GitHub**: https://github.com/letta-ai/letta (21K stars, Apache-2.0)
**Version**: v0.16.4 (2026-01-29)

### Architecture: 3-Tier Memory System

1. **Core Memory** (In-Context)
   - Editable blocks pinned to context window
   - Topics: user prefs, persona, current task
   - Managed by agent or other agents

2. **Archival Memory** (Out-of-Context)
   - Vector DB for long-running memories
   - Semantic search, scales to millions of entries
   - No token usage increase

3. **Recall Memory**
   - Complete interaction history
   - Searchable even when not in context

### Sleeptime Compute (Critical Innovation)

**How it works**:
- Background agents share memory with primary agents
- Transform "raw context" → "learned context" during idle time
- Default trigger: every 5 steps (configurable)
- No performance degradation on benchmarks

**Benefits**:
- Shifts compute from user interaction to idle periods
- Cleans and organizes messy incremental memories
- Mimics human sleep consolidation

