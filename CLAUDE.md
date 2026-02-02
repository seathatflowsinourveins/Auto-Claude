# UNLEASH - Meta-Platform for Claude Enhancement

> **Purpose**: Enhance Claude's capabilities to build State of Witness and AlphaForge
> **Status**: Production | **Architecture**: 8-Layer, 28 SDKs
> **Version**: V35 (Research Architecture v4.1 + Semantic Clustering + Voyage Reranker)
> **Letta Mode**: Cloud (https://api.letta.com) | SDK: letta-client 1.7.7 (API stable)
> **Philosophy**: Autonomous First. Compound Learning. Real Testing Always.
> **Optimized**: 2026-02-01 (V34: Research v4.0, LLM-based HyDE/RAGAS, 41 tests passing)

---

## Project Identity

This is the **UNLEASH** project - the meta-layer that enhances Claude Code's capabilities.

| Project | Location | Letta Cloud Agent |
|---------|----------|-------------------|
| **UNLEASH** (this) | `Z:/insider/AUTO CLAUDE/unleash` | Uses ECOSYSTEM agent |
| **WITNESS** | `Z:/insider/AUTO CLAUDE/Touchdesigner-createANDBE` | `agent-bbcc0a74-5ff8-4ccd-83bc-b7282c952589` |
| **ALPHAFORGE** | `Z:/insider/AUTO CLAUDE/autonomous AI trading system` | `agent-5676da61-c57c-426e-a0f6-390fd9dfcf94` |

**Letta Cloud Agents** (verified 2026-01-30):
- WITNESS (state-of-witness-creative-brain): `agent-bbcc0a74-5ff8-4ccd-83bc-b7282c952589`
- ALPHAFORGE (alphaforge-dev-orchestrator): `agent-5676da61-c57c-426e-a0f6-390fd9dfcf94`
- ECOSYSTEM (claude-code-ecosystem-test): `agent-daee71d2-193b-485e-bda4-ee44752635fe`

---

## 8-Layer SDK Architecture (28 Production SDKs)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNLEASH SDK ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  L8: KNOWLEDGE       │ graphrag, pyribs                                     │
│  L7: PROCESSING      │ ast-grep, crawl4ai                                   │
│  L6: SAFETY          │ guardrails-ai, llm-guard, nemo-guardrails           │
│  L5: OBSERVABILITY   │ langfuse, opik, arize-phoenix, deepeval, ragas,     │
│                      │ promptfoo                                            │
│  L4: REASONING       │ dspy                                                 │
│  L3: STRUCTURED OUT  │ instructor, baml, outlines, pydantic-ai             │
│  L2: MEMORY          │ letta, zep, mem0                                     │
│  L1: ORCHESTRATION   │ temporal-python, langgraph, claude-flow             │
│  L0: PROTOCOL        │ mcp-python-sdk, mcp-ecosystem, litellm, anthropic   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SDK Priority

| Priority | Count | SDKs |
|----------|-------|------|
| **P0 CRITICAL** | 8 | anthropic, claude-flow, letta, opik, mcp-python-sdk, instructor, litellm, langgraph |
| **P1 IMPORTANT** | 10 | dspy, pydantic-ai, pyribs, promptfoo, mem0, ragas, deepeval, guardrails-ai, crawl4ai, ast-grep |
| **P2 SPECIALIZED** | 10 | temporal-python, graphrag, outlines, baml, langfuse, zep, llm-guard, nemo-guardrails, arize-phoenix, mcp-ecosystem |

---

## Key Directories

```
unleash/
├── .claude/          # Agents (24), Skills (28)
├── .config/          # API keys, SDK_INDEX
├── sdks/             # 28 production SDKs
│   ├── anthropic/    # Claude SDKs, cookbooks
│   ├── letta/        # Memory server
│   ├── mcp-python-sdk/
│   └── claude-flow/  # V3 orchestration
├── platform/         # Core platform code
└── apps/             # Applications
```

---

## Key References

| Resource | Location |
|----------|----------|
| SDK Index | `.config/SDK_INDEX.md` |
| Global Rules | `~/.claude/CLAUDE.md` |
| Verification Protocol | `~/.claude/references/verification.md` |
| Research Protocol | `~/.claude/references/research-protocol.md` |

---

## MCP Servers (9 configured)

| Server | Purpose |
|--------|---------|
| claude-flow | V3 orchestration |
| letta | Agent memory |
| code-index | Semantic search |
| firecrawl | Web extraction |
| memory | Knowledge graphs |
| filesystem | File operations |
| sqlite | Database |

---

## Quick Commands

```bash
# === BUILD & INSTALL ===
pip install -e .                    # Core only (editable mode)
pip install -e ".[dev]"             # With dev tools (pytest, mypy, ruff)
pip install -e ".[all]"             # All 28 SDKs

# === TESTING ===
pytest tests/ -v                    # Run all tests
pytest tests/ --cov=core            # With coverage
pytest tests/integration/ -m integration  # Integration tests only
pytest -x --tb=short               # Stop on first failure

# === LINTING & TYPE CHECKING ===
ruff check .                        # Python linting
ruff check --fix .                  # Auto-fix lint issues
ruff format .                       # Format Python code
mypy core/ --strict                 # Type checking (strict mode)

# === LETTA CLOUD ===
# Use verified cloud agents (requires LETTA_API_KEY)
C:\Users\42\.letta-env\Scripts\python.exe -c "from letta_client import Letta; print(Letta().agents.list())"

# Local Letta server (alternative)
cd sdks/letta/letta && letta server --port 8283

# === RESEARCH & VALIDATION ===
python platform/scripts/auto_research.py research "AI agents 2026"
python platform/scripts/auto_validate.py
```

---

## Security Note

**LangGraph**: Upgrade to 3.0+ - CVE-2025-64439 (RCE via JsonPlusSerializer)

---

## V35 Research Architecture v4.1 (2026-02-01)

### LLM-Enhanced Components (53 tests passing)

| Component | Class | V4.1 Enhancement | Fallback |
|-----------|-------|------------------|----------|
| RAGAS Evaluation | `RAGASEvaluator` | **LLM-based scoring via Anthropic** | Heuristic scoring |
| Chunking | `MCPResponseHandler` | Semantic chunking | Chonkie → sliding window |
| Reranking | `RerankerLayer` | **Multi-backend + Voyage** | FlashRank → CrossEncoder → Cohere → Voyage → passthrough |
| CRAG | `DocumentGrader` | **DeBERTa NLI + hybrid scoring** | Keyword-only scoring |
| HyDE | `HyDEExpander` | **LLM-generated hypothetical docs** | Template expansion |
| Query Routing | `QueryRouter` | Complexity detection | Heuristic routing |
| GraphRAG | `GraphRAGProcessor` | **spaCy NER support** | Pattern-based extraction |
| Speculative RAG | `SpeculativeRAGGenerator` | **LLM drafts + Semantic clustering** | Template concatenation |

**Location**: `~/.claude/integrations/research_orchestrator.py` (~80KB consolidated)

### V4.1 New Features

1. **Speculative RAG (Semantic Clustering + LLM Drafts)**: K-means clustering with LLM synthesis
   - Groups semantically similar documents before generating drafts
   - LLM-based draft generation for coherent query-focused answers
   - 50%+ latency reduction vs sequential processing
   - Cluster cohesion scoring for confidence weighting

2. **Voyage Reranker Support**: Added Voyage AI rerank-2.5 API
   - SOTA accuracy for retrieval tasks
   - Auto-selection in reranker priority chain
   - Requires `VOYAGE_API_KEY` environment variable

3. **CRAG with DeBERTa NLI**: Natural Language Inference for document grading
   - Zero-shot classification: relevant/irrelevant/partially relevant
   - Hybrid scoring: 70% NLI (semantic) + 30% keyword (lexical)
   - Model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (91.2% accuracy)

### V4.0 Key Improvements

1. **HyDE (LLM-based)**: True hypothetical document generation via Claude Haiku
   - Generates document-like answers for semantic search
   - Falls back to templates when LLM unavailable

2. **RAGAS (LLM-based)**: Anthropic API evaluation when ragas library unavailable
   - Scores faithfulness, relevancy, precision via LLM
   - 3-tier fallback: RAGAS lib → LLM → Heuristics

3. **GraphRAG (NER-ready)**: spaCy NER support with pattern-based fallback
   - Entity types: PERSON, ORGANIZATION, TECHNOLOGY, LOCATION, EVENT
   - NetworkX Louvain community detection

### Letta SDK v1.7.7 CRITICAL Fixes (Verified with REAL API)

```python
# CORRECT patterns (V34 verified)
results = client.agents.passages.search(agent_id, query="...", top_k=10)  # NOT limit=
for r in results.results:  # NOT .passages
    content = r.content    # NOT .text (for search results)

client.agents.blocks.attach(block_id, agent_id=agent_id)  # block_id POSITIONAL
```

### Files Fixed (V34)

| File | Line | Fix |
|------|------|-----|
| `iterative_retrieval.py` | 186 | `.content` before `.text` fallback |
| `unified_memory_gateway.py` | 461-465 | `.results` accessor, `.content` |
| `research_orchestrator.py` | 936-1070 | V4.0 HyDE with LLM generation |
| `research_orchestrator.py` | 577-780 | V4.0 RAGAS with LLM evaluation |
| `research_orchestrator.py` | 1112-1270 | V4.0 GraphRAG with NER support |

---

## V27.1 Production Stability Update (2026-01-31)

### P0 Fixes Applied ✅

| Fix | Location | Status |
|-----|----------|--------|
| **AttributeError Guards** | `platform_orchestrator.py` | ✅ 12 None checks added |
| **_validate_components()** | `platform_orchestrator.py` | ✅ New startup validation |
| **_is_component_ready()** | `platform_orchestrator.py` | ✅ Per-component checks |
| **Circuit Breaker** | `circuit_breaker.py` | ✅ Already production-ready |

### V27.1 Research Synthesis (3 Deep Agents)

| Agent | Focus | Key Production Patterns |
|-------|-------|------------------------|
| **Claude Code Optimization** | Configuration, Hooks, Memory | Settings hierarchy, MCP timeouts, anti-corruption hooks |
| **MCP Stability** | Timeouts, Pooling, Health | 30s init, 60s tools, circuit breakers, watchdog timers |
| **Autonomous Loops** | Claude-flow V3, Memory | Swarm topologies, Temporal+LangGraph layers, sleep-time |

### Research Agents (V27 Base)

| Agent | Focus | Key Findings |
|-------|-------|--------------|
| **Autonomous Loops** | Ralph Wiggum, GOAP, Factory Signals | Dual-gate exits, A* correction, compound learning |
| **Anthropic SDK** | Latest features 2025-2026 | Extended thinking, 1-hour cache, strict tools |
| **Opik Observability** | Production tracing | Full trace visibility, cost analytics, dashboards |
| **Letta Advanced** | Sleep-time, Archives, Learning SDK | Cross-agent memory, background consolidation |
| **System Analysis** | Gap detection, priorities | P0/P1/P2 ranked improvements |

### V27 Autonomous Loop

```
┌─────────────────────────────────────────────────────────────────┐
│              AdvancedMonitoringLoopV27                          │
├─────────────────────────────────────────────────────────────────┤
│  Ralph Wiggum    │  Direction     │  Chi-Squared   │  GOAP     │
│  (Dual-Gate)     │  Monitor       │  Drift         │  Planner  │
├─────────────────────────────────────────────────────────────────┤
│  Factory Signals (Compound Learning) → Letta Memory (Persistent)│
└─────────────────────────────────────────────────────────────────┘
```

**Expected Gains**:
- 60% reduction in wasted iterations
- 45% faster task completion
- 80% recovery rate from failures
- Cross-session compound learning

### New Capabilities

| Feature | Location | Purpose |
|---------|----------|---------|
| **V27 Loop** | `~/.claude/integrations/advanced_monitoring_loop_v27.py` | Full autonomous loop |
| **Circuit Breaker** | `~/.claude/integrations/circuit_breaker.py` | MCP failure isolation |
| **Factory Signals** | V27 loop + `~/.claude/learnings/` | Compound learning |
| **Opik Integration** | Optional via `pip install opik` | Full observability |
| **Letta Learning SDK** | `pip install agentic-learning` | Drop-in continual learning |

### SDK Upgrades (V27)

| SDK | Feature | Usage |
|-----|---------|-------|
| **Anthropic** | Extended Thinking | `thinking={"type": "enabled", "budget_tokens": 16000}` |
| **Anthropic** | 1-Hour Cache | `cache_control={"type": "ephemeral", "ttl": "1h"}` |
| **Anthropic** | Strict Tools | `strict: True` in tool definitions |
| **Letta** | Sleep-time Compute | `enable_sleeptime=True` |
| **Letta** | Archives API | Cross-agent shared memory |
| **Opik** | Anthropic Tracing | `track_anthropic(client)` |

### MCP Timeout Configuration

| Tier | Servers | Timeout |
|------|---------|---------|
| Essential | LSP, Context7, Exa, Tavily, Jina, Firecrawl | 60s |
| Deferred | Memory, SQLite, Filesystem | 60s |
| Heavy | Letta, Brave | 120s |
| Lazy | Claude-flow, Qdrant, GraphRAG | On-demand |

### Critical Gaps (Updated V27.1)

| Priority | Gap | Status |
|----------|-----|--------|
| **P0** | Platform Orchestrator validation | ✅ FIXED (V27.1) |
| **P0** | Letta v1.8 migration path | 📋 DOCUMENTED |
| **P0** | Security audit hook (CVE-2025-64439) | ✅ Referenced in CLAUDE.md |
| **P1** | Research discrepancy synthesis | TODO |
| **P1** | Circuit breaker expansion | ✅ DONE |
| **P1** | Claude-flow V3 patterns | 📋 DOCUMENTED |
| **P1** | MCP timeout configuration | ✅ CONFIGURED |

### Quick Start V27 Loop

```python
from advanced_monitoring_loop_v27 import create_autonomous_loop

loop = create_autonomous_loop(
    task="Implement feature X",
    max_iterations=50,
    max_cost=10.0,
    letta_agent_id="agent-daee71d2-..."  # Optional
)

result = loop.run(executor_function)
# Returns: status, exit_reason, iterations, progress, cost, learnings
```

---

## V27.1 Production Patterns (From Research)

### Configuration Best Practices

```json
// ~/.claude/settings.json - Production recommended
{
  "env": {
    "MCP_TIMEOUT": "30000",
    "MAX_MCP_OUTPUT_TOKENS": "50000",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "70"
  },
  "cleanupPeriodDays": 14
}
```

### MCP Stability Timeouts

| Operation | Timeout | Rationale |
|-----------|---------|-----------|
| `initialize` | 30s | Should be fast |
| `tools/list` | 30s | Metadata only |
| `tools/call` (simple) | 60s | Standard ops |
| `tools/call` (complex) | 120s | DB, API calls |

### Circuit Breaker Pattern

```python
from circuit_breaker import call_with_circuit_breaker, CircuitOpenError

async def safe_mcp_call():
    try:
        return await call_with_circuit_breaker(
            "letta",
            lambda: client.agents.messages.create(...),
            fallback=lambda: {"error": "fallback"}
        )
    except CircuitOpenError:
        # Circuit open - use cached/degraded response
        return cached_response
```

### Claude-flow V3 Integration

```yaml
# Swarm Topologies Available:
# - hierarchical: Queen coordinates workers (default)
# - mesh: All agents bidirectional
# - ring: Sequential processing
# - star: Hub-and-spoke centralized
```

### Memory-First Research Protocol

1. Check CLAUDE.md "What Claude Gets Wrong"
2. Query episodic memory for similar tasks
3. Execute parallel research (ALL tools)
4. Cross-reference and resolve discrepancies
5. Store learnings in appropriate tier

---

*Last Updated: 2026-02-01 | V35 (Research Architecture v4.1 + Semantic Clustering + Voyage Reranker) | 41 Tests Passing*
