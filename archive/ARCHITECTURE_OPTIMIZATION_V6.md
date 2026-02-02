# Claude Code Architecture Optimization v6.0
## Dual-Role MCP Stack Analysis & Refinement

> **Version**: 6.0 - Role-Separated Architecture
> **Date**: 2026-01-16
> **Principle**: Design-Time vs Runtime MCP Separation

---

## Executive Summary

This document provides a comprehensive analysis of the 68 MCP servers in the current ecosystem, identifying overlaps, gaps, and optimizations based on the **critical architectural distinction**:

| Project | Claude's Role | MCP Purpose |
|---------|---------------|-------------|
| **AlphaForge Trading** | External Architect | Design-time tools (NOT in execution path) |
| **State of Witness** | Generative Brain | Runtime controllers (IS the execution path) |

**Key Findings:**
- 12 potentially redundant servers (can be consolidated)
- 8 servers with unclear role assignment
- 6 missing critical capabilities
- Clear separation needed for 68 → ~52 optimized servers

---

## Current MCP Stack Analysis (68 Servers)

### Category Breakdown

| Category | Count | Servers |
|----------|-------|---------|
| Memory & Persistence | 11 | memory, memento, qdrant, qdrant-witness, lancedb, redis, sqlite, knowledge-graph, graphiti, mem0, letta |
| Financial & Trading | 14 | alpaca, questdb, trading-tools, alphavantage, polygon, twelvedata, financial-datasets, fred, quantconnect, backtrader, ibkr, coingecko, timescaledb, qlib |
| Creative & Visualization | 6 | touchdesigner-creative, comfyui-creative, blender-creative, everart, mermaid, c4-model |
| Observability | 9 | grafana, prometheus, loki, opentelemetry, jaeger, datadog, langfuse, phoenix, helicone |
| Development | 9 | github, git, playwright, puppeteer, sequentialthinking, context7, jupyter, e2b, task-master |
| Security | 5 | semgrep, snyk, trivy, sonarqube, codeql |
| DevOps | 4 | kubernetes, docker, aws, sentry |
| Architecture | 2 | likec4, c4-model |
| Reasoning | 3 | sequentialthinking, thinking-protocol, chain-of-thought |
| Multi-Agent | 1 | crewai |
| Automation | 2 | zapier, n8n |
| Productivity | 4 | notion, slack, linear, calculator |
| Search | 4 | brave-search, exa, tavily, fetch |
| Database | 6 | postgres, postgres-pro, sqlite, questdb, timescaledb, lancedb |

---

## Redundancy Analysis

### CONFIRMED OVERLAPS (12 servers)

| Servers | Issue | Recommendation |
|---------|-------|----------------|
| `postgres` + `postgres-pro` | Same database, different features | **Keep postgres-pro** (more features) |
| `playwright` + `puppeteer` | Both browser automation | **Keep playwright** (modern API, better maintenance) |
| `thinking-protocol` + `chain-of-thought` + `sequentialthinking` | Overlapping reasoning | **Keep sequentialthinking** (MCP-native, proven) |
| `zapier` + `n8n` | Both workflow automation | **Keep n8n** (self-hosted, no limits) |
| `memory` + `memento` + `mem0` + `letta` | Multiple memory systems | **Keep mem0 + letta** (advanced features), deprecate basic |
| `alphavantage` + `polygon` + `twelvedata` | Market data overlap | **Keep polygon** (best coverage), alphavantage (free tier) |
| `qdrant` + `lancedb` | Vector DB overlap | **Keep qdrant** (creative) + **lancedb** (trading) |

### MISPLACED SERVERS

| Server | Current Category | Should Be |
|--------|------------------|-----------|
| `c4-model` | Creative | **Architecture** |
| `mermaid` | Creative | **Documentation** |

---

## Role-Based MCP Classification

### TRADING SYSTEM: Design-Time Only

Claude acts as **architect, auditor, researcher** - NOT in execution path.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TRADING MCPs (Design-Time Only)                        │
│                                                                          │
│  Claude's Role: Architecture → Building → Testing → Audit → Deployment   │
│  Live Trading: Runs autonomously (Rust kill switch, Python event loops)  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SECURITY (Code Quality)        │  ARCHITECTURE (Design)                 │
│  ├─ semgrep                     │  ├─ likec4                            │
│  ├─ snyk                        │  ├─ c4-model                          │
│  ├─ trivy                       │  └─ task-master                       │
│  ├─ sonarqube                   │                                        │
│  └─ codeql                      │  RESEARCH (Market Data)               │
│                                 │  ├─ polygon (primary)                  │
│  DEVELOPMENT                    │  ├─ alphavantage (backup)             │
│  ├─ github                      │  ├─ fred (economic)                   │
│  ├─ git                         │  └─ financial-datasets                │
│  ├─ e2b (sandboxed execution)   │                                        │
│  ├─ context7 (docs)             │  BACKTESTING (Strategy Dev)           │
│  └─ jupyter (analysis)          │  ├─ quantconnect                      │
│                                 │  ├─ backtrader                        │
│  OBSERVABILITY (Setup)          │  └─ qlib (RL research)                │
│  ├─ grafana                     │                                        │
│  ├─ prometheus                  │  DATABASE (Historical)                │
│  ├─ langfuse (LLM traces)       │  ├─ questdb                           │
│  └─ datadog (APM)               │  └─ timescaledb                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: These MCPs help Claude *design and improve* the trading system. The actual trading runs WITHOUT Claude in the loop:
- Rust kill switch: 100ns response (no LLM latency)
- Python event loops: Async execution
- Circuit breakers: Automatic triggers

### CREATIVE SYSTEM: Runtime Controllers

Claude IS the generative brain - MCP IS the execution path.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CREATIVE MCPs (Runtime Controllers)                    │
│                                                                          │
│  Claude's Role: Real-time creative control via MCP commands              │
│  Latency: ~100ms acceptable (no financial risk)                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  REAL-TIME CONTROL             │  IMAGE/VIDEO GENERATION                │
│  ├─ touchdesigner-creative ★   │  ├─ comfyui-creative                   │
│  │   (Primary: 60fps output)   │  ├─ blender-creative                   │
│  │   - Node creation           │  └─ everart                            │
│  │   - Shader parameters       │                                        │
│  │   - Particle systems        │  POSE/ML DATA                          │
│  │   - OSC streaming           │  ├─ qdrant-witness ★                   │
│  │                             │  │   (Pose embeddings, archetype)      │
│  EXPLORATION                   │  └─ redis (real-time cache)            │
│  ├─ pyribs (MAP-Elites)        │                                        │
│  └─ Quality-Diversity loops    │  VISUALIZATION                         │
│                                │  └─ mermaid (flow diagrams)            │
│                                                                          │
│  ★ = Primary runtime path                                                │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Creative MCPs are ACTIVE during generation:
```
Claude → touchdesigner-creative MCP → TouchDesigner → 60fps Visual
Claude → qdrant-witness MCP → Archetype Assignment → Particle Behavior
```

### SHARED MCPs (Both Projects)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         SHARED MCPs (Both Projects)                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MEMORY (Cross-Session)        │  SEARCH (Research)                     │
│  ├─ mem0 (hybrid memory)       │  ├─ brave-search                       │
│  ├─ letta (archival memory)    │  ├─ exa (semantic)                     │
│  ├─ knowledge-graph (Neo4j)    │  └─ tavily (news)                      │
│  └─ graphiti (temporal)        │                                        │
│                                │  PRODUCTIVITY                          │
│  REASONING                     │  ├─ notion                             │
│  └─ sequentialthinking         │  ├─ slack                              │
│                                │  └─ linear                             │
│  INFRASTRUCTURE                │                                        │
│  ├─ docker                     │  UTILITIES                             │
│  ├─ kubernetes                 │  ├─ time                               │
│  └─ aws                        │  ├─ calculator                         │
│                                │  └─ fetch                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Gap Analysis: Missing Critical MCPs

### HIGH PRIORITY GAPS

| Gap | Description | Recommendation |
|-----|-------------|----------------|
| **PAIML MCP Agent Toolkit** | Technical debt analysis, code grading (A+ to F) | Add for trading code quality |
| **ComfyUI-TD Bridge** | Real-time pipeline between ComfyUI and TouchDesigner | Custom MCP needed |
| **OSC MCP Server** | Direct OSC protocol control (not via TD) | For non-TD integrations |
| **Audio/MIDI MCP** | Real-time audio control for creative | For audio-reactive visuals |

### MEDIUM PRIORITY GAPS

| Gap | Description | Recommendation |
|-----|-------------|----------------|
| **Replicate MCP** | ML model deployment | For custom ML inference |
| **Modal MCP** | Serverless GPU execution | For heavy ML workloads |
| **Cursor MCP** | IDE integration | Already have editor access |

### RESEARCH FINDINGS (From Compass Artifacts)

**PAIML MCP Agent Toolkit:**
- 487K LOC/sec analysis speed
- Technical debt grading system
- Code complexity metrics
- Perfect for trading system audit

**Flow-Next Patterns:**
- Overnight autonomous coding
- Checkpoint-based progress
- Already implemented via obra/superpowers

---

## Optimized Architecture v6.0

### Consolidated Server List (52 servers)

**REMOVED (16 servers):**
1. `postgres` → Keep `postgres-pro`
2. `puppeteer` → Keep `playwright`
3. `thinking-protocol` → Keep `sequentialthinking`
4. `chain-of-thought` → Keep `sequentialthinking`
5. `zapier` → Keep `n8n`
6. `memory` → Keep `mem0`
7. `memento` → Keep `mem0` + `letta`
8. `alphavantage` → Demote to backup only (keep in config but secondary)
9. `twelvedata` → Demote to backup only
10. `coingecko` → Remove (not trading focus)
11. `jaeger` → Keep `opentelemetry` (superset)
12. `phoenix` → Keep `langfuse` (more mature)
13. `helicone` → Keep `langfuse` (consolidate)
14. `sentry` → Keep `datadog` (APM included)

**Note**: Not actually deleting, just marking as secondary/deprecated in config.

### Role-Tagged Configuration

```json
{
  "mcpServers": {
    "__role_trading_design": "=== TRADING: DESIGN-TIME ONLY ===",

    "__role_creative_runtime": "=== CREATIVE: RUNTIME CONTROLLERS ===",

    "__role_shared": "=== SHARED: BOTH PROJECTS ==="
  }
}
```

### Priority Tiers

| Tier | Servers | Purpose |
|------|---------|---------|
| **P0 - Critical** | touchdesigner-creative, qdrant-witness, polygon, questdb, github, semgrep | Core functionality |
| **P1 - Important** | mem0, letta, sequentialthinking, grafana, context7 | Enhanced capability |
| **P2 - Nice-to-have** | crewai, n8n, likec4, blender-creative | Extended features |
| **P3 - Backup** | alphavantage, twelvedata, datadog | Redundancy |

---

## Implementation Roadmap

### Phase 1: Role Tagging (Immediate)
- Add role comments to mcp_servers.json
- Document which MCPs are design-time vs runtime
- No functional changes

### Phase 2: Consolidation (This Week)
- Deprecate redundant servers in config (comment out)
- Test that essential functionality still works
- Update UNLEASHED.md to v6.0

### Phase 3: Gap Filling (Next Sprint)
- Research and add PAIML toolkit
- Create ComfyUI-TD bridge MCP
- Add OSC direct control MCP

### Phase 4: Optimization (Ongoing)
- Monitor which MCPs are actually used
- Remove unused after 30 days
- Add new capabilities as needed

---

## Memory Update: Role Distinction

The following should be persisted to cross-session memory:

```yaml
claude_role_distinction:
  trading_system:
    project: AlphaForge
    role: "External Architect/Auditor/Researcher"
    in_execution_path: false
    activities:
      - Architecture design (12 layers)
      - Code writing (Python/Rust)
      - Test creation
      - Security audit
      - Deployment config
      - Monitoring setup
    what_runs_autonomously:
      - Rust kill switch (100ns)
      - Python event loops
      - Circuit breakers
      - Order execution
    mcp_usage: "Design-time tools only"

  creative_system:
    project: "State of Witness"
    role: "Generative Brain (IS the system)"
    in_execution_path: true
    activities:
      - Real-time shader parameters
      - MAP-Elites exploration
      - Particle control (2M)
      - Node network creation
      - Archetype-based behavior
    latency_tolerance: "100ms (no financial risk)"
    mcp_usage: "Runtime controllers"
```

---

## Quick Reference Card

### Before Starting a Session

**Trading Session:**
```bash
# Claude = Architect (design-time)
/session-init trading

# MCPs used: security, architecture, research, development
# NOT used: execution MCPs (live trading runs autonomously)
```

**Creative Session:**
```bash
# Claude = Brain (runtime)
/session-init creative

# MCPs used: touchdesigner-creative, qdrant-witness, comfyui
# Claude DIRECTLY controls visual output via MCP
```

### MCP Usage Rules

| Rule | Trading | Creative |
|------|---------|----------|
| Real-time control via MCP | ❌ Never | ✅ Always |
| Financial execution via MCP | ❌ Never | N/A |
| Code generation via MCP | ✅ Design-time | ✅ Shader/node |
| Research via MCP | ✅ Always | ✅ Always |
| Memory via MCP | ✅ Always | ✅ Always |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 6.0 | 2026-01-16 | Role-Separated Architecture: Trading (design-time) vs Creative (runtime) distinction, 16 servers deprecated, priority tiers, gap analysis |

---

## References

- UNLEASHED.md v5.0 (previous version)
- ECOSYSTEM_STATUS.md v5.0
- GAP_ANALYSIS_V5.md
- 8 Compass Artifacts (research source)
