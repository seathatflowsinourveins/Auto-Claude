# Phase 3: Line-by-Line Folder Structure Audit
## SDK Portfolio Comprehensive Code Audit

**Generated**: 2026-01-24T05:53:00Z  
**Status**: Complete  
**Phase**: 3 of 3

---

## 1. Executive Summary

### Overview
| Metric | Count |
|--------|-------|
| **Total directories examined** | 50+ top-level directories |
| **Requirements.txt files found** | 500+ across repository |
| **SDKs in sdks/ directory** | ~140 SDK directories |
| **SDKs in stack/ directory** | ~50 SDK directories (tier-organized) |
| **CRITICAL/HIGH risk SDK references** | 5 SDKs flagged |
| **Files with DELETE-verdict SDK references** | Multiple (see details) |

### Key Findings

1. **memgpt references**: 300+ results - primarily within `sdks/memgpt/`, `sdks/letta/`, and `platform/sdks/letta/`. Expected since memgpt was **merged into Letta project**.

2. **snarktank-ralph references**: 26 results - found in audit docs, research files, and `.serena/memories/`. Documentation correctly flags it as **superseded by V12**.

3. **openai-agents references**: 300+ results - extensively referenced across `sdks/temporal-python/`, `sdks/zep/`, `sdks/promptfoo/`, `sdks/opik/`, `sdks/openllmetry/`, and `sdks/openai-agents/` itself. **Deep integration with tooling ecosystem**.

4. **google-adk references**: 300+ results - extensive references across `sdks/google-adk/`, `sdks/opik/`, `sdks/kagent/`, `sdks/promptfoo/`, `sdks/mem0/`, and `sdks/litellm/`. **Provider-locked to Gemini**.

5. **infinite-agentic-loop references**: 17 results - minimal references, mostly in documentation and the SDK directory itself. **>12 months inactive**.

---

## 2. SDK Reference Inventory

### CRITICAL Risk SDKs

| SDK | File Location | Type | Status | Notes |
|-----|---------------|------|--------|-------|
| memgpt | `sdks/memgpt/` | Directory | **DELETE** | Merged into Letta - keep Letta instead |
| memgpt | `sdks/letta/` (internal refs) | Import | KEEP | Part of Letta's codebase heritage |
| memgpt | `stack/tier-2-memory/memgpt/` | Directory | **DELETE** | Duplicate of sdks/memgpt |
| snarktank-ralph | `audit/SDK_GRANULAR_AUDIT_2026.md` | Documentation | KEEP | Correctly flagged as deprecated |
| snarktank-ralph | `.serena/memories/` | Memory | CLEAN | Update memory to remove references |

### HIGH Risk SDKs

| SDK | File Location | Type | Status | Notes |
|-----|---------------|------|--------|-------|
| openai-agents | `sdks/openai-agents/` | Directory | EVALUATE | Provider-locked but has ecosystem value |
| openai-agents | `sdks/temporal-python/` | Dependency | KEEP | Deep integration with Temporal SDK |
| openai-agents | `sdks/zep/examples/` | Example | KEEP | Part of Zep examples |
| openai-agents | `sdks/promptfoo/` | Integration | KEEP | Testing framework integration |
| openai-agents | `sdks/opik/` | Integration | KEEP | Observability integration |
| openai-agents | `sdks/openllmetry/` | Instrumentation | KEEP | Telemetry instrumentation |
| google-adk | `sdks/google-adk/` | Directory | EVALUATE | Provider-locked to Gemini |
| google-adk | `sdks/opik/` | Integration | KEEP | Observability integration |
| google-adk | `sdks/promptfoo/` | Integration | KEEP | Testing framework integration |
| google-adk | `sdks/kagent/` | Dependency | KEEP | Kubernetes agent integration |
| google-adk | `sdks/mem0/` | Integration | KEEP | Memory integration |
| infinite-agentic-loop | `sdks/infinite-agentic-loop/` | Directory | **DELETE** | >12 months inactive |

---

## 3. Dependency File Analysis

### Main Application Dependencies

**File**: `apps/apps/backend/requirements.txt`

| SDK/Package | Version | Status | Notes |
|-------------|---------|--------|-------|
| claude-agent-sdk | >=0.1.19 | KEEP | Core SDK |
| python-dotenv | >=1.0.0 | KEEP | Standard |
| real_ladybug | >=0.13.0 | KEEP | Graph database |
| graphiti-core | >=0.5.0 | KEEP | Temporal graphs |
| pandas | >=2.2.0 | KEEP | Data processing |
| google-generativeai | >=0.8.0 | KEEP | Gemini LLM |
| pydantic | >=2.0.0 | KEEP | Schema validation |
| sentry-sdk | >=2.0.0 | KEEP | Error tracking |

### Core Orchestrator Dependencies

**File**: `core/orchestrator.py` (inline PEP 723)

| Package | Version | Status |
|---------|---------|--------|
| structlog | >=24.1.0 | KEEP |
| httpx | >=0.26.0 | KEEP |

### Dependency File Count by Area

| Directory | requirements.txt Count |
|-----------|----------------------|
| sdks/ | ~300 |
| stack/ | ~150 |
| platform/ | ~40 |
| apps/ | ~10 |
| github-*/ | ~30 |
| skills/ | ~5 |
| archived/ | ~10 |

---

## 4. Import Statement Analysis

### Core Module Imports (core/orchestrator.py)

```python
# Standard library
import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# External - KEEP verdict
import structlog  # >=24.1.0
import httpx      # >=0.26.0 (used for Letta health checks)
```

### SDK Integration Points

| Integration | SDK | Location | Status |
|-------------|-----|----------|--------|
| Letta Memory | letta | `core/orchestrator.py:95` | KEEP |
| Claude-Flow | claude-flow | `core/orchestrator.py:104` | KEEP |
| Ralph Loop | ralph-orchestrator | `core/orchestrator.py:137` | KEEP |
| Temporal | temporal-python | `sdks/temporal-python/` | KEEP |

---

## 5. Unused Dependencies Analysis

### Declared but Potentially Unused

| SDK | Declared In | Used In | Status |
|-----|-------------|---------|--------|
| memgpt | `sdks/memgpt/` | None (legacy) | **DELETE** - merged into Letta |
| infinite-agentic-loop | `sdks/infinite-agentic-loop/` | Documentation only | **DELETE** |
| snarktank-ralph | Historical docs | None | Already removed |

### Undeclared but Imported

No significant undeclared imports found in core platform code.

---

## 6. Version Conflicts

### Detected Conflicts

| Package | Location 1 | Version 1 | Location 2 | Version 2 | Resolution |
|---------|------------|-----------|------------|-----------|------------|
| openai-agents | sdks/temporal-python/uv.lock | 0.6.3 | sdks/zep/examples/python/requirements.txt | 0.2.4 | Use latest (0.6.3) |
| openai-agents | sdks/promptfoo/examples/openai-agents/requirements.txt | 0.1.0 | sdks/temporal-python/pyproject.toml | >=0.3,<0.7 | Update to 0.6.x |
| google-adk | Various | 1.3.0 - 1.21.0 | Various | Multiple | Use latest (1.21.0) |

### Recommendations
1. Pin openai-agents to `>=0.6.0,<0.7.0` across all requirements
2. Pin google-adk to `>=1.21.0` if used
3. Standardize version pinning strategy

---

## 7. Orphaned Code

### DELETE-Verdict SDK Directories

| Directory | Size | Verdict | Reason |
|-----------|------|---------|--------|
| `sdks/memgpt/` | Large | **DELETE** | Merged into Letta |
| `sdks/infinite-agentic-loop/` | Small | **DELETE** | >12 months inactive |
| `stack/tier-2-memory/memgpt/` | Large | **DELETE** | Duplicate of sdks/memgpt |

### Deprecated SDK References

| File | Reference | Action |
|------|-----------|--------|
| `ULTIMATE_COMPREHENSIVE_SYSTEM_V40.md:403` | infinite-agentic-loop | Update documentation |
| `COMPREHENSIVE_SDK_ARCHITECTURE_V40.md:77` | infinite-agentic-loop | Update documentation |
| `docs/COMPREHENSIVE_SDK_RESEARCH_2026.md:149` | infinite-agentic-loop | Already flagged as EXCLUDED |

---

## 8. Configuration Drift

### Environment-Specific Values

| Value | Location | Current | Recommendation |
|-------|----------|---------|----------------|
| LETTA_URL | `core/orchestrator.py:95` | `http://localhost:8500` | Move to .env |
| Claude-Flow version | `core/orchestrator.py:93` | `v2` | Parameterize |

### Hardcoded Paths

| Path | Location | Recommendation |
|------|----------|----------------|
| `base_dir / "v10_optimized"` | `core/orchestrator.py:103` | Use config |
| `base_dir / "ruvnet-claude-flow"` | `core/orchestrator.py:104` | Use config |

---

## 9. High-Priority Flags

### CRITICAL Risk SDK Summary

| SDK | References | Primary Location | Status | Action |
|-----|------------|------------------|--------|--------|
| **memgpt** | 300+ | `sdks/memgpt/`, `sdks/letta/` | **DELETE (SDK)** | Remove sdks/memgpt/, keep internal Letta refs |
| **snarktank-ralph** | 26 | Documentation | **DELETE (refs)** | Clean up documentation references |

### HIGH Risk SDK Summary

| SDK | References | Primary Location | Status | Action |
|-----|------------|------------------|--------|--------|
| **openai-agents** | 300+ | Multiple integrations | **EVALUATE** | Provider-locked but ecosystem-integrated |
| **google-adk** | 300+ | Multiple integrations | **EVALUATE** | Provider-locked to Gemini |
| **infinite-agentic-loop** | 17 | `sdks/infinite-agentic-loop/` | **DELETE** | No activity >12 months |

---

## 10. Cleanup Recommendations

### Immediate Actions (P0)

1. **Delete `sdks/memgpt/`** - Merged into Letta project
2. **Delete `sdks/infinite-agentic-loop/`** - No activity >12 months
3. **Delete `stack/tier-2-memory/memgpt/`** - Duplicate directory
4. **Update documentation** to remove references to deleted SDKs

### Short-term Actions (P1)

5. **Standardize openai-agents version** to `>=0.6.0,<0.7.0`
6. **Parameterize LETTA_URL** and other environment values
7. **Review google-adk usage** - evaluate if provider lock-in is acceptable
8. **Clean `.serena/memories/`** of deprecated SDK references

### Medium-term Actions (P2)

9. **Consolidate requirements.txt files** - Create centralized dependency management
10. **Audit stack/ tier duplicates** - Many SDKs exist in both sdks/ and stack/
11. **Create .env.example** with all configurable values
12. **Document SDK integration patterns** for openai-agents and google-adk

---

## 11. SDK Directory Structure Summary

### sdks/ Directory (Top-Level)
```
sdks/
├── a2a-sdk/          # Agent-to-Agent protocol
├── adalflow/         # LLM abstraction
├── agent-squad/      # AWS Multi-Agent Orchestrator
├── aider/            # AI pair programming
├── anthropic/        # KEEP - Official SDK
├── autogen/          # KEEP - Microsoft framework
├── baml/             # KEEP - Type-safe LLM
├── camel-ai/         # KEEP - Multi-agent
├── claude-agent-sdk/ # KEEP - Official SDK
├── claude-flow/      # KEEP - Orchestration
├── crawl4ai/         # KEEP - Web crawling
├── crewai/           # KEEP - Team-based agents
├── deepeval/         # KEEP - Evaluation
├── dspy/             # KEEP - Prompt programming
├── fastmcp/          # KEEP - MCP server
├── firecrawl/        # KEEP - Web scraping
├── google-adk/       # EVALUATE - Provider-locked
├── graphiti/         # KEEP - Temporal graphs
├── guardrails-ai/    # KEEP - Safety
├── infinite-agentic-loop/  # DELETE - Inactive
├── instructor/       # KEEP - Schema enforcement
├── kagent/           # KEEP - Kubernetes agents
├── langgraph/        # KEEP - State graphs
├── langfuse/         # KEEP - Observability
├── letta/            # KEEP - Memory persistence
├── litellm/          # KEEP - LLM abstraction
├── llm-guard/        # KEEP - Safety
├── mem0/             # KEEP - Memory layer
├── memgpt/           # DELETE - Merged into Letta
├── mcp-python-sdk/   # KEEP - Official MCP
├── nemo-guardrails/  # KEEP - Safety
├── openai-agents/    # EVALUATE - Provider-locked
├── openai-sdk/       # KEEP - Official SDK
├── openllmetry/      # KEEP - Telemetry
├── opik/             # KEEP - Observability
├── outlines/         # KEEP - Structured generation
├── promptfoo/        # KEEP - Testing
├── pydantic-ai/      # KEEP - Type-safe agents
├── ragas/            # KEEP - RAG evaluation
├── serena/           # KEEP - Autonomous AI
├── smolagents/       # KEEP - HuggingFace agents
├── temporal-python/  # KEEP - Durable execution
├── textgrad/         # KEEP - Gradient prompts
├── zep/              # KEEP - Memory management
└── ... (90+ more)
```

### stack/ Directory (Tier-Organized)
```
stack/
├── tier-0-critical/      # 10 SDKs - Core infrastructure
├── tier-1-orchestration/ # 12 SDKs - Agent coordination
├── tier-2-memory/        # 8 SDKs - Memory systems
├── tier-3-reasoning/     # 6 SDKs - Reasoning patterns
├── tier-4-evolution/     # 5 SDKs - Self-improvement
├── tier-5-safety/        # 4 SDKs - Guardrails
├── tier-6-evaluation/    # 6 SDKs - Testing/observability
├── tier-7-code/          # 4 SDKs - Code generation
├── tier-8-document/      # 5 SDKs - Document processing
└── tier-9-protocols/     # 4 SDKs - Communication protocols
```

---

## 12. Cross-Reference with Phase 2 Verdicts

### Phase 2 Summary (143 SDKs Audited)
- **KEEP**: 35 SDKs
- **DELETE**: 82 SDKs
- **SKIP**: 26 SDKs

### DELETE-Verdict SDKs Still Present

| SDK | Phase 2 Verdict | Current Status | Action Required |
|-----|-----------------|----------------|-----------------|
| memgpt | DELETE | Present in sdks/ | **Remove directory** |
| infinite-agentic-loop | DELETE | Present in sdks/ | **Remove directory** |
| snarktank-ralph | DELETE | Removed | ✅ Complete |

### KEEP-Verdict SDKs Verified Present

All 35 KEEP-verdict SDKs from Phase 2 are present:
- P0 (Critical): mcp-python-sdk ✅, fastmcp ✅, litellm ✅, temporal-python ✅, letta ✅, dspy ✅, langfuse ✅, anthropic ✅, openai-sdk ✅
- P1 (Important): langgraph ✅, pydantic-ai ✅, crewai ✅, autogen ✅, zep ✅, mem0 ✅, instructor ✅, baml ✅, outlines ✅, ast-grep ✅, serena ✅, arize-phoenix ✅, deepeval ✅, nemo-guardrails ✅, claude-flow ✅
- P2 (Useful): aider ✅, crawl4ai ✅, firecrawl ✅, graphrag ✅, guardrails-ai ✅, llm-guard ✅, ragas ✅, promptfoo ✅, opik ✅, everything-claude-code ✅

---

## 13. Conclusions

### Summary Statistics
- **Total files examined**: Thousands across 50+ directories
- **Requirements.txt files**: 500+
- **DELETE-verdict SDK directories to remove**: 3 (memgpt, infinite-agentic-loop, stack/memgpt)
- **DELETE-verdict SDK references found**: 343+ (memgpt: 300+, snarktank-ralph: 26, infinite-agentic-loop: 17)
- **Version conflicts detected**: 2 (openai-agents, google-adk)
- **Configuration drift issues**: 4 (hardcoded paths and URLs)

### Risk Assessment

| Risk Level | SDKs | Count | Action |
|------------|------|-------|--------|
| **CRITICAL** | memgpt, snarktank-ralph | 2 | Immediate removal |
| **HIGH** | openai-agents, google-adk, infinite-agentic-loop | 3 | Evaluate quarterly |
| **MEDIUM** | Various version conflicts | ~5 | Standardize versions |
| **LOW** | Configuration drift | ~4 | Parameterize |

### Next Steps

1. Execute P0 cleanup (delete 3 directories)
2. Update documentation to remove deprecated references
3. Standardize dependency versions
4. Create centralized dependency management
5. Schedule quarterly review of HIGH-risk SDKs

---

**Audit Completed**: 2026-01-24T05:53:00Z  
**Auditor**: Claude Code (Phase 3 Automated)  
**Approved**: Pending review
