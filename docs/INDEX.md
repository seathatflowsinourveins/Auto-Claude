# UNLEASH Platform - Documentation Index

> **Updated**: 2026-02-04 | **Version**: V66

---

## Quick Start

The UNLEASH platform lives in `/platform/`. Configuration is in `CLAUDE.md` at the project root. Modular rules auto-load from `.claude/rules/`.

```bash
# Run tests (from outside project dir to avoid platform module shadowing)
cd C:\Users\42 && uv run --no-project --with pytest,pytest-asyncio,structlog,httpx,pydantic,rich,aiohttp,numpy python -m pytest "Z:/insider/AUTO CLAUDE/unleash/platform/tests/" -q -k "not hnsw"

# Run Ralph Loop
uv run --no-project --with structlog,httpx,pydantic,rich,aiohttp,python-dotenv,pyyaml python platform/scripts/ralph_loop.py
```

---

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `platform/core/` | Core modules (orchestration, RAG, memory) |
| `platform/adapters/` | SDK adapters (42 hardened + 9 infrastructure) |
| `platform/tests/` | Test suite |
| `platform/scripts/` | Operational scripts (ralph_loop, ecosystem_orchestrator) |
| `platform/data/` | Runtime data (memory blocks, sessions, insights, reports) |
| `platform/api/` | API endpoints |
| `docs/` | Active documentation |
| `docs/essential/` | Core reference docs |
| `docs/gap-resolution/` | Gap analysis and resolution guides |
| `docs/archived/` | Historical docs (architecture, prompts, SDK research, versions, voyage-ai) |
| `.claude/rules/` | Modular project rules (auto-loaded by Claude Code) |
| `.claude/agents/` | Custom agent definitions (explorer, code-reviewer, researcher) |
| `research/iterations/` | Research iteration scripts (155+ topics) |
| `scripts/` | Utility scripts |

---

## Active Documentation

### Architecture & Configuration

| File | Description |
|------|-------------|
| `CLAUDE.md` | Project instructions and configuration (V65) |
| `docs/ULTIMATE_UNLEASHED_ARCHITECTURE.md` | Platform architecture overview |
| `docs/CLAUDE_CODE_CLI_ARCHITECTURE_V2.md` | CLI architecture patterns |
| `docs/CLAUDE_CODE_ADVANCED_PATTERNS_2026.md` | Advanced Claude Code patterns |
| `docs/ADR-027-MCP-CONFIGURATION-OPTIMIZATION.md` | MCP server optimization (23 non-essential identified) |
| `docs/ADR-028-MCP-FLOW-ARCHITECTURE-OPTIMIZATION.md` | MCP flow architecture |

### Research (2026)

| File | Description |
|------|-------------|
| `docs/RESEARCH_SUMMARY_2026.md` | 2026 ecosystem research (Claude Flow V3, Anthropic SDK, MCP, Letta, RAPTOR/CRAG) |
| `docs/V66_RESEARCH_SUMMARY.md` | V66 feature research (Jina Reranker v3, RAPTOR, Batches API, Mem0 Graph) |
| `docs/MULTI_AGENT_ORCHESTRATION_RESEARCH_2026.md` | Multi-agent patterns, RAG, MCP findings |
| `docs/RESEARCH_STACK_2026.md` | Research tool stack details |
| `docs/RAG_ARCHITECTURES_2026.md` | Battle-tested RAG patterns and benchmarks |
| `docs/CLAUDE_FLOW_V3_RESEARCH.md` | Claude Flow V3 swarm patterns |
| `docs/ANTHROPIC_AGENT_SDK_2026_DEEP_DIVE.md` | Anthropic Agent SDK analysis |
| `docs/ANTHROPIC_ECOSYSTEM_2026_RESEARCH.md` | Anthropic ecosystem research |
| `docs/RESEARCH_MISSION_2026_02_04.md` | Current research mission |
| `docs/MEMORY_ARCHITECTURES_2026.md` | Memory system patterns |
| `docs/ARCHITECTURE_IMPROVEMENTS_2026.md` | Architecture improvement plans |
| `docs/ARCHITECTURE_GAP_ANALYSIS_2026.md` | Gap analysis |

### SDK & Integration

| File | Description |
|------|-------------|
| `docs/LATEST_SDK_ANALYSIS.md` | Current SDK version analysis |
| `docs/DEFINITIVE_SDK_REFERENCE_2026.md` | SDK reference guide |
| `docs/SDK_INTEGRATION_GUIDE.md` | Integration guide |
| `docs/LETTA_INTEGRATION_GUIDE.md` | Letta memory integration |
| `docs/AGENT_TO_AGENT_PATTERNS.md` | A2A communication patterns |
| `docs/AGENTIC_WORKFLOW_PATTERNS.md` | Agentic workflow patterns |

### Memory & RAG

| File | Description |
|------|-------------|
| `docs/ADVANCED_MEMORY_PATTERNS.md` | Memory patterns guide |
| `docs/CROSS_SESSION_MEMORY.md` | Cross-session memory |
| `docs/GRAPHRAG_ANALYSIS.md` | GraphRAG analysis |
| `docs/EMBEDDING_MODEL_COMPARISON.md` | Embedding model comparison |
| `docs/CONTEXT7_COMPLETE_GUIDE.md` | Context7 usage guide |
| `docs/CONTEXT7_BEST_PRACTICES.md` | Context7 best practices |

### MCP & Tools

| File | Description |
|------|-------------|
| `docs/MCP_TOOLS_MANIFEST.md` | MCP tools manifest |
| `docs/MCP_FLOW_OPTIMIZATION_SUMMARY.md` | MCP optimization summary |

### Testing & Validation

| File | Description |
|------|-------------|
| `docs/TEST_PLAN_COMPREHENSIVE.md` | Comprehensive test plan |
| `docs/API_VALIDATION_RESULTS.md` | API validation results |
| `docs/INTEGRATION_TEST_RESULTS.md` | Integration test results |
| `docs/MEMORY_BATTLE_TEST_RESULTS.md` | Memory battle test results |
| `docs/CONTEXT7_MCP_TEST_RESULTS.md` | Context7 MCP test results |
| `docs/IMPORT_VALIDATION_REPORT.md` | Import validation report |

### Audits & Reports

| File | Description |
|------|-------------|
| `docs/AUDIT_REPORT_V64_EXHAUSTIVE.md` | V64 exhaustive audit |
| `docs/AUDIT_REPORT_V58.md` | V58 audit report |
| `docs/MEMORY_SYSTEM_AUDIT_REPORT.md` | Memory system audit |
| `docs/PERFORMANCE_BOTTLENECK_ANALYSIS.md` | Performance analysis |
| `docs/ANTHROPIC_SDK_IMPROVEMENT_REPORT.md` | SDK improvement report |

### Swarm & Agents

| File | Description |
|------|-------------|
| `docs/SWARM_PATTERNS_CATALOG.md` | Swarm intelligence patterns |
| `docs/AGENTS.md` | Agent definitions |
| `docs/AGENT_SELECTION_GUIDE.md` | Agent selection guide |

### Other

| File | Description |
|------|-------------|
| `docs/GOALS_TRACKING.md` | Goals tracking |
| `docs/INTEGRATION_ROADMAP.md` | Integration roadmap |
| `docs/IMPLEMENTATION_ROADMAP_2026.md` | Implementation roadmap |
| `docs/QUICK_START_GUIDE.md` | Quick start guide |
| `docs/UNIFIED_CLAUDE_CONFIG.md` | Unified configuration |
| `docs/INTEGRATION_STATUS.md` | Integration status |

---

## Gap Resolution Guides (`docs/gap-resolution/`)

| File | Gap | Status |
|------|-----|--------|
| `00-INDEX.md` | Index of all gaps | -- |
| `01-SILENT-API-FAILURES.md` | Gap01: Silent API failures | RESOLVED |
| `02-FINDINGS-QUALITY.md` | Gap02: Findings quality | RESOLVED |
| `03-VECTOR-PERSISTENCE.md` | Gap03: Vector persistence | RESOLVED |
| `04-STATS-INFLATION.md` | Gap04: Stats inflation | RESOLVED |
| `05-SCRIPT-DUPLICATION.md` | Gap05: Script duplication | PARTIAL |
| `06-SYNTHESIS-PIPELINE.md` | Gap06: Synthesis pipeline | PARTIAL+ |
| `07-DEDUPLICATION.md` | Gap07: Deduplication | RESOLVED |
| `08-ERROR-HANDLING.md` | Gap08: Error handling | RESOLVED |
| `09-ADAPTIVE-DISCOVERY.md` | Gap09: Adaptive discovery | PARTIAL+ |
| `10-CROSS-TOPIC-SYNTHESIS.md` | Gap10: Cross-topic synthesis | PARTIAL+ |
| `11-EVALUATION-INTEGRATION.md` | Gap11: Evaluation integration | PARTIAL+ |
| `12-UNIFIED-ARCHITECTURE.md` | Gap12: Unified architecture | BLUEPRINT |
| `13-FIRECRAWL-ADAPTER-GAPS.md` | Gap13: Firecrawl v2 | RESOLVED |

Research supplements in `docs/gap-resolution/`:
- `ANTHROPIC_AGENT_SDK_PATTERNS.md`, `ANTHROPIC_SDK_2026_DEEP_RESEARCH_PART1.md`
- `BATTLE_TESTED_PATTERNS_2026.md`, `BATTLE_TESTED_PATTERNS_2026_PART1.md`
- `CLAUDE_FLOW_V3_PATTERNS.md`, `CLAUDE_FLOW_V3_SWARM_PATTERNS_2026.md`
- `CONTEXT7_JINA_GAP_ANALYSIS.md`, `LETTA_OPIK_MEMORY_RESEARCH_2026.md`
- `OPIK_EVALUATION_PATTERNS.md`, `RESEARCH_2026_SUMMARY.md`, `RESEARCH_FINDINGS_2026_FEB_PART1.md`

---

## Essential Reference (`docs/essential/`)

| File | Description |
|------|-------------|
| `UNLEASHED_PATTERNS.md` | Production patterns |
| `ECOSYSTEM_STATUS.md` | System status dashboard |
| `HONEST_AUDIT.md` | What works vs needs setup |

---

## Archived Documentation (`docs/archived/`)

Historical docs organized by category:
- `architecture/` - Old architecture docs (2 files)
- `prompts/` - Phase prompt files (16 files)
- `sdk-research-old/` - Old SDK research (7 files)
- `versions/` - Version-specific docs (4 files)
- `voyage-ai/` - Voyage AI plan iterations (8 files)
- `legacy/` - Pre-V40 audit and planning docs (30+ files)

---

## Modular Rules (`.claude/rules/`)

Auto-loaded by Claude Code based on file path context:
- `memory-patterns.md` - Memory block format, Letta integration
- `research-tools.md` - Research API priority and error handling
- `platform-architecture.md` - Module layout, shadowing warnings
- `testing.md` - Test commands, known issues
- `adapters.md` - SDK adapter standards (path-scoped: `platform/adapters/`)
- `research-iterations.md` - Research script standards (path-scoped: `research/iterations/`)
