# SDK Optimization Analysis 2026

> **Generated**: 2026-01-28
> **Purpose**: SDK prioritization and archival recommendations
> **Status**: APPROVED FOR IMPLEMENTATION

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| P0 Critical | 8 | Keep (core platform) |
| P1 Important | 10 | Keep (high value) |
| P2 Specialized | 10 | Keep (domain-specific) |
| P3 Archive | 7 | Archive (redundant) |
| **Total** | **35 → 28** | **-20% maintenance** |

---

## P0 Critical SDKs (8) - NEVER ARCHIVE

| SDK | Purpose | Benchmark |
|-----|---------|-----------|
| **anthropic/** | Claude Agent SDK | 89.0 |
| **claude-flow/** | V3 orchestration (53K files) | Core |
| **letta/** | Agent memory (74% LoCoMo) | 82.3 |
| **opik/** | Observability (7-14x faster) | 72.5 |
| **mcp-python-sdk/** | MCP protocol | 89.2 |
| **instructor/** | Structured output | 85.9 |
| **litellm/** | Multi-provider (100+) | 85.0 |
| **langgraph/** | Stateful workflows | 90.7 |

---

## P1 Important SDKs (10)

| SDK | Purpose | Keep Reason |
|-----|---------|-------------|
| **dspy/** | Prompt optimization | MIPROv2 optimizer |
| **pydantic-ai/** | Type-safe agents | Pydantic team |
| **pyribs/** | Quality-Diversity | WITNESS/TRADING |
| **promptfoo/** | Prompt testing | 51K+ users |
| **mem0/** | Simple memory | Letta alternative |
| **ragas/** | RAG evaluation | SOTA metrics |
| **deepeval/** | CI/CD testing | Pytest-style |
| **guardrails-ai/** | Validation | 20x accuracy |
| **crawl4ai/** | Web scraping | Open-source, fastest |
| **ast-grep/** | Code refactoring | Rust-based, fast |

---

## P2 Specialized SDKs (10)

| SDK | Use Case | When to Use |
|-----|----------|-------------|
| **temporal-python/** | Durable workflows | TRADING reliability |
| **graphrag/** | Knowledge graphs | Multi-hop reasoning |
| **outlines/** | Local models | Self-hosted only |
| **baml/** | Contract-first | Distributed teams |
| **langfuse/** | Open observability | MIT license needed |
| **zep/** | Enterprise memory | SOC 2 compliance |
| **llm-guard/** | Lightweight safety | Quick checks |
| **nemo-guardrails/** | NVIDIA GPU | GPU-accelerated |
| **arize-phoenix/** | Agent debugging | RAG traces |

---

## P3 Archive Candidates (7) - REDUNDANT

| SDK | Replace With | Reason |
|-----|--------------|--------|
| **openai-sdk/** | LiteLLM | Provider-specific unnecessary |
| **fastmcp/** | mcp-python-sdk | Official SDK preferred |
| **firecrawl/** | crawl4ai | Paid vs free+faster |
| **crewai/** | claude-flow | 2.2x slower |
| **autogen/** | claude-flow | Azure lock-in |
| **aider/** | Claude Code CLI | Terminal-only overlap |
| **serena/** | L0/L1 stack | Redundant with pyright |

---

## Project Stack Requirements

### UNLEASH (Meta-Platform)
```
Required: All P0 + dspy, promptfoo, pyribs
Optional: All P2 for docs/examples
```

### WITNESS (Creative AI)
```
Required: P0 + pyribs, crawl4ai, outlines
Optional: graphrag, temporal-python
```

### TRADING (AlphaForge)
```
Required: P0 + pyribs, temporal-python
Optional: ragas, deepeval
```

---

## Storage Analysis

| SDK | Files | Status |
|-----|-------|--------|
| claude-flow | 53K+ | Keep (core) |
| mcp-ecosystem | 18K+ | Keep (protocols) |
| serena | 9K+ | ARCHIVE |
| opik | 7K+ | Keep (observability) |
| litellm | 6K+ | Keep (providers) |
| baml | 5K+ | Keep (contracts) |

---

## Implementation Status

- [x] Analysis complete
- [ ] Archive redundant SDKs
- [ ] Update SDK_INDEX.md
- [ ] Verify no broken imports

---

*Research validated with 2026 sources*
*See DEEP_RESEARCH_SYNTHESIS_2026.md for methodology*
