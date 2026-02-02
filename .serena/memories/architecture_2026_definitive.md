# Unleash Architecture 2026 - Definitive Reference

> **Version**: FINAL | **Date**: 2026-01-25 | **Status**: PRODUCTION

## IMPORTANT: This Replaces V30-V40 Documentation

The V30-V40 architecture documents (28+ layers, 185-220+ SDKs) were WRONG. 
The 2026-01-24 cleanup established the CORRECT architecture:

**8 Layers, 34 SDKs, ~4.5GB** (down from 154 SDKs, 8GB)

## 8-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNLEASH SDK ARCHITECTURE                            │
│                              34 PRODUCTION SDKs                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  L8: KNOWLEDGE       │ graphrag, pyribs                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  L7: PROCESSING      │ aider, ast-grep, crawl4ai, firecrawl               │
├─────────────────────────────────────────────────────────────────────────────┤
│  L6: SAFETY          │ guardrails-ai, llm-guard, nemo-guardrails          │
├─────────────────────────────────────────────────────────────────────────────┤
│  L5: OBSERVABILITY   │ langfuse, opik, arize-phoenix, deepeval, ragas,    │
│                      │ promptfoo                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  L4: REASONING       │ dspy, serena                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  L3: STRUCTURED      │ instructor, baml, outlines, pydantic-ai            │
│     OUTPUT           │                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  L2: MEMORY          │ letta, zep, mem0                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  L1: ORCHESTRATION   │ temporal-python, langgraph, claude-flow, crewai,   │
│                      │ autogen                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  L0: PROTOCOL        │ mcp-python-sdk, fastmcp, litellm, anthropic,       │
│     GATEWAY          │ openai-sdk                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## SDK Priority Classification

### P0 BACKBONE (9 SDKs) - Always Loaded
```
mcp-python-sdk, fastmcp, litellm, temporal-python, 
letta, dspy, langfuse, anthropic, openai-sdk
```

### P1 CORE (15 SDKs) - Primary Capabilities
```
langgraph, pydantic-ai, crewai, autogen, zep, mem0,
instructor, baml, outlines, ast-grep, serena,
arize-phoenix, deepeval, nemo-guardrails, claude-flow
```

### P2 ADVANCED (10 SDKs) - Specialized Use
```
aider, crawl4ai, firecrawl, graphrag, guardrails-ai,
llm-guard, ragas, promptfoo, opik, pyribs
```

## Security Warning

**LangGraph**: Upgrade to 3.0+ immediately - CVE-2025-64439 (RCE via JsonPlusSerializer)

## Key Reference Files

| File | Purpose |
|------|---------|
| `Z:/insider/AUTO CLAUDE/unleash/audit/SDK_KEEP_ARCHITECTURE_2026.md` | Detailed layer definitions |
| `Z:/insider/AUTO CLAUDE/unleash/sdks/` | 34 production SDKs |
| Serena memory: `sdk_cleanup_2026_01_24` | Cleanup decisions |

---

*Updated: 2026-01-25 | Status: DEFINITIVE REFERENCE*
