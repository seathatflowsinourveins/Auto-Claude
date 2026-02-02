# V33 Unified Architecture - Complete Integration Reference

> **Version**: 33.0 | **Date**: 2026-01-25 | **Status**: Production Ready

## Executive Summary

The V33 Architecture provides a **9-layer (L0-L8) unified SDK ecosystem** with ~35 SDKs.
This is the CORRECT architecture following the 2026-01-24 cleanup (154→34 SDKs).

**Key Principle**: Explicit Failures > Silent Stubs

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         V33 UNIFIED ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L8: KNOWLEDGE LAYER                                                  │   │
│  │     GraphRAG, PyRibs (MAP-Elites QD)                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L7: PROCESSING LAYER                                                 │   │
│  │     Crawl4AI, Firecrawl, Aider, AST-Grep                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L6: SAFETY LAYER                                                     │   │
│  │     Guardrails-AI, LLM-Guard, NeMo-Guardrails                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L5: OBSERVABILITY LAYER                                              │   │
│  │     Langfuse, Phoenix, Opik, DeepEval, Ragas, Logfire, OpenTelemetry│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L4: REASONING LAYER                                                  │   │
│  │     DSPy, Serena                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L3: STRUCTURED OUTPUT LAYER                                          │   │
│  │     Instructor, BAML, Outlines, Pydantic-AI                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L2: MEMORY LAYER                                                     │   │
│  │     Letta, Zep, Mem0, CrossSession                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L1: ORCHESTRATION LAYER                                              │   │
│  │     Temporal, LangGraph, Claude-Flow, CrewAI, AutoGen               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L0: PROTOCOL LAYER                                                   │   │
│  │     LiteLLM (100+ providers), FastMCP, Anthropic SDK, OpenAI SDK    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Layer Details

### L0: Protocol Layer (4 SDKs)
Foundation for all LLM communication.

```python
from core.llm_gateway import LLMGateway, quick_complete
from core.mcp_server import create_mcp_server, FASTMCP_AVAILABLE

# Quick completion
response = await quick_complete("Hello", model="claude-sonnet-4-5-20250929")

# MCP server creation
server = create_mcp_server("my-server")
```

### L1: Orchestration Layer (5 Frameworks)
Unified multi-agent orchestration.

```python
from core.orchestration import UnifiedOrchestrator, get_available_frameworks

orchestrator = UnifiedOrchestrator()
result = await orchestrator.run_workflow(
    name="data-pipeline",
    steps=["extract", "transform", "load"],
    prefer="langgraph"  # Falls back to available framework
)

# Available: temporal, langgraph, claude_flow, crewai, autogen
print(get_available_frameworks())
```

### L2: Memory Layer (4 Providers)
Cross-session persistent memory.

```python
from core.memory import UnifiedMemory, get_available_memory_providers

memory = UnifiedMemory()
await memory.save("key_insight", {"topic": "architecture", "data": "..."})
results = await memory.search("architecture patterns", limit=10)

# Available: letta, zep, mem0, cross_session
print(get_available_memory_providers())
```

### L3: Structured Output Layer (4 SDKs)
Type-safe LLM outputs.

```python
from core.structured import InstructorClient, INSTRUCTOR_AVAILABLE
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    confidence: float

if INSTRUCTOR_AVAILABLE:
    client = InstructorClient()
    result = await client.extract(Analysis, "Analyze this text...")
```

### L4: Reasoning Layer (2 SDKs)
Advanced reasoning and optimization.

```python
from core.reasoning import get_dspy_lm, DSPY_AVAILABLE

if DSPY_AVAILABLE:
    lm = get_dspy_lm(model="claude-sonnet-4-5-20250929")
    # Use DSPy modules for optimized prompting
```

### L5: Observability Layer (7 SDKs)
Complete LLM observability stack.

```python
from core.observability import (
    get_langfuse_tracer,
    get_opik_client,
    SDKNotAvailableError,
)

try:
    tracer = get_langfuse_tracer()
    tracer.trace("my-workflow", input="...", output="...")
except SDKNotAvailableError as e:
    print(f"Install: {e.install_cmd}")
```

### L6: Safety Layer (3 SDKs)
Input/output guardrails.

```python
from core.safety import get_guardrails_guard, GUARDRAILS_AVAILABLE

if GUARDRAILS_AVAILABLE:
    guard = get_guardrails_guard()
    validated = await guard.validate(user_input)
```

### L7: Processing Layer (4 SDKs)
Code and web processing tools.

```python
from core.processing import get_crawl4ai_crawler, get_aider_coder

# Web crawling
crawler = get_crawl4ai_crawler()
content = await crawler.crawl("https://example.com")

# AI-assisted code editing
coder = get_aider_coder()
await coder.edit("fix the bug in auth.py")
```

### L8: Knowledge Layer (2 SDKs)
Graph-based RAG and quality-diversity optimization.

```python
from core.knowledge import get_graphrag_client, get_pyribs_archive

# Graph RAG for complex queries
graphrag = get_graphrag_client()
answer = await graphrag.query("How does the auth system work?")

# MAP-Elites for creative exploration
archive = get_pyribs_archive(dims=(20, 20))
archive.update(candidate)
```

## Unified V33 Accessor

```python
from core import V33, get_v33_status, print_v33_status

# Get system status
status = get_v33_status()
print(f"Available: {status.available_layers}/9 layers, {status.available_sdks}/{status.total_sdks} SDKs")

# Pretty print
print_v33_status()

# Unified access
v33 = V33()
if v33.safety.available:
    guard = v33.safety.get_guardrails()
```

## Cross-Session Memory Access

Memory persists across sessions via multiple backends:

| Backend | Access Method | Use Case |
|---------|---------------|----------|
| Serena | `mcp__serena__read_memory` | Project-specific knowledge |
| Episodic | `mcp__episodic_memory__search` | Conversation history |
| Claude-Mem | `mcp__claude-mem__search` | Observations & learnings |
| Letta | Session hooks | Sleeptime consolidation |

## Key Files

| File | Purpose |
|------|---------|
| `core/__init__.py` | V33 unified exports (775 lines) |
| `core/llm_gateway.py` | LiteLLM integration (424 lines) |
| `core/orchestration/__init__.py` | 5-framework orchestration (900+ lines) |
| `core/memory/__init__.py` | 4-backend memory (820+ lines) |
| `core/observability/__init__.py` | 7 SDK integrations (477 lines) |
| `core/safety/__init__.py` | 3 guardrail systems (854 lines) |

## Anthropic Agent Pattern Mapping

The V33 architecture maps to Anthropic's official agent loop:

```
ANTHROPIC PATTERN          V33 LAYER
───────────────────────    ─────────────
1. Gather Context    →     L0-L2 (Protocol, Orchestration, Memory)
2. Take Action       →     L3-L4 (Structured Output, Reasoning)
3. Verify Work       →     L5-L6 (Observability, Safety)
4. Repeat            →     L7-L8 (Processing, Knowledge)
```

## Important Notes

1. **V40 files are LEGACY** - The repo contains V30-V40 files from before cleanup.
   The correct architecture is V33 with 8 layers and ~35 SDKs.

2. **Explicit failures** - Every SDK getter raises `SDKNotAvailableError` if not installed.
   No silent fallbacks or stub patterns.

3. **Compatibility layers** - Phase 12 added compat layers for Python 3.14:
   - `CrewCompat` (LangGraph-based)
   - `OutlinesCompat` (JSON Schema + Regex)
   - `AiderCompat` (Git-aware code modification)
   - `AgentLiteCompat` (Lightweight ReAct)

---
*Created: 2026-01-25 | Source: Deep research integration*
