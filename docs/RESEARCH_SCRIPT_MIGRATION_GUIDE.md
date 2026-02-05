# Research Script Migration Guide

Migration guide for converting legacy research iteration scripts to use `base_executor.py`.

## Overview

The `research/iterations/` directory contains 160 research scripts. As of V65, 10 have been migrated to use `base_executor.py`, with 150 remaining. This guide covers:

1. Why migrate
2. Migration patterns
3. Step-by-step instructions
4. Verification checklist

## Why Migrate?

Legacy scripts duplicate ~170 lines of boilerplate per file. Migrating to `base_executor.py` provides:

| Feature | Legacy | Migrated |
|---------|--------|----------|
| Lines of code | ~200 | ~60 |
| Gap02: Quality filtering | No | Yes |
| Gap04: Honest stats | No | Yes (recomputed from actual data) |
| Gap06: Synthesis | No | Yes (40+ claim indicators) |
| Gap07: 3-layer dedup | No | Yes (URL, content-hash, vector) |
| Gap09: Fallback broadening | No | Yes |
| Gap11: Quality scoring | No | Yes (dashboard + gate retry) |
| L4: Iterative discovery | No | Yes (multi-round) |
| Cross-topic synthesis | No | Yes |
| Contradiction detection | No | Yes |

## Migration Patterns

### Pattern 1: Full Migration (Recommended)

Replace the entire file with base_executor imports. Best for scripts with no custom logic.

**Before (200 lines):**
```python
"""LLMOPS ITERATIONS - LLM Operations..."""

import asyncio
import os
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')
logger = logging.getLogger(__name__)

LLMOPS_TOPICS = [...]

@dataclass
class LLMOpsResult:
    topic: str
    area: str
    sources: list
    findings: list
    vectors: int
    latency: float

class LLMOpsExecutor:
    def __init__(self):
        # ... 15 lines ...

    async def initialize(self):
        # ... 10 lines ...

    async def research(self, topic, area):
        # ... 15 lines ...

    async def _exa(self, topic):
        # ... 15 lines ...

    async def _tavily(self, client, topic):
        # ... 15 lines ...

    async def _perplexity(self, client, topic, area):
        # ... 15 lines ...

    async def _embed(self, client, texts):
        # ... 15 lines ...

    async def run_iteration(self, topic_data, index):
        # ... 15 lines ...

async def main():
    # ... 40 lines ...

if __name__ == "__main__":
    asyncio.run(main())
```

**After (60 lines):**
```python
"""
LLMOPS ITERATIONS - LLM Operations & Deployment
================================================
LLMOps, LLM deployment, LLM infrastructure

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


LLMOPS_TOPICS = [
    {"topic": "LangSmith: LLM observability", "area": "platforms"},
    # ... keep existing topics ...
]


class LLMOpsExecutor(BaseResearchExecutor):
    """Custom executor with LLMOps-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"LLMOps LLM operations best practices: {topic}"


if __name__ == "__main__":
    run_research(
        "llmops",
        "LLMOPS ITERATIONS",
        LLMOPS_TOPICS,
        executor_class=LLMOpsExecutor,
    )
```

### Pattern 2: Adapter Bridge (Incremental)

Use `LegacyScriptAdapter` to wrap existing code without rewriting. Best for scripts with custom logic you want to preserve.

```python
from research.iterations.legacy_adapter import LegacyScriptAdapter

adapter = LegacyScriptAdapter(__file__)

async def main():
    await adapter.initialize()
    topics = adapter.detect_topics() or EXISTING_TOPICS

    for i, topic_data in enumerate(topics, 1):
        await adapter.run_iteration(topic_data, i)

    adapter.save_results()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Pattern 3: Hybrid (Custom Methods)

Inherit from `BaseResearchExecutor` but override specific methods for custom behavior.

```python
from base_executor import BaseResearchExecutor, run_research

class CustomExecutor(BaseResearchExecutor):
    """Custom executor with specialized behavior."""

    # Optional: Custom Perplexity prompt
    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Custom domain research: {topic}"

    # Optional: Custom synthesis (e.g., LLM-based instead of heuristic)
    def synthesize(self, topic, sources, findings):
        # Call parent for heuristic synthesis
        synthesized = super().synthesize(topic, sources, findings)
        # Add custom LLM synthesis here if needed
        return synthesized

    # Optional: Enable RAG evaluation
    enable_rag_evaluation = True
    quality_threshold = 0.4

if __name__ == "__main__":
    run_research("custom", "CUSTOM ITERATIONS", TOPICS, executor_class=CustomExecutor)
```

## Step-by-Step Migration

### Step 1: Identify the TOPICS variable

Find the list of topic dictionaries:
```python
LLMOPS_TOPICS = [
    {"topic": "...", "area": "..."},
    ...
]
```

### Step 2: Identify custom Perplexity prompt

Find the prompt in `_perplexity()`:
```python
f"LLMOps LLM operations: {topic}"
```

### Step 3: Create the new file

```python
"""
SCRIPT_NAME ITERATIONS - Description
====================================
Brief description

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering
- Gap04: Honest stats
- Gap06: Synthesis
- Gap07: 3-layer dedup
- Gap09: Fallback broadening
- Gap11: Quality scoring
"""

from base_executor import BaseResearchExecutor, run_research


TOPICS = [
    # Copy existing topics
]


class CustomExecutor(BaseResearchExecutor):
    """Custom executor with domain-specific prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Your custom prompt: {topic}"


if __name__ == "__main__":
    run_research(
        "collection_name",  # Qdrant collection name
        "TITLE FOR OUTPUT",
        TOPICS,
        executor_class=CustomExecutor,
    )
```

### Step 4: Verify

```bash
# Compile check
uv run --no-project python -m py_compile research/iterations/your_script.py

# Optional: Run to verify functionality
uv run --no-project --with exa_py,qdrant-client,httpx,python-dotenv python research/iterations/your_script.py
```

## Migrated Scripts (10/160)

| Script | Category | Collection |
|--------|----------|------------|
| `llmops_iterations.py` | LLMOps | llmops |
| `advanced_rag_iterations.py` | RAG | rag |
| `agent_memory_iterations.py` | Memory | agent_memory |
| `llm_agents_iterations.py` | Agents | llm_agents |
| `mcp_protocol_iterations.py` | MCP | mcp |
| `vector_databases_iterations.py` | Vector DBs | vectordb |
| `guardrails_safety_iterations.py` | Safety | guardrails |
| `prompt_engineering_iterations.py` | Prompting | prompts |
| `embedding_models_iterations.py` | Embeddings | embeddings |
| `evaluation_benchmarking_iterations.py` | Evaluation | evaluation |

## Remaining Scripts (150)

Scripts in `research/iterations/` not yet migrated can use the adapter pattern for gradual migration:

```python
# Add these 2 lines to any legacy script:
from research.iterations.legacy_adapter import LegacyScriptAdapter
adapter = LegacyScriptAdapter(__file__)
```

## Benefits Summary

After migrating 10 scripts:
- **Code reduction**: ~1,400 lines removed (140 lines per file)
- **Quality improvements**: All Gap fixes automatically applied
- **Consistency**: Same filtering, dedup, and stats across all scripts
- **Maintainability**: Fix once in base_executor, propagate everywhere

## Troubleshooting

### Import Error: "No module named 'base_executor'"

Ensure you're running from the `research/iterations/` directory or add it to Python path:
```python
import sys
sys.path.insert(0, "path/to/research/iterations")
```

### Collection already exists

`base_executor.py` handles this gracefully - existing collections are reused.

### API key errors

Ensure `.config/.env` has all required keys:
- `EXA_API_KEY`
- `TAVILY_API_KEY`
- `PERPLEXITY_API_KEY`
- `JINA_API_KEY`

## Related Documentation

- `research/iterations/base_executor.py` - Core implementation (1228 lines)
- `research/iterations/legacy_adapter.py` - Bridge for gradual migration (622 lines)
- `docs/gap-resolution/` - Gap implementation details
