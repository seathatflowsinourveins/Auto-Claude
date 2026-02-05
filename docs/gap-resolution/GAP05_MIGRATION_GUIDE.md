# Gap05 Migration Guide: Research Iteration Scripts

## Overview

Gap05 addresses the fact that 158 of 160 research iteration scripts in `research/iterations/` were not using the centralized `base_executor.py` infrastructure. This guide documents how to migrate legacy scripts.

## Migration Status

| Status | Count | Description |
|--------|-------|-------------|
| Migrated | 25 | Using base_executor.py (15.7%) |
| Legacy | ~134 | Still using old patterns |
| Complex | 1 | deep_research_iterations.py (context-aware SDK features, keep as-is) |
| Infrastructure | 3 | base_executor.py, legacy_adapter.py, run_iteration.py |

## Why Migrate?

Legacy scripts duplicate 200-400 lines of boilerplate code that base_executor provides:

- **Gap02**: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- **Gap04**: Honest stats recomputation from actual saved data
- **Gap06**: Synthesis with 40+ claim indicators
- **Gap07**: 3-layer dedup (URL, content-hash, vector)
- **Gap09**: Fallback broadening on sparse results
- **Gap11**: Quality scoring and dashboard

## Migration Options

### Option A: Full Migration (Recommended)

Replace the entire legacy script with a minimal version using `run_research()`.

**Before (legacy pattern, ~200 lines):**
```python
import asyncio
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

TOPICS = [
    {"topic": "Topic 1", "area": "area1"},
    {"topic": "Topic 2", "area": "area2"},
]

@dataclass
class MyResult:
    topic: str
    area: str
    sources: list
    findings: list
    vectors: int
    latency: float

class MyExecutor:
    def __init__(self):
        self.exa = None
        self.qdrant = None
        self.keys = {...}
        self.stats = {...}

    async def initialize(self):
        # 20+ lines of setup
        pass

    async def research(self, topic, area):
        # 30+ lines of parallel API calls
        pass

    async def _exa(self, topic):
        # 15+ lines
        pass

    async def _tavily(self, client, topic):
        # 15+ lines
        pass

    async def _perplexity(self, client, topic, area):
        # 15+ lines
        pass

    async def _embed(self, client, texts):
        # 20+ lines
        pass

    async def run_iteration(self, topic_data, index):
        # 15+ lines
        pass

async def main():
    # 30+ lines of orchestration
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

**After (migrated, ~40 lines):**
```python
"""
MY ITERATIONS - Description
============================
Brief description

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


MY_TOPICS = [
    {"topic": "Topic 1", "area": "area1"},
    {"topic": "Topic 2", "area": "area2"},
]


class MyExecutor(BaseResearchExecutor):
    """Custom executor with domain-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Domain-specific research: {topic}"


if __name__ == "__main__":
    run_research(
        "my_collection",
        "MY ITERATIONS",
        MY_TOPICS,
        executor_class=MyExecutor,
    )
```

### Option B: Incremental Migration via Legacy Adapter

For scripts with complex custom logic, use `legacy_adapter.py` to incrementally adopt base_executor features.

```python
from legacy_adapter import LegacyScriptAdapter

adapter = LegacyScriptAdapter(__file__)

# Replace specific methods while keeping custom logic:
async def my_custom_main():
    await adapter.initialize()

    for i, topic_data in enumerate(TOPICS, 1):
        # Use adapter for research (gets Gap02, Gap07, Gap09)
        result = await adapter.research(topic_data["topic"], topic_data["area"])

        # Use adapter for synthesis (gets Gap06)
        synthesized = adapter.synthesize(
            topic_data["topic"],
            result["sources"],
            result["findings"]
        )

        # Use adapter for quality scoring (gets Gap11)
        quality = adapter.score_quality(
            topic_data["topic"],
            result["findings"],
            result["sources"]
        )

        # Custom processing here...

    # Use adapter for saving (gets Gap04)
    adapter.save_results()
```

### Option C: Topic Detection for Gradual Migration

The legacy adapter can detect TOPICS lists via AST parsing:

```python
from legacy_adapter import LegacyScriptAdapter

adapter = LegacyScriptAdapter(__file__)
detected_topics = adapter.detect_topics()  # Parses script for TOPICS

if detected_topics:
    print(f"Found {len(detected_topics)} topics")
```

## Migration Checklist

1. **Add import**: `from base_executor import BaseResearchExecutor, run_research`

2. **Keep TOPICS list**: Rename if needed (e.g., `LLMOPS_TOPICS`)

3. **Create custom executor class** (if custom prompting needed):
   ```python
   class MyExecutor(BaseResearchExecutor):
       def perplexity_prompt(self, topic: str, area: str) -> str:
           return f"Custom prompt: {topic}"
   ```

4. **Replace main block**:
   ```python
   if __name__ == "__main__":
       run_research("collection_name", "TITLE", TOPICS, executor_class=MyExecutor)
   ```

5. **Remove duplicated code**:
   - Remove `@dataclass` for result (use `ResearchResult` from base_executor)
   - Remove `async def _exa()`, `_tavily()`, `_perplexity()`, `_embed()`
   - Remove `async def initialize()`, `research()`, `run_iteration()`
   - Remove `async def main()` and `asyncio.run(main())`

6. **Update docstring** to document migration

7. **Test** by running the script

## What Gets Preserved

- **TOPICS list**: Keep your domain-specific topics
- **Custom prompting**: Override `perplexity_prompt()` in your executor
- **Collection name**: Specify in `run_research()`
- **Script title**: Specify in `run_research()`

## What Gets Upgraded

| Feature | Legacy | Migrated |
|---------|--------|----------|
| Filtering | Basic/none | Gap02: 30-char min, garbage patterns |
| Stats | Often inflated | Gap04: Recomputed from actual data |
| Synthesis | None | Gap06: 40+ claim indicators |
| Deduplication | None | Gap07: URL + content-hash + vector |
| Fallback | None | Gap09: Query broadening |
| Quality | None | Gap11: Scoring + dashboard |

## Generating Migration Diff

Use the legacy adapter to generate a migration diff for any script:

```python
from legacy_adapter import LegacyScriptAdapter
from pathlib import Path

diff = LegacyScriptAdapter.generate_migration_diff(
    Path("research/iterations/my_iterations.py")
)
print(diff)
```

## Testing Migrated Scripts

Run the migration tests:

```bash
cd C:\Users\42 && uv run --no-project --with pytest,pytest-asyncio,structlog,httpx,pydantic,rich,aiohttp,numpy python -m pytest "Z:/insider/AUTO CLAUDE/unleash/platform/tests/test_gap05_migration.py" -v
```

## Common Patterns to Remove

### Pattern 1: Duplicated API Keys
```python
# REMOVE
self.keys = {
    "tavily": os.getenv("TAVILY_API_KEY"),
    "jina": os.getenv("JINA_API_KEY"),
    "perplexity": os.getenv("PERPLEXITY_API_KEY"),
}
```
base_executor handles this.

### Pattern 2: Duplicated Result Dataclass
```python
# REMOVE
@dataclass
class MyResult:
    topic: str
    area: str
    sources: list
    findings: list
    vectors: int
    latency: float
```
Use `ResearchResult` from base_executor.

### Pattern 3: Duplicated Collection Creation
```python
# REMOVE
existing = [c.name for c in self.qdrant.get_collections().collections]
if "my_collection" not in existing:
    self.qdrant.create_collection(...)
```
base_executor handles this.

### Pattern 4: Bare Exception Handling
```python
# REMOVE (Gap01 violation)
except Exception as e:
    return {"sources": [], "findings": []}
```
base_executor logs exceptions properly.

## Scripts Already Migrated (25/159 = 15.7%)

### Batch 1 (V65)
- `llmops_iterations.py`
- `advanced_rag_iterations.py`
- `llm_agents_iterations.py`
- `embedding_models_iterations.py`
- `agentic_rag_iterations.py`
- `cutting_edge_iterations.py`
- `battle_tested_iterations.py`
- `memory_integration_iterations.py`
- `advanced_production_iterations.py`
- `guardrails_safety_iterations.py`
- `vector_databases_iterations.py`
- `mcp_protocol_iterations.py`
- `prompt_engineering_iterations.py`
- `evaluation_benchmarking_iterations.py`
- `agent_memory_iterations.py`

### Batch 2 (V66)
- `knowledge_graphs_iterations.py`
- `reasoning_llm_iterations.py`
- `model_routing_iterations.py`
- `tool_use_iterations.py`
- `long_context_iterations.py`
- `reranking_iterations.py`
- `rlhf_alignment_iterations.py`
- `semantic_caching_iterations.py`
- `code_generation_iterations.py`
- `structured_output_iterations.py`

### Complex (Not Migrated - Custom Features)
- `deep_research_iterations.py` - Uses ResearchContext enum with SDK_FEATURES matrix for context-aware SDK selection

## Anti-Patterns to Avoid

1. **Don't keep duplicated methods**: Remove `_exa`, `_tavily`, `_perplexity`, `_embed` if not customized
2. **Don't use `asyncio.run(main())`**: Use `run_research()` instead
3. **Don't truncate findings on save**: base_executor saves ALL findings (Gap04)
4. **Don't use bare `except:`**: Catch specific exceptions (Gap01)
5. **Don't hardcode collection paths**: base_executor handles paths correctly

## Advanced Customization

For scripts needing more than custom prompts:

```python
class AdvancedExecutor(BaseResearchExecutor):
    """Executor with advanced customization."""

    # Enable/disable features
    enable_iterative_discovery = True  # Gap09 L4
    enable_rag_evaluation = False  # Gap11 RAGEvaluator
    enable_quality_gate = True  # Retry low-quality topics
    quality_threshold = 0.3
    synthesis_max_claims = 5

    def perplexity_prompt(self, topic: str, area: str) -> str:
        """Custom Perplexity prompting."""
        return f"Custom: {topic}"

    def synthesize(self, topic: str, sources: list, findings: list) -> list:
        """Override for custom synthesis logic."""
        # Call base implementation first
        base_synthesis = super().synthesize(topic, sources, findings)

        # Add custom synthesis
        custom = self._my_custom_synthesis(sources)

        return base_synthesis + custom
```

## Metrics

Target: Migrate 50% of scripts (80/159) by V70
Current: 25 migrated (15.7%)
Remaining: ~134 scripts
Complex/Custom: 1 (deep_research_iterations.py - uses context-aware SDK features)
