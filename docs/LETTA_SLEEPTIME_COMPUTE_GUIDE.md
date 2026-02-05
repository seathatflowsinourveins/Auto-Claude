# Letta Sleeptime Compute Integration Guide

> **Version**: V66 | **Updated**: 2026-02-05

## Overview

Sleeptime compute enables AI agents to perform memory consolidation asynchronously during idle periods, achieving:
- **91% latency reduction**: Memory operations don't block user interactions
- **90% token savings**: Intelligent summarization compresses context
- **Continuous learning**: Agents improve through background reflection

Based on Letta's research: https://www.letta.com/blog/sleep-time-compute

## Architecture

### Dual-Agent Pattern

```
+------------------+     +--------------------+
|  Primary Agent   |     |  Sleeptime Agent   |
|------------------|     |--------------------|
| - User messages  |<--->| - Memory editing   |
| - Tool calls     |     | - Summarization    |
| - Search memory  |     | - Insight gen      |
+------------------+     +--------------------+
        |                        |
        +--------+   +-----------+
                 |   |
            +----v---v----+
            |   Memory    |
            |   Blocks    |
            +-------------+
```

When `enable_sleeptime: true`:
1. Primary agent handles conversations with `conversation_search` and `archival_memory_search` tools
2. Sleeptime agent manages memory asynchronously, editing both agents' memory blocks
3. Memory consolidation happens without blocking user interactions

## Configuration

### Environment Variables

```bash
LETTA_API_KEY=your-api-key
LETTA_BASE_URL=https://api.letta.com  # or http://localhost:8500
LETTA_SLEEPTIME_ENABLED=true
LETTA_SLEEPTIME_FREQUENCY=5  # Steps between updates (default: 5)
```

### ralph_production.yaml

```yaml
letta_sleeptime:
  enabled: true
  frequency: 5                       # Steps between sleeptime updates
  agent_id: uap-sleeptime-agent

  integration:
    sync_on_consolidation: true      # Sync during Ralph Loop phase 3
    trigger_after_iteration: true    # Trigger after each iteration
    preserve_working_blocks: true    # Keep working blocks until promoted

  consolidation:
    importance_threshold: 0.3        # Min score for WORKING->LEARNED
    max_blocks: 500                  # Max memory blocks to retain
    min_retention_score: 0.2         # Min score to retain during cleanup
```

## Implementation

### LettaAdapter (platform/adapters/letta_adapter.py)

Sleeptime operations:

| Operation | Description |
|-----------|-------------|
| `create_agent` | Create agent with `enable_sleeptime=True` |
| `get_sleeptime_config` | Get current sleeptime settings |
| `update_sleeptime_config` | Enable/disable or change frequency |
| `trigger_sleeptime` | Manually trigger consolidation |

Example:
```python
from adapters.letta_adapter import LettaAdapter

adapter = LettaAdapter()
await adapter.initialize({
    "api_key": "your-key",
    "sleeptime_enabled": True,
    "sleeptime_frequency": 5,
})

# Create agent with sleeptime
result = await adapter.execute(
    "create_agent",
    name="my-agent",
    enable_sleeptime=True,
    sleeptime_frequency=3,
)

# Trigger consolidation
result = await adapter.execute(
    "trigger_sleeptime",
    agent_id="agent-123",
    consolidation_context="End of session",
)
```

### SleepTimeDaemon (platform/scripts/sleeptime_compute.py)

Local sleeptime compute with optional Letta sync:

```bash
# Check status
uv run sleeptime_compute.py status

# Run consolidation
uv run sleeptime_compute.py consolidate

# Sync with Letta native sleeptime
uv run sleeptime_compute.py letta-sync

# Check Letta sleeptime status
uv run sleeptime_compute.py letta-status

# Enable Letta sleeptime
uv run sleeptime_compute.py letta-enable --frequency 5

# Run as daemon
uv run sleeptime_compute.py daemon --interval 300
```

### Ralph Loop Integration (platform/scripts/ralph_loop.py)

The consolidation phase (phase 3) now supports dual-path consolidation:

1. **Letta Native Sync** (if `LETTA_SLEEPTIME_ENABLED=true`):
   - Triggers Letta's server-side sleeptime agent
   - Benefits: Server-side processing, multi-agent coordination

2. **Local Consolidation** (always runs):
   - Uses importance scoring for WORKING->LEARNED promotion
   - Runs cleanup to prevent unbounded memory growth

## Memory Consolidation Flow

```
WORKING blocks (per iteration)
        |
        v
+-------+--------+
| Importance     |
| Scoring        |
| (recency 0.3 + |
| frequency 0.4 +|
| confidence 0.3)|
+-------+--------+
        |
        v
+-------+--------+
| Filter         |
| threshold=0.3  |
+-------+--------+
        |
        v
+-------+--------+
| Group by topic |
| Summarize      |
| Deduplicate    |
+-------+--------+
        |
        v
LEARNED blocks (consolidated)
```

## Importance Scoring

Formula: `score = 0.3 * recency + 0.4 * frequency + 0.3 * confidence`

| Signal | Weight | Calculation |
|--------|--------|-------------|
| Recency | 0.3 | `0.95 ** age_days` (5% daily decay) |
| Frequency | 0.4 | `log1p(access_count) / log1p(100)` (normalized) |
| Confidence | 0.3 | From metadata (default 0.5) |

Blocks with score < 0.3 are filtered during consolidation.

## Benchmarks

From `test_sleeptime_benchmarks.py`:

| Metric | Result | Target |
|--------|--------|--------|
| Single block scoring | <100us | <100us |
| Batch scoring throughput | >10,000 blocks/s | >10,000 |
| Consolidation (50 blocks) | <500ms | <500ms |
| Token compression | >50% | 90% (Letta native) |
| Full cycle (30 blocks) | <1s | <1s |

Note: 90% token savings requires Letta native sleeptime with LLM-based summarization.

## Testing

```bash
# All sleeptime tests
cd C:\Users\42 && uv run --no-project --with pytest,pytest-asyncio,structlog,httpx,pydantic python -m pytest "Z:/insider/AUTO CLAUDE/unleash/platform/tests/test_sleeptime_integration.py" "Z:/insider/AUTO CLAUDE/unleash/platform/tests/test_sleeptime_benchmarks.py" -v

# Specific test groups
pytest test_sleeptime_integration.py::TestLettaAdapterSleeptime -v  # Adapter ops
pytest test_sleeptime_integration.py::TestSleeptimeComputeMemoryManager -v  # Memory
pytest test_sleeptime_benchmarks.py::TestConsolidationBenchmarks -v  # Performance
```

## Files

| File | Purpose |
|------|---------|
| `platform/adapters/letta_adapter.py` | Letta API adapter with sleeptime ops |
| `platform/scripts/sleeptime_compute.py` | Local sleeptime daemon |
| `platform/scripts/ralph_loop.py` | Orchestration with consolidation |
| `config/ralph_production.yaml` | Production configuration |
| `platform/tests/test_sleeptime_integration.py` | Integration tests (34 tests) |
| `platform/tests/test_sleeptime_benchmarks.py` | Performance benchmarks (12 tests) |

## Research Sources

- [Sleep-time Compute Blog](https://www.letta.com/blog/sleep-time-compute) - Original concept
- [Sleep-time Agents Docs](https://docs.letta.com/guides/agents/architectures/sleeptime/) - API reference
- [Sleep-time Compute Paper](https://arxiv.org/abs/2504.13171) - "Beyond Inference Scaling at Test-time"
- [Letta Memory Docs](https://docs.letta.com/letta-code/memory) - Memory management patterns
