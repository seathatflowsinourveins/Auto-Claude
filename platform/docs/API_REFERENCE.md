# Ultimate Autonomous Platform - API Reference

**Version**: 10.0 (V10 Optimized)
**Last Updated**: January 2026
**Architecture**: Verified, Minimal, Seamless

---

## Table of Contents

1. [Overview](#overview)
2. [Module Categories](#module-categories)
3. [CLI Interface Standards](#cli-interface-standards)
4. [Core Modules](#core-modules)
5. [Script Modules](#script-modules)
6. [Quick Reference](#quick-reference)
7. [Integration Patterns](#integration-patterns)

---

## Overview

The Ultimate Autonomous Platform (UAP) V10 provides a comprehensive suite of modules for autonomous AI development, monitoring, and orchestration. All modules follow consistent patterns:

- **PEP 723 Script Metadata**: Inline dependencies with `uv run`
- **Async-First**: All I/O operations use `asyncio`
- **Structured Logging**: `structlog` for consistent log output
- **Pydantic Models**: Type-safe data structures
- **CLI Standards**: Consistent argument parsing with `argparse`

---

## Module Categories

### Infrastructure
| Module | Purpose | Key Commands |
|--------|---------|--------------|
| `config.py` | Configuration management | `status`, `get`, `set`, `export` |
| `secrets.py` | Encrypted secrets | `set`, `get`, `list`, `rotate` |
| `install.py` | Platform installer | `--check`, `--quick`, `--verbose` |

### Orchestration
| Module | Purpose | Key Commands |
|--------|---------|--------------|
| `ralph_loop.py` | Main iteration loop | `status`, `iterate`, `daemon`, `report` |
| `ecosystem_orchestrator.py` | Health monitoring | `--quick`, `--json`, `--watch` |
| `platform_orchestrator.py` | Unified entry point | `status`, `health`, `benchmark`, `serve` |

### Validation
| Module | Purpose | Key Commands |
|--------|---------|--------------|
| `auto_validate.py` | Validation pipeline | `--quick`, `--json`, `--verbose` |
| `verify_mcp.py` | MCP server verification | `--json` |
| `health_check.py` | Component health | `--json` |

### Memory & Continuity
| Module | Purpose | Key Commands |
|--------|---------|--------------|
| `sleeptime_compute.py` | Sleep-time patterns | `status`, `consolidate`, `warmstart`, `daemon` |
| `session_continuity.py` | Session management | `init`, `status`, `export`, `trinity`, `knowledge` |
| `memory_bridge.py` | Memory sync | `sync`, `status` |

### Performance & Monitoring
| Module | Purpose | Key Commands |
|--------|---------|--------------|
| `performance.py` | Profiling & benchmarks | `status`, `benchmark`, `memory`, `optimize` |
| `metrics.py` | Prometheus metrics | `--port`, `--json` |
| `tracing.py` | OpenTelemetry traces | `--endpoint`, `--sample-rate` |
| `status_dashboard.py` | Status display | `--json`, `--watch` |

### Resilience
| Module | Purpose | Key Commands |
|--------|---------|--------------|
| `rate_limiter.py` | Rate limiting | `token-bucket`, `sliding-window`, `adaptive` |
| `autoscale.py` | HPA metrics | `metrics`, `recommend`, `--port` |

---

## CLI Interface Standards

All V10 modules follow these CLI conventions:

### Standard Flags
```bash
--json, -j     # Output as JSON (machine-readable)
--verbose, -v  # Verbose output
--quick, -q    # Quick mode (skip slow operations)
--help, -h     # Show help
```

### Exit Codes
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Warning (degraded but functional) |
| 2 | Failure (operation failed) |
| 130 | Interrupted (Ctrl+C) |

### Output Format
```
============================================================
MODULE NAME - Description
============================================================
Started: YYYY-MM-DD HH:MM:SS
------------------------------------------------------------
[1/N] Step description... [STATUS] Result
[2/N] Step description... [STATUS] Result
------------------------------------------------------------
RESULT: [STATUS]
Summary: PASS=X, WARN=Y, FAIL=Z
Duration: XXXms
============================================================
```

---

## Core Modules

### `v10_optimized/core/`

#### memory.py - Three-Tier Memory System

```python
from core.memory import CoreMemory, ArchivalMemory, TemporalGraph

# Core memory (in-context)
core = CoreMemory()
core.add_block("system", "You are a helpful assistant")
core.add_block("working", "Current task: implement feature X")

# Archival memory (vector-based)
archival = ArchivalMemory(qdrant_url="http://localhost:6333")
await archival.store("episodic", "User prefers detailed explanations")
results = await archival.search("user preferences", limit=5)

# Temporal graph (knowledge)
graph = TemporalGraph(neo4j_url="bolt://localhost:7687")
await graph.add_fact("Claude", "created_by", "Anthropic")
facts = await graph.query("What did Anthropic create?")
```

#### cooperation.py - Session Handoff

```python
from core.cooperation import SessionHandoff, TaskCoordinator

# Create handoff package
handoff = SessionHandoff()
handoff.add_context("task", "Implementing user auth")
handoff.add_decision("Use JWT tokens instead of sessions")
handoff.add_constraint("Must support SSO")

# Export for next session
package = handoff.export()  # JSON-serializable dict

# Task coordination
coordinator = TaskCoordinator()
coordinator.assign("agent-1", "Implement login endpoint")
coordinator.assign("agent-2", "Write auth middleware")
await coordinator.wait_all()
```

#### harness.py - Agent Harness

```python
from core.harness import ContextWindow, ShiftHandoff, Checkpoint

# Context window management
context = ContextWindow(max_tokens=200000)
context.set_instructions("System prompt here")
context.set_knowledge("Domain knowledge")
context.add_to_history("User: Hello")

# Check capacity
remaining = context.remaining_tokens()

# Create checkpoint
checkpoint = Checkpoint.create(agent_state)
checkpoint.save("checkpoints/iteration_5.json")

# Restore from checkpoint
restored = Checkpoint.load("checkpoints/iteration_5.json")
```

#### executor.py - ReAct Agent Executor

```python
from core.executor import create_executor, ExecutorState

# Create executor with all components
executor = await create_executor(
    mcp_config_path=".mcp.json",
    memory_config={"qdrant_url": "http://localhost:6333"},
    thinking_budget=32000,
)

# Execute task
result = await executor.execute(
    "Analyze the codebase and find security vulnerabilities"
)

# Get execution state
state: ExecutorState = executor.get_state()
print(f"Phase: {state.phase}")
print(f"Iterations: {state.iteration_count}")
```

#### thinking.py - Extended Thinking

```python
from core.thinking import ThinkingEngine, ThinkingStrategy

# Create thinking engine
engine = ThinkingEngine(
    max_tokens=128000,
    strategy=ThinkingStrategy.METACOGNITIVE,
)

# Generate reasoning
chain = await engine.think(
    "Should we use microservices or monolith?",
    context={"team_size": 5, "timeline": "3 months"},
)

# Inspect reasoning
for step in chain.steps:
    print(f"[{step.reasoning_type}] {step.content}")
    print(f"  Confidence: {step.confidence}")
```

---

## Script Modules

### ralph_loop.py - Main Orchestration Loop

The primary entry point for autonomous iterations.

```bash
# Check status
uv run ralph_loop.py status

# Run single iteration
uv run ralph_loop.py iterate

# Run daemon mode (continuous)
uv run ralph_loop.py daemon --interval 300

# View latest report
uv run ralph_loop.py report
```

**Iteration Pipeline:**
1. **Health Check** - Ecosystem orchestrator validation
2. **Validation** - Auto-validate pipeline execution
3. **Consolidation** - Sleep-time memory consolidation
4. **Session Update** - Session continuity refresh

### ecosystem_orchestrator.py - Health Monitoring

Comprehensive ecosystem health dashboard.

```bash
# Quick check
uv run ecosystem_orchestrator.py --quick

# Full check with JSON
uv run ecosystem_orchestrator.py --json

# Watch mode (continuous)
uv run ecosystem_orchestrator.py --watch --interval 30
```

**Checks Performed:**
- Hook syntax validation
- MCP server availability
- Letta server connectivity
- Claude-Flow components
- Infrastructure (Python, uv, Node.js)
- Kill switch status

### auto_validate.py - Validation Pipeline

6-step validation workflow.

```bash
# Full validation
uv run auto_validate.py

# Quick (skip tests)
uv run auto_validate.py --quick

# JSON output
uv run auto_validate.py --json
```

**Validation Steps:**
1. Hook Syntax - Python compile check
2. MCP Packages - npm package verification
3. Infrastructure - Python, uv, Node.js
4. Letta Server - HTTP health check
5. Ecosystem Health - Orchestrator integration
6. Test Suite - pytest execution

### sleeptime_compute.py - Sleep-Time Patterns

Letta-inspired background processing.

```bash
# Check status
uv run sleeptime_compute.py status

# Manual consolidation
uv run sleeptime_compute.py consolidate

# Generate warm start
uv run sleeptime_compute.py warmstart

# Run daemon
uv run sleeptime_compute.py daemon --interval 300
```

**Phases:**
- `IDLE` - Waiting for work
- `CONSOLIDATING` - Memory compaction
- `GENERATING` - Insight generation
- `WARMING` - Warm start preparation

### session_continuity.py - Session Management

Trinity Pattern and knowledge base.

```bash
# Initialize knowledge base
uv run session_continuity.py init

# Check status
uv run session_continuity.py status

# Export for teleportation
uv run session_continuity.py export

# Trinity status
uv run session_continuity.py trinity
```

**9-File Knowledge Base:**
1. `01_style.md` - Coding conventions
2. `02_principles.md` - Development principles
3. `03_architecture.md` - System architecture
4. `04_domain.md` - Domain language
5. `05_workflows.md` - Common workflows
6. `06_decisions.md` - Key decisions
7. `07_patterns.md` - Code patterns
8. `08_testing.md` - Testing strategy
9. `09_context.md` - Project context

### performance.py - Profiling & Benchmarks

Performance analysis toolkit.

```bash
# Current status
uv run performance.py status

# Run benchmarks
uv run performance.py benchmark

# Memory analysis
uv run performance.py memory

# Optimization recommendations
uv run performance.py optimize
```

**Metrics Collected:**
- Execution time per module
- Memory usage (RSS, heap)
- GC object count
- CPU utilization
- Hot function identification

---

## Quick Reference

### Start Everything
```bash
# 1. Install dependencies
uv run install.py --quick

# 2. Start services (Docker)
docker-compose up -d

# 3. Validate installation
uv run auto_validate.py --quick

# 4. Start Ralph Loop
uv run ralph_loop.py daemon
```

### Common Operations
```bash
# Check platform health
uv run ecosystem_orchestrator.py --quick

# Run single iteration
uv run ralph_loop.py iterate

# View performance metrics
uv run performance.py status

# Export session for teleport
uv run session_continuity.py export
```

### Troubleshooting
```bash
# Check all systems
uv run health_check.py --json

# Verify MCP servers
uv run verify_mcp.py --json

# Memory analysis
uv run performance.py memory

# View logs
uv run ecosystem_orchestrator.py --verbose
```

---

## Integration Patterns

### Pattern 1: Subprocess Orchestration

```python
import asyncio
import subprocess

async def run_module(module: str, args: list[str]) -> dict:
    """Run a V10 module as subprocess."""
    result = subprocess.run(
        ["uv", "run", module, "--json"] + args,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(result.stdout) if result.returncode == 0 else None
```

### Pattern 2: Direct Import

```python
from scripts.ecosystem_orchestrator import EcosystemOrchestrator
from scripts.performance import PerformanceRunner

async def analyze():
    # Health check
    orchestrator = EcosystemOrchestrator()
    health = await orchestrator.check_all()

    # Performance
    runner = PerformanceRunner()
    report = await runner.run_benchmark_suite()

    return health, report
```

### Pattern 3: Event-Driven

```python
from scripts.ralph_loop import RalphLoop

async def on_iteration_complete(report):
    """Handle iteration completion."""
    if report.overall_status == "fail":
        await send_alert(report.recommendations)
    else:
        await log_metrics(report)

loop = RalphLoop()
loop.on_complete = on_iteration_complete
await loop.run_daemon(interval=300)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UAP_ENV` | `development` | Environment (dev/staging/prod) |
| `UAP_LOG_LEVEL` | `INFO` | Logging level |
| `LETTA_URL` | `http://localhost:8283` | Letta server URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB |
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j graph DB |
| `REDIS_URL` | `redis://localhost:6379` | Redis cache |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 10.0 | 2026-01 | Initial V10 release |
| 10.1 | 2026-01 | Added performance module |
| 10.2 | 2026-01 | Documentation update |

---

*Generated by Ralph Loop Iteration 18*
