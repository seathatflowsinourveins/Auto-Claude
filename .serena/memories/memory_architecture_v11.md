# Memory Architecture v11 - Unified Cross-Session Integration

## Overview

v11 architecture extends v10 with Phase 6-10 autonomous system integration,
enabling seamless cross-session memory access across ALL subsystems.

## Hook Chain (5 Hooks at Session Start)

```
SESSION START FLOW (v11):
1. echo [SESSION START]          → Log session timestamp
2. letta-session-start.py        → Loads Letta memory blocks with Sleeptime
3. project-bootstrap.py          → Detects project (UNLEASH|WITNESS|TRADING)
4. memory-gateway-hook.py        → Unified pre-task search with ImportanceScorer
5. phase6-session-start.py       → NEW: Phase 6-10 bridge initialization
```

## Session End Flow (3 Hooks)

```
SESSION END FLOW (v11):
1. echo [SESSION END]            → Log session end
2. letta-session-end.py          → Extract learnings to Letta
3. phase6-session-end.py         → NEW: Phase 6-10 persistence
```

## Key Integration Files

| File | Purpose |
|------|---------|
| `phase6_memory_integration.py` | Central orchestrator for Phase 6-10 memory |
| `serena_mcp_bridge.py` | Serena MCP integration with caching |
| `phase6-session-start.py` | Session start hook for Phase 6 bridge |
| `phase6-session-end.py` | Session end hook for learning persistence |
| `memory_gateway.py` | Memory Gateway v2.2 with HyDE/CRAG |

## Memory Systems (5 Backends)

1. **Unified Memory v8** (Letta+Qdrant) - Vector semantic search
2. **Claude-Mem** (MCP) - Observation/decision tracking
3. **Episodic Memory** (MCP) - Conversation history search
4. **Serena Memory** (MCP) - Code-aware persistence
5. **Local JSONL** - Fallback learnings storage

## Phase 6-10 Subsystems

| Subsystem | Memory Strategy |
|-----------|-----------------|
| META_LEARNER | HyDE (research-oriented) |
| CONTINUAL_LEARNING | Direct search |
| SELF_EVOLVER | CRAG (debugging-oriented) |
| WORLD_MODEL | Direct search |
| TELEMETRY | Direct with aggregation |

## ImportanceScorer Formula

```python
final_score = (
    0.35 * type_score +     # decisions(1.0) > patterns(0.85) > validation(0.7)
    0.30 * content_score +   # Based on content relevance
    0.20 * temporal_score +  # Recency boost (24h=1.0, 7d=0.7, 30d=0.4)
    0.15 * source_score      # claude-mem(0.9) > serena(0.85) > episodic(0.8)
)
```

## Project-Specific Memory Keywords

### UNLEASH
- Ralph Loop, self-evolving agents, memory architecture
- SDK research, cross-session, bootstrap, ultrathink

### WITNESS (State of Witness)
- archetype, particle system, TouchDesigner
- shader, pipeline, MediaPipe pose, GLSL

### TRADING (AlphaForge)
- trading system, risk management, kill switch
- paper trading, position sizing, safety manager

## Usage

### Initialize Memory Bridge
```python
from phase6_memory_integration import UnifiedMemoryOrchestrator, MemorySubsystem

async def main():
    orchestrator = UnifiedMemoryOrchestrator()
    await orchestrator.initialize()
    
    # Search across all systems
    response = await orchestrator.search(
        query="memory architecture",
        subsystem=MemorySubsystem.META_LEARNER,
        include_serena=True
    )
    
    # Store new learning
    await orchestrator.store(
        content="Important decision...",
        entry_type="decision",
        subsystem=MemorySubsystem.ORCHESTRATOR
    )
```

### Serena Memory Access
```python
from serena_mcp_bridge import SerenaMCPBridge

async with SerenaMCPBridge() as bridge:
    memories = await bridge.list_memories()
    results = await bridge.search_memories("shader")
```

## Configuration Files

- `~/.claude/settings.json` - Hook registration
- `~/.claude/mcp_servers_OPTIMAL.json` - MCP server config
- `~/.claude/config/project-bootstrap.json` - Project keywords

## Last Updated

January 22, 2026 - Phase 6-10 Integration Complete
