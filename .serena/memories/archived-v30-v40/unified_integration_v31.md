# Unified Integration Layer V31 - Cross-Session Architecture

## Overview
The Unified Integration Layer connects all memory/evolution systems for seamless cross-session persistence.

## Architecture Position
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED INTEGRATION LAYER (V31)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  UserPromptSubmit                                                            │
│  └── unified_integration.pre_task_setup()                                   │
│      ├── TaskComplexityDetector -> thinking_tokens, verbosity_directive     │
│      ├── MemoryGateway.unified_search() -> relevant context                 │
│      └── LettaEvolution.get_guidance() -> patterns to apply                 │
│                                                                              │
│  PostToolUse / Stop                                                          │
│  └── unified_integration.post_task_capture()                                │
│      ├── Extract patterns from completed work                               │
│      ├── Calculate fitness scores                                           │
│      └── LettaEvolution.store_results() -> sleep-time consolidation         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Files
- `~/.claude/hooks/unified_integration.py` - Main integration layer (714 lines)
- `~/.claude/scripts/unleash-integration/task_complexity_detector.py` - P0 complexity detection
- `~/.claude/scripts/unleash-integration/letta_evolution_integration.py` - P1 Letta persistence

## Task Complexity Levels
| Level | Thinking Tokens | Use Case |
|-------|----------------|----------|
| TRIVIAL | 0 | Status, simple reads |
| LOW | 4,000 | Simple edits, lookups |
| MEDIUM | 16,000 | Code changes, analysis |
| HIGH | 64,000 | Architecture, complex bugs |
| ULTRATHINK | 128,000 | System design, critical decisions |

## Hooks Configuration (settings.json)
```json
"UserPromptSubmit": [
  {
    "command": "python unified_integration.py pre_task",
    "timeout": 5
  }
]
"SessionEnd": [
  {
    "command": "python unified_integration.py session_end",
    "timeout": 10
  }
]
```

## Usage
```python
from unified_integration import UnifiedIntegration

integration = UnifiedIntegration()

# Pre-task (on UserPromptSubmit)
context = integration.pre_task_setup(task)
# Returns: thinking config + memory context + guidance

# Post-task (on Stop or PostToolUse)
result = integration.post_task_capture(task, result, files_changed)
# Extracts patterns, calculates fitness, stores to Letta
```

## Integration Points
1. **Task Complexity Detection** (P0): Optimizes thinking tokens per task
2. **Letta Evolution** (P1): Stores patterns for cross-session retrieval
3. **Memory Gateway**: Unified search across all memory backends
4. **Session Hooks**: Automatic pre/post task processing

## Test Results (2026-01-23)
- Task complexity detection: 4/4 correct classifications
- Memory retrieval: 30 memories loaded from learnings
- Letta storage: Successfully stored patterns
- Fitness calculation: Working (0.7 for test case)

## Related Documents
- `ARCHITECTURE_OPTIMIZATION_SYNTHESIS.md` - Full V31 synthesis
- `memory_architecture_v10` - Memory architecture docs
- `cross_session_bootstrap` - Bootstrap system docs
