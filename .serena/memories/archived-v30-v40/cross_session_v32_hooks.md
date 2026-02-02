# Cross-Session Integration V32 - Hook Documentation

## Overview
V32 cross-session integration layer providing:
- HiAgent hierarchical memory (Working → Episodic → Semantic → Procedural)
- Task complexity detection with model suggestions
- Pattern extraction and fitness scoring
- Seamless cross-session persistence

## File Location
`C:\Users\42\.claude\hooks\cross_session_v32.py` (~1,225 lines)

## Hook Entry Points

| Hook | Command | Purpose |
|------|---------|---------|
| SessionStart | `cross_session_v32.py init` | Load HiAgent memory, restore state |
| UserPromptSubmit | `cross_session_v32.py pre_task` | Complexity detection, memory search |
| PostToolUse | `cross_session_v32.py post_tool` | Pattern extraction, fitness scoring |
| PreCompact | `cross_session_v32.py pre_compact` | Save state before compaction |
| SessionEnd | `cross_session_v32.py session_end` | Final sync, summary |

## Task Complexity Levels

| Level | Tokens | Model | Use Case |
|-------|--------|-------|----------|
| TRIVIAL | 0 | haiku | Status, simple reads |
| LOW | 4,000 | haiku | Simple edits, lookups |
| MEDIUM | 16,000 | sonnet | Code changes, analysis |
| HIGH | 64,000 | opus | Architecture, complex bugs |
| ULTRATHINK | 128,000 | opus | System design, critical decisions |

## HiAgent Memory Hierarchy

```
WORKING (≤100 items)
├── Current task context
├── Recent tool outputs
└── Promotes to → EPISODIC on compaction

EPISODIC (≤1000 items)
├── Recent interactions
├── Task outcomes
└── Promotes to → SEMANTIC when accessed 3+ times

SEMANTIC (unlimited)
├── Domain knowledge
├── Consolidated patterns
└── Long-term reference

PROCEDURAL (unlimited)
├── Validated skills
├── High-performing patterns
└── Reusable solutions
```

## Pattern Types

| Type | Fitness | Description |
|------|---------|-------------|
| code | 0.7 | Code modifications |
| decision | 0.6 | Implementation choices |
| insight | 0.75 | Learnings discovered |
| failure | 0.15 | Errors to avoid |
| skill | 0.9 | Validated capabilities |

## State Files

- `~/.claude/state/session_state_v32.json` - Session state
- `~/.claude/state/hiagent_{project}.json` - HiAgent memory per project
- `~/.claude/logs/cross_session_v32.log` - Execution logs

## Integration with V31

V32 runs alongside V31 hooks:
1. V31 unified_integration.py handles basic complexity detection
2. V32 cross_session_v32.py adds HiAgent memory + enhanced patterns
3. Both sync to Letta evolution for sleep-time consolidation

## Usage Example

```python
from cross_session_v32 import get_integration

integration = get_integration()

# Pre-task (automatic via hook)
result = integration.pre_task_setup("Design microservices architecture")
# Returns: thinking config + memory context + guidance

# Post-task (automatic via hook)
result = integration.post_task_capture(
    task="Design task",
    result={"status": "success"},
    files_changed=["architecture.py"]
)
# Returns: patterns extracted + fitness score

# Status check
status = integration.get_status()
# Returns: session stats + memory stats + pattern buffer
```

## Related Documents
- `unified_architecture_v32` - Full V32 architecture
- `unified_integration_v31` - V31 integration layer
- `memory_architecture_v10` - Memory architecture foundation
