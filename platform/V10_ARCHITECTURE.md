# V10 OPTIMIZED ARCHITECTURE
## Seamless Claude Code CLI + Letta Memory System

> **Version**: 10.0.0 | **Status**: Production Ready | **Updated**: January 2026

---

## Design Philosophy

**V10 prioritizes three principles:**

1. **Verified**: Every component has been tested and confirmed working
2. **Minimal**: Only include what's actually needed
3. **Seamless**: Everything works together without manual intervention

### What Changed from V9

| Aspect | V9 APEX | V10 OPTIMIZED |
|--------|---------|---------------|
| MCP Servers | 40+ configured (15 working) | 8 verified (all working) |
| Settings.json | 355 lines | 128 lines |
| Hooks | Referenced missing files | Complete implementations |
| Memory | Theoretical 6-tier cache | Working Letta integration |
| Safety | 18-layer fortress (complex) | Focused guards (simple) |
| Validation | None | Automatic verification |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code CLI                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Session    │  │    Hooks     │  │ Permissions  │           │
│  │   Manager    │  │   Pipeline   │  │   Engine     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘           │
│         │                 │                                      │
│         │    ┌────────────┴────────────┐                        │
│         │    │                         │                        │
│         ▼    ▼                         ▼                        │
│  ┌──────────────┐              ┌──────────────┐                 │
│  │ SessionStart │              │  PreToolUse  │                 │
│  │   Hook       │              │    Hooks     │                 │
│  └──────┬───────┘              └──────┬───────┘                 │
│         │                             │                         │
│         ▼                             ▼                         │
│  ┌──────────────┐              ┌──────────────┐                 │
│  │ letta_sync.py│              │mcp_guard.py  │                 │
│  │   (start)    │              │bash_guard.py │                 │
│  └──────┬───────┘              └──────────────┘                 │
│         │                                                        │
└─────────┼────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Letta Server                                │
│                   (localhost:8283)                               │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Agents     │  │   Memory     │  │  Sleeptime   │           │
│  │              │  │   Blocks     │  │   Agent      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  Memory Blocks:                                                  │
│  • human (user preferences)                                      │
│  • persona (agent identity)                                      │
│  • project-context (current project)                             │
│  • learnings (accumulated knowledge)                             │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server Pool                              │
│                                                                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ filesystem │ │   memory   │ │ sequential │ │  context7  │   │
│  │            │ │            │ │  thinking  │ │            │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
│                                                                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │   eslint   │ │   fetch    │ │   sqlite   │ │   github   │   │
│  │            │ │            │ │            │ │  (gh CLI)  │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
│                                                                  │
│  All servers verified on npm as of January 2026                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Hook Pipeline

The hook system intercepts Claude Code events and executes Python scripts.

```
SessionStart ─────► letta_sync.py start ─────► Load memory to CLAUDE.local.md
     │
     ▼
[User interaction]
     │
     ▼
PreToolUse ───────► mcp_guard.py ────────────► Validate MCP calls
                    bash_guard.py ───────────► Validate Bash commands
     │
     ▼
PostToolUse ──────► audit_log.py ────────────► Log file modifications
     │
     ▼
Stop ─────────────► memory_consolidate.py ───► Trigger sleeptime (every 5 turns)
     │
     ▼
SessionEnd ───────► letta_sync.py end ───────► Save learnings to Letta
```

### 2. Memory Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Session Start                                │
│                                                                  │
│  1. Hook reads Letta agent memory blocks                         │
│  2. Memory formatted as markdown                                 │
│  3. Written to CLAUDE.local.md                                   │
│  4. Claude sees context from previous sessions                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     During Session                               │
│                                                                  │
│  • Turn counter increments on each Stop event                    │
│  • Every 5 turns: Sleeptime agent consolidates memory            │
│  • File changes logged to audit trail                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Session End                                  │
│                                                                  │
│  1. Hook reads transcript (if available)                         │
│  2. Extracts key learnings (tools used, errors, files modified)  │
│  3. Sends summary to Letta agent                                 │
│  4. Letta updates memory blocks                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Safety Guards

**mcp_guard.py** - Validates MCP tool calls:
- Blocks access to sensitive files (.env, secrets, .ssh)
- Prevents path traversal attacks
- Allows CLAUDE.local.md writes (for memory sync)

**bash_guard.py** - Validates Bash commands:
- Blocks destructive commands (rm -rf /, sudo, etc.)
- Warns on risky operations (force push, hard reset)
- Allows development commands (npm, git, python)

**Kill Switch** - Emergency stop:
- Create `~/.claude/KILL_SWITCH` to block ALL operations
- Remove to resume normal operation

---

## File Structure

```
~/.claude/
├── settings.json              # Main configuration
├── CLAUDE.md                  # Global instructions
├── KILL_SWITCH                # Emergency stop (create to activate)
├── .mcp.json                  # MCP server definitions
│
└── v10/
    ├── hooks/
    │   ├── letta_sync.py      # Session start/end memory sync
    │   ├── mcp_guard.py       # MCP tool validation
    │   ├── bash_guard.py      # Bash command validation
    │   ├── memory_consolidate.py  # Sleeptime trigger
    │   └── audit_log.py       # File change logging
    │
    ├── logs/
    │   ├── audit_YYYY-MM-DD.jsonl  # Daily audit logs
    │   └── consolidation.log       # Memory consolidation log
    │
    ├── cache/                 # Local cache
    ├── memory.db              # SQLite for local state
    └── .session_env           # Current session state
```

---

## Verified MCP Servers

### Core (Always Available)

| Server | Package | Purpose | Status |
|--------|---------|---------|--------|
| filesystem | `@modelcontextprotocol/server-filesystem` | File operations | ✅ Verified |
| memory | `@modelcontextprotocol/server-memory` | Key-value store | ✅ Verified |
| sequential-thinking | `@modelcontextprotocol/server-sequential-thinking` | Extended reasoning | ✅ Verified |

### Standard (Development)

| Server | Package | Purpose | Status |
|--------|---------|---------|--------|
| context7 | `@upstash/context7-mcp` | Library documentation | ✅ Verified |
| eslint | `@eslint/mcp` | Code linting | ✅ Verified |
| fetch | `@modelcontextprotocol/server-fetch` | HTTP requests | ✅ Verified |
| sqlite | `@modelcontextprotocol/server-sqlite` | Local database | ✅ Verified |
| github | `gh mcp` (CLI) | Git operations | ✅ Verified |

### Optional (Requires Setup)

| Server | Package | Requires |
|--------|---------|----------|
| qdrant | `mcp-server-qdrant` (uvx) | Docker: qdrant/qdrant |
| brave-search | `@modelcontextprotocol/server-brave-search` | BRAVE_API_KEY |
| playwright | `@anthropic-ai/mcp-server-playwright` | Nothing |

### Confirmed Non-Existent (DO NOT USE)

| Package | Status |
|---------|--------|
| `@letta-ai/mcp-server` | npm 404 |
| `mem0-mcp` | npm 404 |
| `@langfuse/mcp-server` | npm 404 |
| `@anthropic-ai/mcp-server-slack` | npm 404 |
| `@polygon-io/mcp-server` | npm 404 |
| `@alpaca/mcp-server` | npm 404 |

---

## Letta Integration

### Agent Configuration

Each project gets a dedicated Letta agent:

```python
agent = client.agents.create(
    name=f"claude-code-{project_name}",
    model="anthropic/claude-sonnet-4-5-20250929",
    enable_sleeptime=True,
    sleeptime_agent_frequency=5,  # Consolidate every 5 turns
    memory_blocks=[
        {"label": "human", "value": "", "limit": 3000},
        {"label": "persona", "value": "...", "limit": 2000},
        {"label": "project-context", "value": "", "limit": 3000},
        {"label": "learnings", "value": "", "limit": 4000}
    ]
)
```

### Memory Block Purposes

| Block | Purpose | Updated By |
|-------|---------|------------|
| human | User preferences, patterns | Sleeptime agent |
| persona | Agent identity for project | Initial setup |
| project-context | Current project state | Session end hook |
| learnings | Accumulated knowledge | Sleeptime agent |

### API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/health` | GET | Check server status |
| `/v1/agents` | GET/POST | List/create agents |
| `/v1/agents/{id}/memory/blocks` | GET | Load memory |
| `/v1/agents/{id}/messages` | POST | Send messages/trigger memory |

---

## Quick Commands

### Setup

```powershell
# Full installation
cd "Z:\insider\AUTO CLAUDE\unleash\v10_optimized\scripts"
.\setup_v10.ps1 -Mode standard

# Verify packages
uv run verify_mcp.py

# Start Letta
docker run -d --name letta -p 8283:8283 letta/letta:latest
```

### Operations

```powershell
# Check MCP status in Claude Code
/mcp status

# View audit logs
Get-Content ~/.claude/v10/logs/audit_$(Get-Date -Format 'yyyy-MM-dd').jsonl | ConvertFrom-Json | Format-Table

# Check Letta health
Invoke-WebRequest -Uri "http://localhost:8283/v1/health" -UseBasicParsing
```

### Emergency

```powershell
# Activate kill switch (stops ALL operations)
New-Item -Path ~/.claude/KILL_SWITCH -ItemType File

# Deactivate kill switch
Remove-Item ~/.claude/KILL_SWITCH

# View recent errors
Get-Content ~/.claude/v10/logs/*.log | Select-Object -Last 50
```

---

## Model Selection Guide

| Task Type | Recommended | Cost | Trigger |
|-----------|-------------|------|---------|
| Architecture, Research | Opus 4.5 | $15/$75 MTok | `think hard`, `ultrathink` |
| Coding, Debugging | Sonnet 4.5 | $3/$15 MTok | Default |
| Quick questions | Haiku 4.5 | $0.25/$1.25 MTok | Simple queries |

### Extended Thinking

```markdown
<!-- Trigger extended thinking in prompts -->
Please think carefully about this architecture decision.

<!-- Or use explicit triggers -->
/think Design a microservices architecture for...
```

---

## Official Documentation

| Resource | URL |
|----------|-----|
| Claude Code Docs | https://code.claude.com/docs |
| Claude Code Hooks | https://code.claude.com/docs/en/hooks |
| Claude Code Settings | https://code.claude.com/docs/en/settings |
| MCP Specification | https://modelcontextprotocol.io/specification/2025-06-18 |
| MCP Registry | https://registry.modelcontextprotocol.io |
| Letta API | https://docs.letta.com/api/ |
| Letta Sleeptime | https://docs.letta.com/guides/agents/architectures/sleeptime/ |

---

## Comparison: V9 vs V10

### Configuration Complexity

```
V9 APEX:
├── settings.json (355 lines)
├── CLAUDE.md (234 lines)
├── mcp_servers.json (179 lines)
├── model_router_rl.py (722 lines)
├── safety_fortress_v9.py (800+ lines)
└── Total: ~1,500+ lines of config

V10 OPTIMIZED:
├── settings.json (128 lines)
├── CLAUDE.md (143 lines)
├── mcp_servers.json (verified subset)
├── letta_sync.py (331 lines)
├── mcp_guard.py (158 lines)
├── bash_guard.py (167 lines)
└── Total: ~900 lines (40% reduction)
```

### Working Components

```
V9: 15 of 40+ MCP servers (37.5%)
V10: 8 of 8 MCP servers (100%)

V9: 0 of 4 hook files existed
V10: 5 of 5 hook files working
```

---

## Design Decisions

### Why Simplify?

1. **Complexity ≠ Capability**: V9's 18-layer safety fortress added cognitive overhead without proportional benefit
2. **Theoretical vs Practical**: V9 documented features that weren't implemented
3. **Maintenance Burden**: Complex configs break in subtle ways
4. **Letta Does the Heavy Lifting**: Memory management belongs in Letta, not custom code

### Why These Specific MCP Servers?

1. **filesystem**: Essential for any coding task
2. **memory**: Built-in persistence without external deps
3. **sequential-thinking**: Enables complex reasoning
4. **context7**: Documentation lookup (high value)
5. **eslint**: Code quality (high value)
6. **fetch**: HTTP capabilities for testing
7. **sqlite**: Local data persistence
8. **github**: Version control integration

### Why Letta Over Custom Memory?

1. **Proven Architecture**: Based on MemGPT research
2. **Sleeptime**: Background processing without manual triggers
3. **Agent API**: Clean interface for memory operations
4. **Multi-Project**: Each project gets dedicated agent
5. **Scalable**: Can run locally or on infrastructure

---

*V10 OPTIMIZED - Verified. Working. Seamless.*
*January 2026*
