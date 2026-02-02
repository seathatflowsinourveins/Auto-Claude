# V10 OPTIMIZED ARCHITECTURE V2
## Enhanced Claude Code CLI + Letta Memory System

> **Version**: 10.1.0 | **Status**: Production Ready | **Updated**: January 2026

---

## What's New in V10.1

### Key Improvements

| Feature | V10.0 | V10.1 |
|---------|-------|-------|
| Letta Integration | httpx raw API | Official `letta_client` SDK |
| Hook Utilities | Separate implementations | Shared `hook_utils.py` library |
| PreToolUse | Allow/Deny only | Allow/Deny/Modify (v2.0.10+) |
| Context Injection | Not implemented | SessionStart `additionalContext` |
| Health Checking | Manual verification | Comprehensive `health_check.py` |
| Type Safety | Minimal | Full typing with Pyright |

### New Files

```
v10_optimized/
├── hooks/
│   ├── hook_utils.py         # NEW: Shared utilities
│   ├── letta_sync_v2.py      # NEW: SDK-based implementation
│   └── mcp_guard_v2.py       # NEW: Input modification support
└── scripts/
    └── health_check.py        # NEW: System health verification
```

---

## Design Philosophy

**V10 prioritizes three principles:**

1. **Verified**: Every component tested and confirmed working
2. **Minimal**: Only include what's actually needed
3. **Seamless**: Everything works together automatically

### Research-Based Decisions

Based on official documentation research (January 2026):

| Source | Key Pattern Applied |
|--------|-------------------|
| [Claude Code Hooks v2.0.10+](https://code.claude.com/docs/en/hooks) | Input modification via `updatedInput`, `additionalContext` injection, permission decisions: `allow`/`deny`/`ask` |
| [Letta SDK](https://docs.letta.com/api/) | `letta_client.Letta`, `SleeptimeManagerUpdate`, `enable_sleeptime=True` |
| [MCP Protocol 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) | JSON-RPC 2.0, `structuredContent` for typed outputs, capability negotiation |

### MCP Security Best Practices (from specification)

The V10 hooks implement these MCP-mandated security practices:

1. **Human in the Loop**: `mcp_guard_v2.py` validates all MCP tool calls before execution
2. **Input Validation**: All tool inputs are validated against security patterns
3. **Rate Limiting**: Turn counter prevents runaway operations
4. **Audit Logging**: `audit_log.py` logs all tool usage for security review
5. **Timeouts**: All async operations have configurable timeouts
6. **Path Sanitization**: Blocked patterns for sensitive files (`.env`, `.git/`, credentials)

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
│  │letta_sync_v2 │              │mcp_guard_v2  │                 │
│  │   (start)    │              │ + MODIFY     │ ◄── v2.0.10+    │
│  └──────┬───────┘              └──────────────┘                 │
│         │                                                        │
│         ▼ additionalContext injection                           │
└─────────┼────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Letta Server                                │
│                   (localhost:8283)                               │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Agents     │  │   Memory     │  │  Sleeptime   │           │
│  │  (per proj)  │  │   Blocks     │  │   Agent      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  SDK Pattern:                                                    │
│  client = Letta(token="...")                                    │
│  agent = client.agents.create(enable_sleeptime=True)            │
│  client.groups.modify(manager_config=SleeptimeManagerUpdate())  │
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
│  All 8 servers verified on npm as of January 2026               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Shared Hook Utilities (`hook_utils.py`)

New shared library providing:

```python
from hook_utils import (
    HookInput,           # Parse stdin JSON
    HookResponse,        # Build responses with additionalContext
    HookConfig,          # Access CLAUDE_PROJECT_DIR, hooks paths
    SessionState,        # Persist state across hook invocations
    TurnCounter,         # Track turns for sleeptime triggers
    PermissionDecision,  # ALLOW, DENY, MODIFY (v2.0.10+)
    log_event,           # Audit logging
)
```

**Key Features:**
- Environment variable access (`CLAUDE_PROJECT_DIR`, `CLAUDE_HOOKS_DIR`)
- Session state persistence between hook calls
- Consistent JSON response formatting
- Structured logging with `structlog`

### 2. Letta Sync V2 (`letta_sync_v2.py`)

Upgraded implementation using official Letta SDK:

```python
from letta_client import Letta
from letta_client.types import SleeptimeManagerUpdate

client = Letta(token=os.environ.get("LETTA_API_KEY"))

# Create agent with sleeptime
agent = client.agents.create(
    name=f"claude-code-{project_name}",
    memory_blocks=[
        {"label": "human", "value": "", "limit": 3000},
        {"label": "persona", "value": "...", "limit": 2000},
        {"label": "project-context", "value": "", "limit": 3000},
        {"label": "learnings", "value": "", "limit": 4000}
    ],
    model="anthropic/claude-sonnet-4-5-20250929",
    enable_sleeptime=True,
)

# Configure sleeptime frequency
client.groups.modify(
    group_id=agent.multi_agent_group.id,
    manager_config=SleeptimeManagerUpdate(sleeptime_agent_frequency=5)
)
```

**Improvements:**
- Graceful fallback to httpx if SDK not installed
- Context injection via `additionalContext` in response
- Proper session summary extraction from transcripts
- Retry logic for API calls

### 3. MCP Guard V2 (`mcp_guard_v2.py`)

Now supports input modification (v2.0.10+):

```python
# Instead of just blocking, can now modify inputs
if decision == "modify":
    return HookResponse(
        decision=PermissionDecision.MODIFY,
        reason="Input transformed for safety",
        modified_input={
            **tool_input,
            "content": tool_input["content"].replace("\r\n", "\n")  # Normalize line endings
        }
    )
```

**New Capabilities:**
- Line ending normalization
- Path sanitization before execution
- Audit logging of all decisions
- Kill switch integration

### 4. Health Check Script (`health_check.py`)

Comprehensive system verification:

```bash
# Full health check
uv run health_check.py

# Quick check (skip slow tests)
uv run health_check.py --quick

# Output as JSON
uv run health_check.py --json

# Attempt automatic fixes
uv run health_check.py --fix
```

**Checks Performed:**
- Directory structure existence
- Hook file syntax validation
- Configuration file parsing
- MCP package availability
- Letta server connectivity
- Python dependencies
- Kill switch status

---

## Hook Pipeline (V10.1)

```
SessionStart ─────► letta_sync_v2.py start ─────► Load memory
     │                     │
     │                     └─────► additionalContext injection (NEW)
     │
     ▼
[User interaction]
     │
     ▼
PreToolUse ───────► mcp_guard_v2.py ────────────► Validate/Modify MCP calls
                    │
                    └─────► Input modification (v2.0.10+) (NEW)

                    bash_guard.py ───────────► Validate Bash commands
     │
     ▼
PostToolUse ──────► audit_log.py ────────────► Log file modifications
     │
     ▼
Stop ─────────────► memory_consolidate.py ───► Trigger sleeptime (every 5 turns)
     │
     ▼
SessionEnd ───────► letta_sync_v2.py end ───────► Save learnings to Letta
```

---

## Environment Variables

### Standard (Claude Code)

| Variable | Purpose | Example |
|----------|---------|---------|
| `CLAUDE_PROJECT_DIR` | Project root path | `/home/user/project` |
| `CLAUDE_HOOKS_DIR` | Hooks directory | `~/.claude/v10/hooks` |
| `CLAUDE_CODE_REMOTE` | Remote environment flag | `true` |
| `CLAUDE_SESSION_ID` | Current session ID | `abc123` |
| `CLAUDE_ENV_FILE` | Env persistence file | `~/.claude/.env` |

### V10 Custom

| Variable | Purpose | Default |
|----------|---------|---------|
| `LETTA_URL` | Letta server URL | `http://localhost:8283` |
| `LETTA_API_KEY` | Letta API token | (none) |

---

## Quick Commands

### Setup

```powershell
# Full installation
cd "Z:\insider\AUTO CLAUDE\unleash\v10_optimized\scripts"
.\setup_v10.ps1 -Mode standard

# Verify system health
uv run health_check.py

# Start Letta
docker run -d --name letta -p 8283:8283 letta/letta:latest

# Install Python dependencies
uv pip install letta-client httpx structlog rich
```

### Health Check

```powershell
# Quick system check
uv run health_check.py --quick

# Full check with fixes
uv run health_check.py --fix

# JSON output for automation
uv run health_check.py --json > health_report.json
```

### Emergency

```powershell
# Activate kill switch (stops ALL operations)
New-Item -Path ~/.claude/KILL_SWITCH -ItemType File

# Deactivate kill switch
Remove-Item ~/.claude/KILL_SWITCH
```

---

## File Structure (V10.1)

```
~/.claude/
├── settings.json              # Main configuration
├── CLAUDE.md                  # Global instructions
├── KILL_SWITCH                # Emergency stop (create to activate)
├── .mcp.json                  # MCP server definitions
│
└── v10/
    ├── hooks/
    │   ├── hook_utils.py      # Shared utilities (NEW)
    │   ├── letta_sync.py      # Original implementation
    │   ├── letta_sync_v2.py   # SDK-based implementation (NEW)
    │   ├── mcp_guard.py       # Original guard
    │   ├── mcp_guard_v2.py    # With input modification (NEW)
    │   ├── bash_guard.py      # Bash command validation
    │   ├── memory_consolidate.py
    │   └── audit_log.py
    │
    ├── scripts/
    │   ├── setup_v10.ps1      # Installation
    │   ├── verify_mcp.py      # Package verification
    │   └── health_check.py    # System health (NEW)
    │
    ├── logs/
    │   ├── audit_YYYY-MM-DD.jsonl
    │   ├── security_YYYY-MM-DD.jsonl  # NEW
    │   └── consolidation.log
    │
    ├── cache/
    ├── memory.db
    └── .session_env
```

---

## Migration from V10.0

### Using New Components

The V2 hooks are drop-in compatible. To use them:

1. **Update settings.json hooks section:**
   ```json
   {
     "hooks": {
       "SessionStart": [{
         "matcher": "*",
         "hooks": [{
           "type": "command",
           "command": "python \"${CLAUDE_HOOKS_DIR}/letta_sync_v2.py\" start"
         }]
       }]
     }
   }
   ```

2. **Install Letta SDK:**
   ```bash
   uv pip install letta-client
   ```

3. **Run health check:**
   ```bash
   uv run health_check.py
   ```

### Backward Compatibility

- Original hooks (`letta_sync.py`, `mcp_guard.py`) remain functional
- V2 hooks fall back to httpx if `letta_client` not installed
- All environment variables remain the same

---

## Official Documentation

| Resource | URL |
|----------|-----|
| Claude Code Docs | https://code.claude.com/docs |
| Claude Code Hooks | https://code.claude.com/docs/en/hooks |
| Claude Code Settings | https://code.claude.com/docs/en/settings |
| MCP Specification | https://modelcontextprotocol.io/specification/2025-06-18 |
| MCP Python SDK | https://github.com/modelcontextprotocol/python-sdk |
| Letta API | https://docs.letta.com/api/ |
| Letta Sleeptime | https://docs.letta.com/guides/agents/architectures/sleeptime/ |

---

*V10.1 OPTIMIZED - Research-Based. SDK-Powered. Production Ready.*
*January 2026*
