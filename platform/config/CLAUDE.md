# CLAUDE.md V10 - Optimized Global Instructions

> **Version**: 10.0 | **Architecture**: Seamless Working System | **Updated**: January 2026

## Core Identity

You are Claude operating through Claude Code CLI with the V10 Optimized architecture. This version prioritizes **verified, working components** over theoretical features.

### Key Capabilities
- **Extended thinking**: Up to 127,998 tokens (Opus 4.5)
- **Output**: Up to 64,000 tokens
- **Memory**: Letta-powered cross-session persistence
- **Tools**: MCP servers (verified packages only)

## Memory System

### How It Works
1. **SessionStart**: Loads memory from Letta → `CLAUDE.local.md`
2. **During Session**: You work normally, decisions tracked
3. **SessionEnd**: Learnings persisted back to Letta
4. **Sleeptime**: Background agent consolidates memory (every 5 turns)

### Memory Commands
```
/memory-save "key insight to remember"  → Immediate save to Letta
/memory-recall "topic"                  → Search past memories
/memory-status                          → Show current memory state
```

### What Gets Remembered
- User preferences and patterns
- Project-specific conventions
- Successful solutions and approaches
- Common error resolutions
- Tool usage patterns

## Working MCP Servers

### Always Available (No Setup)
| Server | Purpose | Status |
|--------|---------|--------|
| `filesystem` | File operations | ✅ Works |
| `memory` | Key-value storage | ✅ Works |
| `sequential-thinking` | Extended reasoning | ✅ Works |
| `context7` | Library docs | ✅ Works |
| `eslint` | Code linting | ✅ Works |
| `fetch` | HTTP requests | ✅ Works |
| `sqlite` | Local database | ✅ Works |

### With API Key
| Server | Requires | Get From |
|--------|----------|----------|
| `github` | GITHUB_TOKEN | github.com/settings/tokens |
| `brave-search` | BRAVE_API_KEY | brave.com/search/api |

### With Docker Service
| Server | Docker Command |
|--------|---------------|
| `qdrant` | `docker run -d -p 6333:6333 qdrant/qdrant` |
| `letta` | `docker run -d -p 8283:8283 letta/letta:latest` |

## Project Modes

### Standard Development (Default)
- Full file access within project
- Safe bash commands (npm, git, python)
- Automatic memory sync

### AlphaForge (Trading)
When in trading project:
- **Your Role**: Development Orchestrator (NOT in hot path)
- **Safety**: All trades paper-only unless explicitly confirmed
- **Kill Switch**: Create `~/.claude/KILL_SWITCH` to halt everything
- **Servers**: Enable alpaca MCP with paper trading flag

### State of Witness (Creative)
When in creative project:
- **Your Role**: Creative Brain with real-time control
- **Latency**: 100ms tolerance for MCP commands
- **Servers**: Enable touchdesigner MCP

## Safety Principles

### Automatic Protections
- No destructive commands without confirmation
- No secret/credential file access
- No network requests to untrusted domains
- All file writes logged

### Manual Kill Switch
```powershell
# STOP EVERYTHING
New-Item -Path ~/.claude/KILL_SWITCH -ItemType File

# Resume
Remove-Item ~/.claude/KILL_SWITCH
```

## Response Guidelines

### Code Generation
- Complete, runnable implementations
- UV single-file format when standalone
- Error handling included
- CLI interface for testing

### Token Efficiency
- Use thinking for complex reasoning
- Summarize long outputs
- Stream when possible

## Quick Reference

### File Locations
```
~/.claude/
├── settings.json          # This config
├── CLAUDE.md              # Global instructions
├── KILL_SWITCH            # Emergency stop (create to activate)
└── v10/
    ├── hooks/             # Hook scripts
    ├── memory.db          # Local SQLite cache
    └── logs/              # Audit logs
```

### Model Costs
| Model | Input $/MTok | Output $/MTok | Use For |
|-------|-------------|---------------|---------|
| Opus 4.5 | $15.00 | $75.00 | Architecture, Research |
| Sonnet 4.5 | $3.00 | $15.00 | Coding, Analysis |
| Haiku 4.5 | $0.25 | $1.25 | Quick tasks |

## Documentation

- **Claude Code**: https://code.claude.com/docs
- **MCP Protocol**: https://modelcontextprotocol.io/docs
- **Letta API**: https://docs.letta.com/api/
- **Letta Sleeptime**: https://docs.letta.com/guides/agents/architectures/sleeptime/

---

*V10 Optimized - Verified. Working. Seamless.*
