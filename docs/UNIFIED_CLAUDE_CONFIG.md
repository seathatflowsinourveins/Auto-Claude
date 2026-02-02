# Unified Claude Code Configuration
## Unleash Ecosystem - Complete Integration

> **Version**: 1.0 | **Date**: January 18, 2026
> **Compatibility**: Claude Code v2.1.x | V10 Architecture

---

## Quick Start

Copy this file to `~/.claude/CLAUDE.md` to enable the full unleash ecosystem.

---

## Core Configuration

### Environment Variables

```bash
# Memory Systems
LETTA_URL=http://localhost:8283
QDRANT_URL=http://localhost:6333
NEO4J_URL=bolt://localhost:7687
GRAPHITI_ENABLED=true

# Project Paths
UNLEASH_ROOT=Z:/insider/AUTO CLAUDE/unleash
AUTO_CLAUDE_ROOT=Z:/insider/AUTO CLAUDE/unleash/auto-claude
V10_ROOT=~/.claude/v10

# MCP Configuration
MCP_TOOL_SEARCH_MODE=auto
CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000

# Extended Thinking
ULTRATHINK_ENABLED=true
DEEP_RESEARCH_MODE=true
```

---

## Memory Architecture

### Three-Layer Memory System

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: GRAPHITI (Knowledge Graph)                            │
│  ├── Episode types: PATTERN, GOTCHA, INSIGHT, DISCOVERY         │
│  ├── Semantic search via embeddings                             │
│  ├── Cross-project learning (GroupIdMode.PROJECT)               │
│  └── FalkorDB backend (Neo4j compatible)                        │
│                                                                  │
│  Layer 2: LETTA (Long-term Agent Memory)                        │
│  ├── Agent per project                                          │
│  ├── Memory blocks: human, persona, context, learnings          │
│  ├── Sleeptime consolidation (every 5 turns)                    │
│  └── Archival memory with passage search                        │
│                                                                  │
│  Layer 3: FILE-BASED (Fallback)                                 │
│  ├── .auto-claude/memory/*.json                                 │
│  ├── CLAUDE.local.md context injection                          │
│  └── Zero dependencies, always available                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Flow

1. **SessionStart**: Load memory from Letta → inject into context
2. **During Session**: Track patterns, gotchas, discoveries
3. **Every 5 turns**: Sleeptime agent consolidates memory
4. **SessionEnd**: Extract insights → save to all layers

---

## MCP Servers (V10 Verified)

### Core Servers (Always Enabled)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

### Development Servers

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"]
    },
    "eslint": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-eslint"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    }
  }
}
```

### Creative Servers (State of Witness)

```json
{
  "mcpServers": {
    "touchdesigner-creative": {
      "command": "python",
      "args": ["-m", "touchdesigner_mcp"],
      "env": {
        "TD_PORT": "9981"
      }
    },
    "comfyui-creative": {
      "command": "python",
      "args": ["-m", "comfyui_mcp"],
      "env": {
        "COMFYUI_URL": "http://localhost:8188"
      }
    }
  }
}
```

### Memory Servers (Optional)

```json
{
  "mcpServers": {
    "letta-mcp": {
      "command": "npx",
      "args": ["-y", "letta-mcp-server"],
      "env": {
        "LETTA_URL": "http://localhost:8283"
      }
    },
    "qdrant": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

---

## Hooks Configuration (V10.1)

### SessionStart - Memory Loading

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "python ~/.claude/v10/hooks/letta_sync_v2.py",
        "timeout": 10
      }
    ]
  }
}
```

### PreToolUse - Security Guards

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash(*)",
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/v10/hooks/bash_guard.py",
            "timeout": 2
          }
        ]
      },
      {
        "matcher": "mcp__*",
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/v10/hooks/mcp_guard_v2.py",
            "timeout": 2
          }
        ]
      }
    ]
  }
}
```

### PostToolUse - Audit Logging

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/v10/hooks/audit_log.py",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

### Stop - Memory Consolidation

```json
{
  "hooks": {
    "Stop": [
      {
        "type": "command",
        "command": "python ~/.claude/v10/hooks/memory_consolidate.py",
        "timeout": 30
      }
    ]
  }
}
```

---

## Skills Reference (32 Available)

### Architecture & System Design
- `system-design-architect` - Microservices, CQRS, distributed systems
- `complex-system-building` - Event sourcing, DDD, reliability
- `api-design-expert` - REST, GraphQL, gRPC, WebSocket

### Languages & Frameworks
- `python-mastery` - FastAPI, Pydantic v2, async patterns
- `typescript-mastery` - Hono, tRPC, Zod, Bun
- `advanced-coding-patterns` - Design patterns, optimization

### Quality & Testing
- `code-review` - Security (OWASP), performance, architecture
- `tdd-workflow` - Red-green-refactor, property-based testing
- `testing-excellence` - Unit, integration, e2e testing
- `debugging-mastery` - Root cause analysis, profiling

### DevOps & Infrastructure
- `devops-automation` - CI/CD, Docker, Kubernetes, IaC
- `mcp-integration` - MCP server configuration
- `security-audit` - OWASP Top 10, vulnerability scanning

### Data & ML
- `data-engineering` - ETL pipelines, data warehousing
- `financial-data-engineering` - QuestDB, TimescaleDB, tick data
- `ml-ops` - Model deployment, monitoring

### Creative & Visualization
- `touchdesigner-professional` - TOX components, extensions
- `shader-generation` - GLSL 4.60+, particles, volumetrics
- `glsl-visualization` - SDFs, Shadertoy conversion
- `ml-visualization` - Pose tracking, embeddings
- `node-workflow-design` - TD, ComfyUI, Blender nodes

### Exploration & Optimization
- `autonomous-exploration` - MAP-Elites quality-diversity
- `quality-diversity-optimization` - pyribs, QD algorithms
- `map-elites-exploration` - Creative parameter exploration

### Trading & Finance
- `trading-risk-validator` - Position sizing, circuit breakers
- `langgraph-workflows` - Multi-agent trading

### Multi-Agent & Orchestration
- `multi-agent-orchestration` - Claude-Flow, Hive Mind
- `project-memory` - CLAUDE.md hierarchy, ADRs
- `cross-session-memory` - Memory persistence

---

## Project-Specific Configurations

### AlphaForge Trading System

```yaml
# .auto-claude/config.yaml
project_type: trading
safety_level: maximum
mcp_servers:
  - questdb
  - prometheus
  - grafana
hooks:
  PreToolUse:
    - trading_risk_validator
  PostToolUse:
    - audit_log
environment:
  PAPER_TRADING_ONLY: "true"
  KILL_SWITCH_ENABLED: "true"
```

### State of Witness Creative

```yaml
# .auto-claude/config.yaml
project_type: creative
safety_level: relaxed
mcp_servers:
  - touchdesigner-creative
  - comfyui-creative
  - qdrant
hooks:
  SessionStart:
    - letta_sync_v2
  Stop:
    - memory_consolidate
environment:
  CREATIVE_MODE: "true"
  LATENCY_TARGET_MS: "100"
```

---

## Verification Commands

```powershell
# Full system health check
cd ~/.claude/v10/scripts
python health_check.py

# Quick verification
python health_check.py --quick --json

# Memory bridge status
python memory_bridge.py --status

# MCP package verification
python verify_mcp.py

# Letta connectivity
curl http://localhost:8283/v1/agents/
```

---

## Docker Infrastructure

```yaml
# Required containers
services:
  letta:
    image: letta/letta:latest
    ports: ["8283:8283"]

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333", "6334:6334"]

  neo4j:
    image: neo4j:latest
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/password
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| MCP timeout | Server not started | `npx -y @modelcontextprotocol/server-<name>` |
| Letta unreachable | Docker stopped | `docker start letta` |
| Hook errors | Syntax issue | `python -m py_compile <hook.py>` |
| Memory loss | Sync failed | Check audit logs in ~/.claude/v10/logs/ |
| Context missing | CLAUDE.local.md | Run memory_bridge.py --inject-context |

---

## Emergency Procedures

### Kill Switch
```powershell
# Activate kill switch (stops all operations)
New-Item ~/.claude/KILL_SWITCH

# Resume normal operations
Remove-Item ~/.claude/KILL_SWITCH
```

### Reset Memory
```powershell
# Clear Letta agents
curl -X DELETE http://localhost:8283/v1/agents/<agent-id>

# Clear file-based memory
Remove-Item .auto-claude/memory/*.json
```

---

*Unified Claude Code Configuration v1.0*
*Compatible with V10.1 Architecture*
*January 18, 2026*
