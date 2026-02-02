# The Ultimate Claude Code CLI + Letta Memory Integration Guide

> **Transform Claude Code into a persistent, learning development assistant with unlimited memory.**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Complete Infrastructure Setup](#complete-infrastructure-setup)
3. [MCP Server Configuration (All 3 Implementations)](#mcp-server-configuration)
4. [Complete Tool Reference (70+ Tools)](#complete-tool-reference)
5. [Custom Slash Commands for Memory Operations](#custom-slash-commands)
6. [Hooks for Automatic Memory Sync](#hooks-configuration)
7. [CLAUDE.md Integration Patterns](#claudemd-integration)
8. [Skills for Advanced Memory Workflows](#skills-configuration)
9. [Production Deployment Patterns](#production-deployment)
10. [Advanced Usage Patterns](#advanced-usage-patterns)
11. [Troubleshooting Guide](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLAUDE CODE CLI                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Hooks     │  │  Commands   │  │   Skills    │  │  CLAUDE.md  │        │
│  │ (Auto-sync) │  │ (/memory-*) │  │ (Auto-load) │  │  (Context)  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                   │                                          │
│                          MCP Protocol (stdio/http/sse)                       │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │     LETTA MCP SERVER          │
                    │  (oculairmedia/Letta-MCP)     │
                    │                               │
                    │  70+ Tools / 7 Consolidated   │
                    │  • Agent Management           │
                    │  • Memory Operations          │
                    │  • Archival Search            │
                    │  • Tool Integration           │
                    └───────────────┬───────────────┘
                                    │
                         REST API (port 8283)
                                    │
                    ┌───────────────┴───────────────┐
                    │        LETTA SERVER           │
                    │                               │
                    │  ┌─────────┐ ┌─────────────┐  │
                    │  │  Core   │ │  Archival   │  │
                    │  │ Memory  │ │   Memory    │  │
                    │  │(Blocks) │ │ (pgvector)  │  │
                    │  └────┬────┘ └──────┬──────┘  │
                    │       │             │         │
                    │       └──────┬──────┘         │
                    │              │                │
                    │    ┌─────────┴─────────┐      │
                    │    │    PostgreSQL     │      │
                    │    │    + pgvector     │      │
                    │    └───────────────────┘      │
                    └───────────────────────────────┘
```

### Memory Tiers Explained

| Tier | Purpose | Storage | Access Pattern |
|------|---------|---------|----------------|
| **Core Memory** | Always in context | In-prompt blocks | Agent self-edits via tools |
| **Archival Memory** | Long-term semantic storage | pgvector embeddings | Search & retrieve on demand |
| **Recall Memory** | Conversation history | PostgreSQL | Semantic search past chats |

---

## Complete Infrastructure Setup

### Step 1: Deploy Letta Server (Production-Ready)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  letta_db:
    image: ankane/pgvector:v0.5.1
    hostname: letta-db
    restart: always
    environment:
      POSTGRES_USER: ${LETTA_PG_USER:-letta}
      POSTGRES_PASSWORD: ${LETTA_PG_PASSWORD:-letta_secure_password}
      POSTGRES_DB: ${LETTA_PG_DB:-letta}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${LETTA_PG_USER:-letta}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - letta-network

  letta_server:
    image: letta/letta:latest
    hostname: letta-server
    depends_on:
      letta_db:
        condition: service_healthy
    environment:
      LETTA_PG_URI: postgresql://${LETTA_PG_USER:-letta}:${LETTA_PG_PASSWORD:-letta_secure_password}@letta-db:5432/${LETTA_PG_DB:-letta}
      SECURE: "true"
      LETTA_SERVER_PASSWORD: ${LETTA_SERVER_PASSWORD}
      LETTA_UVICORN_WORKERS: ${LETTA_UVICORN_WORKERS:-5}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    ports:
      - "8283:8283"
    restart: unless-stopped
    networks:
      - letta-network

  letta_mcp:
    image: ghcr.io/oculairmedia/letta-mcp-server:latest
    hostname: letta-mcp
    depends_on:
      - letta_server
    environment:
      LETTA_BASE_URL: http://letta-server:8283/v1
      LETTA_PASSWORD: ${LETTA_SERVER_PASSWORD}
      PORT: 3001
      NODE_ENV: production
    ports:
      - "3001:3001"
    restart: unless-stopped
    networks:
      - letta-network

volumes:
  pgdata:
    driver: local

networks:
  letta-network:
    driver: bridge
```

Create `.env` file:

```bash
# Database
LETTA_PG_USER=letta
LETTA_PG_PASSWORD=your_secure_db_password_here
LETTA_PG_DB=letta

# Server Security
LETTA_SERVER_PASSWORD=your_secure_api_password_here
LETTA_UVICORN_WORKERS=5

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Start everything:

```bash
docker compose up -d
docker compose logs -f
```

### Step 2: Create Your First Letta Agent

```python
#!/usr/bin/env python3
"""create_agent.py - Initialize your Claude Code memory agent"""

from letta_client import Letta
import json

# Connect to local server
client = Letta(
    base_url="http://localhost:8283",
    token="your_secure_api_password_here"  # Same as LETTA_SERVER_PASSWORD
)

# Create agent with comprehensive memory blocks
agent = client.agents.create(
    name="claude-code-memory",
    description="Persistent memory agent for Claude Code sessions",
    model="openai/gpt-4.1",  # or "anthropic/claude-sonnet-4-5-20250929"
    embedding="openai/text-embedding-3-small",
    memory_blocks=[
        {
            "label": "human",
            "value": """# Developer Profile
Name: [Your Name]
Role: [Your Role]
Primary Languages: Python, TypeScript, Rust
Preferences:
- Prefers type hints and docstrings
- Uses functional patterns where appropriate
- Values test coverage
Current Focus: [Your current project focus]"""
        },
        {
            "label": "persona",
            "value": """# Claude Code Memory Assistant
I am a persistent memory system for Claude Code sessions.
I remember:
- Past coding decisions and their rationale
- User preferences discovered through interactions
- Project architecture patterns
- Bugs encountered and their solutions
- Performance optimizations applied

I proactively surface relevant memories when they apply to current tasks."""
        },
        {
            "label": "project_context",
            "value": """# Active Projects
## Project 1: [Name]
- Stack: [Technologies]
- Status: [Current status]
- Key decisions: [List]

## Coding Standards
- [Standard 1]
- [Standard 2]"""
        },
        {
            "label": "learnings",
            "value": """# Session Learnings
This block stores insights discovered during coding sessions.
Format: [Date] - [Category] - [Learning]

---
[Learnings will be added here automatically]"""
        }
    ],
    tools=None,  # Add custom tools later
    include_base_tools=True
)

# Save agent ID for Claude Code configuration
config = {
    "agent_id": agent.id,
    "agent_name": agent.name,
    "created_at": str(agent.created_at)
}

with open("letta_agent_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✅ Agent created successfully!")
print(f"   Agent ID: {agent.id}")
print(f"   Agent Name: {agent.name}")
print(f"\n📝 Save this agent ID in your Claude Code configuration")
```

Run it:

```bash
pip install letta-client
python create_agent.py
```

---

## MCP Server Configuration

### Option A: Consolidated Tools (Recommended for Production)

The **nodejs-consolidated-tools** implementation offers 7 mega-tools with 93% SDK coverage:

```bash
# Install globally
npm install -g letta-mcp-server

# Or use Docker
docker pull ghcr.io/oculairmedia/letta-mcp-server:latest
```

**Claude Code CLI Registration:**

```bash
# stdio transport (local development)
claude mcp add letta -s user -- letta-mcp

# HTTP transport (production/remote)
claude mcp add --transport http letta http://localhost:3001/mcp
```

### Option B: Classic Individual Tools (70+ Tools)

Better for learning the API and granular control:

```bash
docker pull ghcr.io/oculairmedia/letta-mcp-server:master
```

### Option C: Rust Implementation (Maximum Performance)

~100-500ms startup, ~10-30MB memory:

```bash
docker pull ghcr.io/oculairmedia/letta-mcp-server-rust:rust-latest
```

### Complete MCP Configuration Files

**Project-scoped `.mcp.json`** (commit to git):

```json
{
  "mcpServers": {
    "letta": {
      "type": "stdio",
      "command": "letta-mcp",
      "args": [],
      "env": {
        "LETTA_BASE_URL": "http://localhost:8283/v1",
        "LETTA_PASSWORD": "${LETTA_PASSWORD}"
      },
      "timeout": 300
    }
  }
}
```

**User-scoped `~/.claude.json`** (personal, all projects):

```json
{
  "mcpServers": {
    "letta": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "letta-mcp-server"],
      "env": {
        "LETTA_BASE_URL": "http://localhost:8283/v1",
        "LETTA_PASSWORD": "your_password"
      }
    }
  }
}
```

**Windows Configuration** (requires cmd wrapper):

```json
{
  "mcpServers": {
    "letta": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "letta-mcp-server"],
      "env": {
        "LETTA_BASE_URL": "http://localhost:8283/v1",
        "LETTA_PASSWORD": "your_password"
      }
    }
  }
}
```

**Remote HTTP Configuration** (for Docker/cloud):

```json
{
  "mcpServers": {
    "letta": {
      "url": "http://your-server:3001/mcp",
      "transport": "http",
      "timeout": 120,
      "alwaysAllow": [
        "list_agents",
        "list_memory_blocks",
        "read_memory_block",
        "list_passages"
      ]
    }
  }
}
```

---

## Complete Tool Reference

### Agent Management Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `create_agent` | Create new Letta agent | `name`, `model`, `memory_blocks[]` |
| `list_agents` | List all agents | `page`, `pageSize` |
| `retrieve_agent` | Get agent by ID | `agent_id` |
| `get_agent_summary` | Get agent summary | `agent_id` |
| `modify_agent` | Update agent config | `agent_id`, `name`, `model` |
| `delete_agent` | Delete agent | `agent_id` |
| `clone_agent` | Clone existing agent | `agent_id`, `new_name` |
| `export_agent` | Export agent backup | `agent_id` |
| `import_agent` | Import from backup | `backup_data` |
| `bulk_delete_agents` | Delete multiple | `agent_ids[]` |

### Memory Block Tools (Core Memory)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_memory_blocks` | List all blocks | `label`, `page`, `include_full_content` |
| `create_memory_block` | Create new block | `name`, `label`, `value`, `metadata` |
| `read_memory_block` | Read block content | `block_id` |
| `update_memory_block` | Update block | `block_id`, `value`, `metadata` |
| `attach_memory_block` | Attach to agent | `block_id`, `agent_id`, `label` |
| `detach_memory_block` | Detach from agent | `block_id`, `agent_id` |

### Passage Tools (Archival Memory)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_passages` | Search archival memory | `agent_id`, `query`, `limit` |
| `create_passage` | Add to archival | `agent_id`, `text`, `metadata` |
| `modify_passage` | Update passage | `passage_id`, `text` |
| `delete_passage` | Remove passage | `passage_id` |

### Agent Messaging Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `prompt_agent` | Send message to agent | `agent_id`, `message`, `stream` |
| `get_agent_messages` | Get message history | `agent_id`, `limit`, `before` |

### Tool Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_agent_tools` | List agent's tools | `agent_id` |
| `attach_tool` | Attach tool to agent | `tool_id`, `agent_id` |
| `detach_tool` | Detach tool | `tool_id`, `agent_id` |
| `upload_tool` | Upload custom tool | `name`, `source_code`, `description` |
| `bulk_attach_tool_to_agents` | Bulk attach | `tool_id`, `agent_ids[]` |

### Model Management

| Tool | Description |
|------|-------------|
| `list_llm_models` | List available LLMs |
| `list_embedding_models` | List embedding models |

### MCP Integration

| Tool | Description |
|------|-------------|
| `list_mcp_servers` | List configured MCP servers |
| `list_mcp_tools_by_server` | List tools from MCP server |
| `add_mcp_tool_to_letta` | Import MCP tool to Letta |

---

## Custom Slash Commands

Create these in `.claude/commands/` for powerful memory workflows:

### `/memory-save.md` - Save Current Context

```markdown
---
description: Save current session learnings to Letta memory
allowed-tools: Bash(python:*)
---

# Save Session Learnings to Letta Memory

Summarize the key learnings, decisions, and insights from this coding session,
then save them to the Letta archival memory.

## Instructions

1. Review the conversation history for:
   - Technical decisions made
   - Bugs discovered and solutions
   - Performance optimizations
   - User preferences learned
   - Architecture patterns used

2. Format as structured entries:
   ```
   [Category]: [Brief description]
   Context: [Why this matters]
   Resolution: [What was done]
   ```

3. Use the Letta MCP tools to:
   - Call `create_passage` to add each learning to archival memory
   - Optionally update the `learnings` memory block if significant

4. Confirm what was saved.

Arguments: $ARGUMENTS
```

### `/memory-recall.md` - Search Past Context

```markdown
---
description: Search Letta memory for relevant past context
allowed-tools: Read, Grep, Glob
---

# Recall Relevant Memories

Search the Letta archival memory for context relevant to: $ARGUMENTS

## Instructions

1. Use `list_passages` with semantic search on the provided topic
2. Also check relevant memory blocks with `read_memory_block`
3. Present found memories in a structured format:
   - Most relevant first
   - Include dates if available
   - Note confidence/relevance scores

4. Suggest how these memories might apply to the current task

If no relevant memories found, say so and suggest what to save for future reference.
```

### `/memory-status.md` - View Memory State

```markdown
---
description: Display current Letta memory status
allowed-tools: Read
---

# Letta Memory Status

Display the current state of all memory systems.

## Show:

1. **Core Memory Blocks**
   - List all blocks with `list_memory_blocks`
   - Show character counts and last updated times

2. **Archival Memory Stats**
   - Total passages count
   - Recent additions (last 5)

3. **Agent Configuration**
   - Current agent ID and name
   - Attached tools
   - Model configuration

4. **Health Check**
   - Connection status
   - Any warnings or issues

Format output clearly with sections and counts.
```

### `/memory-sync.md` - Full Bidirectional Sync

```markdown
---
description: Sync Claude Code context with Letta memory
allowed-tools: Bash(python:*), Read, Write
---

# Full Memory Synchronization

Perform bidirectional sync between Claude Code session and Letta memory.

## Steps:

1. **Pull Phase** - Load relevant context from Letta:
   - Read all core memory blocks
   - Search archival for current project keywords
   - Get recent learnings

2. **Context Integration** - Merge with current session:
   - Identify relevant memories for current task
   - Note any conflicts with current context

3. **Push Phase** - Save session updates to Letta:
   - Update modified memory blocks
   - Create passages for new learnings
   - Update timestamps

4. **Report** - Show sync summary:
   - Memories loaded
   - Updates saved
   - Any conflicts resolved

$ARGUMENTS
```

### `/memory-project-init.md` - Initialize Project Memory

```markdown
---
description: Initialize Letta memory for a new project
allowed-tools: Bash(python:*), Read, Glob
---

# Initialize Project Memory

Set up Letta memory structure for the current project.

## Actions:

1. **Scan Project**
   - Read package.json, pyproject.toml, Cargo.toml
   - Identify tech stack and dependencies
   - Find existing documentation

2. **Create Project Memory Block**
   - Project name and description
   - Technology stack
   - Key file locations
   - Build/test commands

3. **Seed Archival Memory**
   - Architecture decisions from docs
   - README content
   - Existing CLAUDE.md content

4. **Update Agent Configuration**
   - Add project-specific context to persona
   - Configure relevant tools

Project path: $ARGUMENTS (or current directory)
```

---

## Hooks Configuration

Create `.claude/settings.json` for automatic memory operations:

```json
{
  "env": {
    "LETTA_BASE_URL": "http://localhost:8283/v1",
    "LETTA_PASSWORD": "${LETTA_PASSWORD}",
    "LETTA_AGENT_ID": "agent-xxxxx-your-agent-id"
  },
  
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/letta-session-start.py",
            "timeout": 30
          }
        ]
      }
    ],
    
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/letta-prompt-context.py",
            "timeout": 10
          }
        ]
      }
    ],
    
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/letta-track-changes.py",
            "timeout": 5
          }
        ]
      }
    ],
    
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/letta-session-end.py",
            "timeout": 30
          }
        ]
      }
    ]
  },
  
  "permissions": {
    "allow": [
      "Bash(python ~/.claude/hooks/*)",
      "Bash(letta-mcp:*)"
    ]
  }
}
```

### Hook Scripts

Create `~/.claude/hooks/letta-session-start.py`:

```python
#!/usr/bin/env python3
"""Load relevant Letta context at session start"""

import os
import sys
import json
from letta_client import Letta

def main():
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id:
        return
    
    client = Letta(
        base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
        token=os.environ.get("LETTA_PASSWORD", "")
    )
    
    try:
        # Load core memory blocks
        blocks = client.agents.blocks.list(agent_id=agent_id)
        
        context_parts = ["# Loaded from Letta Memory\n"]
        
        for block in blocks:
            context_parts.append(f"## {block.label.title()}\n{block.value}\n")
        
        # Get recent archival memories
        passages = client.agents.passages.list(
            agent_id=agent_id,
            limit=5
        )
        
        if passages:
            context_parts.append("## Recent Learnings\n")
            for p in passages[:5]:
                context_parts.append(f"- {p.text[:200]}...\n")
        
        # Output context (will be shown to Claude)
        print("\n".join(context_parts))
        
    except Exception as e:
        print(f"Letta sync skipped: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
```

Create `~/.claude/hooks/letta-session-end.py`:

```python
#!/usr/bin/env python3
"""Save session learnings to Letta at session end"""

import os
import sys
import json
from datetime import datetime
from letta_client import Letta

def main():
    # Read session data from stdin (provided by Claude Code)
    try:
        session_data = json.load(sys.stdin)
    except:
        session_data = {}
    
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id:
        return
    
    client = Letta(
        base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
        token=os.environ.get("LETTA_PASSWORD", "")
    )
    
    # Extract session summary if available
    summary = session_data.get("conversation_summary", "")
    session_id = session_data.get("session_id", "unknown")
    
    if summary:
        try:
            # Save to archival memory
            client.agents.passages.create(
                agent_id=agent_id,
                text=f"[Session {session_id}] [{datetime.now().isoformat()}]\n{summary}",
                metadata={"type": "session_summary", "session_id": session_id}
            )
            print(f"✅ Session saved to Letta memory")
        except Exception as e:
            print(f"⚠️ Could not save session: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
```

Create `~/.claude/hooks/letta-prompt-context.py`:

```python
#!/usr/bin/env python3
"""Inject relevant context based on user prompt"""

import os
import sys
import json
from letta_client import Letta

def main():
    try:
        input_data = json.load(sys.stdin)
        prompt = input_data.get("prompt", "")
    except:
        return
    
    if not prompt or len(prompt) < 10:
        return
    
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id:
        return
    
    client = Letta(
        base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
        token=os.environ.get("LETTA_PASSWORD", "")
    )
    
    try:
        # Search for relevant memories
        passages = client.agents.passages.list(
            agent_id=agent_id,
            query=prompt[:500],  # Use prompt as search query
            limit=3
        )
        
        if passages:
            print("\n📚 **Relevant memories found:**")
            for p in passages:
                print(f"- {p.text[:150]}...")
            print("")
            
    except Exception as e:
        pass  # Silently skip if search fails

if __name__ == "__main__":
    main()
```

---

## CLAUDE.md Integration

Create a comprehensive `CLAUDE.md` that references Letta memory:

```markdown
# Project: [Your Project Name]

## Memory Integration

This project uses Letta for persistent cross-session memory.

### Agent Configuration
- **Agent ID**: `agent-xxxxx` (set in LETTA_AGENT_ID env var)
- **Base URL**: http://localhost:8283/v1

### Memory Commands
- `/memory-recall [topic]` - Search past context
- `/memory-save` - Save session learnings
- `/memory-status` - View memory state
- `/memory-sync` - Full bidirectional sync

### Automatic Memory Behavior
- Session start: Loads relevant context from Letta
- During session: Searches for relevant memories on complex prompts
- Session end: Saves learnings to archival memory
- File changes: Tracks modifications for future reference

## Core Memory Blocks

### `human` Block
Contains: Developer profile, preferences, current focus
Update when: Preferences change, new skills learned

### `project_context` Block  
Contains: Project stack, architecture decisions, standards
Update when: Major architectural changes, new integrations

### `learnings` Block
Contains: Session-by-session discoveries and solutions
Update when: Automatically at session end

## Guidelines

1. **Before major decisions**: Run `/memory-recall [topic]` to check past context
2. **After solving hard bugs**: Run `/memory-save` to preserve the solution
3. **Starting new features**: Run `/memory-sync` to load full context
4. **Weekly**: Review memory status with `/memory-status`

## Tech Stack
[Your stack details...]

## Commands
[Your build/test commands...]
```

---

## Skills Configuration

Create advanced skills for automatic memory workflows.

### `~/.claude/skills/letta-memory/SKILL.md`

```markdown
---
name: letta-memory
description: Automatically manage Letta memory integration. Use when discussing past decisions, saving learnings, or needing historical context.
allowed-tools: Bash(python:*), Read, Write
---

# Letta Memory Management Skill

## When to Activate
- User references past conversations or decisions
- Complex problem that might have been solved before
- Session contains valuable learnings to preserve
- User asks about project history or patterns

## Capabilities

### Memory Search
When user asks about past decisions or "what did we decide about X":
1. Use `list_passages` tool with semantic query
2. Also check relevant memory blocks
3. Present found context with dates and relevance

### Memory Save
When significant learning or decision occurs:
1. Format as structured passage
2. Use `create_passage` to store in archival
3. Update relevant memory block if needed
4. Confirm what was saved

### Context Loading
At session start or when context seems missing:
1. Load all core memory blocks
2. Search archival for current topic keywords
3. Inject relevant context into conversation

## Memory Block Labels
- `human` - User profile and preferences
- `persona` - Agent behavior configuration
- `project_context` - Current project details
- `learnings` - Session discoveries

## Best Practices
- Always check memory before claiming ignorance
- Save architectural decisions immediately
- Update user preferences when discovered
- Keep passages concise but complete
```

### `~/.claude/skills/context-aware-coding/SKILL.md`

```markdown
---
name: context-aware-coding
description: Use historical context from Letta memory to inform coding decisions. Activates for architecture, patterns, and complex implementations.
allowed-tools: Bash(python:*), Read, Write, Grep, Glob
---

# Context-Aware Coding Skill

## Activation Triggers
- Architecture discussions
- Pattern implementation
- Bug that might have been seen before
- Performance optimization tasks

## Workflow

### Before Implementing
1. Search Letta for similar past implementations
2. Check if patterns were established
3. Look for related bugs/solutions
4. Load relevant project_context

### During Implementation
1. Follow established patterns from memory
2. Note any deviations for later documentation
3. Reference past decisions when relevant

### After Implementation
1. Save new patterns to archival memory
2. Update project_context if needed
3. Document any important learnings

## Integration Points
- Use with `/memory-recall` for explicit searches
- Automatically search on complex prompts
- Save learnings on session end
```

---

## Production Deployment

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: letta-server
  labels:
    app: letta
spec:
  replicas: 2
  selector:
    matchLabels:
      app: letta
  template:
    metadata:
      labels:
        app: letta
    spec:
      containers:
      - name: letta
        image: letta/letta:latest
        ports:
        - containerPort: 8283
        env:
        - name: LETTA_PG_URI
          valueFrom:
            secretKeyRef:
              name: letta-secrets
              key: pg-uri
        - name: LETTA_SERVER_PASSWORD
          valueFrom:
            secretKeyRef:
              name: letta-secrets
              key: server-password
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: letta-secrets
              key: openai-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8283
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: letta-mcp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: letta-mcp
  template:
    spec:
      containers:
      - name: mcp
        image: ghcr.io/oculairmedia/letta-mcp-server:latest
        ports:
        - containerPort: 3001
        env:
        - name: LETTA_BASE_URL
          value: "http://letta-server:8283/v1"
        - name: LETTA_PASSWORD
          valueFrom:
            secretKeyRef:
              name: letta-secrets
              key: server-password
```

### Backup Strategy

```bash
#!/bin/bash
# backup-letta.sh - Run daily via cron

BACKUP_DIR="/backups/letta"
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
docker exec letta-db pg_dump -U letta letta | gzip > "$BACKUP_DIR/db_$DATE.sql.gz"

# Export all agents
for agent_id in $(curl -s http://localhost:8283/v1/agents | jq -r '.[].id'); do
  curl -s "http://localhost:8283/v1/agents/$agent_id/export" > "$BACKUP_DIR/agent_${agent_id}_$DATE.json"
done

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.json" -mtime +30 -delete

echo "Backup completed: $DATE"
```

---

## Advanced Usage Patterns

### Pattern 1: Multi-Project Memory Isolation

Create separate agents per project:

```python
# Create project-specific agent
project_agent = client.agents.create(
    name=f"memory-{project_name}",
    memory_blocks=[
        {"label": "human", "value": "..."},
        {"label": "project", "value": f"Project: {project_name}\nStack: ..."}
    ]
)

# Store agent ID in project's .claude/settings.local.json
```

### Pattern 2: Team Shared Memory

Use shared memory blocks across agents:

```python
# Create shared team knowledge block
team_standards = client.blocks.create(
    label="team_standards",
    value="# Team Coding Standards\n..."
)

# Attach to all team members' agents
for agent_id in team_agent_ids:
    client.agents.blocks.attach(
        agent_id=agent_id,
        block_id=team_standards.id
    )
```

### Pattern 3: Automated Learning Pipeline

```python
# Hook into CI/CD to save deployment learnings
def on_deployment_complete(deployment_info):
    client.agents.passages.create(
        agent_id=AGENT_ID,
        text=f"""Deployment completed:
        - Version: {deployment_info['version']}
        - Environment: {deployment_info['env']}
        - Duration: {deployment_info['duration']}
        - Issues: {deployment_info.get('issues', 'None')}
        """,
        metadata={"type": "deployment", "version": deployment_info['version']}
    )
```

---

## Troubleshooting

### Connection Issues

```bash
# Test Letta server
curl http://localhost:8283/health

# Test MCP server
curl http://localhost:3001/health

# Verify MCP in Claude Code
claude mcp list
```

### Memory Not Persisting

1. Check agent ID is correct in settings
2. Verify PostgreSQL is running: `docker compose ps`
3. Check Letta logs: `docker compose logs letta_server`

### Slow Memory Operations

1. Add indexes to PostgreSQL:
```sql
CREATE INDEX idx_passages_agent ON passages(agent_id);
CREATE INDEX idx_passages_embedding ON passages USING ivfflat (embedding vector_cosine_ops);
```

2. Increase Letta workers: `LETTA_UVICORN_WORKERS=10`

### Windows-Specific Issues

1. Always use `cmd /c` wrapper for npx commands
2. Use forward slashes in paths: `C:/Users/...`
3. Set environment variables in PowerShell:
```powershell
$env:LETTA_BASE_URL = "http://localhost:8283/v1"
$env:LETTA_PASSWORD = "your_password"
```

---

## Quick Reference Card

### Essential Commands

```bash
# Start infrastructure
docker compose up -d

# Add MCP server
claude mcp add letta -s user -- letta-mcp

# Verify connection
claude mcp list

# In Claude Code session
/memory-status          # Check memory state
/memory-recall [topic]  # Search memories
/memory-save            # Save learnings
/memory-sync            # Full sync
```

### Essential Environment Variables

```bash
export LETTA_BASE_URL="http://localhost:8283/v1"
export LETTA_PASSWORD="your_password"
export LETTA_AGENT_ID="agent-xxxxx"
```

### File Locations

| File | Purpose |
|------|---------|
| `~/.claude.json` | User MCP config |
| `.mcp.json` | Project MCP config |
| `~/.claude/settings.json` | User hooks/permissions |
| `.claude/settings.json` | Project hooks/permissions |
| `~/.claude/commands/*.md` | Personal slash commands |
| `.claude/commands/*.md` | Project slash commands |
| `~/.claude/skills/*/SKILL.md` | Personal skills |
| `.claude/skills/*/SKILL.md` | Project skills |
| `CLAUDE.md` | Project context |

---

## Resources

- **Letta GitHub**: https://github.com/letta-ai/letta
- **Letta Python SDK**: https://github.com/letta-ai/letta-python
- **Letta MCP Server**: https://github.com/oculairmedia/Letta-MCP-server
- **Letta Documentation**: https://docs.letta.com
- **Claude Code Documentation**: https://docs.anthropic.com/en/docs/claude-code

---

*This guide enables Claude Code to maintain persistent memory across sessions, learn from past interactions, and provide increasingly contextual assistance over time.*
