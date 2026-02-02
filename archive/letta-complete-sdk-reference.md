# Letta Complete SDK & Advanced Features Reference

## Table of Contents
1. [Required SDKs & Installation](#required-sdks--installation)
2. [Core Architecture Overview](#core-architecture-overview)
3. [Python SDK Complete Reference](#python-sdk-complete-reference)
4. [MCP Server Implementations](#mcp-server-implementations)
5. [Advanced Memory Patterns](#advanced-memory-patterns)
6. [Claude Code Hooks Integration](#claude-code-hooks-integration)
7. [API Endpoints Reference](#api-endpoints-reference)
8. [Production Deployment](#production-deployment)
9. [Official Documentation Links](#official-documentation-links)

---

## Required SDKs & Installation

### Python SDK (Primary)

```bash
# Core Letta Python Client
pip install letta-client

# With all optional dependencies
pip install letta-client[all]

# Version pinning for production (recommended: 1.7.1+)
pip install letta-client==1.7.1
```

**Import patterns:**
```python
from letta_client import Letta
from letta_client.types import Block, Passage, Agent, Message
from letta_client.types import CreateAgentRequest, UpdateBlockRequest
```

### TypeScript/Node.js SDK

```bash
# NPM installation
npm install letta-client

# Or with Yarn
yarn add letta-client

# MCP Server (for Claude Code integration)
npm install -g letta-mcp-server
```

**Import patterns:**
```typescript
import { Letta } from 'letta-client';
import type { Agent, Block, Passage, Message } from 'letta-client';
```

### Docker Images

```bash
# Letta Server (production)
docker pull letta/letta:latest

# PostgreSQL with pgvector (required for embeddings)
docker pull ankane/pgvector:v0.5.1

# MCP Server variants
docker pull ghcr.io/oculairmedia/letta-mcp-server:latest           # Consolidated (7 mega-tools)
docker pull ghcr.io/oculairmedia/letta-mcp-server:master           # Classic (70+ tools)
docker pull ghcr.io/oculairmedia/letta-mcp-server-rust:rust-latest # Rust (max performance)
```

### Anthropic SDKs (for Claude integration)

```bash
# Python
pip install anthropic
pip install anthropic[bedrock]  # AWS Bedrock
pip install anthropic[vertex]   # Google Vertex

# TypeScript
npm install @anthropic-ai/sdk
npm install @anthropic-ai/bedrock-sdk
npm install @anthropic-ai/vertex-sdk
```

---

## Core Architecture Overview

### Memory Tier System

```
┌─────────────────────────────────────────────────────────────────┐
│                     LETTA MEMORY ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    CORE MEMORY                           │    │
│  │  Always in context window • Agent self-edits via tools   │    │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐      │    │
│  │  │ human   │ │ persona │ │ project  │ │ learnings│      │    │
│  │  │ profile │ │ config  │ │ context  │ │ insights │      │    │
│  │  └─────────┘ └─────────┘ └──────────┘ └──────────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │  ARCHIVAL MEMORY  │                        │
│                    │  pgvector store   │                        │
│                    │  semantic search  │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │  RECALL MEMORY    │                        │
│                    │  conversation     │                        │
│                    │  history search   │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

| Tier | Purpose | Storage | Access Pattern |
|------|---------|---------|----------------|
| **Core Memory** | Always in context | In-prompt blocks | Agent self-edits via tools |
| **Archival Memory** | Long-term semantic storage | pgvector embeddings | Search & retrieve on demand |
| **Recall Memory** | Conversation history | PostgreSQL | Semantic search past chats |

### Block Structure (XML-like prepended to prompt)

```xml
<persona>
I am a persistent memory system for development sessions.
I remember past coding decisions, user preferences, and project patterns.
</persona>

<human>
Name: Developer
Role: Senior Engineer
Languages: Python, TypeScript, Rust
Preferences: Type hints, functional patterns, test coverage
</human>

<project_context>
Project: AlphaForge Trading System
Stack: Rust (kill switch), Python (ML), PostgreSQL (data)
Status: Architecture phase
</project_context>

<learnings>
[2025-01-15] - Architecture - Claude CLI external to production trading
[2025-01-16] - Pattern - Hierarchical memory isolation per project
</learnings>
```

---

## Python SDK Complete Reference

### Client Initialization

```python
from letta_client import Letta

# Local development
client = Letta(
    base_url="http://localhost:8283",
    token="your_secure_password"
)

# Cloud deployment
client = Letta(
    base_url="https://your-letta-server.com",
    token="your_api_key",
    timeout=60  # seconds
)
```

### Agent Management

```python
# Create agent with comprehensive memory blocks
agent = client.agents.create(
    name="claude-code-memory",
    description="Persistent memory for Claude Code sessions",
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
Current Focus: [Current project focus]"""
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

I proactively surface relevant memories when applicable."""
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

print(f"Agent created: {agent.id}")
```

```python
# List agents with pagination
agents = client.agents.list(
    page=1,
    page_size=20
)

# Retrieve specific agent
agent = client.agents.retrieve(agent_id="agent-xxxxx")

# Update agent configuration
updated = client.agents.update(
    agent_id=agent.id,
    name="updated-name",
    model="anthropic/claude-sonnet-4-5-20250929"
)

# Enable Sleeptime Memory (async background memory updates)
# Sleeptime offloads memory operations to an async agent that
# passively updates memory blocks in the background, reducing latency
updated = client.agents.update(
    agent_id=agent.id,
    enable_sleeptime=True,  # Enable background memory updates
    context_window_limit=200000,  # Increase context window
    max_tokens=8192  # Increase output limit
)

# Delete agent
client.agents.delete(agent_id=agent.id)

# Clone agent
cloned = client.agents.clone(
    agent_id=agent.id,
    new_name="cloned-agent"
)

# Export/Import for backup
backup = client.agents.export(agent_id=agent.id)
restored = client.agents.import_agent(backup_data=backup)
```

### Core Memory Block Operations

```python
# List all memory blocks for an agent
blocks = client.agents.blocks.list(agent_id=agent.id)
for block in blocks:
    print(f"{block.label}: {len(block.value)} chars")

# Get specific block by label
human_block = client.agents.blocks.get(
    agent_id=agent.id,
    block_label="human"
)

# Update memory block content
client.agents.blocks.update(
    agent_id=agent.id,
    block_label="learnings",
    value="""# Session Learnings
[2025-01-17] - Pattern - Use hierarchical memory blocks for isolation
[2025-01-17] - Decision - Claude CLI external to trading production
"""
)
```

### Standalone Blocks (Shareable)

```python
# Create standalone block (shareable between agents)
shared_block = client.blocks.create(
    label="team_standards",
    value="""# Team Coding Standards
- All functions require docstrings
- Type hints mandatory for public APIs
- Tests required for new features
- PR reviews needed before merge""",
    description="Shared coding standards for all team agents"
)

# Attach block to multiple agents
for agent_id in ["agent-1", "agent-2", "agent-3"]:
    client.agents.blocks.attach(
        agent_id=agent_id,
        block_id=shared_block.id
    )

# Create read-only block (prevents agent modification)
readonly_block = client.blocks.create(
    label="company_policies",
    value="# Company Security Policies\n...",
    read_only=True  # Agent cannot modify
)

# Update shared block (propagates to all attached agents)
client.blocks.update(
    block_id=shared_block.id,
    value="# Updated Team Standards\n..."
)

# List agents attached to a block
attached_agents = client.blocks.list_agents(block_id=shared_block.id)

# Detach block from agent
client.agents.blocks.detach(
    agent_id=agent.id,
    block_id=shared_block.id
)
```

### Archival Memory (Long-term Storage)

```python
# Add passage to archival memory
passage = client.agents.passages.create(
    agent_id=agent.id,
    text="""Discovered that the WebSocket reconnection logic 
    needs exponential backoff with jitter to prevent thundering herd
    when the server restarts. Implemented in ws_client.py""",
    metadata={
        "type": "learning",
        "category": "networking",
        "file": "ws_client.py",
        "timestamp": "2025-01-17T10:30:00Z"
    }
)

# List passages with pagination
passages = client.agents.passages.list(
    agent_id=agent.id,
    limit=20,
    offset=0
)

# Semantic search in archival memory
search_results = client.agents.passages.search(
    agent_id=agent.id,
    query="WebSocket reconnection",
    limit=5
)

for result in search_results:
    print(f"Score: {result.score:.2f}")
    print(f"Text: {result.text[:200]}...")
    print(f"Metadata: {result.metadata}")
    print("---")

# Delete specific passage
client.agents.passages.delete(
    agent_id=agent.id,
    passage_id=passage.id
)
```

### Messaging & Conversation

```python
# Send message to agent (synchronous)
response = client.agents.messages.create(
    agent_id=agent.id,
    messages=[
        {"role": "user", "content": "What do you remember about our project?"}
    ]
)

print(response.messages[-1].content)

# Streaming messages
async def stream_response():
    async with client.agents.messages.stream(
        agent_id=agent.id,
        messages=[{"role": "user", "content": "Analyze this codebase"}]
    ) as stream:
        async for chunk in stream:
            print(chunk.content, end="", flush=True)

# Async message (returns run_id for polling)
run = client.agents.messages.create_async(
    agent_id=agent.id,
    messages=[{"role": "user", "content": "Long-running analysis..."}]
)

# Poll for completion
import time
while True:
    status = client.runs.get(run_id=run.id)
    if status.status == "completed":
        break
    time.sleep(1)

# Get message history
history = client.agents.messages.list(
    agent_id=agent.id,
    limit=50
)

# Reset message history (keep memory blocks)
client.agents.messages.reset(agent_id=agent.id)
```

### Tools Management

```python
# Create custom tool with source code
tool = client.tools.create(
    source_code="""
def analyze_code(code: str, language: str) -> dict:
    \"\"\"
    Analyze code for patterns, issues, and suggestions.

    Args:
        code: Source code to analyze
        language: Programming language (python, typescript, rust)

    Returns:
        Analysis results with patterns, issues, and suggestions
    \"\"\"
    # Implementation here
    return {"patterns": [], "issues": [], "suggestions": []}
""",
    description="Analyze code for patterns and issues",  # Optional override
    pip_requirements=[{"req_string": "pylint>=2.0"}],  # Python dependencies
    tags=["code-analysis", "developer-tools"],  # For organization
    default_requires_approval=False  # Safety flag
)

# List all available tools
all_tools = client.tools.list(limit=50)
for t in all_tools:
    print(f"{t.name}: {t.description}")

# Get tool details including source code
tool_details = client.tools.retrieve(tool_id=tool.id)

# Delete tool
client.tools.delete(tool_id=tool.id)

# Attach tool to agent
client.agents.tools.attach(
    agent_id=agent.id,
    tool_id=tool.id
)

# List agent's tools
tools = client.agents.tools.list(agent_id=agent.id)

# Run tool directly (bypass agent)
result = client.agents.tools.run(
    agent_id=agent.id,
    tool_name="analyze_code",
    arguments={"code": "def foo(): pass", "language": "python"}
)

# Set tool approval requirement
client.agents.tools.set_approval(
    agent_id=agent.id,
    tool_name="dangerous_operation",
    approval_required=True
)

# Detach tool
client.agents.tools.detach(
    agent_id=agent.id,
    tool_id=tool.id
)
```

### Tags-Based Agent Filtering (Recommended Pattern)

The recommended Letta pattern for multi-user and multi-project agent management uses tags.
This pattern is simpler and more flexible than identity-based filtering.

```python
# Create agent with tags (recommended approach)
agent = client.agents.create(
    name="claude-code-project-alpha",
    model="anthropic/claude-sonnet-4-5-20250929",
    tags=["project:alpha", "source:claude-code", "team:backend"],
    memory_blocks=[
        {"label": "human", "value": "User profile..."},
        {"label": "persona", "value": "Assistant persona..."}
    ]
)

# List agents filtered by tag (efficient lookup)
project_agents = client.agents.list(
    tags=["project:alpha"],
    limit=10
)

# Multi-user pattern: one agent per user
def get_or_create_user_agent(user_id: str):
    """Get existing agent or create new one for user."""
    # Check for existing agent using tags
    agents = client.agents.list(tags=[f"user:{user_id}"], limit=1)
    if agents:
        return agents[0]

    # Create new agent for user
    return client.agents.create(
        model="anthropic/claude-sonnet-4-5-20250929",
        tags=[f"user:{user_id}", "source:claude-code"],
        memory_blocks=[
            {"label": "persona", "value": "I am a helpful assistant..."},
            {"label": "human", "value": f"User ID: {user_id}"}
        ]
    )

# Get agent for specific user
user_agent = get_or_create_user_agent("user_123")
```

**Tag naming conventions:**
| Tag Format | Use Case |
|------------|----------|
| `project:{name}` | Project-specific agents |
| `user:{id}` | Per-user agents |
| `source:claude-code` | Provenance tracking |
| `team:{name}` | Team-specific agents |
| `type:user-agent` | Agent categorization |

### Team Shared Memory Pattern

Create shared knowledge blocks that multiple agents can access and update.

```python
# Create shared knowledge block
knowledge = client.blocks.create(
    label="shared_knowledge",
    value="Facts all team agents should know..."
)

# Create multiple agents
agent1 = client.agents.create(
    name="agent-1",
    tags=["team:backend"],
    memory_blocks=[...]
)
agent2 = client.agents.create(
    name="agent-2",
    tags=["team:backend"],
    memory_blocks=[...]
)

# Attach shared block to both agents
client.agents.blocks.attach(agent_id=agent1.id, block_id=knowledge.id)
client.agents.blocks.attach(agent_id=agent2.id, block_id=knowledge.id)

# Now both agents see the same block content!
# Updates to the block are visible to all attached agents

# Update shared block (affects all agents)
client.blocks.update(
    block_id=knowledge.id,
    value="Updated shared knowledge..."
)
```

### Multi-User Identity Management (Legacy)

For complex identity requirements, use the identities API:

```python
# Create identity for user isolation
identity = client.identities.create(
    identifier_key="user_123",
    metadata={"name": "John Developer", "team": "backend"}
)

# Create agent linked to identity
user_agent = client.agents.create(
    name="john-assistant",
    identity_ids=[identity.id],
    memory_blocks=[...]
)

# Filter agents by identity
user_agents = client.agents.list(
    identity_id=identity.id
)
```

### Sleeptime Agents (Background Memory Management)

Sleeptime memory offloads memory operations to an asynchronous agent that passively
updates the primary agent's memory blocks in the background, reducing latency for
the main interaction loop.

```python
# Create agent with sleeptime enabled
sleeptime_agent = client.agents.create(
    name="learning-agent",
    enable_sleeptime=True,  # Enables background memory processing
    sleeptime_agent_frequency=5,  # Trigger every 5 steps
    memory_blocks=[...]
)

# How sleeptime works:
# 1. Multi-agent group is created (primary + sleeptime agent)
# 2. Sleeptime agent runs in background every N steps
# 3. Sleeptime agent has tools to manage primary agent's memory
# 4. Derives learned context from conversation history
# 5. Updates shared memory blocks automatically
```

**Benefits:**
- Lower latency for user interactions (memory ops happen async)
- Automatic context extraction from conversations
- Memory consolidation without blocking main agent

### Core Memory Tools Reference

The agent uses these built-in tools to manage its core memory blocks:

| Tool | Description | Usage |
|------|-------------|-------|
| `memory_insert` | Add new content to a memory block | Append new facts or context |
| `memory_replace` | Replace existing content in a block | Update stale information |
| `memory_rethink` | Rewrite block content with new understanding | Consolidate or restructure |

**Memory block structure in agent context:**
```xml
<memory_blocks>
<persona>
<description>The persona block: Stores details about your current persona...</description>
<metadata>
- chars_current=128
- chars_limit=5000
</metadata>
<value>I am a helpful assistant named Sam.</value>
</persona>
<human>
<description>The human block: Stores key details about the person...</description>
<metadata>
- chars_current=84
- chars_limit=5000
</metadata>
<value>The user's name is Alice.</value>
</human>
</memory_blocks>
```

**Memory tiers overview:**

| Tier | Description | Tools |
|------|-------------|-------|
| Core (In-Context) | Structured sections of the context window (human, persona) | memory_insert, memory_replace, memory_rethink |
| Archival | Long-term vector-store memory for facts and knowledge | archival_memory_insert, archival_memory_search |
| Recall | Conversation history log of all past interactions | conversation_search, conversation_search_date |

---

## MCP Server Implementations

### Option A: Consolidated Tools (Recommended)

```bash
# Install globally
npm install -g letta-mcp-server

# Or via npx
npx -y letta-mcp-server
```

**Claude Code CLI Registration:**
```bash
# stdio transport (local)
claude mcp add letta -s user -- letta-mcp

# HTTP transport (remote/production)
claude mcp add --transport http letta http://localhost:3001/mcp
```

**Project `.mcp.json`:**
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

**Windows Configuration (requires cmd wrapper):**
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

### Tool Reference (Consolidated)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `create_agent` | Create new Letta agent | `name`, `model`, `memory_blocks[]` |
| `list_agents` | List all agents | `page`, `pageSize` |
| `retrieve_agent` | Get agent by ID | `agent_id` |
| `modify_agent` | Update agent config | `agent_id`, `name`, `model` |
| `delete_agent` | Delete agent | `agent_id` |
| `list_memory_blocks` | List all blocks | `label`, `page`, `include_full_content` |
| `create_memory_block` | Create new block | `name`, `label`, `value`, `metadata` |
| `read_memory_block` | Read block content | `block_id` |
| `update_memory_block` | Update block | `block_id`, `value`, `metadata` |
| `attach_memory_block` | Attach to agent | `block_id`, `agent_id`, `label` |
| `list_passages` | List archival passages | `agent_id`, `limit`, `offset` |
| `create_passage` | Add to archival | `agent_id`, `text`, `metadata` |
| `search_passages` | Semantic search | `agent_id`, `query`, `limit` |

---

## Advanced Memory Patterns

### Hierarchical Memory System

Letta uses a hierarchical memory system where memory blocks are structured sections of the agent's context window that persist across all interactions. Blocks are prepended in XML-like format with tags for each block (`<persona>`, `<human>`, etc.). Each block has: label, description, value, and character limit.

### User Identity-Based Isolation

```python
class MultiUserMemoryManager:
    """Manages isolated memory spaces per user."""
    
    def __init__(self, client: Letta):
        self.client = client
    
    def create_user_agent(self, user_id: str, user_name: str):
        """Create isolated agent for specific user."""
        
        # Create identity
        identity = self.client.identities.create(
            identifier_key=user_id,
            metadata={"name": user_name}
        )
        
        # Create agent with identity binding
        agent = self.client.agents.create(
            name=f"{user_name}-assistant",
            identity_ids=[identity.id],
            memory_blocks=[
                {
                    "label": "human",
                    "value": f"# User: {user_name}\nID: {user_id}"
                },
                {
                    "label": "user_preferences",
                    "value": "# Discovered Preferences\n(None yet)"
                }
            ]
        )
        
        return agent, identity
    
    def get_user_agents(self, user_id: str):
        """Retrieve all agents for a specific user."""
        identity = self.client.identities.get(identifier_key=user_id)
        return self.client.agents.list(identity_id=identity.id)
```

### Team Shared Memory Implementation

```python
class TeamMemoryManager:
    """Manages shared memory blocks across team agents."""
    
    def __init__(self, base_url: str, password: str):
        self.client = Letta(base_url=base_url, token=password)
    
    def create_shared_block(
        self,
        label: str,
        value: str,
        description: str = ""
    ) -> dict:
        """Create a shareable memory block."""
        block = self.client.blocks.create(
            label=label,
            value=value,
            description=description
        )
        return {"block_id": block.id, "label": block.label}
    
    def attach_to_agents(
        self,
        block_id: str,
        agent_ids: list
    ) -> dict:
        """Attach shared block to multiple agents."""
        results = {"attached": [], "failed": []}
        
        for agent_id in agent_ids:
            try:
                self.client.agents.blocks.attach(
                    agent_id=agent_id,
                    block_id=block_id
                )
                results["attached"].append(agent_id)
            except Exception as e:
                results["failed"].append({
                    "agent_id": agent_id,
                    "error": str(e)
                })
        
        return results
    
    def create_team_standards(self) -> dict:
        """Create standard team coding standards block."""
        return self.create_shared_block(
            label="team_standards",
            value="""# Team Coding Standards
## Code Quality
- All functions require docstrings
- Type hints mandatory for public APIs
- Maximum cyclomatic complexity: 10

## Testing Requirements
- Unit tests for all business logic
- Integration tests for API endpoints
- Minimum 80% code coverage

## Review Process
- PR reviews required before merge
- Security review for auth changes
- Performance review for DB changes""",
            description="Shared coding standards for all team agents"
        )
```

### Sleeptime Background Learning

Sleeptime agents provide automated learning by running in background to generate learned context from conversation history.

```python
# Enable sleeptime for automated memory management
agent = client.agents.create(
    name="learning-dev-assistant",
    enable_sleeptime=True,
    sleeptime_agent_frequency=5,  # Process every 5 interaction steps
    memory_blocks=[
        {"label": "human", "value": "# Developer\nPreferences: TBD"},
        {"label": "learnings", "value": "# Auto-populated learnings\n"}
    ]
)

# The sleeptime agent:
# 1. Shares memory blocks with primary agent
# 2. Runs in background every N steps
# 3. Has tools to manage memory (update_memory_block, etc.)
# 4. Iteratively derives learned context from conversations
# 5. Saves insights to shared blocks
```

---

## Claude Code Hooks Integration

### Complete Settings Configuration

**`.claude/settings.json`:**
```json
{
  "env": {
    "LETTA_BASE_URL": "http://localhost:8283/v1",
    "LETTA_PASSWORD": "${LETTA_PASSWORD}",
    "LETTA_AGENT_ID": "agent-xxxxx-your-agent-id"
  },
  
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "python ~/.claude/hooks/letta-session-start.py",
        "timeout": 30
      }]
    }],
    
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "python ~/.claude/hooks/letta-prompt-context.py",
        "timeout": 10
      }]
    }],
    
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "python ~/.claude/hooks/letta-track-changes.py",
        "timeout": 5
      }]
    }],
    
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python ~/.claude/hooks/letta-session-end.py",
        "timeout": 30
      }]
    }]
  },
  
  "permissions": {
    "allow": [
      "Bash(python ~/.claude/hooks/*)",
      "Bash(letta-mcp:*)"
    ]
  }
}
```

### Session Start Hook

**`~/.claude/hooks/letta-session-start.py`:**
```python
#!/usr/bin/env python3
"""Load relevant Letta context at session start."""

import os
import sys
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from letta_client import Letta
    LETTA_AVAILABLE = True
except ImportError:
    LETTA_AVAILABLE = False


class LettaSessionLoader:
    def __init__(self):
        self.base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self.password = os.environ.get("LETTA_PASSWORD", "")
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")
        self.project_path = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
        
        self.client = Letta(
            base_url=self.base_url,
            token=self.password
        )
    
    def load_core_memory_blocks(self) -> List[Dict[str, Any]]:
        """Load all core memory blocks for the agent."""
        try:
            blocks = self.client.agents.blocks.list(agent_id=self.agent_id)
            return [
                {
                    "id": block.id,
                    "label": block.label,
                    "value": block.value,
                    "limit": block.limit
                }
                for block in blocks
            ]
        except Exception as e:
            logger.error(f"Failed to load memory blocks: {e}")
            return []
    
    def load_recent_passages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Load recent archival memory passages."""
        try:
            passages = self.client.agents.passages.list(
                agent_id=self.agent_id,
                limit=limit
            )
            return [
                {
                    "id": p.id,
                    "text": p.text[:500],
                    "metadata": p.metadata
                }
                for p in passages
            ]
        except Exception as e:
            logger.error(f"Failed to load passages: {e}")
            return []
    
    def search_project_context(self) -> List[Dict[str, Any]]:
        """Search archival memory for current project context."""
        from pathlib import Path
        project_name = Path(self.project_path).name
        
        try:
            results = self.client.agents.passages.search(
                agent_id=self.agent_id,
                query=f"project {project_name}",
                limit=5
            )
            return [
                {"text": r.text[:300], "score": r.score}
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to search project context: {e}")
            return []


def main():
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id or not LETTA_AVAILABLE:
        return
    
    try:
        loader = LettaSessionLoader()
        
        context_parts = ["# 🧠 Loaded from Letta Memory\n"]
        
        # Load core memory blocks
        blocks = loader.load_core_memory_blocks()
        for block in blocks:
            context_parts.append(f"## {block['label'].title()}\n{block['value'][:500]}\n")
        
        # Load recent learnings
        passages = loader.load_recent_passages(5)
        if passages:
            context_parts.append("## Recent Learnings\n")
            for p in passages[:5]:
                context_parts.append(f"- {p['text'][:200]}...\n")
        
        print("\n".join(context_parts))
        
    except Exception as e:
        print(f"Letta sync skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

### Session End Hook

**`~/.claude/hooks/letta-session-end.py`:**
```python
#!/usr/bin/env python3
"""Save session learnings to Letta at session end."""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

try:
    from letta_client import Letta
    LETTA_AVAILABLE = True
except ImportError:
    LETTA_AVAILABLE = False


class LettaSessionSaver:
    def __init__(self):
        self.client = Letta(
            base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
            token=os.environ.get("LETTA_PASSWORD", "")
        )
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")
        self.project_path = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    
    def save_session_summary(self, summary: str, session_id: str) -> bool:
        """Save session summary to archival memory."""
        if not summary or not self.agent_id:
            return False
        
        try:
            self.client.agents.passages.create(
                agent_id=self.agent_id,
                text=f"""Session Summary ({datetime.now().isoformat()})
Session ID: {session_id}
Project: {Path(self.project_path).name}

{summary}""",
                metadata={
                    "type": "session_summary",
                    "session_id": session_id,
                    "project": Path(self.project_path).name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            return True
        except Exception as e:
            print(f"Failed to save session: {e}", file=sys.stderr)
            return False
    
    def save_file_changes(self, files_modified: list) -> int:
        """Track file modifications in archival memory."""
        if not self.client or not files_modified:
            return 0
        
        saved_count = 0
        project_name = Path(self.project_path).name
        
        for file_path in files_modified[:10]:
            try:
                self.client.agents.passages.create(
                    agent_id=self.agent_id,
                    text=f"File modified: {file_path}\nProject: {project_name}",
                    metadata={
                        "type": "file_change",
                        "file_path": file_path,
                        "project": project_name,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                saved_count += 1
            except Exception:
                pass
        
        return saved_count


def main():
    agent_id = os.environ.get("LETTA_AGENT_ID")
    if not agent_id or not LETTA_AVAILABLE:
        return
    
    # Read session data from stdin (provided by Claude Code)
    try:
        session_data = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except:
        session_data = {}
    
    saver = LettaSessionSaver()
    
    summary = session_data.get("conversation_summary", "")
    session_id = session_data.get("session_id", "unknown")
    
    if summary:
        success = saver.save_session_summary(summary, session_id)
        if success:
            print("✅ Session saved to Letta memory")


if __name__ == "__main__":
    main()
```

---

## API Endpoints Reference

### Complete REST API

**Agent Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/agents/` | List all agents |
| `POST` | `/v1/agents/` | Create new agent |
| `GET` | `/v1/agents/{agent_id}` | Get agent details |
| `PATCH` | `/v1/agents/{agent_id}` | Update agent |
| `DELETE` | `/v1/agents/{agent_id}` | Delete agent |
| `GET` | `/v1/agents/{agent_id}/export` | Export agent backup |
| `POST` | `/v1/agents/import` | Import agent |

**Memory Block Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/agents/{agent_id}/core-memory/blocks` | List agent blocks |
| `GET` | `/v1/agents/{agent_id}/core-memory/blocks/{block_label}` | Get block by label |
| `PATCH` | `/v1/agents/{agent_id}/core-memory/blocks/{block_label}` | Update block |
| `GET` | `/v1/blocks/` | List all standalone blocks |
| `POST` | `/v1/blocks/` | Create standalone block |
| `GET` | `/v1/blocks/{block_id}` | Get block |
| `PATCH` | `/v1/blocks/{block_id}` | Update block |
| `DELETE` | `/v1/blocks/{block_id}` | Delete block |
| `GET` | `/v1/blocks/{block_id}/agents` | List agents using block |
| `PATCH` | `/v1/agents/{agent_id}/core-memory/blocks/attach/{block_id}` | Attach block |
| `PATCH` | `/v1/agents/{agent_id}/core-memory/blocks/detach/{block_id}` | Detach block |

**Archival Memory Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/agents/{agent_id}/archival-memory` | List passages |
| `POST` | `/v1/agents/{agent_id}/archival-memory` | Create passage |
| `GET` | `/v1/agents/{agent_id}/archival-memory/search` | Search passages |
| `DELETE` | `/v1/agents/{agent_id}/archival-memory/{memory_id}` | Delete passage |

**Messaging Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/agents/{agent_id}/messages` | List messages |
| `POST` | `/v1/agents/{agent_id}/messages` | Send message (sync) |
| `POST` | `/v1/agents/{agent_id}/messages/async` | Send message (async) |
| `POST` | `/v1/agents/{agent_id}/messages/stream` | Stream messages |
| `POST` | `/v1/agents/{agent_id}/messages/cancel` | Cancel async message |
| `PATCH` | `/v1/agents/{agent_id}/reset-messages` | Reset message history |

**Conversations API (letta-client 1.7.1+):**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/conversations/` | List all conversations |
| `POST` | `/v1/conversations/` | Create new conversation |
| `GET` | `/v1/conversations/{conversation_id}` | Get conversation details |
| `DELETE` | `/v1/conversations/{conversation_id}` | Delete conversation |
| `GET` | `/v1/conversations/{conversation_id}/messages` | List conversation messages |
| `POST` | `/v1/conversations/{conversation_id}/messages` | Send message to conversation |

**Runs API (Async Job Management):**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/runs/` | List all runs |
| `GET` | `/v1/runs/{run_id}` | Get run status |
| `GET` | `/v1/runs/{run_id}/usage` | Get run token usage |
| `POST` | `/v1/runs/{run_id}/cancel` | Cancel running job |

**Tools Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/tools/` | List all tools |
| `POST` | `/v1/tools/` | Create tool |
| `GET` | `/v1/tools/{tool_id}` | Get tool |
| `PATCH` | `/v1/tools/{tool_id}` | Update tool |
| `DELETE` | `/v1/tools/{tool_id}` | Delete tool |
| `GET` | `/v1/agents/{agent_id}/tools` | List agent tools |
| `PATCH` | `/v1/agents/{agent_id}/tools/attach/{tool_id}` | Attach tool |
| `PATCH` | `/v1/agents/{agent_id}/tools/detach/{tool_id}` | Detach tool |
| `POST` | `/v1/agents/{agent_id}/tools/{tool_name}/run` | Run tool |
| `PATCH` | `/v1/agents/{agent_id}/tools/approval/{tool_name}` | Set approval |

**MCP Server Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/mcp-servers/` | List MCP servers |
| `POST` | `/v1/mcp-servers/` | Register MCP server |
| `GET` | `/v1/mcp-servers/{mcp_server_id}` | Get MCP server |
| `PATCH` | `/v1/mcp-servers/{mcp_server_id}` | Update MCP server |
| `DELETE` | `/v1/mcp-servers/{mcp_server_id}` | Delete MCP server |
| `PATCH` | `/v1/mcp-servers/{mcp_server_id}/refresh` | Refresh tools |

---

## Production Deployment

### Docker Compose Stack

```yaml
version: '3.8'

services:
  letta_db:
    image: ankane/pgvector:v0.5.1
    hostname: letta-db
    restart: always
    environment:
      POSTGRES_USER: ${LETTA_PG_USER:-letta}
      POSTGRES_PASSWORD: ${LETTA_PG_PASSWORD}
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
      LETTA_PG_URI: postgresql://${LETTA_PG_USER:-letta}:${LETTA_PG_PASSWORD}@letta-db:5432/${LETTA_PG_DB:-letta}
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

**.env file:**
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

**Startup:**
```bash
docker compose up -d
docker compose logs -f
```

---

## Official Documentation Links

### Letta Documentation

| Topic | URL |
|-------|-----|
| Getting Started | https://docs.letta.com/introduction |
| Memory Concepts | https://docs.letta.com/concepts/memory |
| Memory Blocks Guide | https://docs.letta.com/guides/agents/memory-blocks |
| Multi-User Guide | https://docs.letta.com/guides/agents/multi-user |
| Sleeptime Agents | https://docs.letta.com/guides/agents/architectures/sleeptime |
| Server Deployment | https://docs.letta.com/server/docker |
| Python SDK | https://docs.letta.com/clients/python |
| API Reference | https://docs.letta.com/api-reference |
| Agent Management API | https://docs.letta.com/api-reference/agents |
| Memory Blocks API | https://docs.letta.com/api-reference/agents/memory |
| Archival Memory API | https://docs.letta.com/api-reference/agents/archival-memory |
| Blocks API | https://docs.letta.com/api-reference/blocks |
| CLI Reference | https://docs.letta.com/letta-code/cli-reference |

### Claude Code Documentation

| Topic | URL |
|-------|-----|
| Introduction | https://docs.anthropic.com/en/docs/claude-code/introduction |
| Configuration | https://docs.anthropic.com/en/docs/claude-code/settings |
| Hooks System | https://docs.anthropic.com/en/docs/claude-code/hooks |
| Custom Commands | https://docs.anthropic.com/en/docs/claude-code/custom-commands |
| Skills | https://docs.anthropic.com/en/docs/claude-code/skills |
| MCP Integration | https://docs.anthropic.com/en/docs/claude-code/mcp |
| CLAUDE.md | https://docs.anthropic.com/en/docs/claude-code/claude-md |

### Anthropic API

| Topic | URL |
|-------|-----|
| API Overview | https://docs.anthropic.com/en/api |
| Python SDK | https://docs.anthropic.com/en/api/client-sdks |
| Extended Thinking | https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking |
| Tool Use | https://docs.anthropic.com/en/docs/build-with-claude/tool-use |

---

## Letta Code CLI Reference

### Basic Usage

```bash
letta                           # Start new conversation with last-used agent
letta --continue, -c            # Resume last session exactly
letta --resume, -r              # Open conversation selector
letta --conversation <id>       # Resume specific conversation
letta --new-agent               # Force create new agent
letta --from-af <path>          # Create from AgentFile
letta --agent <id>, -a          # Use specific agent by ID
letta --name <n>, -n            # Resume agent by name
letta --info                    # Show project info and pinned agents
```

### Model and Configuration

```bash
--model <model>, -m             # Specify model (sonnet-4.5, gpt-5-codex)
--system <preset>               # Use system prompt preset
--system-custom <text>          # Custom system prompt string
--system-append <text>          # Append to resolved system prompt
--toolset <n>                   # Force toolset: default, codex, gemini
--skills <path>                 # Custom skills directory
--sleeptime                     # Enable sleeptime memory management
```

### Headless Mode

```bash
-p "prompt"                     # Run one-off prompt (headless)
--output-format <fmt>           # Output: text, json, stream-json
--input-format <fmt>            # Input: stream-json for bidirectional
--yolo                          # Bypass all permission prompts
--permission-mode <mode>        # Set permission mode
--tools "Tool1,Tool2"           # Limit available tools
--allowedTools "..."            # Allow specific tool patterns
--disallowedTools "..."         # Block specific tool patterns
```

### Memory Configuration

```bash
--init-blocks <names>           # Comma-separated preset block names
--memory-blocks <json>          # JSON array of custom memory blocks
--block-value <label>=<value>   # Set value for preset block
```

### Environment Variables

```bash
LETTA_API_KEY                   # API key for authentication
LETTA_BASE_URL                  # Base URL for self-hosted server
LETTA_DEBUG=1                   # Enable debug logging
LETTA_CODE_TELEM=0              # Disable anonymous telemetry
DISABLE_AUTOUPDATER=1           # Disable auto-updates
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` | Open command autocomplete |
| `@` | Open file autocomplete |
| `!` | Enter bash mode |
| `Tab` | Autocomplete command or file path |
| `Shift+Enter` | Insert newline |
| `↑/↓` | Navigate history or menu items |
| `Esc` | Cancel dialog or clear input |
| `Ctrl+C` | Interrupt operation or exit |
| `Ctrl+V` | Paste content or image |
