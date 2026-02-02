# Letta API Reference

> **Source**: https://docs.letta.com/api-reference
> **Downloaded**: January 17, 2026

## Overview

Letta provides a unified platform for interacting with stateful agents through multiple interfaces. Whether through the ADE's visual interface or programmatically via APIs, you're always connecting to the same agents running in your Letta server.

## Available Interfaces

1. **REST API** - Direct HTTP endpoints for agent management and interaction
2. **Python SDK** - Native client library for Python applications
3. **TypeScript SDK** - Native client library for Node.js/TypeScript applications

## Authentication & Setup

### Python SDK Installation
```bash
pip install letta-client
```

### TypeScript SDK Installation
```bash
npm install @letta-ai/letta-client
```

### Connection Options
- **Local server**: `base_url="http://localhost:8283"`
- **Cloud API**: `api_key` and `project` parameters

## API Resources

- **Agents** - Create, retrieve, update, delete, import/export
- **Messages** - Send messages, stream responses, manage conversation history
- **Tools** - Attach/detach tools, manage approvals, execute tools
- **Blocks** - Manage memory blocks and system components
- **Archives & Passages** - Handle document storage and retrieval
- **Conversations** - Create and manage multi-turn conversations
- **Models** - List available models and embeddings

---

# Sleep-time Agents Architecture

> **Source**: https://docs.letta.com/guides/agents/architectures/sleeptime

## Overview

Sleep-time agents are an experimental Letta feature enabling background agents that share memory with primary agents and asynchronously modify memory blocks.

## Core Concept

The system creates two agents when `enable_sleeptime: true`:

1. **Primary Agent**: Handles direct interactions with tools for `conversation_search` and `archival_memory_search`
2. **Sleep-time Agent**: Runs in background, manages memory blocks asynchronously

## Memory Blocks Foundation

Sleep-time agents generate *learned context* by reflecting on original context (conversation history, files) to derive insights. This learned context stores in labeled memory block sections with character limits, enabling sharing across multiple agents.

## Activation

Set when creating agents:
```python
agent = client.agents.create(
    name="my-agent",
    enable_sleeptime=True
)
```

## Configuration: Frequency Control

Control triggering intervals via `sleeptime_agent_frequency` (default: 5 steps):

### Python
```python
from letta_client.types import SleeptimeManagerUpdate

client.groups.update(
    group_id=group_id,
    manager_config=SleeptimeManagerUpdate(sleeptime_agent_frequency=2)
)
```

### TypeScript
```typescript
await client.groups.update(groupId, {
    manager_config: { sleeptime_agent_frequency: 2 }
});
```

## Memory Block Access

Retrieve blocks by label or ID:

### Python
```python
block = client.agents.blocks.retrieve(
    agent_id=agent_id,
    block_label="persona"
)
```

### TypeScript
```typescript
const block = await client.agents.blocks.retrieve(agentId, "persona");
```

## Performance Recommendation

Keep the frequency relatively high (e.g. 5 or 10) to avoid excessive token consumption and diminishing effectiveness returns.

## Key SDK Resources

- [Python SDK GitHub](https://github.com/letta-ai/letta-python)
- [TypeScript SDK GitHub](https://github.com/letta-ai/letta-node)
