# Letta SDK (formerly MemGPT) - Comprehensive Research Report

*Deep Research Report - 2026-01-19*
*Research Time: 392 seconds*

---

## Overview

Letta SDK is a platform for building AI agents with advanced, stateful memory capabilities. Formerly known as MemGPT, it provides a hierarchical memory architecture with core, recall, and archival memory tiers.

---

## 1. Memory Architecture (Core vs Recall vs Archival)

### Core Memory
- In-context memory visible at each turn
- Consists of **memory blocks** (structured, labeled text segments)
- Labels like `human`, `persona` persist across sessions
- Default 2,000 character limit per block
- Agents edit using `memory_insert` and `memory_replace` tools

### Recall Memory
- Append-only external log of conversational history
- Agents query via date/text-search tools
- Cannot be modified by agents
- Used for retrieving past interactions

### Archival Memory
- Semantically searchable vector database
- Unlimited entries with similarity search
- Tools: `archival_memory_insert`, `archival_memory_search`
- Developer CRUD operations supported

---

## 2. Agent Creation and Configuration

### Key Parameters
| Parameter | Description |
|-----------|-------------|
| `model` | LLM model (e.g., `anthropic/claude-sonnet-4-5-20250929`) |
| `embedding` | Embedding model for archival memory |
| `memory_blocks` | Array of core memory blocks |
| `tools` | List of tool names to attach |
| `context_window_limit` | Maximum context window size |
| `enable_sleeptime` | Enable sleep-time compute |

### Python Example
```python
from letta_client import Letta
import os

client = Letta(api_key=os.getenv("LETTA_API_KEY"))
agent_state = client.agents.create(
    model="anthropic/claude-sonnet-4-5-20250929",
    embedding="openai/text-embedding-3-small",
    memory_blocks=[
        {"label": "human", "value": "Name: Timber. Status: dog."},
        {"label": "persona", "value": "I am a self-improving superintelligence."}
    ],
    tools=["web_search", "run_code"],
    enable_sleeptime=True
)
print(f"Agent created with ID: {agent_state.id}")
```

### Messaging the Agent
```python
response = client.agents.messages.create(
    agent_id=agent_state.id,
    input="What do you know about me?"
)
for msg in response.messages:
    print(msg.content)
```

---

## 3. Memory Blocks and Context Management

Memory blocks are core memory's building blocks with:
- `label` - Identifier (e.g., "human", "persona")
- `description` - Purpose of the block
- `value` - Content string
- `limit` - Character limit

### Shared Memory Blocks
```python
# Create a shared memory block
block = client.blocks.create(
    label="organization",
    description="Information about the organization",
    value="Organization: Letta",
    limit=4000
)

# Attach to multiple agents
agent1 = client.agents.create(
    name="agent1",
    memory_blocks=[{"label":"persona","value":"I am agent 1"}],
    block_ids=[block.id],
    model="openai/gpt-4o-mini"
)
agent2 = client.agents.create(
    name="agent2",
    memory_blocks=[{"label":"persona","value":"I am agent 2"}],
    block_ids=[block.id],
    model="openai/gpt-4o-mini"
)
```

---

## 4. Tool Integration Patterns

### Function-Based Tools
```python
def get_weather(location: str) -> str:
    return f"Weather in {location}: 72°F, sunny"

tool = client.tools.create(func=get_weather)
agent = client.agents.create(
    model="anthropic/claude-sonnet-4-5-20250929",
    tools=[tool.name],
    memory_blocks=[{"label":"persona","value":"Helper"}]
)
```

### Integration Options
- **Function-based tools**: Register Python functions directly
- **API-level interceptors**: Patch LLM API for seamless integration
- **Environment variables**: Securely pass secrets to tools

---

## 5. Persistence and State Management

- Agents persist state server-side
- Recall memory stores conversation logs
- Core memory blocks persist across interactions
- Clients send only new messages
- Shared blocks synchronize context across agents

---

## 6. Sleep-Time Compute and Memory Consolidation

Sleep-time compute allows agents to "think" during idle time:

### Benefits
- Reorganize and refine memory blocks
- Reduce latency during user interactions
- Enhance personalization
- Shift computation from active use to downtime

### Configuration
```python
agent = client.agents.create(
    model="openai/gpt-4o-mini",
    enable_sleeptime=True,
    sleeptime_model="openai/gpt-4",
    sleeptime_frequency="hourly"
)
```

### How It Works
1. Primary conversational agent handles user interactions
2. Asynchronous sleep-time agent manages core memory
3. Sleep-time agent uses stronger models at configurable intervals
4. Memory blocks are consolidated and optimized in background

---

## Integration with Unleash Platform

```python
# In advanced_memory.py
from letta_client import Letta

class AdvancedMemory:
    def __init__(self):
        self.letta = Letta(api_key=os.getenv("LETTA_API_KEY"))

    async def create_agent(self, persona: str, human: str):
        return self.letta.agents.create(
            model="anthropic/claude-sonnet-4-5-20250929",
            memory_blocks=[
                {"label": "persona", "value": persona},
                {"label": "human", "value": human}
            ],
            enable_sleeptime=True
        )

    async def store_archival(self, agent_id: str, content: str):
        return self.letta.agents.archival_memory.insert(
            agent_id=agent_id,
            content=content
        )
```

---

*Sources: [docs.letta.com](https://docs.letta.com), [letta.com/blog](https://www.letta.com/blog), [github.com/letta-ai/letta](https://github.com/letta-ai/letta)*
