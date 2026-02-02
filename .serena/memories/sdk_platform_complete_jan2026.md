# Complete Platform SDK Research - January 2026

## Overview
Full research synthesis of all 15 SDKs in `unleash/platform/sdks/`.

---

## Category 1: Memory & State Management

### Letta (formerly MemGPT)
**Purpose**: Stateful agents with advanced memory that learn and self-improve

```python
from letta_client import Letta

client = Letta(api_key=os.getenv("LETTA_API_KEY"))

# Create stateful agent with memory blocks
agent_state = client.agents.create(
    model="openai/gpt-5.2",  # or claude-opus-4
    memory_blocks=[
        {"label": "human", "value": "User context..."},
        {"label": "persona", "value": "Agent personality..."}
    ],
    tools=["web_search", "fetch_webpage"]
)

# Memory persists across sessions
response = client.agents.messages.create(
    agent_id=agent_state.id,
    input="What do you remember about me?"
)
```

**Key Features**:
- Hierarchical memory blocks (human, persona, system)
- Skills and subagents support
- Letta Code CLI for terminal agents
- Model-agnostic (recommends Opus 4.5, GPT-5.2)

### Mem0
**Purpose**: The Memory Layer for Personalized AI

```python
from mem0 import Memory

memory = Memory()

# Store memories across sessions
def chat_with_memories(message: str, user_id: str = "default"):
    # Retrieve relevant memories
    relevant = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {e['memory']}" for e in relevant["results"])
    
    # Generate response with context
    # ... LLM call with memories ...
    
    # Store new memories
    memory.add(messages, user_id=user_id)
```

**Research Highlights** (arXiv:2504.19413):
- +26% accuracy over OpenAI Memory (LOCOMO benchmark)
- 91% faster responses than full-context
- 90% fewer tokens consumed
- Multi-level memory: User, Session, Agent state

---

## Category 2: Agent Orchestration

### LangGraph (LangChain)
**Purpose**: Low-level orchestration for stateful, long-running agents

```python
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    text: str

def node_a(state: State) -> dict:
    return {"text": state["text"] + "a"}

graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_edge(START, "node_a")

compiled = graph.compile()
result = compiled.invoke({"text": ""})
```

**Core Benefits**:
- Durable execution (persists through failures)
- Human-in-the-loop (inspect/modify state anytime)
- Checkpointing: checkpoint-postgres, checkpoint-sqlite
- LangSmith integration for debugging

**Libraries**:
- `checkpoint` - Base checkpointer interfaces
- `prebuilt` - High-level agent APIs
- `sdk-py/sdk-js` - Client SDKs
- `cli` - Command-line interface

### MCP-Agent (LastMile AI)
**Purpose**: Build agents with Model Context Protocol using composable patterns

```python
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

app = MCPApp(name="my_agent")

async with app.run():
    agent = Agent(
        name="finder",
        instruction="Use filesystem and fetch to answer questions.",
        server_names=["filesystem", "fetch"],
    )
    async with agent:
        llm = await agent.attach_llm(OpenAIAugmentedLLM)
        answer = await llm.generate_str("Summarize README.md")
```

**Key Features**:
- Full MCP support (Tools, Resources, Prompts, OAuth, Sampling)
- Implements ALL patterns from Anthropic's "Building Effective Agents"
- Durable agents via Temporal integration
- Agent-as-MCP-server pattern
- Workflow patterns: orchestrator, evaluator-optimizer, router, swarm

---

## Category 3: Web Crawling & Data Extraction

### Crawl4AI
**Purpose**: LLM-friendly web crawler for RAG, agents, data pipelines

```python
from crawl4ai import AsyncWebCrawler

async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(url="https://example.com")
    print(result.markdown)  # Clean, LLM-ready output
```

**CLI**:
```bash
crwl https://docs.example.com --deep-crawl bfs --max-pages 10
crwl https://example.com -q "Extract all product prices"  # LLM extraction
```

**v0.8.0 Features**:
- Crash recovery with `resume_state`
- Prefetch mode (5-10x faster URL discovery)
- Async browser pool, caching
- 51K+ GitHub stars

### Firecrawl
**Purpose**: API service for clean web data extraction

```python
# Crawl entire website
curl -X POST https://api.firecrawl.dev/v2/crawl \
    -H 'Authorization: Bearer fc-YOUR_API_KEY' \
    -d '{"url": "https://docs.firecrawl.dev", "limit": 10}'
```

**Features**:
- Scrape, Crawl, Map, Search, Extract endpoints
- LLM-ready formats: markdown, structured data, screenshot
- Actions: click, scroll, input before extraction
- Media parsing: PDFs, DOCX, images
- MCP server available: `firecrawl-mcp-server`

---

## Category 4: AI Pair Programming

### Aider
**Purpose**: AI pair programming in terminal

```bash
# Install
pip install aider-install && aider-install

# Use with Claude
aider --model sonnet --api-key anthropic=<key>
```

**Features**:
- Repo map for large codebases
- 100+ programming languages
- Git integration with auto-commits
- Voice-to-code support
- IDE integration (watch mode)
- **88% Singularity**: Aider writes most of its own code!
- Works with Claude 3.7 Sonnet, DeepSeek R1, GPT-4o, o3-mini

---

## Category 5: Advanced Reasoning

### LLM-Reasoners (Maitrix.org)
**Purpose**: Cutting-edge reasoning algorithms for LLMs

```python
from reasoners import WorldModel, SearchConfig, SearchAlgorithm
from reasoners.algorithm import MCTS, BeamSearch, DFS, BFS

# Define world model for state transitions
class MyWorldModel(WorldModel):
    def init_state(self): ...
    def step(self, state, action): ...
    def is_terminal(self, state): ...

# Configure search
config = SearchConfig(reward_fn=my_reward, ...)
reasoner = Reasoner(world_model, config, MCTS())
result = reasoner.run(problem)
```

**Supported Algorithms**:
- Reasoning-via-Planning (MCTS) - Hao et al., 2023
- Tree-of-Thoughts (BFS/DFS) - Yao et al., 2023
- Inference-time Scaling with PRM - Snell et al., 2024
- ReasonerAgent (web research) - Deng et al., 2025
- DRPO, Grace Decoding, Self-Eval, Eurus

**Integrations**:
- SGLang for 100x speedup
- BrowserGym for web planning
- DeepSeek R1 analysis support

---

## SDK Selection Matrix

| Use Case | Primary SDK | Backup |
|----------|-------------|--------|
| **Agent Memory** | Letta | Mem0, Graphiti |
| **Agent Orchestration** | LangGraph | MCP-Agent, EvoAgentX |
| **MCP Integration** | MCP-Agent | - |
| **Web Crawling (OSS)** | Crawl4AI | - |
| **Web Crawling (API)** | Firecrawl | - |
| **Pair Programming** | Aider | Claude Code |
| **Advanced Reasoning** | LLM-Reasoners | ToT, GoT |
| **Prompt Optimization** | DSPy, TextGrad | EvoAgentX |
| **Temporal Knowledge** | Graphiti | - |

---

## Integration Patterns for Core Projects

### AlphaForge Trading
```python
# Memory: Letta for trading agent state
# Orchestration: LangGraph for trading workflows
# Reasoning: LLM-Reasoners MCTS for strategy search
# Data: Crawl4AI for news scraping
```

### State of Witness
```python
# Memory: Mem0 for creative session context
# Evolution: EvoAgentX for workflow generation
# Optimization: TextGrad for aesthetic refinement
# Research: Firecrawl for reference gathering
```

### Unleash Platform
```python
# All SDKs available for meta-enhancement
# Focus: DSPy for prompt optimization
# Focus: LLM-Reasoners for deep thinking
# Focus: Aider for self-improvement
```

---

Last Updated: 2026-01-25
Total SDKs Documented: 15
