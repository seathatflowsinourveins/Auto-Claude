# Cross-Session Bootstrap V35 Memory

## Instant Access Commands

### Agent Creation
```python
from pydantic_ai import Agent
agent = Agent('anthropic:claude-sonnet-4-0', output_type=MyModel)

from smolagents import CodeAgent
agent = CodeAgent(tools=[...], model=model)  # 30% more efficient
```

### Multi-Agent
```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool
orchestrator = AssistantAgent("main", tools=[AgentTool(agent1), AgentTool(agent2)])
```

### MCP 3.0
```python
from fastmcp import FastMCP
mcp = FastMCP("Server")
@mcp.tool
async def my_tool(ctx: Context): ...
```

### Optimization
```python
from dspy.teleprompt import GEPA
optimizer = GEPA(metric=fn, num_candidates=20)
optimized = optimizer.compile(module, trainset=data)

import textgrad as tg
prompt = tg.Variable("...", requires_grad=True)
loss.backward(); tg.TGD([prompt]).step()
```

### State
```python
from langgraph.checkpoint.postgres import PostgresSaver
app = graph.compile(checkpointer=PostgresSaver(conn))
```

### RAG
```python
from crawl4ai import AsyncWebCrawler
from lightrag import LightRAG
```

### Memory
```python
from mem0 import Memory
from letta import create_client
```

### Install All
```bash
pip install pydantic-ai instructor fastmcp litellm autogen-agentchat crewai evoagentx crawl4ai lightrag mem0 letta dspy textgrad pyribs langgraph temporalio nemoguardrails opik smolagents
```

---
*V35 Bootstrap Memory*
