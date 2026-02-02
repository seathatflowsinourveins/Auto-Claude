# Cross-Session Bootstrap V34 Quick Reference

## SDK Quick Access
| SDK | Import | Use Case |
|-----|--------|----------|
| Claude Flow V3 | `npx claude-flow@v3alpha` | 54+ agents, swarm coordination |
| PydanticAI | `from pydantic_ai import Agent` | Type-safe agents, DI |
| EvoAgentX | `from evoagentx.workflow import WorkFlowGenerator` | Self-evolving workflows |
| Instructor | `from instructor import from_anthropic` | Structured LLM output |
| FastMCP | `from fastmcp import FastMCP` | Production MCP servers |
| Crawl4AI | `from crawl4ai import AsyncWebCrawler` | Deep web crawling |
| LightRAG | `from lightrag import LightRAG` | Graph-based RAG |
| Mem0 | `from mem0 import Memory` | Multi-backend memory |
| Opik | `import opik; @opik.track()` | AI observability |
| CrewAI Flows | `from crewai.flow import Flow, start, listen` | Event-driven flows |
| pyribs | `from ribs.archives import GridArchive` | MAP-Elites QD |
| NeMo Guardrails | `from nemoguardrails import RailsConfig` | Safety rails |

## Key Integration Patterns

### 1. PydanticAI Agent
```python
agent = Agent('anthropic:claude-sonnet-4-0', deps_type=MyDeps, output_type=Output)
@agent.tool
async def my_tool(ctx: RunContext[MyDeps], query: str) -> dict: ...
```

### 2. FastMCP Server
```python
mcp = FastMCP("Server")
@mcp.tool
def process(a: int) -> int: return a * 2
```

### 3. Crawl4AI + LightRAG Pipeline
```python
async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(url, config=CrawlerRunConfig(deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=3)))
rag = LightRAG(working_dir="./data")
await rag.ainsert([result.markdown])
response = await rag.aquery(query, param=QueryParam(mode="hybrid"))
```

### 4. CrewAI Flow
```python
class MyFlow(Flow):
    @start()
    def begin(self): return self.state.input
    @listen(begin)
    def process(self, data): return transform(data)
```

### 5. EvoAgentX Workflow
```python
workflow_graph = WorkFlowGenerator(llm=llm).generate_workflow(goal)
agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=config)
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
```

### 6. MAP-Elites with QDAIF
```python
archive = GridArchive(solution_dim=10, dims=[20, 20], ranges=[(-1, 1), (0, 1)])
emitter = EvolutionStrategyEmitter(archive, x0=[0.0]*10, sigma0=0.5)
scheduler = Scheduler(archive, [emitter])
```

## Claude SDK 2026 Optimizations
- Prompt caching: `cache_control: {"type": "ephemeral"}`
- Tool whitelisting: `--allowed-tools Read,Write,Bash`
- Batch processing: `client.messages.batches.create(requests=[...])`

## Full Documentation
- `Z:\insider\AUTO CLAUDE\unleash\CROSS_SESSION_BOOTSTRAP_V34.md`
- `Z:\insider\AUTO CLAUDE\unleash\UNIFIED_ARCHITECTURE_V34_INTEGRATION.md`
- `Z:\insider\AUTO CLAUDE\unleash\sdks\` (137 SDK directories)
