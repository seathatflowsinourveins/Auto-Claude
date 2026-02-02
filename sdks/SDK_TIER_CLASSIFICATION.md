# SDK Tier Classification - ULTIMATE Architecture

> **Generated**: 2026-01-28
> **Evaluation Method**: Deep dive into 36 SDK repositories
> **Purpose**: Optimize UNLEASH architecture with only the best SDKs

---

## Classification Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Documentation Quality** | 25% | Complete, accurate, up-to-date docs |
| **Production Readiness** | 25% | Stable API, versioning, test coverage |
| **Unique Value** | 20% | Capabilities not found elsewhere |
| **Integration Quality** | 15% | Clean APIs, minimal dependencies |
| **Active Maintenance** | 15% | Regular updates, responsive maintainers |

---

## Context7 Verification (2026-01-28)

All TIER 1 Essential SDKs have been verified against Context7 with HIGH reputation sources:

| SDK | Library ID | Benchmark | Snippets | Reputation |
|-----|------------|-----------|----------|------------|
| Claude Agent SDK | `/anthropics/claude-agent-sdk-python` | 89.0 | 59 | HIGH |
| Claude-Flow V3 | *(community, not in Context7)* | - | - | - |
| Letta | `/llmstxt/letta_llms-full_txt` | 82.3 | 17,008 | HIGH |
| Opik | `/comet-ml/opik` | 72.5 | 5,257 | HIGH |
| MCP Python SDK | `/modelcontextprotocol/python-sdk` | 89.2 | 296 | HIGH |
| Instructor | `/instructor-ai/instructor` | 85.9 | 2,810 | HIGH |
| LiteLLM | `/berriai/litellm` | 85.0 | 9,205 | HIGH |
| LangGraph | `/langchain-ai/langgraph` | 90.7 | 254+ | HIGH |

**Cumulative Documentation Coverage**: 34,889+ code snippets across 7 Context7-verified SDKs.

### Key Patterns Retrieved from Context7

**Claude Agent SDK** (`@tool` decorator):
```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("add", "Add two numbers", {"a": float, "b": float})
async def add_numbers(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"{args['a']} + {args['b']} = {result}"}]}
```

**Opik** (`@track` + Hallucination metric):
```python
from opik import track
from opik.evaluation.metrics import Hallucination

@track
def my_llm_pipeline(input: str) -> str:
    return openai_client.chat.completions.create(...)

metric = Hallucination()
score = metric.score(input=question, output=answer, context=[context])
```

**Letta** (shared memory blocks):
```python
from letta_client import Letta, CreateBlock

client = Letta(base_url="http://localhost:8283")
shared_block = client.blocks.create(label="org", value="Shared context", limit=4000)
agent = client.agents.create(memory_blocks=[...], block_ids=[shared_block.id])
```

**MCP Python SDK** (FastMCP server):
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Service")

@mcp.tool()
def get_data(query: str) -> str:
    return f"Results for {query}"

mcp.run()  # stdio transport
```

---

## TIER 1: ESSENTIAL (Core Architecture)

These SDKs form the backbone of the ULTIMATE architecture. Cannot be replaced.

### 1.1 Claude Agent SDK (Python + TypeScript)
**Location**: `anthropic/claude-agent-sdk-python`, `anthropic/claude-agent-sdk-typescript`
**Role**: Official Anthropic agent framework
**Score**: 98/100

| Criteria | Score | Notes |
|----------|-------|-------|
| Documentation | 25/25 | Official Anthropic docs, comprehensive |
| Production Ready | 25/25 | Official SDK, stable API |
| Unique Value | 20/20 | Only official agent framework |
| Integration | 14/15 | Clean in-process MCP servers |
| Maintenance | 14/15 | Anthropic-maintained |

**Key Features**:
- `@tool` decorator for custom tools
- `create_sdk_mcp_server()` for in-process MCP (no subprocess!)
- PreToolUse/PostToolUse hooks for deterministic processing
- ClaudeAgentOptions for comprehensive configuration

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeSDKClient

@tool("search", "Search codebase", {"query": str})
async def search_code(args):
    return {"content": [{"type": "text", "text": f"Results for {args['query']}"}]}

server = create_sdk_mcp_server(name="dev-tools", tools=[search_code])
```

**Verdict**: ✅ ESSENTIAL - Official framework, irreplaceable

---

### 1.2 Claude-Flow V3
**Location**: `mcp-ecosystem/claude-flow`
**Role**: Enterprise multi-agent orchestration
**Score**: 96/100

| Criteria | Score | Notes |
|----------|-------|-------|
| Documentation | 24/25 | Extensive wiki, examples |
| Production Ready | 24/25 | v3.0.0-alpha.184, 13K+ stars |
| Unique Value | 20/20 | Most comprehensive orchestration |
| Integration | 14/15 | 87+ MCP tools, 60+ agents |
| Maintenance | 14/15 | Active development |

**Key Features**:
- 64 specialized agents in 8 categories
- Hive-Mind consensus (Raft, PBFT, Gossip, CRDT)
- SONA self-optimizing neural architecture (<0.05ms)
- 3-tier model routing (WASM → Haiku → Opus)
- Background workers (12 types)

```bash
# Initialize swarm
claude-flow swarm "implement feature" --strategy development --review

# Agent management
claude-flow agent spawn security-architect
```

**Verdict**: ✅ ESSENTIAL - No alternative provides this orchestration depth

---

### 1.3 Letta (formerly MemGPT)
**Location**: `mcp-ecosystem/letta`, `letta/`
**Role**: Stateful agent memory management
**Score**: 95/100

| Criteria | Score | Notes |
|----------|-------|-------|
| Documentation | 24/25 | Comprehensive, cloud + self-host |
| Production Ready | 24/25 | Enterprise-ready, multi-agent |
| Unique Value | 19/20 | Best-in-class memory architecture |
| Integration | 14/15 | Clean API, Python/Node clients |
| Maintenance | 14/15 | Active, MemGPT legacy |

**Key Features**:
- Memory blocks (human, persona, custom)
- Passages for semantic search
- Multi-agent coordination
- Sleep-time compute for memory consolidation
- Tool execution within agents

```python
from letta_client import Letta

client = Letta(api_key=os.getenv("LETTA_API_KEY"))
agent = client.agents.create(
    memory_blocks=[
        {"label": "human", "value": "User context..."},
        {"label": "persona", "value": "Agent personality..."}
    ],
    tools=["web_search", "run_code"]
)
```

**Verdict**: ✅ ESSENTIAL - Unique memory architecture for stateful agents

---

### 1.4 Opik
**Location**: `mcp-ecosystem/opik`, `opik-full/`
**Role**: AI observability and evaluation
**Score**: 94/100

| Criteria | Score | Notes |
|----------|-------|-------|
| Documentation | 24/25 | Extensive, self-host guide |
| Production Ready | 23/25 | Comet-backed, enterprise |
| Unique Value | 19/20 | 50+ evaluation metrics |
| Integration | 14/15 | @track decorator, SDK integrations |
| Maintenance | 14/15 | Actively developed |

**Key Features**:
- @opik.track decorator for tracing
- 50+ evaluation metrics (RAG, agents, bias, heuristic)
- Self-hosted option with Docker
- Claude/OpenAI/LangChain integrations
- Real-time dashboards

```python
import opik
from opik.integrations.anthropic import track_anthropic

client = track_anthropic(anthropic.Anthropic())

@opik.track(name="my_pipeline", tags=["production"])
def my_llm_pipeline(prompt: str):
    return client.messages.create(...)
```

**Verdict**: ✅ ESSENTIAL - No alternative has this evaluation depth

---

### 1.5 MCP Python SDK
**Location**: `mcp-python-sdk/`
**Role**: Official MCP protocol implementation
**Score**: 93/100

| Criteria | Score | Notes |
|----------|-------|-------|
| Documentation | 23/25 | Comprehensive primitives |
| Production Ready | 22/25 | v2 pre-alpha, active |
| Unique Value | 20/20 | Only official MCP SDK |
| Integration | 14/15 | Clean stdio/SSE servers |
| Maintenance | 14/15 | Anthropic-supported |

**Key Features**:
- Resources, Tools, Prompts primitives
- Context for dependency injection
- Completions for inline autocomplete
- Elicitation for user prompts
- stdio + SSE transport

**Verdict**: ✅ ESSENTIAL - Foundation for all MCP servers

---

### 1.6 Instructor
**Location**: `instructor/`
**Role**: Structured LLM outputs with Pydantic
**Score**: 92/100

| Criteria | Score | Notes |
|----------|-------|-------|
| Documentation | 24/25 | Excellent examples |
| Production Ready | 24/25 | Widely used, stable |
| Unique Value | 18/20 | Best structured output lib |
| Integration | 12/15 | Multiple provider support |
| Maintenance | 14/15 | Very active |

**Key Features**:
- Pydantic model validation
- Automatic retries on validation failure
- Streaming support
- Multi-provider (Claude, OpenAI, Gemini, etc.)

```python
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_provider("anthropic/claude-opus-4-5-20251101")
user = client.chat.completions.create(response_model=User, messages=[...])
```

**Verdict**: ✅ ESSENTIAL - Structured outputs are fundamental

---

### 1.7 LiteLLM
**Location**: `litellm/`
**Role**: Unified provider interface
**Score**: 91/100

| Criteria | Score | Notes |
|----------|-------|-------|
| Documentation | 23/25 | Good, many providers |
| Production Ready | 24/25 | 100+ providers |
| Unique Value | 18/20 | Most comprehensive routing |
| Integration | 12/15 | Router + fallbacks |
| Maintenance | 14/15 | Very active |

**Key Features**:
- 100+ LLM providers
- Router with fallback chains
- Cost tracking
- Rate limiting
- Load balancing

```python
from litellm import Router

router = Router(
    model_list=[...],
    routing_strategy="least-busy",
    fallbacks=[{"gpt-4": ["claude-3-opus"]}]
)
```

**Verdict**: ✅ ESSENTIAL - Multi-provider routing is critical

---

### 1.8 LangGraph
**Location**: `langgraph/`
**Role**: Stateful agent orchestration (alternative to Claude-Flow)
**Score**: 91/100

| Criteria | Score | Notes |
|----------|-------|-------|
| Documentation | 23/25 | Good LangChain ecosystem |
| Production Ready | 23/25 | Widely used |
| Unique Value | 17/20 | LangChain native |
| Integration | 14/15 | Clean StateGraph API |
| Maintenance | 14/15 | Active |

**Key Features**:
- StateGraph with START/END
- Checkpointing (MemorySaver, SQLite, Postgres)
- Human-in-the-loop support
- Streaming
- Time-travel debugging

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(AgentState)
graph.add_node("process", process_fn)
app = graph.compile(checkpointer=MemorySaver())
```

**Verdict**: ✅ ESSENTIAL - Complements Claude-Flow for LangChain users

---

## TIER 2: RECOMMENDED (High Value, Optional)

These SDKs add significant value but aren't strictly required.

### 2.1 DSPy
**Location**: `dspy/`
**Role**: Declarative LM programming
**Score**: 88/100

**Key Features**:
- Signatures for typed prompts
- MIPROv2 prompt optimization
- Evaluation-driven development
- Module composition

**Verdict**: ✅ RECOMMENDED - Excellent for prompt optimization

---

### 2.2 Pydantic AI
**Location**: `pydantic-ai/`
**Role**: Type-safe agent framework
**Score**: 87/100

**Key Features**:
- RunContext[T] dependency injection
- Type-safe tool definitions
- Structured outputs built-in

**Verdict**: ✅ RECOMMENDED - Great for type-heavy projects

---

### 2.3 Context7
**Location**: `mcp-ecosystem/context7`
**Role**: Library documentation access
**Score**: 86/100

**Key Features**:
- resolve-library-id + query-docs
- Real-time library documentation
- Code snippet retrieval

**Verdict**: ✅ RECOMMENDED - Essential for accurate SDK usage

---

### 2.4 Exa
**Location**: `mcp-ecosystem/exa-mcp-server`
**Role**: AI-optimized web search
**Score**: 85/100

**Key Features**:
- Neural search
- Content extraction
- Deep research mode

**Verdict**: ✅ RECOMMENDED - Best web search for AI

---

### 2.5 pyribs
**Location**: `pyribs/`
**Role**: Quality-Diversity optimization
**Score**: 84/100

**Key Features**:
- MAP-Elites implementation
- GridArchive, CVTArchive
- Evolution strategy emitters

**Verdict**: ✅ RECOMMENDED for exploration tasks (WITNESS, TRADING)

---

### 2.6 Promptfoo
**Location**: `promptfoo/`
**Role**: LLM evaluation and red-teaming
**Score**: 84/100

**Key Features**:
- YAML-based eval configs
- Red team plugins
- CI/CD integration

**Verdict**: ✅ RECOMMENDED - Critical for prompt testing

---

### 2.7 Mem0
**Location**: `mem0/`
**Role**: User memory layer
**Score**: 83/100

**Key Features**:
- +26% accuracy in benchmarks
- 91% faster than alternatives
- User-scoped memories

**Verdict**: ✅ RECOMMENDED - Alternative to Letta for simpler use cases

---

### 2.8 Ragas
**Location**: `ragas/`
**Role**: RAG evaluation framework
**Score**: 82/100

**Key Features**:
- Faithfulness, relevance, precision metrics
- Testset generation
- CI integration

**Verdict**: ✅ RECOMMENDED for RAG pipelines

---

### 2.9 DeepEval
**Location**: `deepeval/`
**Role**: LLM testing framework
**Score**: 81/100

**Key Features**:
- Pytest-style assertions
- Multiple metrics
- Confident AI integration

**Verdict**: ✅ RECOMMENDED - Good testing alternative

---

### 2.10 Guardrails AI
**Location**: `guardrails-ai/`
**Role**: Output validation
**Score**: 80/100

**Key Features**:
- Validator registry
- Schema enforcement
- Retry mechanisms

**Verdict**: ✅ RECOMMENDED for validation-heavy apps

---

## TIER 3: SPECIALIZED (Domain-Specific)

Use these when you need their specific capabilities.

### 3.1 Aider
**Location**: `aider/`
**Role**: AI pair programming
**Score**: 78/100

**Use When**: Need AI-assisted coding in terminal
**Alternative**: Claude Code CLI itself

---

### 3.2 AST-Grep
**Location**: `ast-grep/`
**Role**: AST-based code search/refactor
**Score**: 77/100

**Use When**: Large-scale code refactoring
**Alternative**: Built-in Grep + tree-sitter

---

### 3.3 Serena
**Location**: `serena/`
**Role**: LSP-based semantic code tools
**Score**: 76/100

**Use When**: Need LSP-level code intelligence
**Alternative**: ULTIMATE L0/L1 layers (pyright, code-index)

---

### 3.4 Temporal Python
**Location**: `temporal-python/`
**Role**: Workflow orchestration
**Score**: 75/100

**Use When**: Need durable, distributed workflows
**Alternative**: LangGraph with checkpointing

---

### 3.5 Crawl4AI
**Location**: `crawl4ai/`
**Role**: LLM-optimized web scraping
**Score**: 74/100

**Use When**: Need structured web extraction
**Alternative**: Firecrawl, Jina

---

### 3.6 Firecrawl
**Location**: `firecrawl/`
**Role**: Website to markdown
**Score**: 73/100

**Use When**: Need markdown from websites
**Alternative**: Crawl4AI

---

### 3.7 GraphRAG
**Location**: `graphrag/`
**Role**: Graph-based RAG
**Score**: 72/100

**Use When**: Complex entity relationships
**Alternative**: Standard RAG with Qdrant

---

### 3.8 Outlines
**Location**: `outlines/`
**Role**: Structured generation
**Score**: 71/100

**Use When**: Local model structured output
**Alternative**: Instructor (for API models)

---

### 3.9 BAML
**Location**: `baml/`
**Role**: Prompting DSL
**Score**: 70/100

**Use When**: Team needs prompt versioning
**Alternative**: DSPy signatures

---

### 3.10 CrewAI
**Location**: `crewai/`
**Role**: Multi-agent framework
**Score**: 69/100

**Use When**: Need hierarchical crews
**Alternative**: Claude-Flow V3 (more capable)

---

### 3.11 AutoGen
**Location**: `autogen/`
**Role**: Microsoft multi-agent
**Score**: 68/100

**Use When**: Azure ecosystem
**Alternative**: Claude-Flow V3

---

### 3.12 Langfuse
**Location**: `langfuse/`
**Role**: LLM observability
**Score**: 68/100

**Use When**: Need @observe decorator style
**Alternative**: Opik (more metrics)

---

### 3.13 Zep
**Location**: `zep/`
**Role**: Long-term memory
**Score**: 67/100

**Use When**: Need fact extraction
**Alternative**: Letta (more features)

---

### 3.14 LLM Guard
**Location**: `llm-guard/`
**Role**: Input/output safety
**Score**: 66/100

**Use When**: Need prompt injection detection
**Alternative**: NeMo Guardrails

---

### 3.15 NeMo Guardrails
**Location**: `nemo-guardrails/`
**Role**: Safety guardrails (NVIDIA)
**Score**: 65/100

**Use When**: NVIDIA ecosystem
**Alternative**: Guardrails AI

---

### 3.16 Arize Phoenix
**Location**: `arize-phoenix/`
**Role**: ML observability
**Score**: 64/100

**Use When**: Need OTEL traces
**Alternative**: Opik

---

## TIER 4: REPLACEABLE/REDUNDANT

These can be replaced by better alternatives.

### 4.1 OpenAI SDK
**Location**: `openai-sdk/`
**Status**: REDUNDANT
**Replace With**: LiteLLM + Instructor

**Reason**: LiteLLM provides the same API with multi-provider support. Instructor adds structured outputs. No need for OpenAI-specific SDK.

---

### 4.2 Tavily Python/JS/MCP
**Location**: `mcp-ecosystem/tavily-*`
**Status**: REPLACEABLE
**Replace With**: Exa

**Reason**: Exa provides superior neural search with content extraction. Tavily duplicates functionality.

---

### 4.3 Jina MCP
**Location**: `mcp-ecosystem/jina-mcp`
**Status**: REPLACEABLE
**Replace With**: Exa + Crawl4AI

**Reason**: Reading/embedding capabilities covered by existing stack.

---

### 4.4 FastMCP
**Location**: `fastmcp/`
**Status**: REDUNDANT
**Replace With**: MCP Python SDK

**Reason**: Official MCP SDK v2 provides same functionality with better support.

---

## Summary Matrix

| Tier | Count | SDKs |
|------|-------|------|
| **TIER 1: ESSENTIAL** | 8 | claude-agent-sdk, claude-flow, letta, opik, mcp-python-sdk, instructor, litellm, langgraph |
| **TIER 2: RECOMMENDED** | 10 | dspy, pydantic-ai, context7, exa, pyribs, promptfoo, mem0, ragas, deepeval, guardrails-ai |
| **TIER 3: SPECIALIZED** | 16 | aider, ast-grep, serena, temporal, crawl4ai, firecrawl, graphrag, outlines, baml, crewai, autogen, langfuse, zep, llm-guard, nemo-guardrails, arize-phoenix |
| **TIER 4: REPLACEABLE** | 4 | openai-sdk, tavily-*, jina-mcp, fastmcp |

---

## ULTIMATE Architecture Stack

Based on this classification, the ULTIMATE production architecture uses:

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER 1: ESSENTIAL                        │
├─────────────────────────────────────────────────────────────┤
│ Claude Agent SDK     │ Official agent framework             │
│ Claude-Flow V3       │ Multi-agent orchestration            │
│ Letta               │ Stateful memory management            │
│ Opik                │ Observability + evaluation            │
│ MCP Python SDK      │ Protocol implementation               │
│ Instructor          │ Structured outputs                    │
│ LiteLLM             │ Multi-provider routing                │
│ LangGraph           │ Stateful workflows                    │
├─────────────────────────────────────────────────────────────┤
│                    TIER 2: RECOMMENDED                      │
├─────────────────────────────────────────────────────────────┤
│ DSPy                │ Prompt optimization                   │
│ Context7            │ Library documentation                 │
│ Exa                 │ Web search                            │
│ pyribs              │ Quality-Diversity (WITNESS/TRADING)   │
│ Promptfoo           │ Evaluation/red-teaming                │
├─────────────────────────────────────────────────────────────┤
│                    TIER 3: AS NEEDED                        │
├─────────────────────────────────────────────────────────────┤
│ Specialized tools loaded on demand per project             │
└─────────────────────────────────────────────────────────────┘
```

---

## Project-Specific Recommendations

### UNLEASH (Meta-Project)
**Required**: All TIER 1 + Context7, Promptfoo, DSPy
**Optional**: pyribs for exploration optimization

### WITNESS (Creative AI)
**Required**: TIER 1 + pyribs, Crawl4AI
**Optional**: Outlines for local model structured output

### TRADING (AlphaForge)
**Required**: TIER 1 + pyribs for strategy diversity
**Optional**: Temporal for workflow durability

---

*Classification complete. 8 Essential, 10 Recommended, 16 Specialized, 4 Replaceable.*
