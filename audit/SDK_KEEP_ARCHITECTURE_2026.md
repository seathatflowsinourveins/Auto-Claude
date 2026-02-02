# SDK KEEP Architecture - 35 Best-of-Breed SDKs
## Ultimate Architecture Design (2026-01-24)

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ULTIMATE SDK ARCHITECTURE                            │
│                              35 KEEP SDKs                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 0: PROTOCOL & GATEWAY                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ mcp-python   │ │   fastmcp    │ │   litellm    │ │  anthropic   │        │
│  │    -sdk      │ │              │ │              │ │   + openai   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 1: ORCHESTRATION & WORKFLOWS                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ temporal-    │ │  langgraph   │ │  claude-flow │ │   crewai/    │        │
│  │   python     │ │              │ │              │ │   autogen    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2: MEMORY & PERSISTENCE                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                         │
│  │    letta     │ │     zep      │ │     mem0     │                         │
│  │  (MemGPT)    │ │              │ │              │                         │
│  └──────────────┘ └──────────────┘ └──────────────┘                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3: STRUCTURED OUTPUT & TYPE SAFETY                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  instructor  │ │    baml      │ │   outlines   │ │ pydantic-ai  │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 4: PROMPTING & REASONING                                              │
│  ┌──────────────┐ ┌──────────────┐                                          │
│  │    dspy      │ │   serena     │                                          │
│  └──────────────┘ └──────────────┘                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 5: OBSERVABILITY & EVALUATION                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   langfuse   │ │    opik      │ │arize-phoenix │ │   deepeval   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐                                          │
│  │    ragas     │ │  promptfoo   │                                          │
│  └──────────────┘ └──────────────┘                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 6: SAFETY & GUARDRAILS                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                         │
│  │guardrails-ai│ │  llm-guard   │ │nemo-guardrails│                         │
│  └──────────────┘ └──────────────┘ └──────────────┘                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 7: CODE & DOCUMENT PROCESSING                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │    aider     │ │   ast-grep   │ │   crawl4ai   │ │  firecrawl   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 8: RAG & KNOWLEDGE                                                    │
│  ┌──────────────┐ ┌──────────────┐                                          │
│  │   graphrag   │ │    pyribs    │                                          │
│  └──────────────┘ └──────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LAYER DEFINITIONS

### LAYER 0: PROTOCOL & GATEWAY (4 SDKs)
The foundation layer handling all LLM communication and tool protocols.

| SDK | Role | Integration Point | Used By |
|-----|------|-------------------|---------|
| **mcp-python-sdk** | MCP Protocol | All MCP tools | All layers |
| **fastmcp** | MCP Server Framework | Tool deployment | Layers 1-7 |
| **litellm** | LLM Gateway | Multi-provider routing | All LLM calls |
| **anthropic** + **openai-sdk** | Provider SDKs | Direct API access | Fallback |

```python
# Example: Layer 0 Integration
from litellm import completion
from mcp import Client

# All LLM calls go through litellm
response = completion(model="claude-3-opus", messages=[...])

# All tools exposed via MCP
mcp_client = Client("fastmcp://localhost:8000")
```

---

### LAYER 1: ORCHESTRATION & WORKFLOWS (5 SDKs)
Manages agent workflows, state machines, and multi-agent coordination.

| SDK | Role | Use Case | Integration |
|-----|------|----------|-------------|
| **temporal-python** | Durable Execution | Long-running workflows | All async jobs |
| **langgraph** | State Machines | Agent graphs, cycles | Complex agents |
| **claude-flow** | Claude Orchestration | Claude-specific patterns | Claude agents |
| **crewai** | Team Agents | Role-based teams | Multi-persona |
| **autogen** | Conversational Agents | Microsoft patterns | Code generation |

```python
# Example: Layer 1 Workflow
from langgraph.graph import StateGraph
from temporal import workflow

@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, task: str):
        # Use langgraph for agent logic
        graph = StateGraph(AgentState)
        # Temporal handles durability
        return await graph.ainvoke({"task": task})
```

---

### LAYER 2: MEMORY & PERSISTENCE (3 SDKs)
Handles agent memory, context management, and state persistence.

| SDK | Role | Memory Type | Integration |
|-----|------|-------------|-------------|
| **letta** | Stateful Agents | Long-term + Self-editing | Primary memory |
| **zep** | Session Memory | Conversation history | Session context |
| **mem0** | Personalization | User preferences | Cross-session |

```python
# Example: Layer 2 Memory Stack
from letta import Agent
from zep_cloud import ZepClient
from mem0 import Memory

class MemoryStack:
    def __init__(self):
        self.letta = Agent()  # Core agent memory
        self.zep = ZepClient()  # Session history
        self.mem0 = Memory()  # Personalization

    async def remember(self, user_id: str, content: str):
        await self.letta.memory.insert(content)
        await self.zep.add_memory(user_id, content)
        await self.mem0.add(content, user_id=user_id)
```

---

### LAYER 3: STRUCTURED OUTPUT & TYPE SAFETY (4 SDKs)
Ensures LLM outputs conform to schemas and types.

| SDK | Role | Approach | Best For |
|-----|------|----------|----------|
| **instructor** | Pydantic Extraction | Retry + Validation | Simple extraction |
| **baml** | Type-safe DSL | Compile-time checks | Complex schemas |
| **outlines** | Grammar Constrained | Token-level control | Precise formats |
| **pydantic-ai** | Type-safe Agents | Agent + Types | Type-safe agents |

```python
# Example: Layer 3 Type Safety
import instructor
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

# Guaranteed to return valid Analysis
client = instructor.from_anthropic()
result = client.chat.completions.create(
    response_model=Analysis,
    messages=[{"role": "user", "content": "Analyze this text..."}]
)
```

---

### LAYER 4: PROMPTING & REASONING (2 SDKs)
Advanced prompt optimization and semantic code operations.

| SDK | Role | Capability | Use Case |
|-----|------|------------|----------|
| **dspy** | Prompt Programming | Optimize prompts | Prompt chains |
| **serena** | Semantic Editing | AST + LSP | Code changes |

```python
# Example: Layer 4 DSPy Chain
import dspy

class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Optimize the pipeline
teleprompter = dspy.BootstrapFewShot()
optimized = teleprompter.compile(RAGPipeline(), trainset=examples)
```

---

### LAYER 5: OBSERVABILITY & EVALUATION (6 SDKs)
Monitoring, tracing, and quality assessment.

| SDK | Role | Metrics | Integration |
|-----|------|---------|-------------|
| **langfuse** | LLM Tracing | Latency, cost, quality | All LLM calls |
| **opik** | AI Observability | 50+ metrics | Comet platform |
| **arize-phoenix** | ML Observability | Embeddings, drift | Production |
| **deepeval** | LLM Evaluation | Hallucination, relevance | CI/CD |
| **ragas** | RAG Evaluation | Retrieval metrics | RAG pipelines |
| **promptfoo** | Prompt Testing | A/B testing | Development |

```python
# Example: Layer 5 Observability
from langfuse import Langfuse
from opik import track
from deepeval.metrics import HallucinationMetric

langfuse = Langfuse()

@track(name="agent_response")
@langfuse.observe()
def agent_respond(query: str):
    response = llm.complete(query)

    # Evaluate response
    hallucination = HallucinationMetric()
    score = hallucination.measure(query, response)

    return response
```

---

### LAYER 6: SAFETY & GUARDRAILS (3 SDKs)
Input validation, output filtering, and safety controls.

| SDK | Role | Protection | Stage |
|-----|------|------------|-------|
| **guardrails-ai** | Validators | Schema + Custom | Input/Output |
| **llm-guard** | Security | PII, Injection | Input scan |
| **nemo-guardrails** | Dialogue Safety | Topical rails | Conversation |

```python
# Example: Layer 6 Safety Stack
from guardrails import Guard
from llm_guard import scan_prompt
from nemoguardrails import RailsConfig, LLMRails

# Input validation
safe_prompt, is_valid, risk_score = scan_prompt(user_input)

# Guardrails validation
guard = Guard().use(ValidJSON(), NoHallucination())
result = guard(llm, prompt=safe_prompt)

# Dialogue rails
rails = LLMRails(RailsConfig.from_path("./config"))
response = rails.generate(messages=conversation)
```

---

### LAYER 7: CODE & DOCUMENT PROCESSING (4 SDKs)
Code manipulation and web content extraction.

| SDK | Role | Capability | Use Case |
|-----|------|------------|----------|
| **aider** | AI Pair Programming | Code changes | Development |
| **ast-grep** | AST Search/Replace | Pattern matching | Refactoring |
| **crawl4ai** | Web Crawling | LLM-friendly | Web scraping |
| **firecrawl** | Web API | Structured scraping | Data extraction |

```python
# Example: Layer 7 Code Processing
from ast_grep_py import SgRoot
from crawl4ai import WebCrawler

# AST-based code search
root = SgRoot(code, "python")
matches = root.find_all("def $NAME($$$ARGS)")

# Web content extraction
crawler = WebCrawler()
result = crawler.crawl("https://docs.example.com")
markdown = result.markdown  # LLM-ready format
```

---

### LAYER 8: RAG & KNOWLEDGE (2 SDKs)
Knowledge graphs and exploration algorithms.

| SDK | Role | Capability | Use Case |
|-----|------|------------|----------|
| **graphrag** | Graph RAG | Knowledge graphs | Document QA |
| **pyribs** | Quality-Diversity | MAP-Elites | Exploration |

```python
# Example: Layer 8 Knowledge
from graphrag import GraphRAG
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

# Graph-based RAG
graphrag = GraphRAG()
graphrag.index(documents)
answer = graphrag.query("What are the key patterns?")

# Quality-Diversity exploration
archive = GridArchive(dims=[100, 100], ranges=[(0, 1), (0, 1)])
emitter = GaussianEmitter(archive, sigma=0.1)
# Explore solution space...
```

---

## SDK INTEGRATION MATRIX

| Layer | Depends On | Called By | Data Flow |
|-------|-----------|-----------|-----------|
| L0 Protocol | None | All | LLM requests, tool calls |
| L1 Orchestration | L0, L2 | Application | Workflow state |
| L2 Memory | L0 | L1, L4 | Context, history |
| L3 Structured | L0 | L1, L4 | Typed outputs |
| L4 Reasoning | L0, L3 | L1 | Optimized prompts |
| L5 Observability | L0 | All (decorators) | Traces, metrics |
| L6 Safety | L0 | L1 (middleware) | Validated I/O |
| L7 Code/Docs | L0 | L1, L4 | Processed content |
| L8 Knowledge | L0, L7 | L1 | Retrieved context |

---

## PROJECT-SPECIFIC CONFIGURATIONS

### State of Witness (Creative)
```yaml
primary_sdks:
  - pyribs        # MAP-Elites exploration
  - langgraph     # Creative workflows
  - letta         # Aesthetic memory
  - opik          # Generation monitoring

optional:
  - outlines      # Shader generation
  - crawl4ai      # Reference gathering
```

### AlphaForge (Trading)
```yaml
primary_sdks:
  - temporal-python  # Durable trading workflows
  - guardrails-ai    # Trade validation
  - langfuse         # Decision tracing
  - deepeval         # Strategy evaluation

optional:
  - dspy           # Signal optimization
  - graphrag       # Market knowledge
```

### Unleash (Meta-Project)
```yaml
primary_sdks:
  - ALL P0 BACKBONE  # Full infrastructure
  - ALL P1 CORE      # All capabilities
  - serena           # Self-modification
  - aider            # Code generation

focus:
  - Self-improvement
  - SDK integration
  - Cross-project coordination
```

---

## STACK TIER REORGANIZATION

After cleanup, the stack/ directory should mirror this architecture:

```
stack/
├── tier-0-protocol/
│   ├── mcp-python-sdk -> ../sdks/mcp-python-sdk
│   ├── fastmcp -> ../sdks/fastmcp
│   └── litellm -> ../sdks/litellm
├── tier-1-orchestration/
│   ├── temporal-python -> ../sdks/temporal-python
│   ├── langgraph -> ../sdks/langgraph
│   ├── claude-flow -> ../sdks/claude-flow
│   ├── crewai -> ../sdks/crewai
│   └── autogen -> ../sdks/autogen
├── tier-2-memory/
│   ├── letta -> ../sdks/letta
│   ├── zep -> ../sdks/zep
│   └── mem0 -> ../sdks/mem0
├── tier-3-structured/
│   ├── instructor -> ../sdks/instructor
│   ├── baml -> ../sdks/baml
│   ├── outlines -> ../sdks/outlines
│   └── pydantic-ai -> ../sdks/pydantic-ai
├── tier-4-reasoning/
│   ├── dspy -> ../sdks/dspy
│   └── serena -> ../sdks/serena
├── tier-5-observability/
│   ├── langfuse -> ../sdks/langfuse
│   ├── opik -> ../sdks/opik
│   ├── arize-phoenix -> ../sdks/arize-phoenix
│   ├── deepeval -> ../sdks/deepeval
│   ├── ragas -> ../sdks/ragas
│   └── promptfoo -> ../sdks/promptfoo
├── tier-6-safety/
│   ├── guardrails-ai -> ../sdks/guardrails-ai
│   ├── llm-guard -> ../sdks/llm-guard
│   └── nemo-guardrails -> ../sdks/nemo-guardrails
├── tier-7-processing/
│   ├── aider -> ../sdks/aider
│   ├── ast-grep -> ../sdks/ast-grep
│   ├── crawl4ai -> ../sdks/crawl4ai
│   └── firecrawl -> ../sdks/firecrawl
└── tier-8-knowledge/
    ├── graphrag -> ../sdks/graphrag
    └── pyribs -> ../sdks/pyribs
```

---

## TOTAL SDK COUNT: 35

| Priority | Count | SDKs |
|----------|-------|------|
| P0 Backbone | 9 | mcp-python-sdk, fastmcp, litellm, temporal-python, letta, dspy, langfuse, anthropic, openai-sdk |
| P1 Core | 15 | langgraph, pydantic-ai, crewai, autogen, zep, mem0, instructor, baml, outlines, ast-grep, serena, arize-phoenix, deepeval, nemo-guardrails, claude-flow |
| P2 Advanced | 11 | aider, crawl4ai, firecrawl, graphrag, guardrails-ai, llm-guard, ragas, promptfoo, opik, opik-full*, pyribs |

*opik-full may be consolidated with opik

---

**Document Version**: 1.0
**Generated**: 2026-01-24
**Architect**: Claude Code
**Status**: ARCHITECTURE DEFINED
