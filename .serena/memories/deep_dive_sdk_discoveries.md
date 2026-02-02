# Deep Dive SDK Discoveries - January 2026

## Executive Summary

This document captures comprehensive SDK discoveries from the unleash SDK ecosystem (118+ repositories) mapped to the AlphaForge Trading System and State of Witness projects.

---

## TIER 1: Knowledge Graph & Memory

### Graphiti (Zep) - Knowledge Graph for AI Agents
**Status**: MCP Server Ready ✅
**Install**: `pip install graphiti-core`

**Key Features**:
- Real-time temporal knowledge graphs
- Bi-temporal data model (event + ingestion time)
- Hybrid retrieval: semantic + keyword (BM25) + graph traversal
- Sub-second query latency (vs GraphRAG's seconds to tens of seconds)
- Multiple backends: Neo4j, FalkorDB, Kuzu, Amazon Neptune
- MCP Server at `/mcp/` endpoint - directly usable with Claude Code!

**Integration Patterns**:
```python
from graphiti_core import Graphiti
graphiti = Graphiti("bolt://localhost:7687", "neo4j", "password")
await graphiti.add_episode(name="Trading Decision", episode_body="...", source="json")
results = await graphiti.search_facts("risk management decisions")
```

**Use Cases**:
- AlphaForge: Trading decision memory, strategy evolution tracking
- State of Witness: Archetype relationship graphs, pose embedding evolution

---

## TIER 2: Evolutionary Computation

### EvoTorch (NNAISENSE) - PyTorch Evolutionary Algorithms
**Install**: `pip install evotorch`

**Algorithms Available**:
| Type | Algorithms |
|------|------------|
| Distribution-based | PGPE, XNES, CMA-ES, SNES, CEM |
| Population-based | GeneticAlgorithm (NSGA-II), CoSyNE, **MAPElites** |

**Key Features**:
- GPU acceleration via PyTorch vectorization
- Ray integration for multi-CPU/GPU/cluster scaling
- Built-in RL support (GymNE for gym environments)
- Visualization with PandasLogger

**MAPElites Integration** (Quality-Diversity):
```python
from evotorch import Problem
from evotorch.algorithms import MAPElites

problem = Problem("max", fitness_fn, solution_length=100)
searcher = MAPElites(problem, num_cells_per_dim=32)
searcher.run(1000)
```

**Use Cases**:
- AlphaForge: Strategy parameter optimization, portfolio allocation QD
- State of Witness: Shader parameter exploration, archetype behavior evolution

---

## TIER 3: Safety & Guardrails

### NeMo Guardrails (NVIDIA) - Programmable LLM Safety
**Install**: `pip install nemoguardrails`

**5 Types of Rails**:
1. **Input Rails**: Reject/modify user input (jailbreak detection, data masking)
2. **Dialog Rails**: Control conversation flow via canonical forms
3. **Retrieval Rails**: Filter RAG chunks before LLM prompting
4. **Execution Rails**: Guard tool/action I/O
5. **Output Rails**: Filter/modify LLM responses

**Colang Language**: Domain-specific language for dialog flow control
```colang
define user express insult
  "You are stupid"

define flow
  user express insult
  bot express calmly willingness to help
```

**Built-in Protections**:
- Jailbreak detection
- Prompt injection detection
- Fact-checking
- Hallucination detection
- Content moderation

**Use Cases**:
- AlphaForge: Trading order validation, risk limit enforcement
- Claude Code: Guard rails for code generation, security checks

---

## TIER 4: Observability

### Langfuse - OpenTelemetry-Based LLM Observability
**Install**: `pip install langfuse`

**Architecture**:
- Built on OpenTelemetry for standardized tracing
- Async-first design with automatic batching
- Auto-instrumentation for OpenAI, LangChain, etc.

**Key Configuration**:
```bash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_DEBUG=true
LANGFUSE_SAMPLE_RATE=0.5
```

**Integration Pattern**:
```python
from langfuse import Langfuse
from langfuse.decorators import observe

@observe()
def my_llm_call():
    # Automatic tracing
    pass
```

### AgentOps - Agent Observability Platform
**Install**: `pip install agentops`

**Key Features**:
- Session replays with step-by-step execution graphs
- LLM cost management
- Native integrations: CrewAI, AG2, LangGraph, Camel AI
- Self-hostable (MIT license)
- MCP Server available!

**Decorator-Based Instrumentation**:
```python
from agentops.sdk.decorators import session, agent, operation

@session
def my_workflow():
    @agent
    class MyAgent:
        @operation
        def process(self):
            pass
```

**Use Cases**:
- All projects: Session replay debugging, cost tracking, multi-agent visualization

---

## TIER 5: Project-Specific Mappings

### AlphaForge Trading System

| Layer | Recommended SDK | Purpose |
|-------|-----------------|---------|
| Memory | Graphiti + Letta | Trading decision graphs, cross-session state |
| Strategy Optimization | EvoTorch MAPElites | QD exploration of strategy space |
| Safety | NeMo Guardrails | Order validation, risk enforcement |
| Observability | AgentOps + Langfuse | Session replay, cost tracking |
| Code Intelligence | Serena | Semantic code navigation (95% token savings) |

### State of Witness Creative System

| Layer | Recommended SDK | Purpose |
|-------|-----------------|---------|
| Memory | Graphiti | Archetype relationship graphs |
| Exploration | EvoTorch + pyribs | MAP-Elites shader/particle exploration |
| Observability | AgentOps | Creative session replay |
| Code Intelligence | Serena | TouchDesigner Python navigation |

---

## TIER 6: Integration Quick Reference

### MCP Servers Ready for Claude Code

| SDK | MCP Status | Endpoint |
|-----|------------|----------|
| Graphiti | ✅ Ready | `http://localhost:8000/mcp/` |
| AgentOps | ✅ Ready | Via Smithery registry |
| Serena | ✅ Active | Local executable |
| Context7 | ✅ Active | Via plugin |
| Exa | ✅ Active | Deep research |

### Token Efficiency Patterns

| Tool | Savings | Technique |
|------|---------|-----------|
| Serena | 95% | LSP-based semantic extraction |
| Context7 | 80% | Indexed documentation retrieval |
| Exa batch | 60% | Combined research queries |
| Graphiti | 70% | Graph traversal vs full context |

---

## Next Steps

1. **Add Graphiti MCP** to `mcp_servers_OPTIMAL.json`
2. **Integrate EvoTorch MAPElites** with existing pyribs workflows
3. **Configure NeMo Guardrails** for AlphaForge trading validation
4. **Set up AgentOps** for session replay in Ralph Loop

---

*Generated: January 2026*
*Maintained by: Claude Code Deep Dive Session*
