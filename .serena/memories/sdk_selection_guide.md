# UNLEASH SDK Selection Guide
## Quick Reference for Claude Code Integration

---

## 🏗️ ARCHITECTURE OVERVIEW

The unleash ecosystem contains **118 SDKs** organized in a **17-layer V29.3 architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Foundation (Serena, LiteLLM, Temporal)             │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2: Agent Frameworks (Pydantic AI, OpenAI Agents)      │
├─────────────────────────────────────────────────────────────┤
│ LAYER 3: Reasoning (LightZero, LLM Reasoners, DSPy)         │
├─────────────────────────────────────────────────────────────┤
│ LAYER 4: Data Acquisition (Crawl4AI, Docling, Firecrawl)    │
├─────────────────────────────────────────────────────────────┤
│ LAYER 5: Knowledge (GraphRAG, LightRAG, LlamaIndex)         │
├─────────────────────────────────────────────────────────────┤
│ LAYER 6: Memory (Mem0, Letta, Graphiti)                     │
├─────────────────────────────────────────────────────────────┤
│ LAYER 7: Evolutionary (EvoTorch, Pyribs, QDax)              │
├─────────────────────────────────────────────────────────────┤
│ LAYER 8: Safety (NeMo Guardrails, LLM Guard, Purple Llama)  │
├─────────────────────────────────────────────────────────────┤
│ LAYER 9: Observability (Langfuse, Arize Phoenix, AgentOps)  │
├─────────────────────────────────────────────────────────────┤
│ LAYER 10: Evaluation (DeepEval, RAGAS, PromptFoo)           │
├─────────────────────────────────────────────────────────────┤
│ LAYER 11: Code Agents (Aider, Cline, SWE-Agent)             │
├─────────────────────────────────────────────────────────────┤
│ LAYER 12: MCP Development (FastMCP, mcp-python-sdk)         │
├─────────────────────────────────────────────────────────────┤
│ LAYER 13: Multi-Agent (CrewAI, AutoGen, LangGraph)          │
├─────────────────────────────────────────────────────────────┤
│ LAYER 14: Vision/Multimodal (BLIP2, Vision Agents)          │
├─────────────────────────────────────────────────────────────┤
│ LAYER 15: Structured Output (Instructor, BAML, Outlines)    │
├─────────────────────────────────────────────────────────────┤
│ LAYER 16: Deployment (Modal, Ray Serve, KServe)             │
├─────────────────────────────────────────────────────────────┤
│ LAYER 17: Security (Rebuff, Any Guardrail)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 PROJECT-SPECIFIC STACKS

### 🏦 AlphaForge Trading System

| SDK | Purpose | Integration Point |
|-----|---------|-------------------|
| **Serena** | Navigate 12-layer architecture | Code intelligence |
| **Temporal** | Durable trade execution | Crash recovery |
| **LightZero** | MuZero for market modeling | Strategic planning |
| **DSPy** | Optimize risk analysis prompts | Prompt optimization |
| **Guardrails AI** | Validate trading decisions | Risk validation |
| **NeMo Guardrails** | Safety rails for automation | Safety layer |
| **LiteLLM** | Route between Claude/GPT | Cost optimization |

**Recommended Flow:**
```
Code Navigation (Serena) → Strategic Planning (LightZero) →
Durable Execution (Temporal) → LLM Routing (LiteLLM) →
Risk Validation (Guardrails)
```

### 🎭 State of Witness (MediaPipe/Real-time)

| SDK | Purpose | Integration Point |
|-----|---------|-------------------|
| **Serena** | Python LSP for MediaPipe | Code navigation |
| **EvoTorch** | Neuroevolution for gestures | Model optimization |
| **Pyribs** | MAP-Elites exploration | Quality-diversity |
| **Vision Agents** | Computer vision tools | Visual understanding |
| **LiveKit Agents** | Real-time AI voice/video | Streaming |
| **Pipecat** | Voice AI pipelines | Audio processing |

**Recommended Flow:**
```
Code Navigation (Serena) → Model Evolution (EvoTorch) →
QD Exploration (Pyribs) → Real-time Processing (LiveKit/Pipecat)
```

---

## 🔑 CRITICAL SDK DETAILS

### 1. FastMCP (Production MCP Framework)
- **Location:** `sdks/fastmcp/`
- **Purpose:** Fast Pythonic MCP server building
- **Key Features:**
  - Enterprise auth (Google, GitHub, Azure, Auth0)
  - Server composition and proxying
  - OpenAPI/FastAPI generation
  - Complete client libraries

```python
from fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
```

### 2. Instructor (Structured LLM Output)
- **Location:** `sdks/instructor/`
- **Purpose:** Type-safe structured outputs from LLMs
- **Supports:** OpenAI, Anthropic, Gemini, Cohere, Mistral, Groq
- **Pattern:** Factory functions with Pydantic validation

```python
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_anthropic(anthropic.Anthropic())
user = client.messages.create(
    model="claude-sonnet-4-20250514",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 25"}]
)
```

### 3. DSPy (Prompt Programming)
- **Location:** `sdks/dspy/`
- **Purpose:** Declarative prompt optimization
- **Key Optimizers:**
  - MIPROv2: Bayesian optimization (10-13% improvement)
  - GEPA: Reflective evolution (8-15% improvement)
  - BootstrapFinetune: Weight updates (15-25% improvement)

### 4. Temporal Python (Durable Execution)
- **Location:** `sdks/temporal-python/`
- **Purpose:** Crash-proof workflows
- **Used By:** OpenAI Codex, Replit Agent 3
- **Key Feature:** Agent state persists through crashes

### 5. LangGraph (Stateful Agent Graphs)
- **Location:** `sdks/langgraph/`
- **Libraries:**
  - `langgraph/` - Core framework
  - `checkpoint-postgres/` - Postgres persistence
  - `checkpoint-sqlite/` - SQLite persistence
  - `sdk-py/` - Python SDK
  - `sdk-js/` - JavaScript SDK

---

## 📊 SDK DECISION TREES

### Agent Framework Selection:
```
Need durable execution? → Temporal + Pydantic AI
Need multi-agent teams? → CrewAI or AutoGen
Need complex state machines? → LangGraph
Need HuggingFace models? → SmolaAgents
Need simple production agent? → Pydantic AI
```

### RAG Pipeline Selection:
```
Web scraping needed? → Crawl4AI or Firecrawl
PDF/DOCX processing? → Docling
Graph-based retrieval? → GraphRAG or LightRAG
Full RAG framework? → LlamaIndex
Temporal knowledge? → Graphiti
```

### Safety Layer Selection:
```
Comprehensive safety? → NeMo Guardrails
Output validation? → Guardrails AI
Input/output scanning? → LLM Guard
Prompt injection defense? → Rebuff
Meta safety suite? → Purple Llama
```

---

## 🚀 QUICK INSTALL COMMANDS

```bash
# Core Stack
pip install dspy-ai litellm temporalio pydantic-ai instructor

# RAG Stack
pip install "crawl4ai[all]" docling llama-index

# Evaluation Stack
pip install deepeval ragas langfuse

# Safety Stack
pip install nemoguardrails guardrails-ai llm-guard

# MCP Stack
pip install fastmcp mcp
```

---

## 🔗 KEY FILE LOCATIONS

| SDK | CLAUDE.md | README | Examples |
|-----|-----------|--------|----------|
| FastMCP | `sdks/fastmcp/AGENTS.md` | `sdks/fastmcp/README.md` | `sdks/fastmcp/examples/` |
| Instructor | `sdks/instructor/CLAUDE.md` | `sdks/instructor/README.md` | `sdks/instructor/examples/` |
| LangGraph | `sdks/langgraph/CLAUDE.md` | `sdks/langgraph/README.md` | `sdks/langgraph/examples/` |
| DSPy | - | `sdks/dspy/README.md` | `sdks/dspy/docs/` |
| Temporal | - | `sdks/temporal-python/README.md` | `sdks/temporal-python/tests/` |

---

*Last Updated: 2026-01-20 | SDK Count: 118 | Architecture: V29.3*
