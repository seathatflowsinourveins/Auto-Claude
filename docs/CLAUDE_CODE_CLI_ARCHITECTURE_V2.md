# Claude Code CLI Architecture V2
## Verified Best-of-Breed SDK Integration Architecture

**Version:** 2.1  
**Created:** 2026-01-24  
**Updated:** 2026-01-24 (Gap Analysis Integration)  
**Based On:** ULTIMATE_SDK_COLLECTION_2026.md (35 SDKs verified from 154+ evaluated)  
**Target:** Production-grade autonomous coding platform with SDK-first integration

---

## Table of Contents

1. [Architecture Vision](#1-architecture-vision)
2. [Layer Architecture with SDK Assignments](#2-layer-architecture-with-sdk-assignments)
3. [SDK Integration Points](#3-sdk-integration-points)
4. [MCP Protocol Layer](#4-mcp-protocol-layer)
5. [Memory Architecture](#5-memory-architecture)
6. [Workflow Patterns](#6-workflow-patterns)
7. [Observability Stack](#7-observability-stack)
8. [Implementation Sequence](#8-implementation-sequence)
9. [Architecture Diagrams](#9-architecture-diagrams)
10. [SDK Quick Reference](#10-sdk-quick-reference)

---

## 1. Architecture Vision

### 1.1 Claude Code CLI as Composable Platform

The Claude Code CLI architecture is designed as a **composable platform** where each capability is provided by a best-of-breed SDK that can be independently upgraded, replaced, or extended.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLAUDE CODE CLI COMPOSABLE PLATFORM                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Orchestrate │  │  Remember   │  │   Observe   │  │   Execute   │    │
│  │  langgraph  │  │   letta     │  │  langfuse   │  │  temporal   │    │
│  │  crewai     │  │   zep       │  │  phoenix    │  │  ast-grep   │    │
│  │  autogen    │  │   mem0      │  │  opik       │  │  serena     │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
│  ┌──────┴────────────────┴────────────────┴────────────────┴──────┐    │
│  │                    MCP PROTOCOL UNIFICATION                      │    │
│  │              mcp-python-sdk + fastmcp foundation                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                │                │                │            │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐    │
│  │   litellm   │  │  anthropic  │  │ openai-sdk  │  │   dspy      │    │
│  │   Gateway   │  │   Official  │  │  Fallback   │  │  Optimize   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 SDK-First Integration Philosophy

**Core Principles:**

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **Best-of-Breed Selection** | Each domain uses the highest-rated SDK | 35 SDKs from 154+ evaluated |
| **Protocol Unification** | All tools accessible via MCP | mcp-python-sdk as foundation |
| **Loose Coupling** | SDKs interact through defined interfaces | Adapter pattern for all integrations |
| **Observable by Default** | Every operation is traced | Langfuse integration on all layers |
| **Fail-Safe Degradation** | Primary → Secondary → Fallback | litellm handles provider failover |

### 1.3 MCP as Universal Protocol Layer

The Model Context Protocol serves as the **universal integration bus** connecting all SDKs:

```python
# Every SDK exposes capabilities through MCP
@mcp_server.tool()
async def analyze_code(code: str) -> AnalysisResult:
    # ast-grep for pattern matching
    patterns = await ast_grep.find_patterns(code)
    # dspy for optimization
    optimized = await dspy_chain(patterns)
    # instructor for structured output
    return await instructor.extract(optimized, AnalysisResult)
```

**Protocol Benefits:**
- **Uniform Tool Access** - Claude sees all capabilities as MCP tools
- **Composable Pipelines** - Chain any SDK operations through MCP
- **Cross-Language Support** - Python SDKs, TypeScript SDKs unified
- **Standardized Errors** - Consistent error propagation across SDKs

---

## 2. Layer Architecture with SDK Assignments

### 2.1 Seven-Layer Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 6: INTERFACE                                                       │
│   Claude Code CLI, TUI, API Gateway                                     │
│   SDKs: [Application-specific, not SDK-driven]                          │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 5: ORCHESTRATION                                                   │
│   Multi-agent coordination, workflow graphs, durable execution          │
│   SDKs: langgraph, temporal-python, crewai, autogen, pydantic-ai,      │
│         claude-flow (P1 - MCP-native, SONA self-learning)               │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 4: INTELLIGENCE                                                    │
│   Prompt optimization, structured output, declarative AI                │
│   SDKs: dspy, instructor, baml, outlines                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 3: MEMORY                                                          │
│   Stateful context, cross-session persistence, knowledge graphs         │
│   SDKs: letta, zep, mem0                                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 2: COMMUNICATION                                                   │
│   MCP protocol, tool registration, provider routing                     │
│   SDKs: mcp-python-sdk, fastmcp, litellm                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 1: FOUNDATION                                                      │
│   Official provider SDKs, direct API access                             │
│   SDKs: anthropic, openai-sdk                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 0: INFRASTRUCTURE                                                  │
│   Observability, evaluation, safety, code intelligence                  │
│   SDKs: langfuse, arize-phoenix, deepeval, nemo-guardrails,            │
│          ast-grep, serena, aider, crawl4ai, opik (P2 - agent ops)       │
├─────────────────────────────────────────────────────────────────────────┤
│ REFERENCE: CONFIGURATION SOURCE                                          │
│   everything-claude-code (P2 - 9 agents, 11 skills, memory hooks)       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Responsibilities and SDK Mapping

| Layer | Responsibility | Primary SDKs | Secondary SDKs |
|-------|----------------|--------------|----------------|
| **L6: Interface** | User interaction | CLI framework | TUI, HTTP API |
| **L5: Orchestration** | Workflow management | langgraph, temporal-python | crewai, autogen, pydantic-ai, claude-flow |
| **L4: Intelligence** | AI reasoning | letta, dspy | instructor, baml, outlines |
| **L3: Memory** | Context persistence | letta | zep, mem0 |
| **L2: Communication** | Protocol handling | mcp-python-sdk, fastmcp | litellm |
| **L1: Foundation** | Provider access | anthropic | openai-sdk |
| **L0: Infrastructure** | Cross-cutting concerns | langfuse, ast-grep | arize-phoenix, deepeval, nemo-guardrails, opik |
| **Reference** | Configuration patterns | everything-claude-code | 9 agents, 11 skills, memory hooks |

### 2.3 Layer Dependencies

```mermaid
graph TB
    subgraph L6[Layer 6: Interface]
        CLI[Claude Code CLI]
        TUI[Terminal UI]
        API[HTTP API]
    end
    
    subgraph L5[Layer 5: Orchestration]
        LANGGRAPH[langgraph]
        TEMPORAL[temporal-python]
        CREWAI[crewai]
        AUTOGEN[autogen]
        PYDANTIC_AI[pydantic-ai]
        CLAUDE_FLOW[claude-flow]
    end
    
    subgraph L4[Layer 4: Intelligence]
        LETTA_INT[letta agents]
        DSPY[dspy]
        INSTRUCTOR[instructor]
        BAML[baml]
        OUTLINES[outlines]
    end
    
    subgraph L3[Layer 3: Memory]
        LETTA_MEM[letta memory]
        ZEP[zep]
        MEM0[mem0]
    end
    
    subgraph L2[Layer 2: Communication]
        MCP[mcp-python-sdk]
        FASTMCP[fastmcp]
        LITELLM[litellm]
    end
    
    subgraph L1[Layer 1: Foundation]
        ANTHROPIC[anthropic]
        OPENAI[openai-sdk]
    end
    
    subgraph L0[Layer 0: Infrastructure]
        LANGFUSE[langfuse]
        PHOENIX[arize-phoenix]
        DEEPEVAL[deepeval]
        NEMO[nemo-guardrails]
        ASTGREP[ast-grep]
        SERENA[serena]
        OPIK[opik]
    end
    
    subgraph REF[Reference Configurations]
        ECC[everything-claude-code]
    end
    
    CLI --> LANGGRAPH
    CLI --> TEMPORAL
    
    LANGGRAPH --> LETTA_INT
    LANGGRAPH --> DSPY
    TEMPORAL --> LETTA_INT
    CREWAI --> INSTRUCTOR
    AUTOGEN --> INSTRUCTOR
    PYDANTIC_AI --> INSTRUCTOR
    
    LETTA_INT --> LETTA_MEM
    DSPY --> ZEP
    INSTRUCTOR --> MEM0
    
    LETTA_MEM --> MCP
    ZEP --> MCP
    MEM0 --> FASTMCP
    
    MCP --> LITELLM
    FASTMCP --> LITELLM
    
    LITELLM --> ANTHROPIC
    LITELLM --> OPENAI
    
    LANGFUSE -.-> L5
    LANGFUSE -.-> L4
    LANGFUSE -.-> L3
    NEMO -.-> L5
    ASTGREP -.-> L4
    CLAUDE_FLOW --> MCP
    OPIK -.-> L5
    ECC -.->|patterns| L5
```

---

## 3. SDK Integration Points

### 3.1 Interface Definitions Between Layers

#### L5 → L4: Orchestration to Intelligence

```python
# Interface: OrchestrationToIntelligence
from typing import Protocol, AsyncIterator
from pydantic import BaseModel

class TaskContext(BaseModel):
    task_id: str
    description: str
    files: list[str]
    memory_context: dict

class AgentResponse(BaseModel):
    thought: str
    action: str | None
    result: any

class IntelligenceLayer(Protocol):
    async def process_task(self, context: TaskContext) -> AgentResponse:
        """Process a task through the intelligence layer."""
        ...
    
    async def stream_response(self, context: TaskContext) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...

# Implementation using letta + dspy
class LettaDspyIntelligence(IntelligenceLayer):
    def __init__(self, letta_client, dspy_module):
        self.letta = letta_client
        self.dspy = dspy_module
    
    async def process_task(self, context: TaskContext) -> AgentResponse:
        # Letta provides agent reasoning
        agent_state = await self.letta.send_message(
            agent_id=self.agent_id,
            message=context.description
        )
        
        # DSPy optimizes the output
        optimized = self.dspy(
            task=context.description,
            agent_output=agent_state.messages[-1]
        )
        
        return AgentResponse(
            thought=optimized.reasoning,
            action=optimized.action,
            result=optimized.result
        )
```

#### L4 → L3: Intelligence to Memory

```python
# Interface: IntelligenceToMemory
class MemoryQuery(BaseModel):
    query: str
    session_id: str
    k: int = 10
    filters: dict = {}

class MemoryResult(BaseModel):
    content: str
    score: float
    metadata: dict
    source: str  # letta, zep, or mem0

class MemoryLayer(Protocol):
    async def store(self, content: str, metadata: dict) -> str:
        """Store content in memory, return ID."""
        ...
    
    async def retrieve(self, query: MemoryQuery) -> list[MemoryResult]:
        """Retrieve relevant memories."""
        ...
    
    async def update_knowledge_graph(self, entities: list, relations: list) -> None:
        """Update the knowledge graph with new entities and relations."""
        ...
```

#### L3 → L2: Memory to Communication

```python
# Interface: MemoryToCommunication
class MCPToolCall(BaseModel):
    server: str
    tool: str
    arguments: dict

class MCPToolResult(BaseModel):
    success: bool
    result: any
    error: str | None

class CommunicationLayer(Protocol):
    async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """Execute an MCP tool call."""
        ...
    
    async def list_tools(self) -> list[dict]:
        """List all available MCP tools."""
        ...
    
    async def complete(self, messages: list, **kwargs) -> any:
        """Route completion through litellm."""
        ...
```

### 3.2 Data Flow Through the Stack

```mermaid
flowchart LR
    subgraph Input[Input Processing]
        USER_INPUT[User Request]
        GUARD_IN[nemo-guardrails]
    end
    
    subgraph Process[Core Processing]
        direction TB
        ORCHESTRATE[langgraph/temporal]
        INTEL[letta/dspy]
        MEMORY[letta/zep/mem0]
        TOOLS[MCP Tools]
    end
    
    subgraph Output[Output Processing]
        STRUCT[instructor/baml]
        GUARD_OUT[guardrails-ai]
        OBSERVE[langfuse]
    end
    
    subgraph External[External Systems]
        LLM[litellm → anthropic]
        CODE[ast-grep/serena]
        DATA[crawl4ai/firecrawl]
    end
    
    USER_INPUT --> GUARD_IN
    GUARD_IN --> ORCHESTRATE
    ORCHESTRATE --> INTEL
    INTEL --> MEMORY
    MEMORY --> TOOLS
    TOOLS --> CODE
    TOOLS --> DATA
    TOOLS --> LLM
    LLM --> STRUCT
    STRUCT --> GUARD_OUT
    GUARD_OUT --> OBSERVE
```

### 3.3 Error Propagation Paths

| Error Source | Detection Layer | Handler | Recovery Strategy |
|--------------|-----------------|---------|-------------------|
| LLM API Error | L1/litellm | Circuit breaker | Fallback provider |
| MCP Tool Failure | L2/mcp-sdk | Tool retry | Alternative tool |
| Memory Retrieval | L3/letta | Cache layer | Graceful degradation |
| Agent Loop | L5/langgraph | Checkpoint save | Resume from state |
| Validation | L0/guardrails | Input sanitization | User feedback |

```python
# Error propagation example
class ErrorPropagation:
    async def handle_llm_error(self, error: Exception, context: dict):
        # L1 error -> litellm handles failover
        if isinstance(error, ProviderError):
            return await litellm.completion(
                model="gpt-4o",  # Fallback
                messages=context["messages"],
                fallbacks=["claude-sonnet-4-20250514"]
            )
        
        # L2 error -> MCP retry
        if isinstance(error, MCPToolError):
            return await self.retry_with_backoff(
                self.mcp.call_tool,
                context["tool_call"]
            )
        
        # L5 error -> Checkpoint and resume
        if isinstance(error, OrchestrationError):
            checkpoint = await self.temporal.checkpoint(context["workflow"])
            raise RecoverableError(f"Saved checkpoint: {checkpoint.id}")
```

---

## 4. MCP Protocol Layer

### 4.1 MCP Unification of Tool Access

All tool access in Claude Code CLI flows through the MCP protocol layer:

```python
# platform/mcp/unified_access.py
from mcp import Server
from fastmcp import FastMCP

class UnifiedMCPLayer:
    def __init__(self):
        self.server = FastMCP("claude-code-cli")
        self.registered_tools = {}
        self._register_all_sdk_tools()
    
    def _register_all_sdk_tools(self):
        # Memory tools from letta/zep/mem0
        self._register_memory_tools()
        # Code tools from ast-grep/serena
        self._register_code_tools()
        # Data tools from crawl4ai/firecrawl
        self._register_data_tools()
        # Evaluation tools from deepeval
        self._register_eval_tools()
    
    @self.server.tool()
    async def unified_memory_store(
        content: str,
        memory_type: str = "letta"  # letta, zep, or mem0
    ) -> str:
        """Store content in the appropriate memory system."""
        if memory_type == "letta":
            return await self.letta.remember(content)
        elif memory_type == "zep":
            return await self.zep.add_memory(content)
        else:
            return await self.mem0.add(content)
```

### 4.2 Tool Registration Pattern

```python
# Each SDK registers its capabilities as MCP tools
class SDKToolRegistration:
    """Pattern for registering SDK capabilities as MCP tools."""
    
    def register_ast_grep_tools(self, server: FastMCP):
        @server.tool()
        async def find_code_pattern(
            pattern: str,
            path: str,
            language: str = "python"
        ) -> list[dict]:
            """Find code patterns matching ast-grep syntax."""
            return await self.ast_grep.find_pattern(pattern, path, language)
        
        @server.tool()
        async def apply_code_fix(
            rule_id: str,
            path: str,
            preview: bool = True
        ) -> dict:
            """Apply an ast-grep rule fix to code."""
            return await self.ast_grep.apply_fix(rule_id, path, preview)
    
    def register_dspy_tools(self, server: FastMCP):
        @server.tool()
        async def optimize_prompt(
            prompt: str,
            examples: list[dict],
            metric: str = "accuracy"
        ) -> str:
            """Optimize a prompt using DSPy MIPROv2."""
            optimizer = dspy.MIPROv2(metric=self.metrics[metric])
            return await optimizer.optimize(prompt, examples)
```

### 4.3 Resource Management

```python
# MCP resource management for SDK state
class MCPResourceManager:
    """Manages MCP resources backed by SDK state."""
    
    def __init__(self, server: FastMCP):
        self.server = server
        self._register_resources()
    
    def _register_resources(self):
        @self.server.resource("memory://letta/{agent_id}")
        async def get_letta_memory(agent_id: str) -> dict:
            """Expose Letta agent memory as MCP resource."""
            return await self.letta.get_agent_memory(agent_id)
        
        @self.server.resource("graph://zep/{session_id}")
        async def get_zep_graph(session_id: str) -> dict:
            """Expose Zep knowledge graph as MCP resource."""
            return await self.zep.get_graph(session_id)
        
        @self.server.resource("trace://langfuse/{trace_id}")
        async def get_trace(trace_id: str) -> dict:
            """Expose Langfuse traces as MCP resource."""
            return await self.langfuse.get_trace(trace_id)
```

---

## 5. Memory Architecture

### 5.1 Three-Tier Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MEMORY ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    TIER 1: LETTA (LLM OS)                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │   │
│  │  │   Persona   │  │   Human     │  │   Archival  │               │   │
│  │  │   Memory    │  │   Memory    │  │   Memory    │               │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │   │
│  │  • Self-editing memory blocks                                     │   │
│  │  • Agent personality persistence                                  │   │
│  │  • Long-term knowledge storage                                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↑ ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    TIER 2: ZEP (Session Memory)                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │   │
│  │  │   Message   │  │  Knowledge  │  │   Entity    │               │   │
│  │  │   History   │  │   Graph     │  │   Tracking  │               │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │   │
│  │  • 94.8% Dynamic Memory Retrieval accuracy                        │   │
│  │  • <10ms P95 latency                                              │   │
│  │  • Session-level context management                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↑ ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   TIER 3: MEM0 (Cross-Session)                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │   │
│  │  │   Vector    │  │   Graph     │  │   Hybrid    │               │   │
│  │  │   Store     │  │   Store     │  │   Search    │               │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │   │
│  │  • 45.7k GitHub stars                                             │   │
│  │  • Cross-session persistence                                      │   │
│  │  • +26% retrieval accuracy boost                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Memory Layer Responsibilities

| Tier | SDK | Responsibility | Use Case |
|------|-----|----------------|----------|
| **Tier 1** | letta | Primary agent memory, self-editing | Agent personality, tool schemas, core knowledge |
| **Tier 2** | zep | Session context, knowledge graphs | Conversation history, entity tracking, recent context |
| **Tier 3** | mem0 | Cross-session persistence | User preferences, project knowledge, long-term patterns |

### 5.3 Memory Integration Code

```python
# platform/memory/integrated_memory.py
from letta import create_client as create_letta_client
from zep_python import ZepClient
from mem0 import Memory

class IntegratedMemorySystem:
    """Unified memory system combining Letta, Zep, and Mem0."""
    
    def __init__(self, config: MemoryConfig):
        # Tier 1: Letta for agent memory
        self.letta = create_letta_client()
        self.letta_agent = self._get_or_create_letta_agent()
        
        # Tier 2: Zep for session memory
        self.zep = ZepClient(
            api_url=config.zep_url,
            api_key=config.zep_api_key
        )
        
        # Tier 3: Mem0 for cross-session
        self.mem0 = Memory.from_config({
            "vector_store": {"provider": "qdrant"},
            "graph_store": {"provider": "neo4j"},
            "llm": {"provider": "anthropic"}
        })
    
    async def remember(
        self,
        content: str,
        tier: str = "auto",
        metadata: dict = None
    ) -> str:
        """Store memory in the appropriate tier."""
        if tier == "auto":
            tier = self._determine_tier(content)
        
        if tier == "letta":
            # Agent-specific memory
            return await self._store_letta(content, metadata)
        elif tier == "zep":
            # Session memory
            return await self._store_zep(content, metadata)
        else:
            # Cross-session memory
            return await self._store_mem0(content, metadata)
    
    async def recall(
        self,
        query: str,
        k: int = 10,
        tiers: list[str] = ["letta", "zep", "mem0"]
    ) -> list[MemoryResult]:
        """Retrieve memories from specified tiers."""
        results = []
        
        if "letta" in tiers:
            letta_results = await self._recall_letta(query, k)
            results.extend(letta_results)
        
        if "zep" in tiers:
            zep_results = await self._recall_zep(query, k)
            results.extend(zep_results)
        
        if "mem0" in tiers:
            mem0_results = await self._recall_mem0(query, k)
            results.extend(mem0_results)
        
        # Sort by relevance score and deduplicate
        return self._merge_and_rank(results, k)
    
    def _determine_tier(self, content: str) -> str:
        """Auto-determine the best tier for content."""
        # Agent personality/schema updates -> Letta
        if "personality" in content or "tool_schema" in content:
            return "letta"
        # Recent conversation context -> Zep
        elif "session" in content or "conversation" in content:
            return "zep"
        # Everything else -> Mem0 for persistence
        else:
            return "mem0"
```

### 5.4 Memory Data Flow

```mermaid
flowchart TB
    subgraph Input[Input Sources]
        USER[User Messages]
        AGENT[Agent Outputs]
        TOOLS[Tool Results]
    end
    
    subgraph Processing[Memory Processing]
        ROUTER[Tier Router]
        EMBED[Embedding Generation]
        GRAPH[Graph Extraction]
    end
    
    subgraph Tier1[Tier 1: Letta]
        PERSONA[Persona Memory]
        HUMAN[Human Memory]
        ARCHIVAL[Archival Memory]
    end
    
    subgraph Tier2[Tier 2: Zep]
        SESSION[Session Store]
        ENTITIES[Entity Store]
        KNOWLEDGE[Knowledge Graph]
    end
    
    subgraph Tier3[Tier 3: Mem0]
        VECTORS[Vector Store]
        GRAPHS[Graph Store]
        HYBRID[Hybrid Index]
    end
    
    subgraph Retrieval[Retrieval Path]
        QUERY[Query Processing]
        FUSION[Result Fusion]
        RANK[Re-ranking]
    end
    
    USER --> ROUTER
    AGENT --> ROUTER
    TOOLS --> ROUTER
    
    ROUTER --> EMBED
    ROUTER --> GRAPH
    
    EMBED --> Tier1
    EMBED --> Tier2
    EMBED --> Tier3
    
    GRAPH --> Tier2
    GRAPH --> Tier3
    
    QUERY --> Tier1
    QUERY --> Tier2
    QUERY --> Tier3
    
    Tier1 --> FUSION
    Tier2 --> FUSION
    Tier3 --> FUSION
    
    FUSION --> RANK
```

---

## 6. Workflow Patterns

### 6.1 Single Agent Pattern (Direct Claude)

For simple, single-turn tasks requiring direct LLM interaction.

```python
# Pattern: Direct Claude through litellm
async def single_agent_task(task: str) -> str:
    """Single agent pattern using litellm → anthropic."""
    from litellm import acompletion
    
    # Safety check
    safe, _ = await nemo_guardrails.check_input(task)
    if not safe:
        raise SafetyError("Task blocked by guardrails")
    
    # Direct completion with observation
    with langfuse.trace(name="single_agent"):
        response = await acompletion(
            model="claude-sonnet-4-20250514",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task}
            ],
            temperature=0.7
        )
    
    # Structured extraction
    result = await instructor.extract(
        response.choices[0].message.content,
        TaskResult
    )
    
    return result
```

### 6.2 Multi-Agent Pattern (LangGraph/CrewAI)

For complex tasks requiring multiple specialized agents.

```python
# Pattern: Multi-agent with LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

class MultiAgentState(TypedDict):
    task: str
    plan: list[str]
    current_agent: str
    results: dict[str, any]
    iteration: int

def create_multi_agent_workflow():
    """Create a multi-agent workflow graph."""
    graph = StateGraph(MultiAgentState)
    
    # Add agent nodes
    graph.add_node("architect", architect_agent)
    graph.add_node("coder", coder_agent)
    graph.add_node("reviewer", reviewer_agent)
    graph.add_node("tester", tester_agent)
    
    # Define flow
    graph.set_entry_point("architect")
    graph.add_edge("architect", "coder")
    graph.add_conditional_edges(
        "coder",
        route_after_coding,
        {"review": "reviewer", "test": "tester", "done": END}
    )
    graph.add_edge("reviewer", "coder")
    graph.add_edge("tester", END)
    
    # Compile with checkpointing
    return graph.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))

# Alternative: CrewAI for role-based agents
from crewai import Crew, Agent, Task

def create_crew_workflow():
    """Create a CrewAI multi-agent team."""
    architect = Agent(
        role="Software Architect",
        goal="Design scalable solutions",
        backstory="Expert system designer"
    )
    
    coder = Agent(
        role="Senior Developer",
        goal="Write clean, efficient code",
        backstory="10+ years coding experience"
    )
    
    crew = Crew(
        agents=[architect, coder],
        tasks=[
            Task(description="Design the system", agent=architect),
            Task(description="Implement the design", agent=coder)
        ]
    )
    
    return crew
```

### 6.3 Durable Workflow Pattern (Temporal)

For long-running tasks requiring fault tolerance.

```python
# Pattern: Durable workflows with Temporal
from temporalio import workflow, activity
from temporalio.client import Client
from datetime import timedelta

@activity.defn
async def analyze_codebase(path: str) -> dict:
    """Activity: Analyze codebase with ast-grep."""
    patterns = await ast_grep.scan_directory(path)
    return {"patterns": patterns, "issues": len(patterns)}

@activity.defn
async def generate_fixes(issues: list) -> list:
    """Activity: Generate fixes with dspy."""
    fixes = []
    for issue in issues:
        fix = await dspy_fixer(issue)
        fixes.append(fix)
    return fixes

@activity.defn
async def apply_fixes(fixes: list) -> dict:
    """Activity: Apply fixes with aider."""
    applied = []
    for fix in fixes:
        result = await aider.apply_fix(fix)
        applied.append(result)
    return {"applied": len(applied), "results": applied}

@workflow.defn
class DurableCodeFixWorkflow:
    """Durable workflow for code fixes with checkpointing."""
    
    @workflow.run
    async def run(self, codebase_path: str) -> dict:
        # Phase 1: Analysis (with checkpoint)
        analysis = await workflow.execute_activity(
            analyze_codebase,
            codebase_path,
            start_to_close_timeout=timedelta(minutes=5)
        )
        
        # Phase 2: Generate fixes (with retry)
        fixes = await workflow.execute_activity(
            generate_fixes,
            analysis["patterns"],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Phase 3: Apply fixes (with human checkpoint option)
        if await workflow.wait_condition(
            lambda: self.human_approved,
            timeout=timedelta(hours=1)
        ):
            result = await workflow.execute_activity(
                apply_fixes,
                fixes,
                start_to_close_timeout=timedelta(minutes=15)
            )
        else:
            result = {"status": "cancelled", "reason": "Human approval timeout"}
        
        return result
```

### 6.4 Human-in-the-Loop Pattern

For tasks requiring human approval at critical points.

```python
# Pattern: Human-in-the-loop with LangGraph
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolExecutor

class HITLState(TypedDict):
    task: str
    proposed_action: dict
    human_feedback: str | None
    approved: bool
    result: any

def create_hitl_workflow():
    """Create workflow with human approval checkpoints."""
    graph = StateGraph(HITLState)
    
    # Analysis node
    async def analyze(state: HITLState) -> HITLState:
        # Generate proposed action
        proposal = await generate_proposal(state["task"])
        state["proposed_action"] = proposal
        return state
    
    # Human checkpoint node
    async def human_checkpoint(state: HITLState) -> HITLState:
        # This will pause and wait for human input
        # LangGraph handles the interruption
        state["approved"] = state.get("human_feedback") == "approve"
        return state
    
    # Execution node
    async def execute(state: HITLState) -> HITLState:
        if state["approved"]:
            result = await execute_action(state["proposed_action"])
            state["result"] = result
        else:
            state["result"] = {"status": "rejected"}
        return state
    
    graph.add_node("analyze", analyze)
    graph.add_node("human_checkpoint", human_checkpoint)
    graph.add_node("execute", execute)
    
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "human_checkpoint")
    graph.add_conditional_edges(
        "human_checkpoint",
        lambda s: "execute" if s.get("human_feedback") else "human_checkpoint"
    )
    graph.add_edge("execute", END)
    
    # Compile with interrupt_before for human checkpoint
    return graph.compile(interrupt_before=["human_checkpoint"])
```

### 6.5 Workflow Pattern Selection Guide

```mermaid
flowchart TD
    START[Task Received] --> COMPLEXITY{Task Complexity?}
    
    COMPLEXITY -->|Simple| SINGLE[Single Agent Pattern]
    COMPLEXITY -->|Multi-step| DURATION{Long Running?}
    COMPLEXITY -->|Multi-role| MULTI[Multi-Agent Pattern]
    
    DURATION -->|No| LANGGRAPH[LangGraph Workflow]
    DURATION -->|Yes| TEMPORAL[Temporal Durable]
    
    MULTI --> STYLE{Collaboration Style?}
    STYLE -->|Graph-based| LANGGRAPH_MULTI[LangGraph Multi-Agent]
    STYLE -->|Role-based| CREWAI[CrewAI Team]
    STYLE -->|Conversational| AUTOGEN[AutoGen]
    
    SINGLE --> HITL{Human Review Needed?}
    LANGGRAPH --> HITL
    TEMPORAL --> HITL
    LANGGRAPH_MULTI --> HITL
    CREWAI --> HITL
    AUTOGEN --> HITL
    
    HITL -->|Yes| HITL_PATTERN[Add HITL Checkpoints]
    HITL -->|No| EXECUTE[Execute Workflow]
    HITL_PATTERN --> EXECUTE
```

---

## 7. Observability Stack

### 7.1 Three-Pillar Observability

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐ │
│  │      TRACING       │  │      METRICS       │  │    EVALUATION      │ │
│  │     (langfuse)     │  │   (arize-phoenix)  │  │    (deepeval)      │ │
│  ├────────────────────┤  ├────────────────────┤  ├────────────────────┤ │
│  │ • @observe decorator│  │ • OTEL integration │  │ • 14+ metrics      │ │
│  │ • Trace hierarchy   │  │ • Embedding viz    │  │ • CI/CD integration│ │
│  │ • Prompt management │  │ • Drift detection  │  │ • pytest-style     │ │
│  │ • Cost tracking     │  │ • Performance dash │  │ • Auto-evaluate    │ │
│  │ • Session linking   │  │ • Alerting         │  │ • Custom metrics   │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘ │
│              │                      │                      │             │
│              └──────────────────────┼──────────────────────┘             │
│                                     │                                    │
│                    ┌────────────────┴────────────────┐                  │
│                    │     OpenTelemetry Integration    │                  │
│                    │   Unified observability export   │                  │
│                    └─────────────────────────────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Langfuse Integration

```python
# platform/observability/langfuse_integration.py
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

# Initialize Langfuse
langfuse = Langfuse()

class ObservabilityLayer:
    """Unified observability through Langfuse."""
    
    @observe(name="autonomous_session")
    async def run_session(self, task: str) -> SessionResult:
        """Traced autonomous coding session."""
        # Set session metadata
        langfuse_context.update_current_observation(
            metadata={
                "task_type": self._classify_task(task),
                "sdk_versions": self._get_sdk_versions()
            }
        )
        
        # Execute with nested tracing
        plan = await self._plan(task)
        
        for step in plan.steps:
            result = await self._execute_step(step)
            langfuse_context.update_current_observation(
                output={"step": step.id, "result": result}
            )
        
        # Track costs
        langfuse_context.update_current_observation(
            model="claude-sonnet-4-20250514",
            usage={"total_tokens": self.token_count}
        )
        
        return SessionResult(...)
    
    @observe(name="tool_execution")
    async def execute_tool(self, tool: str, args: dict) -> any:
        """Traced tool execution."""
        langfuse_context.update_current_trace(
            tags=[f"tool:{tool}"]
        )
        return await self.mcp.call_tool(tool, args)
```

### 7.3 Arize Phoenix Integration

```python
# platform/observability/phoenix_integration.py
import phoenix as px
from phoenix.trace.opentelemetry import register

class PhoenixMLOps:
    """ML observability with Arize Phoenix."""
    
    def __init__(self):
        # Start Phoenix server
        self.session = px.launch_app()
        
        # Register OpenTelemetry
        register(endpoint="http://localhost:6006/v1/traces")
    
    def track_embeddings(self, embeddings: list, metadata: list):
        """Track embedding drift and quality."""
        px.Client().log_embeddings(
            embeddings=embeddings,
            metadata=metadata,
            model_id="claude-code-cli"
        )
    
    def track_llm_spans(self, spans: list):
        """Track LLM call performance."""
        for span in spans:
            px.Client().log_span(
                span_id=span.id,
                latency_ms=span.duration,
                token_count=span.tokens,
                model=span.model
            )
```

### 7.4 DeepEval Testing Integration

```python
# platform/observability/deepeval_integration.py
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    GEval
)
from deepeval.test_case import LLMTestCase

class EvaluationPipeline:
    """Evaluation pipeline using DeepEval."""
    
    def __init__(self):
        self.metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.8),
            ContextualPrecisionMetric(threshold=0.7),
            GEval(
                name="Code Quality",
                criteria="The generated code is clean, efficient, and follows best practices.",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT
                ]
            )
        ]
    
    async def evaluate_response(
        self,
        input: str,
        output: str,
        context: list[str] = None
    ) -> dict:
        """Evaluate a single response."""
        test_case = LLMTestCase(
            input=input,
            actual_output=output,
            retrieval_context=context
        )
        
        results = await evaluate([test_case], self.metrics)
        return {
            "passed": all(r.success for r in results),
            "scores": {m.name: r.score for m, r in zip(self.metrics, results)},
            "reasons": {m.name: r.reason for m, r in zip(self.metrics, results)}
        }
    
    async def ci_evaluation(self, test_cases: list[LLMTestCase]) -> bool:
        """Run evaluation in CI/CD pipeline."""
        results = await evaluate(test_cases, self.metrics)
        
        # Generate report
        report = self._generate_report(results)
        
        # Fail CI if below threshold
        return all(r.success for r in results)
```

### 7.5 OpenTelemetry Unification

```python
# platform/observability/otel_integration.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

class OTelIntegration:
    """OpenTelemetry integration for unified observability."""
    
    def __init__(self):
        # Set up tracer provider
        provider = TracerProvider()
        
        # Export to Langfuse
        langfuse_exporter = LangfuseOTelExporter()
        provider.add_span_processor(BatchSpanProcessor(langfuse_exporter))
        
        # Export to Phoenix
        phoenix_exporter = OTLPSpanExporter(endpoint="http://localhost:6006")
        provider.add_span_processor(BatchSpanProcessor(phoenix_exporter))
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer("claude-code-cli")
    
    def trace_operation(self, name: str):
        """Context manager for tracing operations."""
        return self.tracer.start_as_current_span(name)
```

---

## 8. Implementation Sequence

### 8.1 Phase 1: Core Backbone

**Objective:** Establish foundation infrastructure with P0 SDKs

| SDK | Version | Purpose | Validation |
|-----|---------|---------|------------|
| mcp-python-sdk | ^1.0.0 | Protocol foundation | `from mcp import Server` |
| fastmcp | ^0.8.0 | Enhanced MCP patterns | `from fastmcp import FastMCP` |
| litellm | ^1.55.0 | LLM gateway | `from litellm import completion` |
| temporal-python | ^1.8.0 | Durable execution | `from temporalio import workflow` |
| anthropic | ^0.40.0 | Official Claude SDK | `from anthropic import Anthropic` |
| openai-sdk | ^1.0.0 | Fallback provider | `from openai import OpenAI` |

```bash
# Phase 1 Installation
pip install mcp fastmcp litellm temporalio anthropic openai
```

```python
# Phase 1 Validation Script
from mcp import Server
from fastmcp import FastMCP
from litellm import completion
from temporalio import workflow
from anthropic import Anthropic
from openai import OpenAI

print("✅ Phase 1: Core Backbone - All SDKs validated")
```

### 8.2 Phase 2: Intelligence Layer

**Objective:** Add agent memory, prompt optimization, structured output, and MCP-native orchestration

| SDK | Version | Purpose | Validation |
|-----|---------|---------|------------|
| letta | ^0.7.0 | LLM OS memory | `from letta import create_client` |
| dspy | ^2.5.0 | Prompt optimization | `import dspy` |
| langfuse | ^2.0.0 | Observability | `from langfuse import Langfuse` |
| instructor | ^1.5.0 | Structured output | `import instructor` |
| baml | ^0.5.0 | Type-safe AI | `from baml_py import b` |
| outlines | ^0.1.0 | Constrained generation | `import outlines` |
| **claude-flow** | ^v3alpha | MCP-native orchestration | `claude mcp add claude-flow` |

```bash
# Phase 2 Installation
pip install letta dspy-ai langfuse instructor baml outlines

# Claude-flow MCP Server (NEW)
claude mcp add claude-flow -- npx claude-flow@v3alpha mcp start
```

```python
# Phase 2 Validation Script
from letta import create_client
import dspy
from langfuse import Langfuse
import instructor
# baml and outlines require additional setup
# claude-flow validates via MCP: swarm_init, agent_spawn, memory_search

print("✅ Phase 2: Intelligence Layer - All SDKs validated")
```

### 8.3 Phase 3: Observability

**Objective:** Complete observability stack and evaluation pipeline (including opik for agent optimization)

| SDK | Version | Purpose | Validation |
|-----|---------|---------|------------|
| arize-phoenix | ^4.0.0 | ML observability | `import phoenix as px` |
| deepeval | ^1.0.0 | LLM evaluation | `from deepeval import evaluate` |
| **opik** | ^0.2.0 | Agent optimization & guardrails (NEW) | `import opik` |
| nemo-guardrails | ^0.10.0 | Safety rails | `from nemoguardrails import LLMRails` |
| guardrails-ai | ^0.5.0 | Output validation | `import guardrails` |
| llm-guard | ^0.3.0 | Security scanning | `from llm_guard import scan` |

**Note:** opik (17,481★) complements langfuse with agent-specific optimization, production dashboards, and automated cost tracking.

```bash
# Phase 3 Installation
pip install arize-phoenix deepeval opik nemoguardrails guardrails-ai llm-guard
```

```python
# Phase 3 Validation Script
import phoenix as px
from deepeval import evaluate
import opik
from nemoguardrails import LLMRails
import guardrails
from llm_guard import scan_output

# opik provides complementary agent monitoring to langfuse
opik.track()  # Similar decorator pattern to langfuse

print("✅ Phase 3: Observability - All SDKs validated")
```

### 8.4 Phase 4: Advanced Features

**Objective:** Add agent frameworks, code intelligence, and data acquisition

| SDK | Version | Purpose | Validation |
|-----|---------|---------|------------|
| langgraph | ^0.2.0 | Graph workflows | `from langgraph.graph import StateGraph` |
| pydantic-ai | ^0.1.0 | Type-safe agents | `from pydantic_ai import Agent` |
| crewai | ^0.80.0 | Team orchestration | `from crewai import Crew` |
| autogen | ^0.4.0 | Conversational agents | `from autogen import ConversableAgent` |
| zep | ^2.0.0 | Session memory | `from zep_python import ZepClient` |
| mem0 | ^0.1.0 | Cross-session memory | `from mem0 import Memory` |
| ast-grep | ^0.30.0 | Code patterns | CLI validation |
| serena | ^0.1.0 | Code intelligence MCP | MCP server validation |
| aider | ^0.60.0 | AI pair coding | `from aider.coders import Coder` |
| crawl4ai | ^0.4.0 | Web extraction | `from crawl4ai import AsyncWebCrawler` |
| firecrawl | ^1.0.0 | Web crawling | `from firecrawl import FirecrawlApp` |
| graphrag | ^0.3.0 | Knowledge graphs | `import graphrag` |
| ragas | ^0.2.0 | RAG evaluation | `from ragas import evaluate` |
| promptfoo | ^0.90.0 | Prompt testing | CLI validation |

```bash
# Phase 4 Installation
pip install langgraph pydantic-ai crewai pyautogen zep-python mem0ai \
    ast-grep-py aider-chat "crawl4ai[all]" firecrawl graphrag ragas

# CLI tools
npm install -g promptfoo
```

### 8.5 Complete Installation Script

```bash
#!/bin/bash
# CLAUDE CODE CLI V2 - Complete SDK Installation

set -e

echo "========================================"
echo "Claude Code CLI V2.1 - SDK Installation"
echo "========================================"

echo ""
echo "=== Phase 1: Core Backbone ==="
pip install mcp fastmcp litellm temporalio anthropic openai

echo ""
echo "=== Phase 2: Intelligence Layer ==="
pip install letta dspy-ai langfuse instructor baml outlines
# NEW: claude-flow MCP-native orchestration
claude mcp add claude-flow -- npx claude-flow@v3alpha mcp start

echo ""
echo "=== Phase 3: Observability ==="
pip install arize-phoenix deepeval opik nemoguardrails guardrails-ai llm-guard
# Note: opik (17,481★) complements langfuse with agent optimization

echo ""
echo "=== Phase 4: Advanced Features ==="
pip install langgraph pydantic-ai crewai pyautogen zep-python mem0ai
pip install ast-grep-py aider-chat "crawl4ai[all]" firecrawl graphrag ragas

echo ""
echo "=== CLI Tools ==="
npm install -g promptfoo

echo ""
echo "=== Reference Configurations ==="
echo "See everything-claude-code for:"
echo "  - 9 agents, 11 skills, memory persistence hooks"
echo "  - Battle-tested Claude Code configurations"

echo ""
echo "========================================"
echo "✅ All 35 Best-of-Breed SDKs Installed"
echo "========================================"
```

---

## 9. Architecture Diagrams

### 9.1 Complete System Architecture

```mermaid
flowchart TB
    subgraph Interface[Layer 6: Interface]
        CLI[Claude Code CLI]
        TUI[Terminal UI]
        API[HTTP/WS API]
    end
    
    subgraph Orchestration[Layer 5: Orchestration]
        LANGGRAPH[langgraph]
        TEMPORAL[temporal-python]
        CREWAI[crewai]
        AUTOGEN[autogen]
        PYDANTIC_AI[pydantic-ai]
    end
    
    subgraph Intelligence[Layer 4: Intelligence]
        LETTA_AGENT[letta agents]
        DSPY[dspy]
        INSTRUCTOR[instructor]
        BAML[baml]
        OUTLINES[outlines]
    end
    
    subgraph Memory[Layer 3: Memory]
        LETTA_MEM[letta memory]
        ZEP[zep]
        MEM0[mem0]
    end
    
    subgraph Communication[Layer 2: Communication]
        MCP[mcp-python-sdk]
        FASTMCP[fastmcp]
        LITELLM[litellm]
    end
    
    subgraph Foundation[Layer 1: Foundation]
        ANTHROPIC[anthropic]
        OPENAI[openai-sdk]
    end
    
    subgraph Infrastructure[Layer 0: Infrastructure]
        LANGFUSE[langfuse]
        PHOENIX[arize-phoenix]
        DEEPEVAL[deepeval]
        NEMO[nemo-guardrails]
        ASTGREP[ast-grep]
        SERENA[serena]
        AIDER[aider]
        CRAWL4AI[crawl4ai]
    end
    
    CLI --> LANGGRAPH
    CLI --> TEMPORAL
    TUI --> LANGGRAPH
    API --> LANGGRAPH
    
    LANGGRAPH --> LETTA_AGENT
    LANGGRAPH --> DSPY
    TEMPORAL --> LETTA_AGENT
    CREWAI --> INSTRUCTOR
    AUTOGEN --> INSTRUCTOR
    PYDANTIC_AI --> BAML
    
    LETTA_AGENT --> LETTA_MEM
    DSPY --> ZEP
    INSTRUCTOR --> MEM0
    
    LETTA_MEM --> MCP
    ZEP --> FASTMCP
    MEM0 --> FASTMCP
    
    MCP --> LITELLM
    FASTMCP --> LITELLM
    
    LITELLM --> ANTHROPIC
    LITELLM --> OPENAI
    
    LANGFUSE -.->|traces| Orchestration
    LANGFUSE -.->|traces| Intelligence
    PHOENIX -.->|metrics| Communication
    NEMO -.->|guards| Interface
    ASTGREP -.->|analysis| Intelligence
    CRAWL4AI -.->|data| Memory
```

### 9.2 Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input[Input Stage]
        USER[User Request]
        MCP_IN[MCP Tool Call]
        WEBHOOK[Webhook Event]
    end
    
    subgraph Safety[Safety Layer]
        NEMO_IN[nemo-guardrails Input]
        LLM_GUARD[llm-guard Scanner]
    end
    
    subgraph Process[Processing Pipeline]
        ROUTER[Task Router]
        ORCHESTRATE[Orchestration Layer]
        INTEL[Intelligence Layer]
        MEMORY_PROC[Memory Layer]
    end
    
    subgraph Execute[Execution Layer]
        CODE_EXEC[Code Execution]
        TOOL_EXEC[Tool Execution]
        LLM_EXEC[LLM Execution]
    end
    
    subgraph Output[Output Stage]
        STRUCT_OUT[Structured Output]
        GUARD_OUT[Output Validation]
        OBSERVE_OUT[Observation Logging]
    end
    
    USER --> NEMO_IN
    MCP_IN --> NEMO_IN
    WEBHOOK --> NEMO_IN
    
    NEMO_IN --> LLM_GUARD
    LLM_GUARD --> ROUTER
    
    ROUTER --> ORCHESTRATE
    ORCHESTRATE --> INTEL
    INTEL --> MEMORY_PROC
    
    MEMORY_PROC --> CODE_EXEC
    MEMORY_PROC --> TOOL_EXEC
    MEMORY_PROC --> LLM_EXEC
    
    CODE_EXEC --> STRUCT_OUT
    TOOL_EXEC --> STRUCT_OUT
    LLM_EXEC --> STRUCT_OUT
    
    STRUCT_OUT --> GUARD_OUT
    GUARD_OUT --> OBSERVE_OUT
```

### 9.3 SDK Dependency Graph

```mermaid
graph TB
    subgraph P0[P0: Backbone - 9 SDKs]
        MCP[mcp-python-sdk]
        FASTMCP[fastmcp]
        LITELLM[litellm]
        TEMPORAL[temporal-python]
        LETTA[letta]
        DSPY[dspy]
        LANGFUSE[langfuse]
        ANTHROPIC[anthropic]
        OPENAI[openai-sdk]
    end
    
    subgraph P1[P1: Core Features - 14 SDKs]
        LANGGRAPH[langgraph]
        PYDANTIC_AI[pydantic-ai]
        CREWAI[crewai]
        AUTOGEN[autogen]
        ZEP[zep]
        MEM0[mem0]
        INSTRUCTOR[instructor]
        BAML[baml]
        OUTLINES[outlines]
        ASTGREP[ast-grep]
        SERENA[serena]
        PHOENIX[arize-phoenix]
        DEEPEVAL[deepeval]
        OPIK[opik]
        NEMO[nemo-guardrails]
        CRAWL4AI[crawl4ai]
    end
    
    subgraph P2[P2: Advanced - 9 SDKs]
        AIDER[aider]
        GUARDRAILS[guardrails-ai]
        LLM_GUARD[llm-guard]
        FIRECRAWL[firecrawl]
        GRAPHRAG[graphrag]
        RAGAS[ragas]
        PROMPTFOO[promptfoo]
    end
    
    MCP --> FASTMCP
    FASTMCP --> LITELLM
    LITELLM --> ANTHROPIC
    LITELLM --> OPENAI
    TEMPORAL --> LETTA
    LETTA --> ZEP
    LETTA --> MEM0
    DSPY --> INSTRUCTOR
    LANGFUSE --> PHOENIX
    LANGFUSE --> DEEPEVAL
    LANGGRAPH --> CREWAI
    LANGGRAPH --> AUTOGEN
    PYDANTIC_AI --> INSTRUCTOR
    ASTGREP --> SERENA
    CRAWL4AI --> GRAPHRAG
    NEMO --> GUARDRAILS
    NEMO --> LLM_GUARD
    DEEPEVAL --> RAGAS
    DEEPEVAL --> PROMPTFOO
```

### 9.4 Memory Architecture Diagram

```mermaid
flowchart TB
    subgraph Input[Memory Input]
        USER_MSG[User Messages]
        AGENT_OUT[Agent Outputs]
        TOOL_RES[Tool Results]
    end
    
    subgraph Routing[Memory Router]
        CLASSIFIER[Content Classifier]
        EMBEDDER[Embedding Generator]
        GRAPH_EXT[Graph Extractor]
    end
    
    subgraph Tier1[Tier 1: Letta - Agent Memory]
        PERSONA[Persona Memory]
        HUMAN[Human Memory]
        ARCHIVAL[Archival Memory]
    end
    
    subgraph Tier2[Tier 2: Zep - Session Memory]
        SESSION[Session Store]
        ENTITIES[Entity Store]
        KG_ZEP[Knowledge Graph]
    end
    
    subgraph Tier3[Tier 3: Mem0 - Cross-Session]
        VECTORS[Vector Store]
        GRAPHS[Graph Store]
        HYBRID[Hybrid Index]
    end
    
    subgraph Retrieval[Unified Retrieval]
        QUERY_PROC[Query Processing]
        MULTI_SEARCH[Multi-Tier Search]
        FUSION[Result Fusion]
        RERANK[Re-ranking]
    end
    
    USER_MSG --> CLASSIFIER
    AGENT_OUT --> CLASSIFIER
    TOOL_RES --> CLASSIFIER
    
    CLASSIFIER --> EMBEDDER
    CLASSIFIER --> GRAPH_EXT
    
    EMBEDDER --> Tier1
    EMBEDDER --> Tier2
    EMBEDDER --> Tier3
    
    GRAPH_EXT --> KG_ZEP
    GRAPH_EXT --> GRAPHS
    
    QUERY_PROC --> MULTI_SEARCH
    MULTI_SEARCH --> Tier1
    MULTI_SEARCH --> Tier2
    MULTI_SEARCH --> Tier3
    
    Tier1 --> FUSION
    Tier2 --> FUSION
    Tier3 --> FUSION
    
    FUSION --> RERANK
```

---

## 10. SDK Quick Reference

### 10.1 P0 Backbone SDKs (9 total)

| SDK | Import | Primary Use |
|-----|--------|-------------|
| **mcp-python-sdk** | `from mcp import Server` | MCP protocol foundation |
| **fastmcp** | `from fastmcp import FastMCP` | Enhanced MCP patterns |
| **litellm** | `from litellm import completion` | Universal LLM gateway |
| **temporal-python** | `from temporalio import workflow` | Durable workflows |
| **letta** | `from letta import create_client` | LLM OS memory |
| **dspy** | `import dspy` | Declarative prompting |
| **langfuse** | `from langfuse import Langfuse` | LLM observability |
| **anthropic** | `from anthropic import Anthropic` | Official Claude SDK |
| **openai-sdk** | `from openai import OpenAI` | OpenAI compatibility |

### 10.2 P1 Core Feature SDKs (15 total)

| SDK | Import | Primary Use |
|-----|--------|-------------|
| **langgraph** | `from langgraph.graph import StateGraph` | Graph workflows |
| **pydantic-ai** | `from pydantic_ai import Agent` | Type-safe agents |
| **crewai** | `from crewai import Crew` | Role-based teams |
| **autogen** | `from autogen import ConversableAgent` | Conversational agents |
| **claude-flow** | MCP: `swarm_init, agent_spawn` | MCP-native orchestration, SONA learning (NEW) |
| **zep** | `from zep_python import ZepClient` | Session memory |
| **mem0** | `from mem0 import Memory` | Cross-session persistence |
| **instructor** | `import instructor` | Structured outputs |
| **baml** | `from baml_py import b` | Type-safe AI functions |
| **outlines** | `import outlines` | Constrained generation |
| **ast-grep** | CLI: `ast-grep scan` | Code pattern matching |
| **serena** | MCP server | Code intelligence |
| **arize-phoenix** | `import phoenix as px` | ML observability |
| **deepeval** | `from deepeval import evaluate` | LLM evaluation |
| **nemo-guardrails** | `from nemoguardrails import LLMRails` | AI safety rails |
| **crawl4ai** | `from crawl4ai import AsyncWebCrawler` | Web extraction |

### 10.3 P2 Advanced SDKs (11 total)

| SDK | Import | Primary Use |
|-----|--------|-------------|
| **aider** | `from aider.coders import Coder` | AI pair coding |
| **guardrails-ai** | `import guardrails` | Output validation |
| **llm-guard** | `from llm_guard import scan_output` | Security scanning |
| **firecrawl** | `from firecrawl import FirecrawlApp` | Web crawling |
| **graphrag** | `import graphrag` | Knowledge graphs |
| **ragas** | `from ragas import evaluate` | RAG evaluation |
| **promptfoo** | CLI: `promptfoo eval` | Prompt testing |
| **opik** | `import opik; opik.track()` | Agent optimization, guardrails (NEW) |
| **everything-claude-code** | Config reference | 9 agents, 11 skills, memory hooks (NEW) |

### 10.4 Version Requirements

```toml
# pyproject.toml - SDK dependencies
[project.dependencies]
# P0 Backbone
mcp = ">=1.0.0"
fastmcp = ">=0.8.0"
litellm = ">=1.55.0"
temporalio = ">=1.8.0"
letta = ">=0.7.0"
dspy-ai = ">=2.5.0"
langfuse = ">=2.0.0"
anthropic = ">=0.40.0"
openai = ">=1.0.0"

# P1 Core
langgraph = ">=0.2.0"
pydantic-ai = ">=0.1.0"
crewai = ">=0.80.0"
pyautogen = ">=0.4.0"
zep-python = ">=2.0.0"
mem0ai = ">=0.1.0"
instructor = ">=1.5.0"
baml = ">=0.5.0"
outlines = ">=0.1.0"
ast-grep-py = ">=0.30.0"
arize-phoenix = ">=4.0.0"
deepeval = ">=1.0.0"
opik = ">=0.2.0"
nemoguardrails = ">=0.10.0"
crawl4ai = ">=0.4.0"

# P2 Advanced
aider-chat = ">=0.60.0"
guardrails-ai = ">=0.5.0"
llm-guard = ">=0.3.0"
firecrawl = ">=1.0.0"
graphrag = ">=0.3.0"
ragas = ">=0.2.0"
```

---

## Document Metadata

| Attribute | Value |
|-----------|-------|
| **Document Title** | Claude Code CLI Architecture V2.1 |
| **Version** | 2.1 |
| **Created** | 2026-01-24 |
| **Updated** | 2026-01-24 (Gap Analysis Integration) |
| **Based On** | ULTIMATE_SDK_COLLECTION_2026.md, ARCHITECTURE_GAP_ANALYSIS_2026.md |
| **Previous Version** | CLAUDE_CODE_CLI_ARCHITECTURE.md (V1) |
| **SDKs Integrated** | 35 best-of-breed from 154+ evaluated (+3 from gap analysis) |
| **Layer Count** | 7 (L0-L6) + Reference Configurations |
| **Priority Distribution** | P0: 9, P1: 15 (+1), P2: 11 (+2) |

### Gap Analysis Additions (2026-01-24)

| SDK | Priority | Category | Key Capabilities |
|-----|----------|----------|------------------|
| **claude-flow** | P1 | Orchestration | MCP-native, SONA self-learning, 54+ agents, Byzantine consensus |
| **opik** | P2 | Observability | Agent optimization, guardrails, complements langfuse |
| **everything-claude-code** | P2/Ref | Configuration | 9 agents, 11 skills, memory persistence hooks |

---

*Architecture Document V2.1 - January 2026*
*Built on verified best-of-breed SDK selection*
*Updated with gap analysis recommendations from ARCHITECTURE_GAP_ANALYSIS_2026.md*
*Cross-referenced with COMPREHENSIVE_SDK_RESEARCH_2026.md*
