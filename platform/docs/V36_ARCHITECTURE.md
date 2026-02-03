# UNLEASH Platform V36 Architecture

> **Version**: 36.0.0 | **Updated**: February 2026 | **SDKs**: 39

## Overview

UNLEASH is a unified SDK integration platform that aggregates 39 best-in-class AI/ML SDKs into a coherent 8-layer architecture for building production-grade AI agent systems.

```
┌─────────────────────────────────────────────────────────────────┐
│                         UNLEASH V36                              │
│                                                                  │
│   Protocol → Orchestration → Memory → Intelligence → Safety     │
│      ↓            ↓            ↓           ↓           ↓        │
│   Gateway     Multi-Agent   Persistent   Reasoning   Guardrails │
│   Routing     Coordination  Context      Optimization Content   │
│                                                                  │
│   With: Health-aware routing, circuit breakers, observability   │
└─────────────────────────────────────────────────────────────────┘
```

## 8-Layer SDK Architecture

| Layer | Name | SDKs | Purpose |
|-------|------|------|---------|
| **L0** | Protocol | anthropic, litellm, mcp, portkey, a2a | LLM APIs, gateways, MCP |
| **L1** | Orchestration | langgraph, temporal, openai-agents, strands | Agent frameworks, workflows |
| **L2** | Memory | letta, graphiti, mem0, simplemem | Persistent context, knowledge |
| **L3** | Structured | instructor, baml, outlines, pydantic-ai | Output generation |
| **L4** | Reasoning | dspy, agot | Optimization, reasoning |
| **L5** | Observability | langfuse, opik, braintrust, phoenix | Monitoring, evaluation |
| **L6** | Safety | guardrails-ai, llm-guard, nemo | Content moderation |
| **L7** | Processing | ragflow, ragatouille, crawl4ai, ast-grep | RAG, retrieval |
| **L8** | Knowledge | cognee, graphrag, pyribs | Knowledge graphs |

## V36 New Adapters (11)

### P0 Critical
- **OpenAI Agents SDK** (L1) - Multi-agent orchestration with handoffs
- **Cognee V36** (L8) - 90% multi-hop reasoning accuracy
- **MCP Apps** (L0) - Server lifecycle management

### P1 Important
- **Graphiti** (L2) - Temporal knowledge graphs (replaces Zep)
- **Strands Agents** (L1) - AWS enterprise agent building
- **A2A Protocol** (L0) - Google inter-agent communication
- **RAGFlow** (L7) - Production RAG pipelines

### P2 Specialized
- **SimpleMem** (L2) - 30x context compression
- **RAGatouille** (L7) - ColBERT late interaction
- **Braintrust** (L5) - LLM evaluation platform
- **Portkey Gateway** (L0) - Unified LLM gateway

## Core Interfaces

### SDKAdapter (Base Interface)

```python
class SDKAdapter(ABC):
    @property
    def sdk_name(self) -> str: ...
    @property
    def layer(self) -> SDKLayer: ...
    @property
    def available(self) -> bool: ...

    async def initialize(self, config: Dict) -> AdapterResult: ...
    async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
    async def health_check(self) -> AdapterResult: ...
    async def shutdown(self) -> AdapterResult: ...
```

### AdapterResult (Response Type)

```python
@dataclass
class AdapterResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
```

## Memory Architecture

```
┌─────────────────────────────────────────────────┐
│              MEMORY TIERS (V36)                  │
├─────────────────────────────────────────────────┤
│ MAIN_CONTEXT   │ In LLM context (8K tokens)     │
│ CORE_MEMORY    │ Essential facts (Letta blocks) │
│ RECALL_MEMORY  │ Recent history (24h TTL)       │
│ ARCHIVAL_MEMORY│ Long-term (Qdrant + HNSW)      │
└─────────────────────────────────────────────────┘
```

### HNSW Configuration
- **M**: 48 (connections per node)
- **efConstruction**: 200 (build quality)
- **efSearch**: 100 (query accuracy)

## Configuration Files

| File | Purpose |
|------|---------|
| `platform/.mcp.json` | MCP server definitions |
| `platform/config/settings.yaml` | Unified V36 settings |
| `platform/config/memory_config.yaml` | Memory tier config |
| `platform/config/letta_config.yaml` | Letta connection |
| `.claude/settings.json` | Hooks and swarm config |
| `CLAUDE.md` | Claude Code instructions |

## Swarm Configuration

```yaml
topology: hierarchical-mesh
max_agents: 15
consensus: raft
memory_backend: hybrid
```

## Usage Examples

### Initialize Adapter

```python
from platform.adapters.simplemem_adapter import SimpleMemAdapter

adapter = SimpleMemAdapter()
await adapter.initialize({"max_tokens": 4096})

result = await adapter.execute("compress", context="Your text here...")
print(result.data["compression_ratio"])  # ~30x

await adapter.shutdown()
```

### Multi-Adapter Pipeline

```python
from platform.adapters.portkey_gateway_adapter import PortkeyGatewayAdapter
from platform.adapters.simplemem_adapter import SimpleMemAdapter
from platform.adapters.braintrust_adapter import BraintrustAdapter

# L0: Gateway
gateway = PortkeyGatewayAdapter()
await gateway.initialize({})

# L2: Memory
memory = SimpleMemAdapter()
await memory.initialize({})

# L5: Observability
tracker = BraintrustAdapter()
await tracker.initialize({"project": "my-project"})

# Pipeline: Gateway -> Memory -> Track
response = await gateway.execute("chat", messages=[...])
compressed = await memory.execute("compress", context=response.data["content"])
await tracker.execute("log", input="query", output=compressed.data)
```

## Testing

```bash
# All tests
cd platform && pytest

# Contract tests (adapter compliance)
pytest tests/contract/ -v

# Integration tests
pytest tests/integration/ -v

# E2E pipeline tests
pytest tests/e2e/ -v
```

## Dependencies

```bash
# Install V36 bundle
pip install -e ".[v36]"

# Individual layers
pip install -e ".[protocol,orchestration,memory]"
```

## Security

- CVE-2025-64439: Fixed (langgraph>=1.0.1)
- CVE-2025-68664: Fixed (langchain-core>=1.2.5)
- Circuit breakers on all adapters
- Health-aware routing with fallbacks

## Performance Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| Adapter init | <50ms | ~10ms |
| Health check | <10ms | ~1ms |
| Memory compress | <100ms | ~20ms |
| Vector search | <20ms | ~15ms |

---

*UNLEASH V36 - Unified SDK Integration Platform*
