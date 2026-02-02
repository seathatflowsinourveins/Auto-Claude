# SDK Ecosystem V30 - Cross-Session Reference

**Last Updated**: 2026-01-23
**Purpose**: Quick-access SDK patterns for all sessions

---

## Essential SDKs Quick Reference

### Core Agent Stack
| SDK | Install | Primary Use |
|-----|---------|-------------|
| DSPy | `pip install dspy-ai` | GEPA prompt optimization |
| LiteLLM | `pip install litellm` | 100+ LLM unified API, A2A bridge |
| Pydantic-AI | `pip install pydantic-ai` | Type-safe agents, graphs |
| LangGraph | `pip install langgraph` | Stateful workflows |
| Temporal | `pip install temporalio` | Durable execution |
| CrewAI | `pip install crewai` | Flows + Crews |

### RAG Stack
| SDK | Install | Primary Use |
|-----|---------|-------------|
| LlamaIndex | `pip install llama-index` | Full RAG framework, 300+ integrations |
| LightRAG | `pip install lightrag-hku` | Knowledge graph RAG |
| Crawl4AI | `pip install crawl4ai` | Web→Markdown |
| Chonkie | `pip install chonkie` | 9 chunker types |
| GraphRAG | `pip install graphrag` | Microsoft graph-based RAG |

### Safety Stack
| SDK | Install | Primary Use |
|-----|---------|-------------|
| NeMo Guardrails | `pip install nemoguardrails` | Colang DSL, 5 rail types |
| LLM-Guard | `pip install llm-guard` | Input/output scanners |
| Guardrails AI | `pip install guardrails-ai` | Hub validators |

### Evaluation Stack
| SDK | Install | Primary Use |
|-----|---------|-------------|
| Opik | `pip install opik` | 50+ metrics, 16 integrations |
| DeepEval | `pip install deepeval` | LLM evals, red teaming |
| RAGAS | `pip install ragas` | RAG evaluation |

---

## Key Integration Patterns

### A2A Protocol (Agent-to-Agent)
```python
from litellm.a2a_protocol import A2AClient
a2a = A2AClient()
response = await a2a.send_task(agent_id="...", task="...")
```

### Durable Execution (Temporal)
```python
from temporal import workflow
@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, data):
        return await workflow.execute_activity(process, data)
```

### Safety Pipeline (NeMo + LLM-Guard)
```python
from nemoguardrails import LLMRails
from llm_guard import scan_prompt, scan_output
# 3-layer: input scan → guardrails → output scan
```

### Unified Evaluation
```python
import opik
from deepeval.metrics import GEval
from ragas.metrics import DiscreteMetric
# Combine for comprehensive coverage
```

---

## Project-Specific Stacks

### UNLEASH
- Orchestration: Temporal + LangGraph
- Agents: Pydantic-AI + CrewAI
- Optimization: DSPy GEPA + pyribs

### WITNESS  
- Exploration: pyribs MAP-Elites + EvoTorch
- Real-time: LiveKit + Pipecat
- MCP: touchdesigner-creative

### TRADING
- Orchestration: Temporal (crash-proof)
- Safety: NeMo Guardrails (trading rules)
- Validation: Guardrails AI

---

## Documentation Locations
- Full patterns: `unleash/sdks/SDK_INTEGRATION_PATTERNS_V30.md`
- SDK index: `unleash/sdks/SDK_INDEX.md`
- Quick reference: `unleash/sdks/SDK_QUICK_REFERENCE.md`
- Deep dive: `unleash/DEEP_DIVE_SDK_REFERENCE.md`
