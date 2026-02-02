# V33++ Iteration 2 Research Synthesis

**Date**: 2026-01-23
**Ralph Loop**: Iteration 2/25
**Purpose**: Deep integration findings from Exa research + local SDK exploration

---

## Quick Reference: New Patterns Discovered

### 1. Advanced Hook Patterns
- **PermissionRequest hooks**: RBAC integration with `authorize.py`
- **Session lifecycle**: SessionStart, SessionEnd, SubagentStart, SubagentStop
- **PostToolUse context injection**: `additionalContext` field for observations

### 2. NeMo Guardrails + LangGraph
- **5 Rail Types**: Input, Retrieval, Dialog, Execution, Output
- **RunnableRails**: `prompt | (guardrails | llm)` pattern
- **Passthrough mode**: Essential for tool calling
- **Multi-agent safety**: Per-agent guardrail configurations

### 3. pyribs Quality-Diversity (17+ Algorithms)
- **QDAIF**: QD for AI Feedback - perfect for LLM prompt evolution
- **QDHF**: QD for Human Feedback
- **BOP-Elites**: Bayesian Optimization + MAP-Elites
- **Optuna integration**: CMA-MAE Sampler via OptunaHub

### 4. Temporal + PydanticAI Durable Execution
- **TemporalAgent**: Wraps agents for crash-proof workflows
- **Deterministic workflows**: Activities handle I/O
- **Model pre-registration**: Runtime model selection
- **Streaming**: `event_stream_handler` for partial results

### 5. Opik Unified Observability
- **LiteLLM**: `litellm.callbacks = ["opik"]` automatic tracing
- **RAGAS**: `OpikTracer()` callback for RAG evaluation
- **50+ metrics**: RAG, Agents, Bias, Heuristic, Hallucination

---

## Key Code Patterns

### NeMo + LangGraph
```python
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
guardrails = RunnableRails(config=config, passthrough=True)
runnable = prompt | (guardrails | llm)
```

### QDAIF for Prompts
```python
from ribs.archives import GridArchive
archive = GridArchive(dims=[50, 50], ranges=[(0, 1), (0, 1)])
# creativity x correctness behavioral space
```

### Temporal Durable Agent
```python
from pydantic_ai.durable_exec.temporal import TemporalAgent
temporal_agent = TemporalAgent(agent)
```

### Opik Tracing
```python
litellm.callbacks = ["opik"]
# OR
from ragas.integrations.opik import OpikTracer
```

---

## Files to Reference

| Topic | Path |
|-------|------|
| Full V33++ Doc | `unleash/UNIFIED_ARCHITECTURE_V33_PLUS.md` |
| pyribs Algorithms | `sdks/pyribs/docs/supported-algorithms.md` |
| NeMo + LangGraph | `sdks/nemo-guardrails/docs/integration/langchain/langgraph-integration.md` |
| Rail Types | `sdks/nemo-guardrails/docs/about/rail-types.md` |
| Temporal + Pydantic | `platform/docs/official/sdks/core-mcp/pydantic-ai/docs/durable_execution/temporal.md` |
| Opik + LiteLLM | `sdks/litellm/docs/my-website/docs/observability/opik_integration.md` |
| Opik + RAGAS | `sdks/ragas/docs/howtos/integrations/_opik.md` |

---

*Cross-session memory for V33++ architecture access*
