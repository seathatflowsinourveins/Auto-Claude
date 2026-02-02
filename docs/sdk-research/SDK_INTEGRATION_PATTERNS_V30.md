# SDK Integration Patterns V30 - Unified Architecture

**Created**: 2026-01-23
**Purpose**: Unified SDK integration patterns for all three core projects

---

## 1. EXECUTIVE SUMMARY: Key Findings from SDK Review

### 1.1 Critical Evolution Patterns (2025-2026)

| Pattern | Old Way | New Way | SDKs Adopting |
|---------|---------|---------|---------------|
| **Agent-to-Agent (A2A)** | Direct LLM calls | Protocol-based agent communication | LiteLLM, Pydantic-AI |
| **MCP Integration** | Custom tool definitions | Standardized MCP servers | FastMCP v2, LiteLLM |
| **Durable Execution** | Fire-and-forget | Checkpointed workflows | Temporal, Pydantic-AI, LangGraph |
| **Enterprise Auth** | API keys only | OAuth, OIDC, WorkOS, Auth0 | FastMCP v2 |
| **Server Composition** | Single servers | Multi-server mounting | FastMCP v2 |
| **Quality-Diversity** | Grid search | MAP-Elites archives | pyribs, QDax |

### 1.2 SDK Maturity Assessment

| SDK | Version | Production Ready | Key Strength |
|-----|---------|------------------|--------------|
| **DSPy** | 2.6+ | ✅ Yes | GEPA optimizer, declarative prompts |
| **LiteLLM** | 1.60+ | ✅ Yes | 100+ LLM unified API, A2A bridge |
| **Pydantic-AI** | 0.1+ | ✅ Yes | Type-safe agents, graph support |
| **LangGraph** | Latest | ✅ Yes | Stateful workflows, Pregel-based |
| **Temporal** | 1.0+ | ✅ Yes | Durable orchestration, crash-proof |
| **FastMCP** | v2 | ✅ Yes | Enterprise auth, server composition |
| **CrewAI** | Latest | ✅ Yes | Flows + Crews, 100k+ developers |
| **NeMo Guardrails** | 0.19.0 | ✅ Yes | Colang DSL, 5 rail types |
| **LLM-Guard** | Latest | ✅ Yes | 15+ input scanners, 20+ output scanners |
| **DeepEval** | Latest | ✅ Yes | LLM evals, red teaming, CI/CD |
| **RAGAS** | Latest | ✅ Yes | RAG evaluation, test generation |
| **Opik** | Latest | ✅ Yes | 50+ metrics, 16 integrations |

---

## 2. IMPROVEMENT OPPORTUNITIES IDENTIFIED

### 2.1 Gap Analysis

| Gap | Impact | Solution |
|-----|--------|----------|
| **No unified A2A protocol** | Agents can't interoperate | Standardize on LiteLLM A2A bridge |
| **Scattered MCP configs** | Tool duplication | Create central MCP registry |
| **Missing durable execution** | Workflow failures lost | Adopt Temporal for critical paths |
| **No safety layer** | Security vulnerabilities | Integrate NeMo + LLM-Guard pipeline |
| **Evaluation fragmentation** | No unified metrics | Combine Opik + DeepEval + RAGAS |
| **Memory inconsistency** | Cross-session amnesia | Unified Mem0 + Letta integration |

### 2.2 Recommended Improvements

#### Improvement 1: Unified Agent Communication Layer
```python
# Current: Direct LLM calls
response = llm.generate(prompt)

# Improved: A2A Protocol with routing
from litellm.a2a_protocol import A2AClient

a2a = A2AClient()
response = await a2a.send_task(
    agent_id="trading-risk-analyst",
    task="Evaluate portfolio risk for AAPL position",
    context={"position_size": 1000, "current_price": 185.50}
)
```

#### Improvement 2: Durable Workflow Pattern
```python
# Current: Fire-and-forget execution
def process_trade(order):
    validate(order)
    execute(order)
    confirm(order)  # If this fails, previous steps lost

# Improved: Temporal durable execution
@workflow.defn
class TradeWorkflow:
    @workflow.run
    async def run(self, order: Order) -> TradeResult:
        await workflow.execute_activity(
            validate_order, order,
            start_to_close_timeout=timedelta(seconds=30)
        )
        result = await workflow.execute_activity(
            execute_trade, order,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        return result
```

#### Improvement 3: Unified Safety Pipeline
```python
# Combine NeMo Guardrails + LLM-Guard
from nemoguardrails import LLMRails, RailsConfig
from llm_guard import scan_prompt, scan_output
from llm_guard.input_scanners import PromptInjection, Toxicity
from llm_guard.output_scanners import FactualConsistency

class UnifiedSafetyPipeline:
    def __init__(self):
        self.rails = LLMRails(RailsConfig.from_path("./guardrails"))
        self.input_scanners = [PromptInjection(), Toxicity()]
        self.output_scanners = [FactualConsistency()]

    async def safe_generate(self, prompt: str) -> str:
        # Layer 1: LLM-Guard input scan
        sanitized, valid, risk = scan_prompt(self.input_scanners, prompt)
        if not valid:
            raise SecurityError(f"Input rejected: {risk}")

        # Layer 2: NeMo Guardrails
        response = await self.rails.generate_async(
            messages=[{"role": "user", "content": sanitized}]
        )

        # Layer 3: LLM-Guard output scan
        _, valid, risk = scan_output(self.output_scanners, prompt, response["content"])
        if not valid:
            raise SecurityError(f"Output rejected: {risk}")

        return response["content"]
```

#### Improvement 4: Unified Evaluation Framework
```python
# Combine Opik + DeepEval + RAGAS
import opik
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from ragas.metrics import DiscreteMetric

class UnifiedEvaluation:
    """Combined evaluation using best-of-breed metrics."""

    @opik.track(name="unified_eval", tags=["evaluation"])
    async def evaluate(self, input: str, output: str, context: list[str]) -> dict:
        results = {}

        # DeepEval G-Eval for custom criteria
        correctness = GEval(
            name="Correctness",
            criteria="Determine if output is factually correct",
            threshold=0.7
        )
        test_case = LLMTestCase(input=input, actual_output=output, context=context)
        correctness.measure(test_case)
        results["correctness"] = correctness.score

        # RAGAS for structured evaluation
        accuracy_metric = DiscreteMetric(
            name="accuracy",
            allowed_values=["accurate", "inaccurate"],
            prompt="Evaluate if response is accurate. Response: {response}"
        )
        ragas_score = await accuracy_metric.ascore(response=output)
        results["ragas_accuracy"] = ragas_score.value

        return results
```

---

## 3. PROJECT-SPECIFIC INTEGRATION STACKS

### 3.1 UNLEASH (Meta-Development) Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    UNLEASH ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│  ORCHESTRATION    │  Temporal + LangGraph                   │
│  AGENTS           │  Pydantic-AI + CrewAI                   │
│  OPTIMIZATION     │  DSPy GEPA + pyribs MAP-Elites          │
│  ROUTING          │  LiteLLM (A2A + MCP bridge)             │
│  MEMORY           │  Mem0 + Letta + Graphiti                │
│  SAFETY           │  NeMo Guardrails + LLM-Guard            │
│  EVALUATION       │  Opik + DeepEval + RAGAS                │
│  MCP              │  FastMCP v2 (server composition)        │
└─────────────────────────────────────────────────────────────┘
```

**Key Integrations:**
```python
# Unleash Ralph Loop with full SDK integration
from dspy import GEPA
from pydantic_ai import Agent
from litellm import completion
from temporal.workflow import defn as workflow

@workflow
class RalphLoopWorkflow:
    """Self-improving autonomous loop with durable execution."""

    async def run(self, objective: str) -> dict:
        # Phase 1: Research with DSPy optimization
        optimized_prompt = await GEPA.optimize(objective)

        # Phase 2: Execute with Pydantic-AI agent
        agent = Agent("claude-opus-4-5-20251101", deps_type=RalphDeps)
        result = await agent.run(optimized_prompt)

        # Phase 3: Evaluate with unified framework
        evaluation = await unified_eval(result)

        # Phase 4: Store learnings with durable checkpoint
        await workflow.execute_activity(store_learnings, result, evaluation)

        return {"result": result, "evaluation": evaluation}
```

### 3.2 WITNESS (Creative/TouchDesigner) Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    WITNESS ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│  ORCHESTRATION    │  LangGraph (streaming)                  │
│  AGENTS           │  Pydantic-AI (type-safe)                │
│  EXPLORATION      │  pyribs MAP-Elites + EvoTorch           │
│  VISION           │  Vision Agents + MediaPipe              │
│  REAL-TIME        │  LiveKit Agents + Pipecat               │
│  ROUTING          │  LiteLLM (MCP for TouchDesigner)        │
│  MEMORY           │  Letta (creative patterns)              │
│  MCP              │  touchdesigner-creative MCP server      │
└─────────────────────────────────────────────────────────────┘
```

**Key Integrations:**
```python
# Witness creative exploration with pyribs
from pyribs import GridArchive, CVT_MAP_Elites
from pydantic_ai import Agent

class WitnessExplorer:
    """MAP-Elites exploration for shader parameters."""

    def __init__(self):
        # 2D behavioral space: energy, coherence
        self.archive = GridArchive(
            dims=[50, 50],
            ranges=[(0, 1), (0, 1)]
        )
        self.agent = Agent("claude-opus-4-5-20251101")

    async def explore(self, archetype: str) -> dict:
        # Generate shader parameters via Claude
        params = await self.agent.run(
            f"Generate GLSL parameters for {archetype} archetype"
        )

        # Evaluate fitness and behavioral descriptors
        fitness, (energy, coherence) = await self.evaluate_aesthetic(params)

        # Add to archive (quality-diversity)
        self.archive.add_single([energy, coherence], fitness, params)

        return {"params": params, "fitness": fitness}
```

### 3.3 TRADING (AlphaForge) Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│  ORCHESTRATION    │  Temporal (crash-proof workflows)       │
│  AGENTS           │  Pydantic-AI + Guardrails AI            │
│  REASONING        │  LightZero MCTS + DSPy                  │
│  ROUTING          │  LiteLLM (cost optimization)            │
│  SAFETY           │  NeMo Guardrails (trading rules)        │
│  VALIDATION       │  Guardrails AI (order validation)       │
│  EVALUATION       │  DeepEval (decision quality)            │
│  MEMORY           │  Mem0 (market patterns)                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Integrations:**
```python
# Trading workflow with full safety
from temporal import workflow, activity
from nemoguardrails import LLMRails
from guardrails import Guard
from guardrails.hub import ValidRange

@workflow.defn
class TradingWorkflow:
    """Durable trading execution with safety rails."""

    @workflow.run
    async def execute_trade(self, order: TradeOrder) -> TradeResult:
        # Layer 1: NeMo Guardrails for trading rules
        rails = LLMRails(RailsConfig.from_path("./trading_rails"))
        validated = await rails.generate_async(
            messages=[{"role": "system", "content": f"Validate order: {order}"}]
        )

        # Layer 2: Guardrails AI for structured validation
        guard = Guard().use(
            ValidRange(min=0, max=order.max_position_size, on_fail="exception")
        )
        guard.validate(order.quantity)

        # Layer 3: Temporal durable execution
        result = await workflow.execute_activity(
            execute_order_activity,
            order,
            retry_policy=RetryPolicy(maximum_attempts=3),
            start_to_close_timeout=timedelta(seconds=30)
        )

        return result
```

---

## 4. UNIFIED CONFIGURATION TEMPLATES

### 4.1 Central MCP Registry
```yaml
# ~/.claude/mcp-registry.yml
version: "2.0"

servers:
  # Core servers (all projects)
  litellm:
    command: "litellm --mcp"
    env:
      LITELLM_API_KEY: "${LITELLM_API_KEY}"

  memory:
    command: "mem0-mcp"
    env:
      MEM0_API_KEY: "${MEM0_API_KEY}"

  # Project-specific servers
  touchdesigner-creative:
    project: witness
    command: "python -m touchdesigner_mcp"
    env:
      TD_PORT: "7000"

  trading-data:
    project: trading
    command: "python -m trading_mcp"
    env:
      ALPACA_API_KEY: "${ALPACA_API_KEY}"
```

### 4.2 Unified Safety Configuration
```yaml
# ~/.claude/safety-config.yml
version: "1.0"

layers:
  - name: llm-guard-input
    scanners:
      - PromptInjection
      - Toxicity
      - Secrets
      - InvisibleText
    action: reject

  - name: nemo-guardrails
    config_path: "./guardrails/config.yml"
    rails:
      input:
        - check jailbreak
        - mask sensitive data
      output:
        - self check facts
        - self check hallucination

  - name: llm-guard-output
    scanners:
      - FactualConsistency
      - Bias
      - MaliciousURLs
    action: warn
```

### 4.3 Evaluation Pipeline Configuration
```yaml
# ~/.claude/eval-config.yml
version: "1.0"

tracing:
  provider: opik
  workspace: "unleash"
  tags: ["production"]

metrics:
  rag:
    - AnswerRelevancy
    - ContextPrecision
    - Faithfulness

  agents:
    - TaskCompletion
    - ToolCorrectness
    - TrajectoryAccuracy

  safety:
    - Hallucination
    - Toxicity
    - Bias

benchmarks:
  - MMLU
  - TruthfulQA
  - GSM8K
```

---

## 5. MIGRATION CHECKLIST

### 5.1 From Current to V30 Architecture

- [ ] **Install Core SDKs**
  ```bash
  pip install dspy-ai litellm temporalio pydantic-ai langgraph
  pip install nemoguardrails llm-guard guardrails-ai
  pip install deepeval ragas opik
  pip install mem0ai pyribs evotorch
  pip install "fastmcp[auth]"
  ```

- [ ] **Configure A2A Protocol**
  - Enable LiteLLM A2A bridge
  - Define agent discovery endpoints
  - Set up inter-agent authentication

- [ ] **Enable Durable Execution**
  - Deploy Temporal server
  - Convert critical workflows to `@workflow.defn`
  - Add retry policies and timeouts

- [ ] **Implement Safety Pipeline**
  - Configure NeMo Guardrails Colang rules
  - Set up LLM-Guard scanners
  - Add Guardrails AI validators

- [ ] **Unify Evaluation**
  - Configure Opik tracing
  - Define DeepEval test suites
  - Set up RAGAS for RAG evaluation

- [ ] **Centralize MCP**
  - Create mcp-registry.yml
  - Migrate server configs
  - Enable FastMCP v2 composition

---

## 6. QUICK REFERENCE: SDK SELECTION GUIDE

### When to Use What

| Task | Primary SDK | Alternative |
|------|-------------|-------------|
| **Prompt optimization** | DSPy (GEPA) | TextGrad |
| **Agent orchestration** | Pydantic-AI | CrewAI |
| **Multi-agent systems** | CrewAI | AutoGen |
| **Durable workflows** | Temporal | LangGraph |
| **LLM routing** | LiteLLM | - |
| **RAG pipeline** | LlamaIndex | LightRAG |
| **Web scraping** | Crawl4AI | Firecrawl |
| **Chunking** | Chonkie | LLMLingua |
| **Safety rails** | NeMo Guardrails | Guardrails AI |
| **Input/output scanning** | LLM-Guard | Purple Llama |
| **LLM evaluation** | DeepEval | - |
| **RAG evaluation** | RAGAS | - |
| **Observability** | Opik | Langfuse |
| **MCP servers** | FastMCP v2 | mcp-python-sdk |
| **Quality-diversity** | pyribs | QDax |
| **Evolutionary** | EvoTorch | - |

---

*Document Version: 30.0 | Created: 2026-01-23*
