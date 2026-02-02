# SDK Deep Research - Platform SDKs (January 2026)

## Overview
Comprehensive research into 6 advanced LLM orchestration SDKs from `unleash/platform/sdks/`.

---

## 1. DSPy (Stanford NLP) - Declarative Prompt Optimization

### Core Concept
"Prompts as weights" - treat prompts like neural network parameters to optimize automatically.

### Key Patterns

```python
# Signature Pattern - Declarative I/O specification
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")

# Modules
dspy.ChainOfThought(signature)      # Step-by-step reasoning
dspy.ProgramOfThought(signature)    # Code-based problem solving
dspy.ReAct(signature, tools=[])     # Reasoning + Acting loop
dspy.Parallel(num_threads=N)        # Parallel execution
dspy.Refine(module, N=3, reward_fn) # Iterative refinement

# Optimizers
BootstrapFewShot(metric)            # Create few-shot examples
MIPROv2(metric, auto="light")       # Auto-tuned optimization
COPRO(metric, depth=3)              # Coordinate ascent prompts
SIMBA(metric)                       # Meta-optimization
```

### Integration Points
- Use for optimizing Claude prompts systematically
- Combine with evaluation metrics for continuous improvement
- Export optimized prompts for production use

---

## 2. Graphiti (Zep) - Temporal Knowledge Graphs

### Core Concept
Bi-temporal knowledge graphs that track both when events occurred AND when knowledge was acquired.

### Architecture
```
graphiti_core/
├── graphiti.py          # Main class
├── driver/              # Neo4j, FalkorDB
├── llm_client/          # OpenAI, Anthropic, Gemini
├── embedder/            # Embedding clients
├── search/              # Hybrid retrieval
```

### Key Patterns
```python
# Hybrid retrieval (semantic + BM25 + graph)
results = graphiti.search(
    query="user preferences",
    entity_types=["Preference", "Procedure"],
    limit=10
)

# MCP Server best practices
# 1. Always search before adding new memory
# 2. Use specific entity type filters
# 3. Store immediately with add_memory
```

### Integration Points
- Replace simple vector stores with temporal-aware graphs
- Track knowledge evolution over sessions
- Enable "time travel" queries for historical context

---

## 3. TextGrad (Stanford - Nature 2025) - Gradient-Based Text Optimization

### Core Concept
PyTorch-like API for differentiating through text - backprop via LLM feedback.

### Key Patterns
```python
import textgrad as tg

# Variables (like tensors)
answer = tg.Variable(text, role_description="concise answer", requires_grad=True)

# Textual Gradient Descent
optimizer = tg.TGD(parameters=[answer])
loss_fn = tg.TextLoss("Evaluate critically")

# Backprop loop
loss = loss_fn(answer)
loss.backward()
optimizer.step()

# Prompt optimization
system_prompt = tg.Variable("You are helpful.", requires_grad=True)
model = tg.BlackboxLLM(engine, system_prompt=system_prompt)
optimizer = tg.TGD(parameters=list(model.parameters()))
```

### Integration Points
- Optimize Claude system prompts via gradient feedback
- Fine-tune responses iteratively
- Combine with DSPy for comprehensive optimization

---

## 4. Tree-of-Thought (Princeton) - Deliberate Problem Solving

### Core Concept
BFS/DFS search over thought trees with explicit state evaluation.

### Key Patterns
```python
from tot.methods.bfs import solve

args = argparse.Namespace(
    method_generate='propose',  # propose (sequential) vs sample (independent)
    method_evaluate='value',    # value (independent) vs vote (together)
    method_select='greedy',
    n_generate_sample=1,
    n_evaluate_sample=3,
    n_select_sample=5           # breadth in ToT+BFS
)

ys, infos = solve(args, task, idx)
```

### Integration Points
- Use for complex multi-step reasoning tasks
- Implement custom evaluation functions for domain-specific scoring
- Combine with Claude's extended thinking for ultra-deep reasoning

---

## 5. Graph-of-Thoughts (ETH Zurich - AAAI 2024) - Flexible Thought Graphs

### Core Concept
Graph of Operations (GoO) - more flexible than linear or tree structures.

### Key Patterns
```python
from graph_of_thoughts import controller, operations

# Build operation graph
gop = operations.GraphOfOperations()
gop.append_operation(operations.Generate())
gop.append_operation(operations.Score(scoring_fn))
gop.append_operation(operations.GroundTruth(test_fn))

# Controller pattern
ctrl = controller.Controller(
    lm, gop, Prompter(), Parser(),
    {"original": input, "current": "", "method": "got"}
)
ctrl.run()
```

### Integration Points
- Model complex reasoning with non-linear dependencies
- Implement aggregation/refinement operations
- Visualize thought graphs for debugging

---

## 6. EvoAgentX - Self-Evolving Agent Ecosystem

### Core Concept
Automatic workflow generation + continuous self-improvement via evolution.

### Key Patterns
```python
from evoagentx.workflow import WorkFlowGenerator, WorkFlow
from evoagentx.agents import AgentManager

# Automatic workflow from goal
goal = "Generate Tetris game in HTML"
workflow_graph = WorkFlowGenerator(llm).generate_workflow(goal)

agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config)

workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager)
output = workflow.execute()

# Evolution algorithms
# - TextGrad: Gradient-based prompt optimization
# - MIPRO: Multi-instruction prompt optimization
# - AFlow: Automatic workflow evolution
# - EvoPrompt: Evolutionary prompt search

# Human-in-the-Loop
from evoagentx.hitl import HITLManager, HITLInterceptorAgent
hitl_manager = HITLManager()
hitl_manager.activate()
```

### Integration Points
- Enable autonomous workflow creation from high-level goals
- Implement continuous improvement loops
- Use HITL for safety-critical decision points

---

## Cross-SDK Integration Strategy

### Layer 1: Foundation
- **Graphiti** for persistent knowledge (temporal graphs)
- **DSPy** for prompt optimization (signatures + optimizers)

### Layer 2: Reasoning
- **ToT/GoT** for complex problem solving
- **TextGrad** for iterative refinement

### Layer 3: Orchestration
- **EvoAgentX** for multi-agent coordination
- **HITL** for safety gates

### Layer 4: Optimization Loop
```
Input → DSPy Signature → ToT/GoT Reasoning → TextGrad Refinement
                              ↓
                        EvoAgentX Evolution
                              ↓
                    Graphiti Memory Storage → Next Input
```

---

## Application to Core Projects

### AlphaForge (Trading)
- DSPy for strategy optimization
- GoT for complex trade analysis
- Graphiti for market event memory
- HITL for high-risk decisions

### State of Witness (Creative)
- TextGrad for aesthetic refinement
- EvoAgentX for workflow evolution
- ToT for compositional exploration

### Unleash (Meta)
- All SDKs for capability enhancement
- Continuous self-improvement loops
- Cross-project knowledge sharing

---

Last Updated: 2026-01-25
