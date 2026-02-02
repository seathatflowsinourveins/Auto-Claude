# Continuous Optimization Protocols

**Purpose**: Persistent self-improvement patterns for every session

---

## AUTO-INVOCATION TRIGGERS

### When to Use Each Agent (Proactive)

| Trigger | Agent | Action |
|---------|-------|--------|
| Complex feature request | **planner** | Create implementation plan BEFORE coding |
| Code just written/modified | **code-reviewer** | Immediate quality review |
| Bug fix or new feature | **tdd-guide** | Write tests FIRST |
| Auth/input/API code | **security-reviewer** | Security analysis |
| Build fails | **build-error-resolver** | Fix errors quickly |
| Critical user flow | **e2e-runner** | Playwright testing |
| Architectural decision | **architect** | ADR documentation |
| Code maintenance | **refactor-cleaner** | Dead code removal |

### Verification Gates

**ALWAYS run `/verify` after:**
- Completing a feature
- Before creating a PR
- After refactoring
- After fixing bugs

**ALWAYS run `/tdd` for:**
- New features (tests FIRST)
- Bug fixes (reproduction test FIRST)
- Refactoring (coverage BEFORE changes)

---

## SKILL INVOCATION PATTERNS

### Pattern: Feature Development
```
1. /plan → Create implementation plan, WAIT for confirmation
2. /tdd → Write tests first
3. [implement] → Write minimal code to pass tests
4. /code-review → Review changes
5. /verify → Full verification
6. /commit → Create commit
```

### Pattern: Bug Fix
```
1. tdd-guide → Write reproduction test
2. [fix] → Implement fix
3. /verify → Ensure no regressions
4. /code-review → Review fix
```

### Pattern: Security-Sensitive Changes
```
1. /plan → Identify security considerations
2. security-reviewer → Pre-implementation review
3. [implement] → With security patterns
4. security-reviewer → Post-implementation audit
5. /verify pre-pr → Full security scan
```

---

## CROSS-SESSION MEMORY PROTOCOL

### Start of Session
1. Read `enhancement_loop_research_jan2026` for patterns
2. Read `cross_session_bootstrap` for SDK access
3. Apply ImportanceScorer to prioritize context

### During Session
- Track decisions in working memory
- Note patterns that could be reused
- Identify learnings for extraction

### End of Session
- continuous-learning skill extracts patterns
- Significant learnings → save to memory
- Update enhancement_loop_research if major insights

---

## PROJECT-SPECIFIC OPTIMIZATIONS

### WITNESS (Creative/TouchDesigner)
- **Memory**: Mem0 for session context, Letta for long-term creative memory
- **Evolution**: EvoAgentX for automatic workflow generation
- **Optimization**: TextGrad for aesthetic refinement, DSPy for prompt tuning
- **Reasoning**: Graph-of-Thoughts for compositional exploration
- **Research**: Crawl4AI + Firecrawl for reference gathering
- **Evaluation**: Opik for aesthetic quality metrics

### TRADING (AlphaForge)
- **Orchestration**: LangGraph for durable trading workflows
- **Memory**: Letta for trading agent state persistence
- **Reasoning**: LLM-Reasoners MCTS for strategy search
- **Optimization**: DSPy MIPROv2 for strategy prompts
- **Data**: Crawl4AI for news/sentiment scraping
- **Safety**: CRITICAL - circuit breakers, risk limits, HITL gates

### UNLEASH (Meta-Project)
- **All SDKs**: Full access to entire platform ecosystem
- **Pair Programming**: Aider for self-improvement
- **Optimization**: DSPy + TextGrad for capability enhancement
- **Reasoning**: Full LLM-Reasoners suite (MCTS, ToT, GoT)
- **Memory**: Graphiti temporal graphs + Mem0
- **Orchestration**: MCP-Agent + LangGraph patterns

---

## ADVANCED SDK INTEGRATION PATTERNS

### Pattern: Gradient-Based Improvement
```python
# TextGrad for iterative refinement
import textgrad as tg
tg.set_backward_engine("claude-opus-4-5")
answer = tg.Variable(response, requires_grad=True)
loss = tg.TextLoss("Evaluate critically")(answer)
loss.backward()
optimizer.step()
```

### Pattern: Declarative Prompt Optimization
```python
# DSPy for systematic prompt tuning
import dspy
class MyTask(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

optimized = MIPROv2(metric=my_metric, auto="light").compile(
    dspy.ChainOfThought(MyTask),
    trainset=examples
)
```

### Pattern: Temporal Knowledge Persistence
```python
# Graphiti for bi-temporal knowledge
graphiti.add_episode(
    content="User prefers dark mode",
    source="conversation",
    occurred_at=datetime.now()
)
results = graphiti.search("user preferences", entity_types=["Preference"])
```

### Pattern: Advanced Reasoning Search
```python
# LLM-Reasoners for complex problems
from reasoners import WorldModel, Reasoner
from reasoners.algorithm import MCTS

reasoner = Reasoner(
    world_model=MyWorldModel(),
    search_config=MySearchConfig(),
    search_algorithm=MCTS()
)
solution = reasoner.run(problem)
```

---

## ENHANCED AGENT ORCHESTRATION

### MCP-Agent Workflow Patterns
| Pattern | Use Case |
|---------|----------|
| orchestrator-worker | Task delegation and aggregation |
| evaluator-optimizer | Iterative quality improvement |
| router | Intent classification and dispatch |
| swarm | Multi-agent collaboration |
| parallel | Concurrent independent tasks |

### LangGraph Checkpointing
```python
# Durable execution with state persistence
from langgraph.checkpoint.postgres import PostgresSaver

graph = StateGraph(State)
checkpointer = PostgresSaver(conn_string)
compiled = graph.compile(checkpointer=checkpointer)
# Survives crashes, resumes from exact state
```

---

## CAPABILITY ENHANCEMENT TRIGGERS

### Recognize and Apply

| Scenario | Enhancement |
|----------|-------------|
| Complex reasoning needed | Extended thinking (128K tokens) |
| Multi-step task | TodoWrite + agent orchestration |
| Multiple valid approaches | AskUserQuestion for preference |
| External data needed | MCP server (brave-search, exa) |
| Code quality concern | Immediate code-reviewer |
| Performance issue | Profiling + optimization patterns |

### Never Skip

- Security review for auth/input/API changes
- TDD for new features
- Verification before PR
- Memory persistence for significant decisions

---

## METRICS FOR SELF-IMPROVEMENT

### Track Per Session
- Patterns extracted (continuous-learning)
- Verification pass rate
- Agent invocations (proactive vs reactive)
- Memory retrievals (hit rate)

### Optimize For
- First-time success (reduce iterations)
- Pattern reuse (cross-domain pollination)
- Proactive agent use (before problems occur)
- Memory relevance (right context at right time)

---

*These protocols are ALWAYS ACTIVE across all sessions*
