# Enhancement Loop Research Synthesis (January 2026)

**Purpose**: Cross-session persistent knowledge base for continuous capability optimization

---

## 1. EVERYTHING-CLAUDE-CODE PATTERNS (Local Repo Deep Dive)

### Agents (9 Total)
| Agent | Description | Model | Key Tools |
|-------|-------------|-------|-----------|
| **planner** | Implementation planning, risk assessment | Opus | Read, Grep, Glob |
| **architect** | System design, ADRs, scalability | Opus | Read, Grep, Glob |
| **code-reviewer** | Quality, security, maintainability | Opus | Read, Grep, Glob, Bash |
| **security-reviewer** | OWASP Top 10, secrets, injection | Opus | Read, Write, Edit, Bash, Grep, Glob |
| **tdd-guide** | Test-first development, 80%+ coverage | Opus | Read, Write, Edit, Bash, Grep |
| **build-error-resolver** | Fix build/type errors quickly | - | Read, Write, Edit, Bash, Grep, Glob |
| **e2e-runner** | Playwright E2E testing | - | Read, Write, Edit, Bash, Grep, Glob |
| **refactor-cleaner** | Dead code cleanup | - | Read, Write, Edit, Bash, Grep, Glob |
| **doc-updater** | Documentation maintenance | - | Read, Write, Edit, Bash, Grep, Glob |

### Skills (11 Total)
| Skill | Purpose |
|-------|---------|
| **continuous-learning** | Stop hook pattern extraction, saves to ~/.claude/skills/learned/ |
| **verification-loop** | 6-phase verification: Build→Types→Lint→Tests→Security→Diff |
| **eval-harness** | Eval-Driven Development, pass@k/pass^k metrics |
| **backend-patterns** | Repository/Service/Middleware patterns, caching, rate limiting |
| **tdd-workflow** | Red-Green-Refactor cycle, 80%+ coverage |
| **frontend-patterns** | React/Next.js patterns |
| **coding-standards** | Universal code quality standards |
| **cross-session-memory** | Memory persistence patterns |
| **security-review** | Security checklist patterns |
| **strategic-compact** | Manual context compaction |
| **data-engineering** | ETL and pipeline patterns |

### Commands (14 Total)
- `/plan` - Create implementation plan, WAIT for user confirmation
- `/tdd` - Test-driven development workflow
- `/verify` - 6-phase verification (quick|full|pre-commit|pre-pr)
- `/eval` - Run evaluation harness
- `/code-review` - Review code changes
- `/build-fix` - Fix build errors
- `/checkpoint` - Save/restore state
- `/orchestrate` - Multi-agent orchestration
- `/e2e` - End-to-end testing
- `/refactor-clean` - Dead code removal
- `/update-codemaps` - Update code documentation
- `/update-docs` - Documentation updates
- `/test-coverage` - Coverage analysis
- `/security-review` - Security analysis

### Rules (8 Total)
- **security.md** - No hardcoded secrets, input validation, OWASP compliance
- **coding-style.md** - Immutability, small files, error handling
- **git-workflow.md** - Conventional commits, PR workflow
- **testing.md** - 80% coverage, unit/integration/E2E
- **patterns.md** - API response format, hooks, repository pattern
- **performance.md** - Model selection, context management
- **hooks.md** - PreToolUse, PostToolUse, Stop hooks
- **agents.md** - Agent orchestration patterns

---

## 2. CLAUDE CODE CLI (v2.1.15 - January 2026)

### Key Capabilities
- **3,000+ MCP services** via MCP.so index
- **Specialized subagents** for task delegation
- **Hooks system**: SessionStart → UserPromptSubmit → PreToolUse → PostToolUse → Stop → SessionEnd
- **Skills system**: YAML frontmatter, slash command invocation
- **Plugins** (beta): Package commands/agents/MCP/hooks
- **Model switching**: Haiku (fast) ↔ Opus (powerful)

### v2.1.x Features
- Skill hot-reload
- Unified skills/commands
- Agent resilience
- MCP tool search auto mode
- npm deprecation (native binary preferred)

### Extended Thinking
```python
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    thinking={"type": "enabled", "budget_tokens": 128000},
    betas=["interleaved-thinking-2025-05-14"]  # For streaming
)
```

---

## 3. MCP PROTOCOL (2025-11-25 Specification)

### Core Architecture
- **JSON-RPC 2.0** based protocol
- **Server features**: Tools, Resources, Prompts
- **Capability negotiation** on connect
- **Lazy loading**: MCP Tool Search reduces context usage

### Essential Servers
```json
{
  "mcpServers": {
    "brave-search": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"]},
    "memory": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem"]}
  }
}
```

---

## 4. SDK ECOSYSTEM (34 Production SDKs - 8 Layers)

### Platform SDKs (unleash/platform/sdks)
| SDK | Category | Purpose |
|-----|----------|---------|
| **langgraph** | Orchestration | StateGraph, checkpointing, HITL |
| **letta** | Memory | Memory blocks, archival, sleep-time |
| **dspy** | Reasoning | Signatures, MIPROv2, GEPA optimizers |
| **serena** | Reasoning | Symbolic editing, project context |
| **aider** | Processing | Code editing, git integration |
| **crawl4ai** | Processing | Async web crawling |
| **firecrawl** | Processing | Web scraping |
| **mem0** | Memory | Cross-session memory |
| **graphiti** | Knowledge | Temporal knowledge graphs |
| **mcp-agent** | Protocol | MCP client patterns |
| **EvoAgentX** | Orchestration | Evolutionary agents |
| **graph-of-thoughts** | Reasoning | GoT prompting |
| **tree-of-thought-llm** | Reasoning | ToT prompting |
| **llm-reasoners** | Reasoning | Reasoning frameworks |
| **textgrad** | Processing | Gradient-based text optimization |

### Core Layers (unleash/core)
- cli, compat, knowledge, memory, observability
- orchestration, performance, processing, providers
- reasoning, safety, structured, tools

---

## 5. ADVANCED SDK RESEARCH (Exa Deep Dives)

### Opik - AI Observability
```python
import opik
from opik.integrations.anthropic import track_anthropic

# Auto-trace Claude calls
client = track_anthropic(anthropic.Anthropic())

# 50+ evaluation metrics
from opik.evaluation.metrics import (
    Hallucination, AnswerRelevance, ContextPrecision,  # RAG
    AgentTaskCompletionJudge, TrajectoryAccuracy,       # Agents
    GenderBiasJudge, PoliticalBiasJudge,                # Bias
    ROUGE, BERTScore, Readability                       # Heuristic
)
```

### PyRibs - Quality Diversity
```python
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

# QDAIF for LLM exploration
for iteration in range(1000):
    solutions = scheduler.ask()
    objectives = llm_evaluate_quality(solutions)
    measures = llm_extract_diversity_features(solutions)
    scheduler.tell(objectives, measures)
```

### Guardrails AI (v0.7.2)
```python
from guardrails import Guard
from guardrails.hub import DetectPII, ToxicLanguage

guard = Guard().use_many(
    DetectPII(on_fail="exception"),
    ToxicLanguage(on_fail="reask")
)
validated = guard(llm_output)
```

### LangGraph 3.0+
```python
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

pool = ConnectionPool(conn_string, max_size=10)
checkpointer = PostgresSaver(pool)
# SECURITY: Use langgraph>=3.0 (CVE-2025-64439 fixed)
```

### Letta Memory System
- **Memory blocks**: XML-structured, always in context, 2000 char default
- **Archival memory**: Semantic search via vector DB, unlimited
- **Tools**: archival_memory_insert, archival_memory_search
- **Sleep-time**: Background memory consolidation

### DSPy 3.0
- **MIPROv2**: Bayesian optimization for instructions + few-shot
- **GEPA**: Genetic-Pareto optimizer (outperforms MIPROv2 by +14%)
- **Signatures**: Declarative I/O definitions

---

## 6. CONTINUOUS OPTIMIZATION PROTOCOLS

### Session Start Protocol
1. Load cross-session memory (this file + bootstrap)
2. Check project context (UNLEASH|WITNESS|TRADING)
3. Initialize relevant agents based on task type
4. Apply memory gateway ImportanceScorer

### During Session
1. Track patterns via continuous-learning skill
2. Run verification-loop after significant changes
3. Use appropriate agents proactively (security after auth code, tdd for features)
4. Apply eval-harness for capability testing

### Session End Protocol
1. Stop hook extracts learnings
2. Pattern detection: error_resolution, user_corrections, workarounds
3. Save to ~/.claude/skills/learned/ if significant
4. Update memory with session insights

### Cross-Domain Pollination
- WITNESS patterns → apply visual evaluation to TRADING charts
- TRADING risk patterns → apply circuit breakers to WITNESS resource usage
- UNLEASH SDK patterns → shared across all projects

---

## 7. KEY REFERENCE DOCUMENTS

| Document | Location | Purpose |
|----------|----------|---------|
| cross_session_bootstrap | Serena memory | 8-layer SDK quick reference |
| architecture_2026_definitive | Serena memory | Full architecture docs |
| DEEP_DIVE_SDK_REFERENCE.md | unleash/ | SDK usage patterns |
| everything-claude-code-full | unleash/ | Production agents/skills/commands |

---

---

## 8. CLAUDE AGENT SDK PATTERNS (January 2026)

### Core Architecture
- **Orchestrator + Subagents**: Give each subagent ONE job, orchestrator coordinates
- **Agent Loop**: Context → Thought → Action → Observation → Loop
- **Hooks & Deterministic Overrides**: Version and validate as production code

### Anthropic's 6 Composable Patterns
1. **Prompt Chaining** - Sequential steps with validation gates
2. **Routing** - Classify input, dispatch to specialized handlers
3. **Parallelization** - Fan-out/fan-in for independent tasks
4. **Orchestrator-Workers** - Central planner delegates to specialized agents
5. **Evaluator-Optimizer** - Generate → Evaluate → Improve loop
6. **Autonomous Agent** - Full agency with tool use and memory

### Subagent Best Practices
```python
# Orchestrator: global planning, delegation, state management
# Subagents: single goal, clear I/O, narrow permissions

# Pipeline pattern (deterministic):
analyst → architect → implementer → tester → security_audit

# Parallel pattern (independent):
[ui_agent, api_agent, db_agent] → merger
```

---

## 9. LANGGRAPH PRODUCTION PATTERNS (2026)

### State Persistence
```python
class State(TypedDict):
    messages: list
    user_id: str
    task_status: str

# Checkpoint at every node transition
# Resume after crashes, continue multi-turn conversations
```

### Human-in-the-Loop (HITL)
```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent

# Interrupt patterns for human approval
graph.add_edge("proposal", "human_review")  # Pause for approval
graph.add_edge("human_review", "execute")   # Continue after approval
```

### Production Checklist
- [ ] PostgresSaver with ConnectionPool (not SQLite)
- [ ] langgraph>=3.0 (CVE-2025-64439 fixed)
- [ ] State snapshots for debugging/replay
- [ ] Interrupt patterns for critical decisions
- [ ] Timeout handling for long approvals

---

## 10. AGENTIC WORKFLOW PRINCIPLES

### Key Insight (Anthropic Research)
> "Agents are just workflows with feedback loops. The sophistication comes from how you architect those loops."

### When to Add Complexity
| Complexity Level | Use When |
|-----------------|----------|
| Simple Tool Use | Single-step tasks |
| Prompt Chaining | Multi-step, deterministic |
| Routing | Input classification needed |
| Parallel Agents | Independent subtasks |
| Orchestrator | Complex coordination |
| Full Autonomy | Open-ended exploration |

### Production Safety
- Narrow tool permissions for orchestrators ("read and route")
- Explicit I/O contracts for subagents
- Deterministic pipelines where possible
- Interrupt patterns for critical decisions
- Audit logging for all actions

---

*Last Updated: 2026-01-25 | Enhancement Loop Research Synthesis v2*
