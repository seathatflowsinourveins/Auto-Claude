# UNLEASHED: Ultimate Claude Code Ecosystem Guide

> **Version FINAL** - Ultimate Unified Architecture
> Two projects, one ecosystem, seamless integration.
> **AlphaForge Trading** (Claude designs) + **State of Witness** (Claude drives)
> 70 MCP servers | 67 skills | 17 plugins | 210K+ lines of code

---

## Executive Summary

This document consolidates the complete Claude Code ecosystem for two major projects:

| Project | Purpose | Claude's Role |
|---------|---------|---------------|
| **AlphaForge** | Autonomous AI trading (138K lines) | Designs & builds the system |
| **State of Witness** | ML-powered computational art (2M particles) | Drives real-time output |

**Key Numbers:**
- **210,000+ lines** of production code
- **70 MCP servers** configured
- **67 custom skills** available
- **18 slash commands**
- **17 plugins enabled**
- **128K thinking tokens** (extended thinking)
- **64K output tokens** (max output)
- **Subscription mode** (unlimited budget)

**FINAL Version Highlights:**
- **Complete Unified Architecture** - Both projects fully documented in ULTIMATE_UNIFIED_FINAL.md
- **Seamless Integration** - Shared memory, QD patterns, observability across both projects
- **Session Modes** - `/session-init trading`, `/session-init creative`, `/session-init both`
- **Verified Working** - Memory systems tested, configs validated, integration points confirmed

**Core Technologies:**
- **Trading**: LangGraph orchestration, ML ensemble, QuestDB time-series, PostgreSQL checkpoints
- **Creative**: Sapiens 2B pose, DINOv3+SigLIP2 embeddings, HDBSCAN clustering, GLSL 430 particles
- **Shared**: pyribs MAP-Elites, episodic-memory, claude-mem, Langfuse observability

**v4.0 Additions:**
- **Claude Task Master** (24.3k stars) - PRD parsing, task generation
- **SecOpsAgentKit** - 25+ security skills (Trivy, SonarQube, CodeQL)
- **LikeC4 MCP** - Natural language architecture queries
- **Knowledge Graph Memory** - Neo4j + Graphiti cross-session memory
- **QuantConnect + IBKR** - Professional backtesting & execution
- **6-Tier Extended Thinking** - From 4K (think) to 128K (full budget)
- **C4 Architecture Diagrams** - Automated diagram generation

**Previous Version Additions:**
- Sequential Thinking MCP for structured reasoning
- obra/superpowers integration (12.3k+ GitHub stars)
- LangGraph 1.0.6 multi-agent orchestration
- TimesFM 2.5 foundation model for forecasting
- pyribs MAP-Elites for quality-diversity optimization
- FinRL CVaR-PPO risk-aware RL trading
- Claude Agent SDK production patterns
- Advanced Tool Use (Tool Search, Programmatic Calling)

---

## Claude's Dual Roles

### AlphaForge: Development Lifecycle Orchestrator

Claude operates the **full development lifecycle**:
```
Architecture → Building → Testing → Audit → Deployment → Monitoring
```

**What Claude Does:**
- Designs 12-layer architecture
- Writes production Python/Rust code
- Creates comprehensive test suites
- Performs security audits
- Generates K8s deployment configs
- Sets up Grafana monitoring

**What Runs Autonomously (No LLM):**
- Rust kill switch (100ns response)
- Python event loops
- Circuit breakers
- Order execution

### State of Witness: Generative Creative Brain

Claude **IS** the creative system:
```
Claude Code → MCP Commands → TouchDesigner → 60fps Visual Output
```

**What Claude Does:**
- Generates shader parameters in real-time
- Explores aesthetic space with MAP-Elites
- Controls 2M particle behaviors
- Creates node networks live
- Evaluates and iterates compositions

**Why This Works:**
- Creative tolerates 100ms latency
- No financial risk
- Exploration is the goal

---

## MCP Server Ecosystem

### By Category (42+ Total)

#### Memory & Persistence
| Server | Purpose | Command |
|--------|---------|---------|
| memory | Knowledge graph | `npx -y @modelcontextprotocol/server-memory` |
| memento | Temporal memory with semantic search | `npx -y @gannonh/memento-mcp` |
| qdrant | Vector embeddings | `uvx mcp-server-qdrant` |
| lancedb | Hybrid vector search | `uvx lancedb-mcp` |
| redis | Caching, pub/sub | `uvx mcp-server-redis` |

#### Financial & Trading
| Server | Purpose | Critical |
|--------|---------|----------|
| alpaca | Order execution | YES |
| questdb | Time-series tick data | YES |
| trading-tools | Custom trading ops | YES |
| alphavantage | Market data | NO |
| polygon | Alternative data | NO |
| twelvedata | Fundamentals | NO |
| financial-datasets | Income statements, etc. | NO |

#### Creative & Visualization
| Server | Purpose | Critical |
|--------|---------|----------|
| touchdesigner-creative | TD node control | YES |
| qdrant-witness | Pose embeddings | YES |
| comfyui-creative | Image generation | NO |
| blender-creative | 3D assets | NO |
| everart | AI art | NO |
| mermaid | Diagram generation | NO |

#### Observability
| Server | Purpose |
|--------|---------|
| grafana | Dashboards |
| prometheus | Metrics |
| loki | Logs |
| opentelemetry | Distributed tracing |
| jaeger | Trace visualization |
| datadog | APM |

#### Development
| Server | Purpose |
|--------|---------|
| github | Version control |
| playwright | Browser automation |
| puppeteer | Headless browser |
| sequentialthinking | Complex reasoning |
| context7 | Documentation |
| jupyter | Notebook integration |
| e2b | Code execution sandbox |

#### Security
| Server | Purpose |
|--------|---------|
| semgrep | SAST scanning |
| snyk | Vulnerability detection |

#### DevOps
| Server | Purpose |
|--------|---------|
| kubernetes | K8s management |
| docker | Container ops |
| aws | Cloud services |

---

## Skills Library

### Trading Skills (12)
- `trading-architecture` - 12-layer design patterns
- `trading-risk-validator` - Pre-trade validation
- `langgraph-workflows` - Multi-agent orchestration
- `financial-data-engineering` - QuestDB, time-series
- `trading-system` - General trading patterns
- `complex-system-building` - Event sourcing, DDD

### Creative Skills (10)
- `pose-analyzer` - MediaPipe extraction
- `touchdesigner-professional` - TD development
- `glsl-visualization` - Shader generation
- `ml-visualization` - ML output rendering
- `node-workflow-design` - Node networks
- `shader-generation` - GLSL patterns
- `quality-diversity-optimization` - MAP-Elites

### Architecture Skills (6)
- `system-design-architect` - Distributed systems
- `api-design-expert` - REST, GraphQL, gRPC
- `ecosystem-orchestrator` - Cross-project coordination
- `professional-software-patterns` - Production patterns

### Quality Skills (6)
- `tdd-workflow` - Test-driven development
- `testing-excellence` - All testing types
- `debugging-mastery` - Root cause analysis
- `code-review` - Security, performance
- `security-audit` - OWASP, vulnerabilities

### Memory Skills (3)
- `cross-session-memory` - Persistence
- `project-memory` - CLAUDE.md patterns
- `episodic-memory` - Past conversations

---

## Slash Commands

### Available Commands

| Command | Purpose |
|---------|---------|
| `/session-init [trading\|creative\|both]` | Initialize session with context |
| `/analyze-trading [area]` | Deep trading analysis |
| `/analyze-creative [component]` | Creative system analysis |
| `/start-exploration` | MAP-Elites exploration |
| `/create-node` | TD/ComfyUI node creation |
| `/speckit.*` | Specification workflows |

### Usage Examples

```bash
# Start a trading-focused session
/session-init trading

# Analyze the risk management layer
/analyze-trading risk-management

# Check pose pipeline health
/analyze-creative pose-extraction

# Start creative exploration
/start-exploration
```

---

## Architecture Patterns

### AlphaForge 12-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ L11 Dashboard     │ Real-time P&L, drawdown alerts, controls    │
├───────────────────┼─────────────────────────────────────────────┤
│ L10 Self-Improve  │ Optuna optimization, Reflexion learning     │
├───────────────────┼─────────────────────────────────────────────┤
│ L9  Archive       │ PostgreSQL, QuestDB, immutable audit logs   │
├───────────────────┼─────────────────────────────────────────────┤
│ L8  Observability │ Prometheus, Grafana, Langfuse               │
├───────────────────┼─────────────────────────────────────────────┤
│ L7  Execution     │ Smart order routing, rate limiting          │
├───────────────────┼─────────────────────────────────────────────┤
│ L6  AEGIS Risk    │ DAKC sizing, CVaR, circuit breakers         │
├───────────────────┼─────────────────────────────────────────────┤
│ L5  Apex Governor │ LangGraph 14-node workflow, fast path       │
├───────────────────┼─────────────────────────────────────────────┤
│ L4  Cognition     │ 4-model ensemble (CatBoost, Chronos, etc.)  │
├───────────────────┼─────────────────────────────────────────────┤
│ L3  Causal        │ Feature attribution (planned)               │
├───────────────────┼─────────────────────────────────────────────┤
│ L2  Opportunity   │ S-Score calculation, 5-tier classification  │
├───────────────────┼─────────────────────────────────────────────┤
│ L1  Data Fabric   │ WebSocket ingestion, QuestDB 4M rows/sec    │
├───────────────────┼─────────────────────────────────────────────┤
│ L0  Safety        │ Rust kill switch 100ns, adaptive rate limit │
└─────────────────────────────────────────────────────────────────┘
```

### State of Witness Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  Camera → MediaPipe Holistic → 33 Pose + 21 Hand + 468 Face     │
├─────────────────────────────────────────────────────────────────┤
│                      PROCESSING LAYER                            │
│  Normalization → PCA (33D→8D) → K-Means → Archetype Assignment  │
├─────────────────────────────────────────────────────────────────┤
│                      ANALYSIS LAYER                              │
│  Embedding (128D) → Cosine Similarity → Archetype Probability   │
├─────────────────────────────────────────────────────────────────┤
│                    COMMUNICATION LAYER                           │
│  OSC (7000) ←→ WebSocket (8080) ←→ Qdrant (6333)               │
├─────────────────────────────────────────────────────────────────┤
│                      RENDERING LAYER                             │
│  TouchDesigner → GLSL Compute → 2M Particles → 60fps Output     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8 Archetypes (State of Witness)

| Archetype | Color | Physics | Description |
|-----------|-------|---------|-------------|
| WARRIOR | (255,0,0) Red | mass:1.2, damp:0.1, grav:9.8 | Aggressive, forceful |
| NURTURER | (255,105,180) Pink | mass:1.4, damp:0.6, grav:6.0 | Centered, grounded |
| SAGE | (0,255,255) Cyan | mass:0.8, damp:0.8, grav:2.0 | Deliberate, aligned |
| JESTER | (255,255,0) Yellow | mass:0.6, damp:0.2, grav:12.0 | Erratic, high energy |
| LOVER | (255,20,147) Pink | mass:0.7, damp:0.4, grav:7.0 | Fluid, hip-centric |
| MAGICIAN | (138,43,226) Violet | mass:0.9, damp:0.5, grav:7.5 | Precise gestures |
| INNOCENT | (0,255,127) Green | mass:0.5, damp:0.15, grav:14.0 | Bouncy, light |
| EVERYMAN | (192,192,192) Gray | mass:1.0, damp:0.4, grav:9.8 | Neutral, balanced |

---

## Performance Benchmarks

### Trading System
- Rust kill switch: 100ns response
- Rate limiter: 1,000 req/min Alpaca limit
- QuestDB ingestion: 4M+ rows/sec
- LangGraph fast path: 200ms (vs 1,900ms normal)
- Test suite: 1,967 tests, 99.5% pass rate

### Creative System
- Target frame rate: 60 FPS (min 30)
- Capture-to-viz latency: <100ms
- Pose confidence threshold: >=0.5
- Tracking jitter: <=5%
- Particle count: 2M
- GPU utilization: 40-70% optimal

---

## Configuration Reference

### Environment Variables

```powershell
# Trading
$env:ALPACA_API_KEY = "..."
$env:ALPACA_SECRET_KEY = "..."
$env:QUESTDB_URL = "http://localhost:9000"
$env:GRAFANA_URL = "http://localhost:3000"

# Creative
$env:TOUCHDESIGNER_PORT = "9981"
$env:QDRANT_URL = "http://localhost:6333"
$env:COMFYUI_URL = "http://localhost:8188"

# Claude Code
$env:MAX_THINKING_TOKENS = "127998"
$env:ANTHROPIC_MODEL = "claude-opus-4-5-20251101"
$env:COST_LIMIT_DAILY = "50.0"
```

### Key File Locations

```
C:\Users\42\.claude\
├── settings.json           # Main configuration
├── mcp_servers.json        # MCP server definitions
├── CLAUDE.md               # Global instructions
├── skills/                 # 54+ custom skills
│   ├── trading-*/
│   ├── pose-analyzer/
│   └── ecosystem-orchestrator/
├── commands/               # Slash commands
│   ├── analyze-trading.md
│   ├── analyze-creative.md
│   └── session-init.md
└── plugins/               # Installed plugins
    ├── superpowers/
    ├── episodic-memory/
    └── claude-mem/

Z:\insider\AUTO CLAUDE\
├── autonomous AI trading system\
│   └── antigravity-omega-v12-ultimate\  # AlphaForge
└── Touchdesigner-createANDBE\           # State of Witness
```

---

## Quick Start Workflows

### Trading Development Session

```bash
# 1. Initialize session
/session-init trading

# 2. Check implementation status
# (Auto-loaded from memory)

# 3. Work on priority item
# Use TDD workflow: test first, then implement

# 4. Validate changes
pytest tests/ -x --tb=short

# 5. Save learnings
/memory
```

### Creative Exploration Session

```bash
# 1. Initialize session
/session-init creative

# 2. Check TouchDesigner connection
# (Auto-verified via MCP ping)

# 3. Start exploration
/start-exploration

# 4. Evaluate results with fitness functions

# 5. Refine and iterate
```

### Cross-Project Session

```bash
# 1. Initialize both contexts
/session-init both

# 2. Use ecosystem-orchestrator skill
# Claude will coordinate context switching

# 3. Work on shared infrastructure
# (Qdrant, Redis, monitoring)
```

---

## Best Practices

### DO
- Always run `/session-init` at start
- Use TodoWrite for complex tasks
- Run health checks before critical work
- Save session state before context switching
- Use episodic-memory for past decisions
- Test trading code with TDD workflow
- Use MAP-Elites for creative exploration

### DON'T
- Skip MCP server verification
- Mix trading and creative code in same session
- Deploy trading changes without paper testing
- Ignore circuit breaker states
- Use float for financial calculations (use Decimal)
- Forget to update CLAUDE.md with learnings
- Run autonomous trading without monitoring

---

## Troubleshooting

### Ghost Sessions in Resume List
**Symptom:** Many sessions starting with "Context: This summary will be shown..."
**Cause:** claude-mem Stop hook creating summarization cascade
**Fix:** Disable Stop hook in `hooks.json` (already done)

### MCP Server Connection Failed
**Symptom:** "Connection closed" errors on Windows
**Cause:** Missing `cmd /c` wrapper for npx commands
**Fix:** Use `"command": "cmd"` with `"args": ["/c", "npx", ...]`

### TouchDesigner Not Responding
**Symptom:** MCP ping fails
**Fix:**
1. Check TD is running
2. Verify MCP TOX is loaded
3. Check port 9981 is open

### Trading Tests Failing
**Symptom:** Test suite errors after changes
**Fix:**
1. Run `pytest tests/ -x --tb=long`
2. Check for race conditions
3. Verify mock data is current

---

---

## Advanced Capabilities (v2.0)

### High-End Thinking Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| **deep-research** | Multi-source research with synthesis | New technology evaluation, due diligence |
| **advanced-code-builder** | Production-grade component building | New subsystems, complex features |
| **architecture-analyzer** | System analysis with RADAR method | Before modifying unfamiliar code |
| **ultrathink-patterns** | Extended reasoning frameworks | Complex decisions, debugging |

### New Slash Commands

| Command | Purpose |
|---------|---------|
| `/research <topic>` | Deep multi-source research |
| `/build <component>` | Build with TDD and validation |
| `/ultrathink <problem>` | Extended reasoning with full token budget |
| `/analyze-architecture <path>` | Comprehensive architecture analysis |

### Ultrathink Reasoning Frameworks

1. **Multi-Perspective Analysis** - Multiple stakeholder viewpoints
2. **Constraint Satisfaction** - Hard/soft requirements optimization
3. **Causal Chain Analysis** - Root cause investigation
4. **Scenario Planning** - Uncertainty management
5. **Red Team / Blue Team** - Proposal evaluation
6. **Fermi Estimation** - Quantitative reasoning

### Code Building Pipeline

```
Design (10%) → Tests First (20%) → Implementation (40%) → Validation (20%) → Refinement (10%)
```

**Quality Gates:**
1. Syntax validation (py_compile)
2. Type checking (pyright)
3. Unit tests (pytest)
4. Coverage check (≥80%)
5. Linting (ruff)

### Research Methodology

```
Question Decomposition → Multi-Source Collection → Evidence Matrix → Validation → Report
```

**Source Priority:**
1. Official documentation
2. Academic papers
3. GitHub repositories
4. Technical blogs

### Configuration Updates

**Subscription Mode:**
```json
{
  "autonomy": {
    "maxIterations": 500,
    "subscriptionMode": true,
    "unlimitedBudget": true,
    "maxParallelAgents": 16,
    "enableUltrathink": true,
    "enableDeepResearch": true
  }
}
```

**Environment Variables:**
```bash
SUBSCRIPTION_MODE=true
ULTRATHINK_ENABLED=true
MAX_THINKING_TOKENS=127998
DEEP_RESEARCH_MODE=true
```

---

---

## Deep Research Insights (v3.0)

### Sequential Thinking MCP Integration

The [Sequential Thinking MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking) provides structured problem-solving through thought tracking and analysis.

**Key Capabilities:**
- **Thought Tracking**: Records sequential thoughts with metadata (sequence numbers, progress)
- **Branching & Revisions**: Supports alternative thinking paths and thought revisions
- **Progress Monitoring**: Tracks position in thought sequence and completion status
- **Summary Generation**: Produces comprehensive summaries of thinking processes
- **Thread-Safe Persistence**: Automatically saves sessions for continuity

**Integration Pattern:**
```typescript
// Add to mcp_servers.json
{
  "sequentialthinking": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
    "env": {}
  }
}
```

**Use Cases:**
- Complex architectural decisions
- Multi-step debugging workflows
- Strategic planning sessions
- Research synthesis tasks

---

### obra/superpowers Integration

The [superpowers framework](https://github.com/obra/superpowers) (12.3k+ stars) provides battle-tested skills for agentic development.

**Core Skills Categories:**
| Category | Skills | Purpose |
|----------|--------|---------|
| Testing | TDD, async testing, anti-patterns | RED/GREEN test-first development |
| Debugging | systematic-debugging, root-cause-tracing | 4-phase root cause analysis |
| Collaboration | brainstorming, planning, code-review | Structured development workflows |
| Development | git-worktrees, finishing-branches | Isolated feature development |

**TDD Philosophy:**
```
RED → GREEN → REFACTOR
Write failing test → Minimal implementation → Clean up
```

**Key Commands:**
- `/superpowers:brainstorm` - Before complex features
- `/superpowers:write-plan` - For migrations/refactors
- `/superpowers:execute-plan` - Run plans in batches

**Integration (Already Installed):**
```bash
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
```

---

### LangGraph 1.0 Multi-Agent Orchestration

[LangGraph](https://www.langchain.com/langgraph) (v1.0.6) provides production-grade multi-agent orchestration.

**Architecture Patterns:**

```
┌─────────────────────────────────────────────────────────────┐
│                     SUPERVISOR NODE                         │
│  Coordinates agent routing, state management, decisions     │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Agent A        │  Agent B        │  Agent C                │
│  (Analysis)     │  (Execution)    │  (Validation)           │
└────────┬────────┴────────┬────────┴────────┬────────────────┘
         │                 │                 │
         └────────────────►│◄────────────────┘
                    STATE GRAPH
```

**Key Features:**
- **Durable Execution**: Persists through failures, auto-resumes
- **Human-in-the-Loop**: Inspect/modify agent state at any point
- **Comprehensive Memory**: Short-term + long-term persistent memory
- **Fast Path**: 200ms (vs 1,900ms normal) for time-critical operations

**AlphaForge Integration (L5 Apex Governor):**
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# 14-node workflow graph
workflow = StateGraph(TradingState)
workflow.add_node("market_analysis", analyze_market)
workflow.add_node("risk_check", validate_risk)
workflow.add_node("fast_path", execute_fast)  # 200ms path
workflow.add_node("slow_path", execute_slow)  # Full analysis
# ... conditional routing based on opportunity score
```

---

### Extended Thinking Optimization

Claude's [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) enables deep reasoning with up to 128K tokens.

**Token Budget Strategy:**

| Task Type | Budget | Use Case |
|-----------|--------|----------|
| Quick Answer | 1K-4K | Simple queries, lookups |
| Code Generation | 4K-10K | Standard implementations |
| Debugging | 10K-20K | Complex bug analysis |
| Architecture | 20K-50K | System design decisions |
| Deep Research | 50K-80K | Comprehensive investigation |
| Strategic Planning | 80K-128K | Long-term roadmapping |

**Performance Benchmarks:**
- GPQA Physics: **96.5%** accuracy with extended thinking
- Math improvement: **logarithmic** with thinking tokens
- Interleaved thinking: Enables reasoning between tool calls

**Trigger Patterns:**
```markdown
# Lightweight thinking (~4K tokens)
"Think about..."

# Medium thinking (~10K tokens)
"Analyze thoroughly..."

# Deep thinking (~32K tokens)
"Use megathink for..."

# Maximum thinking (~128K tokens)
"Use ultrathink to deeply reason about..."
```

**Important API Constraint:**
- Thinking tokens: Up to **128K** (internal reasoning)
- Output tokens: Maximum **64K** (what gets returned)
- These are separate budgets - don't conflate them!

---

### TimesFM Foundation Models for Trading

Google's [TimesFM](https://github.com/google-research/timesfm) provides state-of-the-art time series forecasting.

**Model Evolution:**

| Version | Parameters | Context | Key Feature |
|---------|------------|---------|-------------|
| TimesFM 1.0 | 200M | 512 | Zero-shot forecasting |
| TimesFM 2.0 | 500M | 2048 | 25% better, 4x context |
| TimesFM 2.5 | 200M | 15K | In-context fine-tuning |

**Integration with AlphaForge:**
```python
from timesfm import TimesFM

# Initialize model
tfm = TimesFM.from_pretrained("google/timesfm-2.5-200m-pytorch")

# Forecast with context
forecast = tfm.predict(
    context=historical_prices[-2048:],  # 2K context window
    horizon=30,  # 30-day forecast
    freq="D"
)
```

**Key Advantages:**
- **Zero-Shot**: No task-specific training required
- **400B+ Timepoints**: Pretrained on massive dataset
- **BigQuery Integration**: Native support in Google Cloud
- **ICF Mode**: Few-shot learning from related time series

---

### pyribs Quality-Diversity Optimization

[pyribs](https://pyribs.org/) implements MAP-Elites and other QD algorithms for creative exploration.

**Core Concept:**
Quality-Diversity optimizes for BOTH fitness AND diversity, producing an archive of diverse high-quality solutions.

```
┌─────────────────────────────────────────────────────┐
│            BEHAVIORAL SPACE (Archive)               │
│  ┌───┬───┬───┬───┬───┐                             │
│  │ ★ │   │ ★ │   │ ★ │  ★ = High-quality solution │
│  ├───┼───┼───┼───┼───┤                             │
│  │   │ ★ │   │ ★ │   │  Each cell = unique niche  │
│  ├───┼───┼───┼───┼───┤                             │
│  │ ★ │   │   │   │ ★ │  Goal: Fill archive with   │
│  └───┴───┴───┴───┴───┘  diverse excellent solutions│
└─────────────────────────────────────────────────────┘
```

**State of Witness Application:**
```python
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

# Define behavioral dimensions
archive = GridArchive(
    solution_dim=128,  # Embedding dimension
    dims=[8, 8],       # 8x8 archetype grid
    ranges=[(0, 1), (0, 1)],  # Normalized space
)

# Explore parameter space
emitter = EvolutionStrategyEmitter(archive, sigma0=0.5)
scheduler = Scheduler(archive, [emitter])

for iteration in range(1000):
    solutions = scheduler.ask()
    objectives, measures = evaluate_poses(solutions)
    scheduler.tell(objectives, measures)
```

**Trading Application (Strategy QD):**
- Explore diverse trading strategies
- Optimize for Sharpe AND strategy diversity
- Avoid overfitting to single market regime

---

### FinRL CVaR-PPO Risk-Aware Trading

[FinRL](https://github.com/AI4Finance-Foundation/FinRL) provides state-of-the-art RL for trading with risk management.

**CPPO Algorithm (CVaR Proximal Policy Optimization):**
- Integrates Conditional Value-at-Risk into PPO objective
- Optimizes for worst-case scenarios
- Reduces tail risk in trading decisions

**Hierarchical Reinforced Trader (HRT) Pattern:**
```
┌─────────────────────────────────────────────────────┐
│           HIGH-LEVEL CONTROLLER (PPO)               │
│           Strategic stock selection                  │
├─────────────────────────────────────────────────────┤
│           LOW-LEVEL CONTROLLER (DDPG)               │
│           Trade execution optimization               │
└─────────────────────────────────────────────────────┘
```

**Performance:**
- Sampling speed: **227K+ samples/sec** (A100 GPU, 2048 parallel envs)
- DeepSeek integration: Sentiment + risk level features
- Dynamic model selection based on market conditions

---

### Claude Agent SDK Best Practices

[Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) provides production-grade autonomous agent capabilities.

**Agent Loop Pattern:**
```
GATHER CONTEXT → TAKE ACTION → VERIFY WORK → REPEAT
     ↑                                         │
     └─────────────────────────────────────────┘
```

**Long-Running Agent Architecture:**
```python
# Initializer Agent (first run)
- Create init.sh for environment setup
- Create claude-progress.txt for state tracking
- Initial git commit for baseline

# Coding Agent (incremental progress)
- Read claude-progress.txt for context
- Make incremental changes
- Update progress artifacts
- Commit changes with clear messages
```

**Critical Best Practices:**
1. **Don't give tools they don't need** - Increases confusion
2. **Use clear tool names** - Not "handler" or "processor"
3. **Validate inputs** - Clear error messages
4. **Mark side effects** - Make them explicit
5. **Verify before marking complete** - Use browser automation

**Guardrail Agents (Security):**
```python
# Monitor main agent actions in real-time
guardrail = GuardrailAgent(
    allowed_operations=["read", "write_code", "git_commit"],
    blocked_patterns=["rm -rf", "curl | bash", "eval("],
    sandbox_mode=True
)
```

---

### Advanced Tool Use Patterns (2026)

Anthropic's [advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) features:

**Tool Search Tool:**
- Access thousands of tools without context overhead
- Claude searches for relevant tools dynamically

**Programmatic Tool Calling:**
- Invoke tools in code execution environment
- Build complex tool pipelines programmatically

**Automatic Tool Call Clearing:**
- Clears old results approaching token limits
- Enables efficient multi-turn tool conversations

**OSWorld Benchmark Progress:**
- 2024: 14.9% (initial)
- 2026: **60%+** (current iterations)

---

### MCP Ecosystem Expansion

**Critical New Servers:**

| Server | Purpose | Use Case |
|--------|---------|----------|
| sequentialthinking | Structured reasoning | Complex decisions |
| context7 | Documentation access | API learning |
| e2b | Code sandbox | Safe execution |
| playwright | Browser automation | Verification |
| semgrep | SAST scanning | Security audit |

**MCP + LangGraph Integration:**
```
LangGraph (Graph-first orchestration) + MCP (Semantic transport)
= Programmable agents with rich tool access
```

---

## Maximum Power Stack (v4.0)

### Claude Task Master Integration

[Claude Task Master](https://github.com/eyaltoledano/claude-task-master) (24.3k+ stars) provides PRD-to-tasks conversion and intelligent project management.

**Key Capabilities:**
- **PRD Parsing**: Convert product requirements into structured tasks
- **Core Mode**: 70% token reduction for efficient task generation
- **Task Dependencies**: Automatic dependency chain detection
- **Complexity Analysis**: AI-powered task complexity scoring

**MCP Integration:**
```json
{
  "task-master": {
    "command": "npx",
    "args": ["-y", "task-master-ai", "--projectPath", "Z:\\insider\\AUTO CLAUDE"],
    "env": {
      "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
      "MODEL": "claude-sonnet-4-20250514"
    }
  }
}
```

**Workflow:**
```
PRD Document → Task Master → Structured Tasks → Implementation
```

---

### SecOpsAgentKit - Comprehensive Security

The SecOpsAgentKit provides 25+ security skills for defense-in-depth coverage.

**Tool Stack:**

| Category | Tools | Purpose |
|----------|-------|---------|
| **SAST** | Semgrep, SonarQube, CodeQL, Bandit | Static code analysis |
| **Container** | Trivy, Snyk, Grype, Checkov | CVE scanning, SBOM, IaC |
| **DAST** | OWASP ZAP, Nuclei, Nikto | Dynamic testing |
| **Compliance** | CIS Benchmarks, OWASP Top 10 | Standards verification |

**Security Levels:**

| Level | Checks | Use Case |
|-------|--------|----------|
| **Development** | Semgrep auto, Trivy basic | Local development |
| **Staging** | Full SAST, container scan, DAST baseline | Pre-production |
| **Production** | All + CodeQL semantic, penetration testing | Trading system |

**Trading-Specific Checks:**
```python
TRADING_SECURITY_CHECKS = {
    "authentication": ["api_key_exposure", "jwt_validation", "mfa_enforcement"],
    "authorization": ["privilege_escalation", "order_permission_bypass"],
    "data_protection": ["pii_exposure", "credentials_in_logs"],
    "financial_integrity": ["race_conditions_orders", "decimal_precision", "overflow_underflow"],
    "rate_limiting": ["api_throttling", "order_flooding", "ddos_protection"]
}
```

---

### LikeC4 Architecture Queries

[LikeC4](https://likec4.dev/) enables natural language architecture exploration.

**Query Examples:**
```
"Show me the data flow from market ingestion to order execution"
"What components depend on the Redis cache?"
"Display the L6 AEGIS Risk layer interactions"
```

**MCP Configuration:**
```json
{
  "likec4": {
    "command": "npx",
    "args": ["-y", "@likec4/mcp"],
    "env": {
      "LIKEC4_WORKSPACE": "Z:\\insider\\AUTO CLAUDE\\autonomous AI trading system\\antigravity-omega-v12-ultimate\\docs\\architecture"
    }
  }
}
```

**C4 Architecture Generation:**
```json
{
  "c4-model": {
    "command": "npx",
    "args": ["-y", "mcp-server-c4-diagram"],
    "env": {
      "OUTPUT_DIR": "C:\\Users\\42\\.claude\\data\\diagrams"
    }
  }
}
```

---

### Knowledge Graph Memory

Persistent memory across sessions using graph databases.

**Stack:**
| Component | Purpose | Backend |
|-----------|---------|---------|
| **knowledge-graph** | Entity-relation storage | Neo4j |
| **graphiti** | Temporal graph with embeddings | Neo4j + OpenAI |

**Benefits:**
- Entity relationships persist across sessions
- Temporal reasoning about past decisions
- Semantic similarity search on concepts
- Automatic knowledge extraction from conversations

**MCP Configuration:**
```json
{
  "knowledge-graph": {
    "command": "npx",
    "args": ["-y", "@anthropic-ai/mcp-server-knowledge-graph"],
    "env": {
      "NEO4J_URL": "${NEO4J_URL}"
    }
  },
  "graphiti": {
    "command": "uvx",
    "args": ["graphiti-mcp"],
    "env": {
      "NEO4J_URI": "${NEO4J_URL}",
      "OPENAI_API_KEY": "${OPENAI_API_KEY}"
    }
  }
}
```

---

### Advanced Trading Stack

Professional-grade trading infrastructure.

**QuantConnect Integration:**
```json
{
  "quantconnect": {
    "command": "uvx",
    "args": ["quantconnect-mcp"],
    "env": {
      "QC_USER_ID": "${QC_USER_ID}",
      "QC_API_TOKEN": "${QC_API_TOKEN}"
    }
  }
}
```

**Capabilities:**
- 20+ years of equity data
- Options, futures, crypto support
- Paper trading with realistic fills
- Live deployment to cloud
- Backtesting with transaction costs

**Interactive Brokers (IBKR):**
```json
{
  "ibkr": {
    "command": "uvx",
    "args": ["ibkr-mcp"],
    "env": {
      "IB_HOST": "127.0.0.1",
      "IB_PORT": "7497"
    }
  }
}
```

**Backtrader Integration:**
```json
{
  "backtrader": {
    "command": "python",
    "args": ["-m", "mcp_backtrader"],
    "env": {
      "DATA_PATH": "Z:\\insider\\AUTO CLAUDE\\autonomous AI trading system\\data\\historical"
    }
  }
}
```

---

### 6-Tier Extended Thinking

Calibrated thinking depth for different task complexities.

| Tier | Tokens | Trigger Keywords | Use Case |
|------|--------|------------------|----------|
| **Level 1** | ~4K | "think", "consider" | Standard analysis |
| **Level 2** | ~8K | "think harder", "reason through" | Complex debugging |
| **Level 3** | ~10K | "megathink", "analyze deeply" | Security audit |
| **Level 4** | ~32K | "ultrathink", "exhaustive analysis" | Architecture decisions |
| **Level 5** | ~64K | "gigathink", "explore all possibilities" | Novel domains |
| **Level 6** | ~128K | "full budget", "deepest analysis" | Foundational architecture |

**Automatic Escalation Rules:**
```python
ESCALATION_TRIGGERS = {
    "security_critical": "ultrathink",      # Level 4
    "financial_risk": "gigathink",          # Level 5
    "architectural_foundation": "gigathink", # Level 5
    "novel_domain": "full_budget",          # Level 6
    "multi_system_integration": "ultrathink" # Level 4
}
```

**Slash Command:**
```bash
/ultrathink --depth=gigathink "Evaluate kill switch reliability"
```

---

### Advanced Reasoning MCPs

Additional reasoning support beyond Sequential Thinking.

**Thinking Protocol:**
```json
{
  "thinking-protocol": {
    "command": "npx",
    "args": ["-y", "mcp-thinking-protocol"],
    "env": {
      "THINKING_MODE": "extended",
      "MAX_DEPTH": "10"
    }
  }
}
```

**Chain-of-Thought:**
```json
{
  "chain-of-thought": {
    "command": "npx",
    "args": ["-y", "mcp-server-cot"],
    "env": {
      "COT_STYLE": "step-by-step",
      "REFLECTION_ENABLED": "true"
    }
  }
}
```

---

### Complete MCP Server Inventory (58 Total)

| Category | Count | Servers |
|----------|-------|---------|
| **Memory & Persistence** | 8 | memory, memento, qdrant, qdrant-witness, lancedb, redis, sqlite, knowledge-graph, graphiti |
| **Financial & Trading** | 13 | alpaca, questdb, trading-tools, alphavantage, polygon, twelvedata, financial-datasets, fred, quantconnect, backtrader, ibkr, coingecko, timescaledb |
| **Creative & Visualization** | 6 | touchdesigner-creative, comfyui-creative, blender-creative, everart, mermaid, c4-model |
| **Observability** | 6 | grafana, prometheus, loki, opentelemetry, jaeger, datadog |
| **Development** | 9 | github, git, playwright, puppeteer, sequentialthinking, context7, jupyter, e2b, task-master |
| **Security** | 5 | semgrep, snyk, trivy, sonarqube, codeql |
| **DevOps** | 4 | kubernetes, docker, aws, sentry |
| **Architecture** | 2 | likec4, c4-model |
| **Reasoning** | 3 | sequentialthinking, thinking-protocol, chain-of-thought |
| **Productivity** | 4 | notion, slack, linear, calculator |
| **Search** | 4 | brave-search, exa, tavily, fetch |

---

## Research & Gap Analysis (v5.0)

### Gap Analysis Summary

Comprehensive research identified **23 gaps** and **47 potential improvements** across the ecosystem:

| Priority | Count | Categories |
|----------|-------|------------|
| **Critical** | 5 | Memory (Mem0), Safety (Guardrails), Observability (Langfuse), Trading (QLib), Tooling (LSP) |
| **High** | 9 | Letta, CrewAI, Phoenix, RFMs, ComfyUI-TD, TDD-Guard, Flow-Next, Memory Stack, Actor-Critic |
| **Medium** | 6 | AutoGen, Swarm, VEO-3, Sora 2, ContextKit, Zapier |
| **Enhancement** | 3 | Helicone, Continuous-Claude, ContextKit |

**Full Gap Analysis**: `Z:\insider\AUTO CLAUDE\unleash\GAP_ANALYSIS_V5.md`

---

### Mem0 Hybrid Memory Layer (CRITICAL)

Research finding: Mem0 provides **26% accuracy boost**, **91% lower latency**, **90% token savings** through hybrid datastore architecture.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    MEM0 HYBRID MEMORY                        │
├──────────────────┬──────────────────┬───────────────────────┤
│  Vector Store    │   Key-Value      │   Graph Store         │
│  (Embeddings)    │   (Fast Access)  │   (Relationships)     │
└──────────────────┴──────────────────┴───────────────────────┘
```

**MCP Configuration:**
```json
{
  "mem0": {
    "command": "uvx",
    "args": ["mem0-mcp"],
    "env": {
      "MEM0_API_KEY": "${MEM0_API_KEY}",
      "MEM0_ORG_ID": "${MEM0_ORG_ID}"
    }
  }
}
```

**Key Capabilities:**
- Dynamic information extraction and consolidation
- Real-time memory retrieval with sub-100ms latency
- Automatic context relevance scoring
- Cross-session knowledge persistence

---

### Letta Archival Memory

Letta persists beyond context windows with visual Agent Development Environment (ADE).

**Key Features:**
- Archival memory with dynamic compilation
- Visual workspace for building agents
- API-first architecture for integration
- Temporal awareness for versioned facts

**MCP Configuration:**
```json
{
  "letta": {
    "command": "npx",
    "args": ["-y", "@letta-ai/mcp-server"],
    "env": {
      "LETTA_API_KEY": "${LETTA_API_KEY}",
      "LETTA_BASE_URL": "http://localhost:8283"
    }
  }
}
```

---

### Langfuse LLM Observability (CRITICAL)

Dedicated LLM observability with tracing, evaluations, and prompt management. **19k+ GitHub stars**, MIT licensed.

**Capabilities:**
- Complete trace visualization for agent decisions
- Evaluation pipelines for quality assurance
- Prompt version management
- Cost tracking and attribution
- Integration with OpenTelemetry

**MCP Configuration:**
```json
{
  "langfuse": {
    "command": "npx",
    "args": ["-y", "@langfuse/mcp-server"],
    "env": {
      "LANGFUSE_PUBLIC_KEY": "${LANGFUSE_PUBLIC_KEY}",
      "LANGFUSE_SECRET_KEY": "${LANGFUSE_SECRET_KEY}",
      "LANGFUSE_HOST": "https://cloud.langfuse.com"
    }
  }
}
```

---

### Arize Phoenix Agent Evaluation

Multi-step agent trace evaluation on OpenTelemetry standards.

**Features:**
- Complete agent decision trace capture
- Evaluation over time for drift detection
- OpenTelemetry-native integration
- Visual trace explorer

**MCP Configuration:**
```json
{
  "phoenix": {
    "command": "uvx",
    "args": ["arize-phoenix"],
    "env": {
      "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:6006"
    }
  }
}
```

---

### QLib Reinforcement Learning (CRITICAL)

Microsoft's QLib provides state-of-the-art RL support for quantitative investment.

**Key Features:**
- Actor-Critic methods (A2C, A3C, SAC)
- Market-specific optimizations
- Integration with backtesting infrastructure
- Production-ready RL pipelines

**Research Insight:**
> Actor-Critic category shows significant promise with state-of-the-art performance but remains comparatively under-researched, suggesting prime territory for breakthrough improvements.

**MCP Configuration:**
```json
{
  "qlib": {
    "command": "python",
    "args": ["-m", "qlib.contrib.mcp_server"],
    "cwd": "Z:\\insider\\AUTO CLAUDE\\autonomous AI trading system\\antigravity-omega-v12-ultimate",
    "env": {
      "QLIB_DATA_PATH": "Z:\\insider\\AUTO CLAUDE\\autonomous AI trading system\\data\\qlib"
    }
  }
}
```

**Trading RL Patterns:**
- **Relational Foundation Models (RFMs)**: Graph Transformers for entity-relation signals
- **Multimodal Financial Models (MFFMs)**: Audio/video/tabular unified embedding
- **GRPO**: Group Relative Policy Optimization for step-by-step reasoning

---

### CrewAI Multi-Agent Orchestration

Role-playing collaborative AI agents with specific "crews" for team-style workflows.

**Key Concepts:**
- Define agent roles (Researcher, Analyst, Writer)
- Automatic task delegation based on roles
- Collaborative completion of complex tasks
- Memory sharing between crew members

**MCP Configuration:**
```json
{
  "crewai": {
    "command": "uvx",
    "args": ["crewai-tools"],
    "env": {
      "OPENAI_API_KEY": "${OPENAI_API_KEY}",
      "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
    }
  }
}
```

---

### Guardrail Agent Skill (CRITICAL)

Real-time monitoring agent for security, blocking dangerous operations before execution.

**Blocked Pattern Categories:**

| Category | Examples |
|----------|----------|
| **System Destruction** | `rm -rf /`, `format C:`, fork bombs |
| **Code Injection** | `curl \| bash`, `eval(`, `exec(` |
| **Database Destruction** | `DROP TABLE`, `TRUNCATE`, `DELETE FROM` |
| **Trading Safety** | `bypass_risk`, `kill_switch.*disable`, `circuit_breaker.*bypass` |
| **Credential Exposure** | `api_key = "..."`, `echo > .env` |

**Safety Layers (AlphaForge):**
```
┌──────────────────────────────────────────────────────────────┐
│                    SAFETY LAYERS                              │
├──────────────────────────────────────────────────────────────┤
│ L1: Guardrail Agent     │ Claude-level, pre-execution        │
│ L2: Python Risk Checks  │ Business logic validation          │
│ L3: API Rate Limiter    │ Broker-level protection            │
│ L4: Rust Kill Switch    │ 100ns hardware-level circuit break │
└──────────────────────────────────────────────────────────────┘
```

**Skill Location:** `C:\Users\42\.claude\skills\guardrail-agent\SKILL.md`

---

### LSP Plugin Expansion

Full language server coverage for real-time type information and error detection.

**Enabled Plugins:**
| Plugin | Language | Purpose |
|--------|----------|---------|
| `pyright-lsp` | Python | Type checking, completion |
| `rust-analyzer-lsp` | Rust | Type info for kill switch code |
| `gopls-lsp` | Go | Go infrastructure tooling |
| `tdd-guard` | All | Test-first enforcement |

**Benefits:**
- Real-time type information in context
- "Single biggest productivity gain" per research
- Immediate error detection before running code
- Intelligent completions with type awareness

---

### Workflow Automation

Cross-app automation for extended ecosystem integration.

**Zapier Integration:**
```json
{
  "zapier": {
    "command": "npx",
    "args": ["-y", "zapier-mcp"],
    "env": {
      "ZAPIER_API_KEY": "${ZAPIER_API_KEY}"
    }
  }
}
```
- Connects 6,000+ services
- Automated workflow triggers
- Cross-platform data flow

**n8n Integration:**
```json
{
  "n8n": {
    "command": "npx",
    "args": ["-y", "@n8n/mcp-server"],
    "env": {
      "N8N_API_KEY": "${N8N_API_KEY}",
      "N8N_URL": "http://localhost:5678"
    }
  }
}
```
- Self-hosted workflow automation
- Custom node development
- Complex multi-step workflows

---

### Complete MCP Server Inventory (68 Total)

| Category | Count | New in v5.0 |
|----------|-------|-------------|
| **Memory & Persistence** | 11 | mem0, letta |
| **Financial & Trading** | 14 | qlib |
| **Creative & Visualization** | 6 | - |
| **Observability** | 9 | langfuse, phoenix, helicone |
| **Development** | 9 | - |
| **Security** | 5 | - |
| **DevOps** | 4 | - |
| **Architecture** | 2 | - |
| **Reasoning** | 3 | - |
| **Multi-Agent** | 1 | crewai |
| **Automation** | 2 | zapier, n8n |
| **Productivity** | 4 | - |
| **Search** | 4 | - |

---

### Research Sources

**Memory Frameworks:**
- [Mem0 Research: 26% Accuracy Boost](https://mem0.ai/research)
- [Survey of AI Agent Memory Frameworks](https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks)
- [Graphiti Knowledge Graph Guide](https://medium.com/@saeedhajebi/building-ai-agents-with-knowledge-graph-memory)

**Agentic Frameworks:**
- [Agentic AI Frameworks: Top 8 Options 2026](https://www.instaclustr.com/education/agentic-ai/agentic-ai-frameworks-top-8-options-in-2026/)
- [Top 10 LangGraph Alternatives](https://www.ema.co/additional-blogs/addition-blogs/langgraph-alternatives-to-consider)

**Observability:**
- [Top 5 AI Agent Observability Platforms 2026](https://o-mega.ai/articles/top-5-ai-agent-observability-platforms-the-ultimate-2026-guide)
- [Langfuse vs Arize Comparison](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)

**Trading AI:**
- [Emerging AI Patterns in Finance 2026](https://gradientflow.com/emerging-ai-patterns-in-finance-what-to-watch-in-2026/)
- [RL for Quantitative Trading Survey](https://dl.acm.org/doi/10.1145/3582560)

**Claude Code:**
- [awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code)
- [Top 10 MCP Servers 2026](https://apidog.com/blog/top-10-mcp-servers-for-claude-code/)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 6.0 | 2026-01-16 | Role-Separated Architecture: Clear Design-Time (trading) vs Runtime (creative) MCP distinction, PAIML technical debt analysis (A+ to F grading, 487K LOC/sec), OSC direct control MCP, ComfyUI-TD bridge, redundancy removal (zapier removed), ARCHITECTURE_OPTIMIZATION_V6.md analysis doc, 70 total MCP servers |
| 5.0 | 2026-01-16 | Research & Gap Analysis Edition: Mem0 hybrid memory (26% accuracy boost), Letta archival memory, Langfuse observability (19k stars), Phoenix agent evaluation, Helicone cost tracking, QLib RL for trading, CrewAI multi-agent, Guardrail Agent skill, LSP plugins (rust-analyzer, gopls), TDD-Guard enforcement, n8n automation, 68 total MCP servers, 17 plugins |
| 4.0 | 2026-01-16 | Maximum Power Stack: Claude Task Master, SecOpsAgentKit (Trivy, SonarQube, CodeQL), LikeC4 architecture, Knowledge Graph Memory, QuantConnect/IBKR trading, 6-tier extended thinking, C4 diagrams, 58 total MCP servers |
| 3.0 | 2026-01-16 | Deep Research Edition: Sequential Thinking MCP, obra/superpowers, LangGraph 1.0, Extended Thinking optimization, TimesFM 2.5, pyribs QD, FinRL CPPO, Agent SDK patterns, Advanced Tool Use |
| 2.0 | 2026-01-16 | Added deep-research, advanced-code-builder, architecture-analyzer, ultrathink-patterns skills; New /research, /build, /ultrathink, /analyze-architecture commands; Subscription mode configuration |
| 1.0 | 2026-01-16 | Initial synthesis from 6 Compass artifacts |

---

## References

### Agentic Frameworks
- [obra/superpowers](https://github.com/obra/superpowers) - TDD, debugging skills (12.3k+ stars)
- [LangGraph](https://www.langchain.com/langgraph) - Multi-agent orchestration (v1.0.6)
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) - Autonomous agent development
- [Sequential Thinking MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking) - Structured reasoning

### Trading & Finance
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - Financial reinforcement learning
- [TimesFM](https://github.com/google-research/timesfm) - Time series foundation model
- [QuestDB](https://questdb.io/) - Time-series database (4M+ rows/sec)
- [Alpaca](https://alpaca.markets/) - Commission-free trading API

### Creative & ML
- [MediaPipe TouchDesigner](https://github.com/torinmb/mediapipe-touchdesigner) - Pose tracking
- [pyribs](https://pyribs.org/) - Quality-diversity optimization
- [TouchDesigner](https://derivative.ca/) - Real-time visual development

### Infrastructure
- [PyO3](https://pyo3.rs/) - Rust/Python bindings
- [MCP Protocol](https://modelcontextprotocol.io/) - Model Context Protocol
- [Docker/K8s](https://kubernetes.io/) - Container orchestration

### Anthropic Documentation
- [Extended Thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) - 128K reasoning
- [Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use) - Tool Search, Programmatic Calling
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices) - Agentic coding
