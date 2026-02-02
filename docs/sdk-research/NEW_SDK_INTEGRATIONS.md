# New SDK Integrations (2026-01-22)

## Overview
Two powerful additions to the Unleash SDK ecosystem:

---

## 1. Opik - AI Observability & Evaluation Platform

**Location**: `unleash/sdks/opik-full/`
**Source**: https://github.com/comet-ml/opik
**Purpose**: End-to-end LLM observability, evaluation, and optimization

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Tracing** | Deep tracing of LLM calls, conversations, agent activity |
| **Evaluation** | LLM-as-a-judge metrics, experiment management |
| **Production Monitoring** | 40M+ traces/day, dashboards, online evaluation rules |
| **Agent Optimizer** | Automatic prompt and agent optimization |
| **Guardrails** | Safety and responsible AI features |

### Structure
```
opik-full/
├── sdks/
│   ├── python/           # Python SDK (pip install opik)
│   └── typescript/       # TypeScript SDK
├── apps/
│   ├── opik-backend/     # Java backend service
│   ├── opik-frontend/    # React frontend
│   ├── opik-documentation/
│   ├── opik-guardrails-backend/
│   └── opik-python-backend/
├── deployment/
│   ├── docker-compose/   # Local deployment
│   └── helm_chart/       # Kubernetes deployment
└── tests_end_to_end/
```

### Quick Start
```bash
# Install Python SDK
pip install opik

# Configure (cloud or self-hosted)
opik configure

# Or programmatically
import opik
opik.configure(api_key="YOUR_KEY", workspace="YOUR_WORKSPACE")
```

### Integration with Unleash Projects

| Project | Use Case |
|---------|----------|
| **AlphaForge** | Trace trading decision pipelines, evaluate risk analysis accuracy |
| **State of Witness** | Monitor MediaPipe processing, evaluate pose classification |
| **Unleash** | Track Ralph Loop iterations, evaluate self-improvement |

### LLM-as-a-Judge Metrics
- Hallucination detection
- Answer relevance
- Context precision
- Moderation
- Faithfulness

### Self-Hosting
```bash
# Local with Docker Compose
cd opik-full
./opik.sh                    # Full suite
./opik.sh --backend          # Backend only
./opik.sh --guardrails       # With guardrails

# Access at http://localhost:5173
```

---

## 2. Everything Claude Code - Production Configs

**Location**: `unleash/everything-claude-code-full/`
**Source**: https://github.com/affaan-m/everything-claude-code
**Purpose**: Battle-tested Claude Code configs from Anthropic hackathon winner

### Key Components

#### Agents (9 specialized subagents)
| Agent | Purpose |
|-------|---------|
| `planner.md` | Feature implementation planning |
| `architect.md` | System design decisions |
| `tdd-guide.md` | Test-driven development |
| `code-reviewer.md` | Quality and security review |
| `security-reviewer.md` | Vulnerability analysis |
| `build-error-resolver.md` | Fix build errors |
| `e2e-runner.md` | Playwright E2E testing |
| `refactor-cleaner.md` | Dead code cleanup |
| `doc-updater.md` | Documentation sync |

#### Skills (11 workflow domains)
| Skill | Purpose |
|-------|---------|
| `coding-standards/` | Language best practices |
| `backend-patterns/` | API, database, caching |
| `frontend-patterns/` | React, Next.js patterns |
| `continuous-learning/` | Auto-extract patterns |
| `strategic-compact/` | Manual compaction |
| `tdd-workflow/` | TDD methodology |
| `security-review/` | Security checklist |
| `eval-harness/` | Verification evaluation |
| `verification-loop/` | Continuous verification |
| `clickhouse-io/` | ClickHouse patterns |

#### Commands (14 slash commands)
| Command | Purpose |
|---------|---------|
| `/plan` | Implementation planning |
| `/tdd` | Test-driven development |
| `/e2e` | E2E test generation |
| `/code-review` | Quality review |
| `/build-fix` | Fix build errors |
| `/refactor-clean` | Dead code removal |
| `/learn` | Extract patterns mid-session |
| `/checkpoint` | Save verification state |
| `/verify` | Run verification loop |
| `/eval` | Run evaluation harness |
| `/orchestrate` | Multi-agent orchestration |
| `/test-coverage` | Coverage analysis |
| `/update-codemaps` | Update code maps |
| `/update-docs` | Update documentation |

#### Rules (6 always-follow guidelines)
- `security.md` - Mandatory security checks
- `coding-style.md` - Immutability, file organization
- `testing.md` - TDD, 80% coverage requirement
- `git-workflow.md` - Commit format, PR process
- `agents.md` - When to delegate to subagents
- `performance.md` - Model selection, context management

#### Hooks
- Memory persistence hooks (session lifecycle)
- Strategic compaction suggestions
- Pre/Post tool use automations

### Installation Options

#### Option 1: As Claude Code Plugin
```bash
# Add marketplace
/plugin marketplace add affaan-m/everything-claude-code

# Install
/plugin install everything-claude-code@everything-claude-code
```

#### Option 2: Manual Integration
Copy desired components to `~/.claude/`:
- agents → `~/.claude/agents/`
- skills → `~/.claude/skills/`
- commands → `~/.claude/commands/`
- rules → `~/.claude/rules/`
- hooks → `~/.claude/hooks/`

#### Option 3: Reference from Unleash
Components available at:
```
Z:\insider\AUTO CLAUDE\unleash\everything-claude-code-full\
```

---

## Integration Matrix

| Component | Opik | Everything Claude Code |
|-----------|------|------------------------|
| **Tracing** | Full LLM tracing | - |
| **Evaluation** | LLM-as-judge, experiments | Eval harness, verification loop |
| **Agents** | - | 9 specialized subagents |
| **Skills** | - | 11 domain skills |
| **Commands** | - | 14 slash commands |
| **Memory** | Trace storage | Session persistence hooks |
| **CI/CD** | PyTest integration | Verification loops |

## Synergy Recommendations

### For AlphaForge Trading
1. Use Opik to trace trading decision pipelines
2. Use `security-reviewer` agent for code audits
3. Use `verification-loop` skill for continuous validation

### For State of Witness
1. Use Opik to monitor MediaPipe accuracy
2. Use `backend-patterns` for OSC/WebSocket code
3. Use `continuous-learning` to capture pose patterns

### For Unleash Meta-Development
1. Use Opik to evaluate Ralph Loop iterations
2. Use `eval-harness` for self-improvement metrics
3. Use all agents for code quality

---

*Integrated: 2026-01-22*
