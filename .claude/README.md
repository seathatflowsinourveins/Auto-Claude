# UNLEASH Agent & Skill Registry

> **Generated**: 2026-01-28
> **Purpose**: Central documentation for all agents and skills in the UNLEASH platform

---

## Quick Reference

| Category | Count | Location |
|----------|-------|----------|
| Agents | 24 types | `.claude/agents/` |
| Skills | 28 types | `.claude/skills/` |
| Commands | varies | `.claude/commands/` |
| Helpers | varies | `.claude/helpers/` |

---

## 1. Agent Types (24 Total)

### Core Agents
| Agent | Purpose | Priority |
|-------|---------|----------|
| `core/` | Base agent infrastructure | P0 |
| `templates/` | Agent templates and scaffolding | P1 |

### Architecture & Design
| Agent | Purpose | Priority |
|-------|---------|----------|
| `architecture/` | System design, ADRs, scalability | P1 |
| `analysis/` | Code analysis, metrics, patterns | P2 |
| `optimization/` | Performance tuning, bottleneck detection | P2 |

### Development
| Agent | Purpose | Priority |
|-------|---------|----------|
| `development/` | Core development workflow | P1 |
| `testing/` | Test generation, coverage | P2 |
| `documentation/` | Docs generation, maintenance | P3 |
| `devops/` | CI/CD, deployment, infrastructure | P2 |

### Multi-Agent Coordination
| Agent | Purpose | Priority |
|-------|---------|----------|
| `swarm/` | Swarm intelligence patterns | P1 |
| `hive-mind/` | Collective decision making | P1 |
| `consensus/` | Byzantine fault tolerance, voting | P1 |
| `goal/` | Goal-directed orchestration | P2 |

### Specialized Domains
| Agent | Purpose | Priority |
|-------|---------|----------|
| `github/` | GitHub operations, PRs, issues | P1 |
| `payments/` | Payment processing domain | P2 |
| `data/` | Data engineering, ETL | P2 |
| `specialized/` | Domain-specific agents | P3 |

### Advanced Frameworks
| Agent | Purpose | Priority |
|-------|---------|----------|
| `v3/` | Claude-Flow V3 agents | P0 |
| `sona/` | SONA architecture agents | P1 |
| `sparc/` | SPARC methodology agents | P2 |
| `flow-nexus/` | Flow coordination | P2 |
| `sublinear/` | Sublinear complexity optimization | P3 |
| `custom/` | User-defined custom agents | P3 |

---

## 2. Skills (28 Total)

### AgentDB Skills (5)
| Skill | Purpose | Use When |
|-------|---------|----------|
| `agentdb-advanced` | Advanced AgentDB operations | Complex queries |
| `agentdb-learning` | Pattern learning, ML integration | Training agents |
| `agentdb-memory-patterns` | Memory management patterns | Memory optimization |
| `agentdb-optimization` | AgentDB performance tuning | Slow queries |
| `agentdb-vector-search` | Vector similarity search | Semantic search |

### GitHub Skills (5)
| Skill | Purpose | Use When |
|-------|---------|----------|
| `github-code-review` | PR review automation | Code reviews |
| `github-multi-repo` | Multi-repository operations | Cross-repo work |
| `github-project-management` | Project boards, issues | Project tracking |
| `github-release-management` | Release automation | Deployments |
| `github-workflow-automation` | GitHub Actions workflows | CI/CD setup |

### V3 Skills (9)
| Skill | Purpose | Use When |
|-------|---------|----------|
| `v3-cli-modernization` | CLI interface updates | CLI development |
| `v3-core-implementation` | Core DDD module implementation | Core development |
| `v3-ddd-architecture` | Domain-Driven Design patterns | Architecture design |
| `v3-integration-deep` | Deep SDK integration | SDK work |
| `v3-mcp-optimization` | MCP server optimization | MCP tuning |
| `v3-memory-unification` | Memory system consolidation | Memory architecture |
| `v3-performance-optimization` | Performance tuning | Optimization |
| `v3-security-overhaul` | Security hardening | Security audit |

### ReasoningBank Skills (2)
| Skill | Purpose | Use When |
|-------|---------|----------|
| `reasoningbank-agentdb` | Reasoning + AgentDB integration | Complex reasoning |
| `reasoningbank-intelligence` | Intelligence amplification | Analysis tasks |

### Workflow Skills (7)
| Skill | Purpose | Use When |
|-------|---------|----------|
| `hooks-automation` | Hook implementation patterns | Adding hooks |
| `pair-programming` | Collaborative coding patterns | Pair sessions |
| `skill-builder` | Creating new skills | Skill development |
| `sparc-methodology` | SPARC framework | Structured approach |
| `stream-chain` | Streaming data pipelines | Data processing |
| `swarm-advanced` | Advanced swarm patterns | Complex coordination |
| `swarm-orchestration` | Swarm task distribution | Multi-agent tasks |

---

## 3. Agent Priority Matrix

```
P0 (Critical - Always Active)
├── core/
├── v3/
└── templates/

P1 (High - Frequently Used)
├── development/
├── architecture/
├── swarm/
├── hive-mind/
├── consensus/
├── sona/
└── github/

P2 (Medium - On-Demand)
├── testing/
├── analysis/
├── optimization/
├── devops/
├── goal/
├── data/
├── payments/
├── sparc/
└── flow-nexus/

P3 (Low - Specialized)
├── documentation/
├── specialized/
├── sublinear/
└── custom/
```

---

## 4. Skill Trigger Patterns

### Development Triggers
```
"implement" → v3-core-implementation
"design" → v3-ddd-architecture
"optimize" → v3-performance-optimization
"security" → v3-security-overhaul
"cli" → v3-cli-modernization
```

### GitHub Triggers
```
"review PR" → github-code-review
"create release" → github-release-management
"multi-repo" → github-multi-repo
"workflow" → github-workflow-automation
```

### Memory Triggers
```
"memory" → agentdb-memory-patterns
"vector search" → agentdb-vector-search
"learning" → agentdb-learning
```

### Orchestration Triggers
```
"swarm" → swarm-orchestration
"multi-agent" → swarm-advanced
"hooks" → hooks-automation
"stream" → stream-chain
```

---

## 5. Usage Examples

### Spawn an Agent
```bash
Task("Architecture review", "Review system architecture", "architecture")
Task("Code review", "Review PR #123", "github")
Task("Swarm task", "Distribute work across agents", "swarm")
```

### Invoke a Skill
```bash
# V3 Development
/v3-core-implementation
/v3-ddd-architecture

# GitHub Operations
/github-code-review
/github-release-management

# Memory Operations
/agentdb-vector-search
```

---

## 6. Model Routing

| Agent Category | Default Model | When to Override |
|----------------|---------------|------------------|
| Core, V3 | Sonnet | Opus for complex architecture |
| Development | Sonnet | Haiku for simple fixes |
| Testing | Haiku | Sonnet for test design |
| Documentation | Haiku | - |
| Architecture | Opus | - |
| Security | Opus | - |
| Swarm workers | Haiku | - |

---

## 7. Directory Structure

```
.claude/
├── agents/
│   ├── analysis/
│   ├── architecture/
│   ├── consensus/
│   ├── core/
│   ├── custom/
│   ├── data/
│   ├── development/
│   ├── devops/
│   ├── documentation/
│   ├── flow-nexus/
│   ├── github/
│   ├── goal/
│   ├── hive-mind/
│   ├── optimization/
│   ├── payments/
│   ├── sona/
│   ├── sparc/
│   ├── specialized/
│   ├── sublinear/
│   ├── swarm/
│   ├── templates/
│   ├── testing/
│   └── v3/
│
├── skills/
│   ├── agentdb-advanced/
│   ├── agentdb-learning/
│   ├── agentdb-memory-patterns/
│   ├── agentdb-optimization/
│   ├── agentdb-vector-search/
│   ├── github-code-review/
│   ├── github-multi-repo/
│   ├── github-project-management/
│   ├── github-release-management/
│   ├── github-workflow-automation/
│   ├── hooks-automation/
│   ├── pair-programming/
│   ├── reasoningbank-agentdb/
│   ├── reasoningbank-intelligence/
│   ├── skill-builder/
│   ├── sparc-methodology/
│   ├── stream-chain/
│   ├── swarm-advanced/
│   ├── swarm-orchestration/
│   ├── v3-cli-modernization/
│   ├── v3-core-implementation/
│   ├── v3-ddd-architecture/
│   ├── v3-integration-deep/
│   ├── v3-mcp-optimization/
│   ├── v3-memory-unification/
│   ├── v3-performance-optimization/
│   └── v3-security-overhaul/
│
├── commands/
│   └── [Custom CLI commands]
│
└── helpers/
    └── [Utility scripts]
```

---

## 8. Related Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| CLAUDE.md | Root | Project configuration |
| SDK_INDEX.md | .config/ | SDK reference |
| ARCHITECTURE.md | platform/ | Platform architecture |
| CONTRIBUTING.md | apps/ | Contribution guidelines |

---

*Registry maintained by UNLEASH platform*
*Last updated: 2026-01-28*
