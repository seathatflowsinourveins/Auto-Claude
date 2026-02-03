# UNLEASH Agents Registry

> **Quick Access**: Full registry at [.claude/README.md](./.claude/README.md)
> **Count**: 24 agent types | 28 skills | See priority matrix below

---

## Priority Matrix

| Priority | Agents | When to Use |
|----------|--------|-------------|
| **P0 Critical** | `core/`, `v3/` | Core infrastructure, Claude-Flow V3 |
| **P1 Important** | `architecture/`, `development/`, `swarm/`, `hive-mind/`, `consensus/`, `github/`, `sona/`, `templates/` | Architecture, coordination, GitHub ops |
| **P2 Specialized** | `analysis/`, `optimization/`, `testing/`, `devops/`, `goal/`, `payments/`, `data/`, `sparc/`, `flow-nexus/` | Domain-specific work |
| **P3 Optional** | `documentation/`, `specialized/`, `sublinear/`, `custom/` | Nice-to-have |

---

## Quick Reference

### Core Agents (Always Available)
```
.claude/agents/core/       # Base infrastructure
.claude/agents/v3/         # Claude-Flow V3 patterns
.claude/agents/templates/  # Agent scaffolding
```

### Multi-Agent Coordination
```
.claude/agents/swarm/      # Swarm intelligence
.claude/agents/hive-mind/  # Collective decisions
.claude/agents/consensus/  # Byzantine fault tolerance
```

### Development Workflow
```
.claude/agents/development/  # Core dev workflow
.claude/agents/testing/      # Test generation
.claude/agents/github/       # GitHub operations
```

---

## Skills (28 Total)

| Category | Skills | Purpose |
|----------|--------|---------|
| **V3** (9) | v3-cli-modernization, v3-core-implementation, v3-ddd-architecture, v3-integration-deep, v3-mcp-optimization, v3-memory-unification, v3-performance-optimization, v3-security-overhaul, v3-testing-comprehensive | Claude-Flow V3 development |
| **GitHub** (5) | github-code-review, github-multi-repo, github-project-management, github-release-management, github-workflow-automation | GitHub operations |
| **AgentDB** (5) | agentdb-advanced, agentdb-learning, agentdb-memory-patterns, agentdb-optimization, agentdb-vector-search | Agent database ops |
| **SONA** (3) | sona-* | SONA architecture |
| **Multi-Agent** (3) | swarm-*, consensus-*, hive-* | Coordination patterns |
| **Other** (3) | Various specialized | Domain-specific |

---

## Usage

```bash
# Spawn an agent
# In Claude Code, agents are automatically available via Task tool

# Example: Use architecture agent for system design
# Task tool with subagent_type="architecture"

# Example: Use v3 agent for Claude-Flow work
# Task tool with subagent_type="v3"
```

---

*Full documentation: [.claude/README.md](./.claude/README.md)*
