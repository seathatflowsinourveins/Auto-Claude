# UNIFIED WORKFLOW GUIDE
## Claude Code Dual-Project Ecosystem - Essential Operations

> **Version**: 1.0 | **Date**: 2026-01-16 | **Status**: PRODUCTION-READY
> **Projects**: AlphaForge Trading System + State of Witness Creative

---

## Quick Reference

### Session Initialization

```powershell
# Trading Mode
.\init.ps1 -Mode trading

# Creative Mode
.\init.ps1 -Mode creative

# Dual Mode (default)
.\init.ps1 -Mode dual
```

### Essential Skills Matrix

| Skill | Domain | When to Use |
|-------|--------|-------------|
| `ecosystem-orchestrator` | Both | Context switching, system health |
| `deep-research` | Both | Technical investigation, due diligence |
| `trading-risk-validator` | Trading | Order validation, risk checks |
| `guardrail-agent` | Both | Security monitoring, blocked patterns |
| `touchdesigner-professional` | Creative | TD node networks, performance |
| `glsl-visualization` | Creative | Shader development, visual effects |
| `map-elites-exploration` | Creative | Quality-diversity optimization |
| `system-design-architect` | Both | Architecture decisions |
| `cross-session-memory` | Both | Context persistence |
| `code-review` | Both | Quality assurance |

### MCP Servers (Optimized 14-Server Stack)

**Tier 1 - Zero Config (Always Available)**
- `filesystem` - File access to project directories
- `memory` - Knowledge graph persistence
- `sequentialthinking` - Extended reasoning chains
- `context7` - Library documentation lookup
- `sqlite` - Local database operations
- `time` - Temporal operations
- `fetch` - Web content retrieval
- `calculator` - Mathematical operations

**Tier 2 - API Keys Required**
- `github` - Version control operations
- `brave-search` - Web search capabilities

**Tier 3 - Docker Services**
- `qdrant` - Vector embeddings
- `postgres` - Structured data

**Tier 4 - Creative Services**
- `touchdesigner-creative` - TD real-time control

---

## AlphaForge Trading Workflow

### Claude's Role: DEVELOPMENT ORCHESTRATOR

```
Claude writes code → Tests → Deploys → System runs autonomously
```

**Claude DOES:**
- Design architecture (11 layers, 233 modules)
- Write production Python/Rust code
- Create test suites (1,967+ tests)
- Perform security audits
- Generate deployment configs
- Set up monitoring dashboards

**Claude DOES NOT:**
- Execute trades in real-time
- Make live trading decisions
- Override kill switch (100ns Rust)

### Development Cycle

```
1. /session-init trading
2. Review features.json for current status
3. Implement next milestone (TDD workflow)
4. Run pytest suite
5. Security audit with guardrail-agent
6. Update claude-progress.txt
7. Git commit with proper message
```

### Critical Files

| File | Purpose |
|------|---------|
| `claude-progress.txt` | Session state persistence |
| `features.json` | Feature completion tracking |
| `CLAUDE.md` | Project context |
| `src/safety/kill_switch.rs` | 100ns emergency halt |

---

## State of Witness Creative Workflow

### Claude's Role: THE CREATIVE SYSTEM ITSELF

```
Claude Code → MCP → TouchDesigner → 60fps Visual Output
```

**Why Claude in the hot path works here:**
- 100ms latency acceptable (imperceptible at 60fps)
- No financial risk
- Exploration is the goal

### Creative Cycle

```
1. /session-init creative
2. Ping TouchDesigner via MCP
3. Run MAP-Elites exploration
4. Generate parameter variations
5. Evaluate with fitness functions
6. Store successful configs in memory
7. Update features.json
```

### Critical Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| `particle_count` | 100K-2M | 2,000,000 |
| `archetype_count` | 1-8 | 8 |
| `fps_target` | 30-60 | 60 |
| `latency_max_ms` | 50-200 | 100 |

---

## Quality-Diversity Optimization (MAP-Elites)

### Configuration

```python
# pyribs integration
archive_dimensions = ["color_warmth", "visual_complexity"]
fitness_weights = {
    "image_reward": 0.35,
    "aesthetic_score": 0.25,
    "clip_alignment": 0.20,
    "diversity_bonus": 0.20
}
archive_coverage_target = 0.70
```

### Exploration Loop

```
1. Initialize archive with random solutions
2. Select parent from archive
3. Mutate parameters (shader, physics, color)
4. Evaluate with fitness functions
5. Add to archive if elite
6. Repeat until coverage target
```

---

## Context Switching Protocol

### Trading → Creative

```python
1. Save trading session state
2. Verify TD MCP connection
3. Load creative context
4. Ping TouchDesigner
```

### Creative → Trading

```python
1. Save creative session state
2. Verify trading services (QuestDB, Redis)
3. Load trading context
4. Check market status
```

---

## File Organization

```
C:\Users\42\.claude\
├── settings.json           # Master configuration
├── mcp_servers_OPTIMAL.json # 14-server optimized stack
├── init.ps1                # Bootstrap script
├── env.ps1                 # Environment variables
├── skills/                 # 34 skill definitions
│   ├── ecosystem-orchestrator/
│   ├── deep-research/
│   ├── trading-risk-validator/
│   ├── guardrail-agent/
│   └── ...
├── archive/                # Historical configs
│   ├── stale_todos/
│   ├── mcp_servers_OLD.json
│   └── mcp_servers_WORKING_OLD.json
└── logs/                   # Operation logs
    ├── mcp_calls.log
    ├── trading_ops.log
    └── sessions.log

Z:\insider\AUTO CLAUDE\
├── autonomous AI trading system/
│   ├── claude-progress.txt
│   ├── features.json
│   └── antigravity-omega-v12-ultimate/
│       └── CLAUDE.md
├── Touchdesigner-createANDBE/
│   ├── claude-progress.txt
│   ├── features.json
│   └── CLAUDE.md
└── unleash/
    ├── OPTIMAL_ARCHITECTURE_2026.md
    ├── ECOSYSTEM_STATUS.md
    ├── UNIFIED_WORKFLOW.md (this file)
    ├── HONEST_AUDIT.md
    └── archive/
```

---

## Safety & Guardrails

### Blocked Patterns (Always)

```python
SYSTEM_BLOCKED = [
    "rm -rf /", "rm -rf *", "del /s /q C:",
    "format C:", "sudo rm", ":(){:|:&};:"
]

TRADING_BLOCKED = [
    "submit_order.*validate=False",
    "execute_trade.*bypass_risk",
    "kill_switch.*disable",
    "circuit_breaker.*bypass"
]
```

### Circuit Breakers

| Limit | Value |
|-------|-------|
| Daily loss | 2% |
| Position limit | $10,000 |
| Max drawdown | 5% |
| Kill switch latency | 100ns |

---

## Verification Checklist

### Daily Startup
- [ ] Run init.ps1 for appropriate mode
- [ ] Verify MCP servers responding
- [ ] Check features.json for current status
- [ ] Review recent commits

### Before Trading Development
- [ ] All tests passing
- [ ] Security audit clean
- [ ] Kill switch verified
- [ ] Risk parameters validated

### Before Creative Session
- [ ] TouchDesigner running and responding
- [ ] Qdrant vector store accessible
- [ ] Archive coverage baseline set
- [ ] Camera/input devices ready

---

## Related Documents

- `OPTIMAL_ARCHITECTURE_2026.md` - Research-backed architecture
- `ECOSYSTEM_STATUS.md` - Current deployment status
- `HONEST_AUDIT.md` - Self-assessment
- Project-level `CLAUDE.md` files - Context per project
