# COMPASS SYNTHESIS - Consolidated Reference
> **Synthesized from**: 8 compass artifacts | **Date**: January 17, 2026

This document consolidates research from 8 compass artifact files into a single organized reference for the Claude Code ecosystem supporting both AlphaForge Trading and State of Witness creative systems.

---

## Table of Contents
1. [MCP Server Ecosystem](#1-mcp-server-ecosystem)
2. [Trading & Financial Data](#2-trading--financial-data)
3. [Code Quality & Security](#3-code-quality--security)
4. [Agent Frameworks](#4-agent-frameworks)
5. [Memory & Knowledge Systems](#5-memory--knowledge-systems)
6. [Skills & Plugins](#6-skills--plugins)
7. [Creative & Visualization](#7-creative--visualization)
8. [Architecture Tools](#8-architecture-tools)
9. [Performance Optimization](#9-performance-optimization)
10. [Recommended Stacks](#10-recommended-stacks)

---

## 1. MCP Server Ecosystem

### Official Statistics (January 2026)
- **2,000+ servers** in official MCP registry
- **7,880+ servers** on PulseMCP discovery platform
- **~30 official vendor-maintained** servers recommended for production
- MCP donated to **Linux Foundation's Agentic AI Foundation** (December 2025)

### Core Working Servers (No Setup)
| Server | Purpose | Status |
|--------|---------|--------|
| filesystem | Local file access | Works |
| memory | Built-in memory | Works |
| fetch | HTTP fetching | Works |
| sequentialthinking | Extended reasoning | Works |
| context7 | Documentation lookup | Works |
| time | System time | Works |
| sqlite | Local database | Works |
| calculator | Math operations | Works |
| coingecko | Free crypto data | Works |
| mermaid | Diagram rendering | Works |
| memento | Local sqlite memory | Works |
| osc | Local OSC protocol | Works |

### Windows MCP Pattern
```json
{
  "mcpServers": {
    "server-name": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@package/server-name"]
    }
  }
}
```

### Verified NPM Packages
| Server | Package | Verified |
|--------|---------|----------|
| sequentialthinking | @modelcontextprotocol/server-sequential-thinking | Yes |
| github | @modelcontextprotocol/server-github | Yes |
| postgres | @modelcontextprotocol/server-postgres | Yes |
| context7 | @upstash/context7-mcp | Yes |
| qdrant | mcp-server-qdrant | Yes |
| alpaca | alpaca-mcp-server (uvx) | Yes |

### Non-Existent Packages (Remove from Config)
- @letta-ai/mcp-server (npm 404)
- mem0-mcp (npm 404)
- @langfuse/mcp-server (npm 404)
- @anthropic-ai/mcp-server-slack (npm 404)

---

## 2. Trading & Financial Data

### Official Trading MCP Servers
| Server | Stars | Type | Key Capability |
|--------|-------|------|----------------|
| **Alpaca MCP** | 379 | Trading | Multi-asset execution, OAuth 2.0, paper trading |
| **Polygon/Massive** | - | Market Data | 35+ tools, all asset classes |
| **Alpha Vantage** | - | Technical Analysis | 50+ indicators, fundamentals |
| **QuantConnect** | - | Algo Trading | Full backtest-to-live pipeline |
| **Twelve Data** | - | Streaming | Real-time WebSocket data |
| **CoinGecko** | - | Crypto | 15K+ coins, DEX data |
| **InfluxDB** | - | Time-Series | Natural language DB queries |

### Broker Integrations
- **Interactive Brokers**: 3 community implementations (rcontesti, code-rabi, GaoChX)
- **MetaTrader MCP** v0.2.5: MT5 forex/CFD (Windows only)
- **Zerodha/AngelOne**: Indian markets
- **TradeStation**: US markets with auto token refresh
- **Binance/CCXT**: Crypto futures and multi-exchange

### Time-Series Databases
| Database | MCP Support | Performance |
|----------|-------------|-------------|
| QuestDB | Community | 11M+ rows/sec, ASOF JOIN |
| InfluxDB | Official | Natural language queries |
| TimescaleDB | Via PostgreSQL | No dedicated MCP |

### Performance Libraries
| Library | Speedup | Use Case |
|---------|---------|----------|
| Numba @njit | 50-100x | Algorithm hot loops |
| Polars | 2-10x vs Pandas | Data processing |
| uvloop | 2x | AsyncIO event loop |
| Apache Arrow | Zero-copy | Inter-process data |
| msgspec | 6.9x vs Pydantic | Hot path serialization |

### Risk Management
- **Circuit Breakers**: 5 failure threshold, 60s recovery
- **HMM Regime Detection**: Reduce positions in high volatility
- **Kelly Criterion**: Position sizing
- **VaR/CVaR**: 95% and 99% confidence levels
- **Rust Kill Switch**: Sub-millisecond independent checks

### Compliance Requirements
- FINRA Rule 15-09: Pre-trade risk checks
- SEC Rule 15c3-5: Market access controls
- Audit trails: Microsecond timestamps, 12-month retention
- Series 57: Developer registration requirement

---

## 3. Code Quality & Security

### Code Analysis Tools
| Tool | Stars | Key Feature |
|------|-------|-------------|
| **PAIML Toolkit** v2.210.0 | 109 | TDG scoring A-F, 487K LOC/sec |
| **SonarQube MCP** | Official | Enterprise quality gates |
| **Semgrep MCP** | 617 | 5,000+ rules, 30+ languages |
| **Snyk MCP** | Official | SAST, SCA, container, AIBOM |
| **Trivy MCP** v0.0.18 | Official | Vuln, misconfig, secrets |
| **ESLint MCP** | 70M/wk | v9.34.0 multithreaded |

### PAIML Technical Debt Grading
Six metrics: cyclomatic complexity, cognitive complexity, code duplication, SATD markers, test coverage, maintainability index.

Quality gates:
- Max cyclomatic complexity: 10 (NIST)
- Max cognitive complexity: 7
- Zero SATD tolerance (TODO, FIXME, HACK)

### Security Skill Collections
**SecOpsAgentKit** (github.com/AgentSecOps/SecOpsAgentKit):
- **AppSec**: sast-semgrep, sast-bandit, dast-nuclei, api-spectral
- **DevSecOps**: sca-trivy, container-grype, iac-checkov, secrets-gitleaks
- **Compliance**: policy-opa for SOC2/PCI-DSS
- **Threat Modeling**: pytm for STRIDE

---

## 4. Agent Frameworks

### Production-Ready Frameworks (v1.0+)
| Framework | Version | Date | Stars | Best For |
|-----------|---------|------|-------|----------|
| **LangGraph** | 1.0.6 | Jan 12, 2026 | 23.4K | Complex workflows, checkpointing |
| **Pydantic-AI** | 1.43.0 | Jan 16, 2026 | 15M dl | Type-safe, MCP native |
| **CrewAI** | 1.8.0 | Jan 8, 2026 | 30.5K | Role-based multi-agent |
| **MS Agent Framework** | Preview | Oct 2025 | - | Enterprise (AutoGen+SK unified) |

### SDK Versions
| SDK | Version | Notes |
|-----|---------|-------|
| Anthropic Python | 0.76.0 | Streaming, tools, batching |
| Anthropic TS | 0.71.2 | 3.7M weekly npm downloads |
| Claude Agent SDK | 0.2.9 | Programmatic agent building |
| OpenAI Agents | 0.6.6 | GPT-5.1, Temporal integration |
| PyO3 | 0.27.2 | Python 3.14 support |
| Maturin | 1.11.5 | Zero-config Rust wheels |

### Model Pricing (January 2026)
| Model | Input $/MTok | Output $/MTok |
|-------|--------------|---------------|
| Opus 4.5 | $15.00 | $75.00 |
| Sonnet 4.5 | $3.00 | $15.00 |
| Haiku 4.5 | $0.25 | $1.25 |

### Thinking Token Triggers
| Trigger | Budget | Use Case |
|---------|--------|----------|
| `think` | ~4,000 | Small reasoning boost |
| `think hard`, `megathink` | ~10,000 | Medium problems |
| `ultrathink` | ~32,000 | Complex architecture |

---

## 5. Memory & Knowledge Systems

### Memory MCP Servers
| Server | Backend | Best For |
|--------|---------|----------|
| @modelcontextprotocol/server-memory | JSONL | Basic project conventions |
| @gannonh/memento-mcp | Neo4j + OpenAI | Semantic search, temporal decay |
| obra/episodic-memory | SQLite + vectors | Session recall |
| Graphiti (Zep) | Neo4j | Multi-tenant teams |
| Obsidian MCP | Obsidian vault | Personal knowledge bases |

### CLAUDE.md Hierarchy
1. `~/.claude/CLAUDE.md` - Global (personal)
2. `./CLAUDE.md` - Project (team)
3. `./CLAUDE.local.md` - Private (gitignored)
4. `.claude/rules/*.md` - Modular rules

### Memory Commands
```
/memory-save    → Persist decisions, patterns
/memory-recall  → Semantic search past memories
/memory-status  → Memory system dashboard
/memory-sync    → Sync across backends
```

---

## 6. Skills & Plugins

### Essential Skill Collections
| Collection | Stars | Quality | Focus |
|------------|-------|---------|-------|
| **obra/superpowers** | 12.3K | 9/10 | TDD, debugging, git workflows |
| **anthropics/skills** | 40.2K | Official | xlsx, webapp-testing, mcp-builder |
| **alirezarezvani/claude-code-tresor** | 571 | 8/10 | 19 commands, 8 agents |
| **SecOpsAgentKit** | - | 8/10 | 25+ security skills |

### obra/superpowers Key Skills
- `systematic-debugging`: 4-phase root cause
- `test-driven-development`: RED-GREEN-REFACTOR
- `brainstorming`: Socratic design refinement
- `writing-plans`: Bite-sized tasks
- `requesting-code-review`: Pre-review checklist
- `finishing-a-development-branch`: Feature completion
- `using-git-worktrees`: Parallel sessions

### Skill Development Pattern
**Progressive disclosure (4 levels)**:
1. Level 1 (~100 tokens): YAML frontmatter - loaded at startup
2. Level 2 (<5k tokens): SKILL.md body - loaded on trigger
3. Level 3: `references/` - loaded on-demand
4. Level 4: `scripts/` - executed, only output consumed

```yaml
---
name: trading-risk-validator  # Max 64 chars
description: Validates orders against risk limits. Triggers on "check risk", "validate order".  # Max 1024 chars
allowed-tools: Read, Grep, Glob, Bash(python:*)
---
```

---

## 7. Creative & Visualization

### TouchDesigner Integration
| Server | Stars | Version | Capability |
|--------|-------|---------|------------|
| **8beeeaaat/touchdesigner-mcp** | 147 | v1.4.2 | Node CRUD, Python scripts |

**Tools**: `create_td_node`, `update_td_node_parameters`, `execute_python_script`

**Installation**:
```bash
claude mcp add-json "touchdesigner" '{"command":"npx","args":["-y","touchdesigner-mcp-server@latest","--stdio"]}'
```

### MediaPipe Integration
- **Keypoints**: 33 pose + 21 hand + 468 face landmarks
- **Plugin**: torinmb/mediapipe-touchdesigner
- **ONNX Runtime**: 43 FPS YOLO inference on RTX 3080Ti

### OSC Streaming Pattern
```python
from pythonosc import udp_client
client = udp_client.SimpleUDPClient("127.0.0.1", 5005)
for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
    client.send_message(f"/pose/{idx}/xyz",
        [landmark.x, landmark.y, landmark.z, landmark.visibility])
```

### GLSL Shader Skills
| Skill | Content |
|-------|---------|
| shader-noise | Perlin, Simplex, Worley, FBM, domain warping |
| shader-fundamentals | Coordinate spaces, transforms |
| algorithmic-art | p5.js patterns, flow fields, particles |

### MAP-Elites Quality Diversity
```python
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
archive = GridArchive(solution_dim=10, dims=[20, 20], ranges=[(-1, 1), (-1, 1)])
```
Used in Llama v3 and AlphaEvolve for creative exploration.

---

## 8. Architecture Tools

### C4 Model Tools
| Tool | Type | Capability |
|------|------|------------|
| **LikeC4 MCP** | npm | Natural language queries, Git tracking |
| **Structurizr** | Enterprise | Governance-focused C4 |
| **PlantUML MCP** | npm | Auto syntax correction |
| **ArchiMate MCP** | - | 7 EA layers |

### Diagram Generation
- **mcp-mermaid** (hustcc): All syntax, base64/SVG/URL
- **claude-mermaid** (veelenga): WebSocket live preview
- **mermaid-mcp-server** (peng-shawn): Puppeteer PNG/SVG

### Event-Driven Architecture
- **Axon Framework**: 70M+ downloads, Java CQRS/ES
- **EventStoreDB/KurrentDB**: Purpose-built event storage
- **Chronicle Queue**: 30μs latency Java
- **Context Mapper**: DDD DSL with 11+ refactorings

### Architecture Gaps
No dedicated MCP servers for:
- Domain-Driven Design
- Event Sourcing
- CQRS patterns
- EventStore integration

---

## 9. Performance Optimization

### High-Frequency Trading Stack
| Component | Technology | Performance |
|-----------|------------|-------------|
| Backtesting | NautilusTrader + Rust | 5M rows/sec |
| Data | QuestDB + Polars | 11M+ rows/sec |
| Serialization | msgspec | 6.9x faster |
| Network | DPDK/Kernel bypass | 4x throughput |
| Event Loop | uvloop | 2x throughput |

### hftbacktest Pattern
```python
from numba import njit
from hftbacktest import BUY, SELL, GTX, LIMIT

@njit
def market_making_algo(hbt):
    while hbt.elapse(10_000_000) == 0:  # 10ms
        depth = hbt.depth(0)
        mid = (depth.best_bid + depth.best_ask) / 2.0
        position = hbt.position(0)
        reservation_price = mid - 0.01 * position
        hbt.submit_buy_order(0, reservation_price - spread, qty, GTX, LIMIT)
```

### Kubernetes Deployment
- **StatefulSets** for persistent state bots
- **Pod Disruption Budgets**: minAvailable: 2
- **Blue-green deployment** for instant rollback
- **EC2 Cluster Placement Groups**: ~50μs latency
- **AWS Local Zones** (NYC/Chicago): <1ms exchange access

---

## 10. Recommended Stacks

### AlphaForge Trading System
```json
{
  "mcpServers": {
    "sequential-thinking": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]},
    "alpaca": {"command": "uvx", "args": ["alpaca-mcp-server", "serve"]},
    "qdrant": {"command": "uvx", "args": ["mcp-server-qdrant"]},
    "semgrep": {"command": "semgrep", "args": ["mcp"]}
  }
}
```

**Skills**: obra/superpowers, security-auditor, test-driven-development

**Databases**: QuestDB (tick data), Redis (cache), PostgreSQL (checkpoints)

### State of Witness Creative
```json
{
  "mcpServers": {
    "touchdesigner": {"command": "npx", "args": ["-y", "touchdesigner-mcp-server@latest", "--stdio"]},
    "memory": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
    "qdrant": {"command": "uvx", "args": ["mcp-server-qdrant"]}
  }
}
```

**Skills**: algorithmic-art, shader-noise, shader-fundamentals

**Integration**: OSC (port 5005), WebSocket, MediaPipe plugin

---

## Source Artifacts
This synthesis was created from:
1. compass_artifact_wf-d3536e39 - Windows ML ecosystem
2. compass_artifact_wf-306e0e02 - State of Witness architecture
3. compass_artifact_wf-55f914ed - Windows power user setup
4. compass_artifact_wf-ec136f08 - Trading systems toolkit
5. compass_artifact_wf-3d7e3ac6 - Skills and memory solutions
6. compass_artifact_wf-f6a65b8b - Autonomous trading guide
7. compass_artifact_wf-31a438fe - Systematic thinking tools
8. compass_artifact_wf-fe14f7c4 - Trading tools and code quality

*Last updated: January 17, 2026*
