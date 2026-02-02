# Comprehensive SDK Inventory - Unleash Directory
## Updated: 2026-01-18 (v2 - With Claude-Flow)

---

## 📊 Overview

| Organization | Repos | Size | Purpose |
|--------------|-------|------|---------|
| **RuvNet** | 1 | 573 MB | **Claude-Flow v3** - 54+ Agents, Swarm Orchestration |
| **Anthropic** | 10 | 386 MB | Claude SDKs, Plugins, Cookbooks, Skills |
| **Zep/Graphiti** | 4 | 465 MB | Knowledge Graphs, Memory Systems |
| **Letta-AI** | 12 | 195 MB | Stateful Agents, Memory Platform |
| **MCP Official** | 8 | 118 MB | Model Context Protocol Ecosystem |
| **Auto-Claude** | 1 | 91 MB | Autonomous Multi-Session Coding |
| **Exa Labs** | 8 | 12 MB | Semantic Search, Research Tools |
| **TOTAL** | **44** | **~1.9 GB** | |

> **NEW**: See `ULTIMATE_ARCHITECTURE.md` for complete integration guide!

---

## 🔥 TIER 0: THE ORCHESTRATOR (New!)

### RuvNet Claude-Flow v3 (1 repo - 573 MB)
| Repository | Stars | Purpose |
|------------|-------|---------|
| `ruvnet-claude-flow` | 15k+ | **Enterprise AI Orchestration Platform** |

**Core Capabilities**:
- 54+ specialized agents (coder, tester, reviewer, architect, security)
- Hive Mind swarm coordination (Queen-led hierarchies)
- 5 consensus algorithms (Raft, Byzantine, Gossip, CRDT, Majority)
- Q-Learning router with 89% task routing accuracy
- SONA self-learning (<0.05ms adaptation)
- HNSW vector memory (150-12,500x faster retrieval)
- Multi-provider support (Claude, GPT, Gemini, Cohere, Ollama)
- 84.8% SWE-Bench solve rate

```bash
# Quick Start
npx claude-flow@v3alpha init
npx claude-flow@v3alpha mcp start
npx claude-flow@v3alpha --agent coder --task "Build feature"
```

---

## 🔥 TIER 1: Critical SDKs (Must Know)

### Anthropic Official (10 repos)
| Repository | Stars | Purpose |
|------------|-------|---------|
| `skills` | 44,499 | **THE** skills reference - all official skills |
| `claude-cookbooks` | 31,257 | Recipes & patterns for Claude |
| `claude-quickstarts` | 13,560 | Quick start templates |
| `claude-code-action` | 5,191 | GitHub Actions integration |
| `claude-plugins-official` | 4,146 | Official plugin directory |
| `claude-agent-sdk-python` | 4,195 | Python Agent SDK |
| `claude-code-security-review` | 2,843 | Security scanning action |
| `anthropic-sdk-typescript` | 1,521 | TypeScript API SDK |
| `claude-agent-sdk-demos` | 1,185 | SDK examples |
| `claude-agent-sdk-typescript` | 646 | TypeScript Agent SDK |

### MCP Official (8 repos)
| Repository | Stars | Purpose |
|------------|-------|---------|
| `servers` | 76,552 | **ALL** official MCP servers |
| `python-sdk` | 21,178 | Python SDK for MCP |
| `typescript-sdk` | 11,361 | TypeScript SDK for MCP |
| `inspector` | 8,332 | Visual MCP testing tool |
| `modelcontextprotocol` | 6,929 | MCP specification |
| `registry` | 6,289 | Server registry |
| `go-sdk` | 3,647 | Go SDK for MCP |
| `quickstart-resources` | 961 | Tutorial resources |

### Zep/Graphiti (4 repos)
| Repository | Stars | Purpose |
|------------|-------|---------|
| `graphiti` | 22,075 | Real-time knowledge graphs |
| `zep` | 3,976 | Agent memory platform |
| `zep-python` | 175 | Python SDK |
| `zep-js` | 69 | JavaScript SDK |

### Letta-AI (12 repos)
| Repository | Stars | Purpose |
|------------|-------|---------|
| `letta` | 20,695 | Core stateful agent platform |
| `agent-file` | 983 | .af format - agent serialization |
| `letta-code` | 860 | Memory-first coding agent |
| `sleep-time-compute` | 118 | Background memory consolidation |
| `letta-python` | 48 | Official Python SDK |
| `letta-node` | 40 | Official TypeScript SDK |
| `ai-memory-sdk` | 40 | Experimental memory SDK |
| `learning-sdk` | 37 | Continual learning |
| `letta-evals` | 35 | Evaluation kit |
| `skills` | 31 | Shared skills |
| `deep-research` | 19 | Deep research agent |
| `create-letta-app` | 14 | App scaffolding |

---

## 🔍 TIER 2: Essential Tools

### Exa Labs (8 repos)
| Repository | Stars | Purpose |
|------------|-------|---------|
| `exa-mcp-server` | 3,571 | MCP for semantic web search |
| `company-researcher` | 1,368 | Automated company analysis |
| `exa-deepseek-chat` | 721 | Chat with reasoning |
| `exa-hallucination-detector` | 309 | LLM output verification |
| `exa-py` | 187 | Python SDK |
| `exa-js` | 112 | JavaScript SDK |
| `ai-sdk` | 33 | Vercel AI integration |
| `benchmarks` | 29 | Performance benchmarks |

### Auto-Claude (AndyMik90)
| Feature | Description |
|---------|-------------|
| Purpose | Autonomous multi-session AI coding |
| Key Files | CLAUDE.md, guides/, scripts/, apps/ |
| Integration | CodeRabbit, pre-commit hooks |

---

## 📁 Directory Structure

```
unleash/
├── anthropic/                 # Anthropic Official (386 MB)
│   ├── skills/                # 44k stars - THE skills reference
│   ├── claude-cookbooks/      # 31k stars - Recipes
│   ├── claude-quickstarts/    # 13k stars - Templates
│   ├── claude-code-action/    # 5k stars - GitHub Action
│   ├── claude-plugins-official/# 4k stars - Plugin directory
│   ├── claude-agent-sdk-python/# Agent SDK
│   ├── claude-agent-sdk-typescript/
│   ├── claude-agent-sdk-demos/
│   ├── claude-code-security-review/
│   └── anthropic-sdk-typescript/
│
├── mcp-official/              # MCP Ecosystem (118 MB)
│   ├── servers/               # 76k stars - ALL MCP servers
│   ├── python-sdk/            # 21k stars
│   ├── typescript-sdk/        # 11k stars
│   ├── inspector/             # 8k stars - Visual testing
│   ├── modelcontextprotocol/  # Spec
│   ├── registry/              # Server registry
│   ├── go-sdk/
│   └── quickstart-resources/
│
├── zep-graphiti/              # Knowledge & Memory (465 MB)
│   ├── graphiti/              # 22k stars - Knowledge graphs
│   ├── zep/                   # 4k stars - Memory platform
│   ├── zep-python/
│   └── zep-js/
│
├── letta-ai/                  # Stateful Agents (195 MB)
│   ├── letta/                 # 21k stars - Core platform
│   ├── letta-code/            # Coding agent
│   ├── agent-file/            # .af format
│   ├── letta-python/
│   ├── letta-node/
│   ├── learning-sdk/
│   ├── ai-memory-sdk/
│   ├── sleep-time-compute/
│   ├── deep-research/
│   ├── skills/
│   ├── letta-evals/
│   └── create-letta-app/
│
├── exa-labs/                  # Semantic Search (12 MB)
│   ├── exa-mcp-server/        # 3.6k stars - MCP
│   ├── company-researcher/    # 1.4k stars
│   ├── exa-hallucination-detector/
│   ├── exa-deepseek-chat/
│   ├── exa-py/
│   ├── exa-js/
│   ├── ai-sdk/
│   └── benchmarks/
│
├── auto-claude/               # Autonomous Coding (91 MB)
│   └── (AndyMik90/Auto-Claude)
│
├── archived/                  # Old versions
├── active/                    # Active implementations
├── v10_optimized/             # V10 setup guide
└── docs/                      # Documentation
```

---

## 🔑 Key Integration Points

### For AlphaForge Trading:
- `mcp-official/servers` - Financial MCP servers
- `zep-graphiti/graphiti` - Temporal knowledge graphs
- `letta-ai/letta` - Agent state management
- `anthropic/claude-agent-sdk-python` - Building agents

### For State of Witness:
- `exa-labs/exa-mcp-server` - Research capabilities
- `letta-ai/letta-code` - Creative coding assistance
- `anthropic/skills` - Skill patterns
- `mcp-official/servers` - TouchDesigner integration

---

## 🔄 Update Commands

```powershell
# Update all repos in a directory
function Update-Repos($dir) {
    Get-ChildItem -Path $dir -Directory | ForEach-Object {
        Write-Host "Updating $($_.Name)..."
        Set-Location $_.FullName
        git pull 2>$null
        Set-Location ..
    }
}

# Update specific ecosystem
Update-Repos "Z:\insider\AUTO CLAUDE\unleash\anthropic"
Update-Repos "Z:\insider\AUTO CLAUDE\unleash\mcp-official"
Update-Repos "Z:\insider\AUTO CLAUDE\unleash\letta-ai"
Update-Repos "Z:\insider\AUTO CLAUDE\unleash\exa-labs"
Update-Repos "Z:\insider\AUTO CLAUDE\unleash\zep-graphiti"
```

---

## 📊 Star Count Summary

| Range | Count | Examples |
|-------|-------|----------|
| 50k+ | 2 | MCP servers (76k), Anthropic skills (44k) |
| 20-50k | 3 | claude-cookbooks (31k), graphiti (22k), letta (21k) |
| 10-20k | 3 | claude-quickstarts (13.5k), MCP python-sdk (21k), MCP ts-sdk (11k) |
| 5-10k | 3 | inspector (8k), registry (6k), MCP spec (6.9k) |
| 1-5k | 10 | claude-code-action, exa-mcp-server, zep, etc. |
| <1k | 22 | SDKs, tools, examples |

**Total Stars Represented: ~280,000+**

---

## 🚀 Quick Reference

### Most Important Files to Read:
1. `anthropic/skills/README.md` - Skill development patterns
2. `mcp-official/servers/README.md` - All available MCP servers
3. `anthropic/claude-cookbooks/` - Implementation recipes
4. `zep-graphiti/graphiti/README.md` - Knowledge graph setup
5. `letta-ai/letta/README.md` - Stateful agent basics

### Essential Documentation:
- MCP Spec: `mcp-official/modelcontextprotocol/`
- Agent SDK: `anthropic/claude-agent-sdk-python/`
- Memory: `zep-graphiti/zep/` and `letta-ai/letta/`

---

*Generated: 2026-01-18*
*Total Repositories: 43*
*Total Size: ~1.3 GB*
*Sources: github.com/anthropics, github.com/modelcontextprotocol, github.com/getzep, github.com/letta-ai, github.com/exa-labs, github.com/AndyMik90*
