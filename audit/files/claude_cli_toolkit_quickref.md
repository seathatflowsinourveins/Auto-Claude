# Ultimate Claude CLI Toolkit - Quick Reference Card
> Complete 7-Part Series Summary | AlphaForge + State of Witness

---

## 🎯 Core Architecture

```
Claude CLI = BUILDER TOOL
├── AlphaForge (Trading): CLI is EXTERNAL builder, NOT in production
└── State of Witness (Creative/ML): CLI IS INTEGRATED as generative brain
```

**Critical Rule**: Maximum 7 MCP servers active for optimal performance

---

## ⚡ Quick Start

```bash
# 1. Install toolkit
chmod +x install-claude-cli-toolkit.sh
./install-claude-cli-toolkit.sh

# 2. Set environment variables
source ~/.bashrc

# 3. Verify installation
./validate-toolkit.sh

# 4. Start Claude Code
claude
```

---

## 📦 Tier 1: Core MCP Servers (Always Active)

| Server | Command | Tokens | Purpose |
|--------|---------|--------|---------|
| **github** | `npx -y @anthropic-ai/mcp-server-github` | ~1,500 | Version control |
| **context7** | URL: `https://mcp.context7.com/sse` | ~1,500 | Documentation |
| **filesystem** | `npx -y @anthropic-ai/mcp-server-filesystem` | ~1,000 | File access |

---

## 📊 Tier 2: Trading MCPs (AlphaForge Focus)

| Server | Command | Tokens | Purpose |
|--------|---------|--------|---------|
| **alpaca** | `uvx alpaca-mcp-server serve` | ~3,000 | Trading execution |
| **polygon** | `uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon` | ~3,500 | Market data |
| **snyk** | `npx -y snyk-mcp-server` | ~1,500 | Security scanning |

⚠️ **CRITICAL**: Always set `ALPACA_PAPER_TRADE=true`

---

## 🎨 Tier 3: Creative/ML MCPs (State of Witness Focus)

| Server | Command | Tokens | Purpose |
|--------|---------|--------|---------|
| **touchdesigner** | `npx touchdesigner-mcp-server@latest --stdio` | ~2,500 | Real-time visuals |
| **comfyui** | `python -m mcp_comfyui` | ~2,000 | Image generation |
| **mlflow** | `mlflow mcp` | ~1,500 | Experiment tracking |
| **wandb** | URL: `https://mcp.withwandb.com/mcp` | ~2,000 | Model tracking |

---

## 🐍 Python SDKs (No MCP - Use via CLI)

### Trading
```bash
pip install vectorbt riskfolio-lib ta-lib skfolio
```

### Creative/ML
```bash
pip install diffusers transformers accelerate mediapipe ribs
```

---

## 📁 Directory Structure

```
~/.claude/
├── CLAUDE.md              # Global instructions
├── settings.json          # Global settings
├── commands/              # Global slash commands
└── agents/                # Global subagents

project/
├── CLAUDE.md              # Project memory
├── .mcp.json              # MCP config (git-tracked)
└── .claude/
    ├── settings.json      # Project settings
    ├── settings.local.json # Personal (gitignored)
    ├── commands/          # Project commands
    ├── agents/            # Project agents
    ├── skills/            # Project skills
    └── rules/             # Modular rules
```

---

## 🔧 Essential Settings.json

```json
{
  "permissions": {
    "allowedTools": ["Read", "Write", "Edit", "Bash(git *)"],
    "deny": ["Read(.env*)", "Bash(*SECRET*=*)"]
  },
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write(*.py)",
      "hooks": [{"type": "command", "command": "ruff format \"$file\""}]
    }]
  }
}
```

---

## 🪝 Hook Events

| Event | Trigger | Use Case |
|-------|---------|----------|
| `SessionStart` | Session begins | Load environment |
| `PreToolUse` | Before tool executes | Block dangerous operations |
| `PostToolUse` | After tool executes | Auto-format, auto-test |
| `Stop` | Session ends | Capture learnings |

---

## 🚀 Project Configurations

### AlphaForge (.mcp.json)
```json
{
  "mcpServers": {
    "github": {...},
    "alpaca": {...},
    "polygon": {...},
    "snyk": {...},
    "context7": {...}
  }
}
```

### State of Witness (.mcp.json)
```json
{
  "mcpServers": {
    "github": {...},
    "touchdesigner": {...},
    "comfyui": {...},
    "mlflow": {...},
    "wandb": {...},
    "context7": {...}
  }
}
```

---

## 🔐 Environment Variables

```bash
# Core
export ANTHROPIC_API_KEY="..."
export GITHUB_TOKEN="..."

# Trading
export ALPACA_API_KEY="..."
export ALPACA_SECRET_KEY="..."
export ALPACA_PAPER_TRADE="true"  # ALWAYS
export POLYGON_API_KEY="..."

# Security
export SNYK_TOKEN="..."
export SENTRY_AUTH_TOKEN="..."

# Creative/ML
export WANDB_API_KEY="..."
export MLFLOW_TRACKING_URI="http://localhost:5000"
export TD_HOST="127.0.0.1"
export TD_PORT="9981"
```

---

## 📋 Essential Commands

```bash
# Session management
claude                    # Start interactive
claude --resume           # Resume last session
claude --continue         # Continue recent

# MCP management
claude mcp list           # List servers
claude mcp add <n> -- <cmd> # Add server
claude mcp remove <n>  # Remove server
claude --mcp-debug        # Debug connections

# Memory
claude /memory            # Edit memory
claude /init              # Bootstrap CLAUDE.md
# + # prefix to add quick memory

# Configuration
claude /config            # View config
claude /hooks             # Manage hooks
```

---

## 🛡️ Safety Checklist

- [ ] `ALPACA_PAPER_TRADE=true` verified
- [ ] API keys in environment variables only
- [ ] `.env` files in `.gitignore`
- [ ] Production commands blocked by hooks
- [ ] Security MCP enabled for code audits
- [ ] Never commit secrets to git

---

## 📊 Token Budget Summary

| Category | Tokens | Notes |
|----------|--------|-------|
| Core MCPs | ~4,000 | Always active |
| Trading MCPs | ~6,500 | AlphaForge mode |
| Creative MCPs | ~8,000 | State of Witness mode |
| Security MCPs | ~1,500 | Optional |
| **Target Budget** | **~15,000** | Leave room for context |

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| MCP not connecting | Check env vars, verify server running |
| Hook not triggering | Check matcher pattern |
| Too slow | Reduce to 5-7 MCP servers |
| Memory not loading | Check CLAUDE.md location |
| Permission denied | Add to allowedTools |

```bash
# Debug commands
claude mcp list
claude --mcp-debug
claude --verbose
```

---

## 📚 Document Reference

| Part | Topic | Key Content |
|------|-------|-------------|
| 1-4 | Foundation | Plugin audit, SDK migration, memory, security |
| 5 | Trading | Alpaca, Polygon, VectorBT, Riskfolio-Lib |
| 6 | Creative/ML | TouchDesigner, ComfyUI, MLflow, W&B |
| 7 | Integration | Scripts, hooks, validation, complete configs |

---

## 🎯 Development Lifecycle Coverage

| Phase | Trading Tools | Creative Tools |
|-------|---------------|----------------|
| **Design** | Riskfolio-Lib, skfolio | TouchDesigner MCP, Blender |
| **Research** | Polygon MCP, VectorBT | MLflow, W&B |
| **Audit** | Snyk MCP, SonarQube | - |
| **Implement** | Alpaca MCP, TA-Lib | ComfyUI MCP, diffusers |
| **Debug** | Sentry MCP, Grafana | MLflow traces |
| **Deploy** | GitHub MCP | GitHub MCP |

---

*Generated: January 2026 | Claude CLI Toolkit v10 OPTIMIZED*
