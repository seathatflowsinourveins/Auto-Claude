# Ultimate Claude CLI Development Toolkit
## Complete Consolidated Guide (Parts 1-7)

> **Version**: 10.0 OPTIMIZED  
> **For**: AlphaForge Trading + State of Witness Creative/ML Development  
> **Date**: January 2026

---

## Executive Summary

This document consolidates the complete 7-part research series on optimizing Claude Code CLI for full development lifecycle work. Key achievements:

- **40% reduction** in configuration complexity (V9 → V10)
- **100% MCP server reliability** (from 35% working)
- **7-server optimal limit** established for tool selection
- **Complete project configurations** for trading and creative/ML systems

### Core Architecture

```
Claude CLI = BUILDER TOOL for Full Dev Lifecycle
├── AlphaForge (Trading): Claude CLI is EXTERNAL builder
│   └── NOT part of production architecture - system runs autonomously
│
└── State of Witness (Creative/ML): Claude CLI is INTEGRATED
    └── Functions as generative brain with real-time MCP control
```

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Core MCP Servers](#2-core-mcp-servers)
3. [Trading Toolkit](#3-trading-toolkit)
4. [Creative/ML Toolkit](#4-creativeml-toolkit)
5. [Memory & Hooks System](#5-memory--hooks-system)
6. [Security Configuration](#6-security-configuration)
7. [Project Configurations](#7-project-configurations)
8. [Validation & Troubleshooting](#8-validation--troubleshooting)

---

## 1. Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)
- UV package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

```bash
# Download and run master installation script
chmod +x install-claude-cli-toolkit.sh
./install-claude-cli-toolkit.sh

# Set environment variables (edit template)
cat ~/.claude/env-template.sh >> ~/.bashrc
# Edit ~/.bashrc with your API keys
source ~/.bashrc

# Validate installation
~/.claude/validate-toolkit.sh

# Check MCP servers
claude mcp list
```

### Project Setup

```bash
# For AlphaForge trading project
./setup-alphaforge.sh ~/projects/alphaforge

# For State of Witness creative project
./setup-state-of-witness.sh ~/projects/state-of-witness
```

---

## 2. Core MCP Servers

### Tier 1: Always Active (All Projects)

| Server | Installation | Tokens | Purpose |
|--------|--------------|--------|---------|
| **github** | `claude mcp add github -s user -- npx -y @anthropic-ai/mcp-server-github` | ~1,500 | Version control, PRs, issues |
| **context7** | `claude mcp add context7 -s user --url https://mcp.context7.com/sse` | ~1,500 | Library documentation |
| **filesystem** | `claude mcp add filesystem -s user -- npx -y @anthropic-ai/mcp-server-filesystem --allow ~` | ~1,000 | File access |

### Configuration (.mcp.json)
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "context7": {
      "url": "https://mcp.context7.com/sse"
    }
  }
}
```

---

## 3. Trading Toolkit

### MCP Servers

| Server | Installation | Tokens | Purpose |
|--------|--------------|--------|---------|
| **alpaca** | `uvx alpaca-mcp-server serve` | ~3,000 | Multi-asset trading execution |
| **polygon** | `uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon` | ~3,500 | Market data (Polygon/Massive) |
| **snyk** | `npx -y snyk-mcp-server` | ~1,500 | Security scanning |

### Alpaca MCP Configuration
```json
{
  "alpaca": {
    "command": "uvx",
    "args": ["alpaca-mcp-server", "serve"],
    "env": {
      "ALPACA_API_KEY": "${ALPACA_API_KEY}",
      "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
      "ALPACA_PAPER_TRADE": "true"
    }
  }
}
```

⚠️ **CRITICAL**: Always set `ALPACA_PAPER_TRADE=true` during development

### Python SDKs (No MCP - Use via CLI)

```bash
# Install trading SDKs
pip install vectorbt riskfolio-lib skfolio

# TA-Lib requires system library first
# macOS: brew install ta-lib
# Ubuntu: apt-get install libta-lib-dev
pip install ta-lib
```

| SDK | Purpose | Example |
|-----|---------|---------|
| **VectorBT** | Vectorized backtesting | `import vectorbt as vbt` |
| **Riskfolio-Lib** | Portfolio optimization | `import riskfolio as rp` |
| **TA-Lib** | Technical indicators | `import talib` |
| **skfolio** | scikit-learn portfolio | `from skfolio import ...` |

### Trading Lifecycle Coverage

| Phase | Tools |
|-------|-------|
| **Design** | Riskfolio-Lib (portfolio architecture) |
| **Research** | Polygon MCP (data), VectorBT (backtest) |
| **Audit** | Snyk MCP (security), SonarQube |
| **Execute** | Alpaca MCP (paper trading) |
| **Debug** | Sentry MCP (errors) |

---

## 4. Creative/ML Toolkit

### MCP Servers

| Server | Installation | Tokens | Purpose |
|--------|--------------|--------|---------|
| **touchdesigner** | `npx touchdesigner-mcp-server@latest --stdio` | ~2,500 | Real-time visual control |
| **comfyui** | `python -m mcp_comfyui` | ~2,000 | Image generation workflows |
| **mlflow** | `mlflow mcp` | ~1,500 | Experiment tracking |
| **wandb** | URL: `https://mcp.withwandb.com/mcp` | ~2,000 | Model performance tracking |

### TouchDesigner MCP Configuration
```json
{
  "touchdesigner": {
    "command": "npx",
    "args": ["touchdesigner-mcp-server@latest", "--stdio"],
    "env": {
      "TD_HOST": "${TD_HOST:-127.0.0.1}",
      "TD_PORT": "${TD_PORT:-9981}"
    }
  }
}
```

**Requirements**: TouchDesigner with WebServer DAT running on port 9981

### Python SDKs (No MCP - Use via CLI)

```bash
# Install creative/ML SDKs
pip install diffusers transformers accelerate
pip install mediapipe
pip install ribs  # pyribs for quality-diversity
```

| SDK | Purpose |
|-----|---------|
| **diffusers** | Stable Diffusion pipelines |
| **MediaPipe** | Real-time face/pose/hand tracking |
| **pyribs** | Quality-diversity algorithms (MAP-Elites) |

### State of Witness Architecture

```
Claude CLI (Generative Brain)
    │
    ├──► TouchDesigner MCP ──► Real-time Visuals (port 9981)
    │
    ├──► ComfyUI MCP ──► Generated Assets (port 8188)
    │
    ├──► MLflow MCP ──► Experiment Tracking (port 5000)
    │
    └──► W&B MCP ──► Model Performance
```

---

## 5. Memory & Hooks System

### Directory Structure

```
~/.claude/
├── CLAUDE.md              # Global instructions (all projects)
├── settings.json          # Global settings
├── settings.local.json    # Personal (gitignored)
├── commands/              # Global slash commands
├── agents/                # Global subagents
└── session.log            # Session history

project/
├── CLAUDE.md              # Project memory
├── .mcp.json              # MCP servers (git-tracked)
└── .claude/
    ├── settings.json      # Project settings (git-tracked)
    ├── settings.local.json # Personal (gitignored)
    ├── commands/          # Project slash commands
    ├── agents/            # Project subagents
    ├── skills/            # Domain knowledge
    └── rules/             # Scoped rules
```

### Memory Hierarchy (Loaded in Order)

1. `~/.claude/CLAUDE.md` - Global (all projects)
2. `project/CLAUDE.md` - Project-specific
3. `project/.claude/rules/*.md` - Scoped rules
4. Subdirectory `CLAUDE.md` - On-demand when accessing files

### Essential Hooks Configuration

```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "startup",
      "hooks": [{
        "type": "command",
        "command": "echo \"[$(date)] Session started\" >> ~/.claude/session.log",
        "timeout": 5
      }]
    }],
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qi 'production'; then echo '{\"feedback\": \"⚠️ Production keyword detected\"}'; fi",
        "timeout": 5
      }]
    }],
    "PostToolUse": [{
      "matcher": "Write(*.py)",
      "hooks": [{
        "type": "command",
        "command": "ruff format \"$file\" --quiet && ruff check \"$file\" --fix --quiet || true",
        "timeout": 15
      }]
    }]
  }
}
```

### Hook Events Reference

| Event | Trigger | Use Case |
|-------|---------|----------|
| `SessionStart` | Session begins | Load environment, log start |
| `PreToolUse` | Before tool | Block dangerous commands |
| `PostToolUse` | After tool | Auto-format, lint, test |
| `Stop` | Session ends | Capture learnings, log end |

### Quick Memory Commands

```bash
# Add quick memory (during session)
# prefix with # to add to memory instantly
# example: # Always use Pydantic for validation

# Edit memory
/memory                    # Opens CLAUDE.md in editor

# Bootstrap new project
/init                      # Generates initial CLAUDE.md
```

---

## 6. Security Configuration

### Global Settings (Deny Rules)

```json
{
  "permissions": {
    "deny": [
      "Read(.env)",
      "Read(.env.*)",
      "Read(**/secrets/**)",
      "Read(**/*_key.json)",
      "Write(.env*)",
      "Write(**/secrets/**)",
      "Bash(rm -rf /)",
      "Bash(*SECRET*=*)",
      "Bash(*API_KEY*=*)"
    ]
  }
}
```

### Trading-Specific Security

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qiE 'alpaca.*live|production|--live'; then echo '{\"block\": true, \"message\": \"🛑 BLOCKED: Live trading not allowed\"}' >&2; exit 2; fi",
        "timeout": 5
      }]
    }]
  }
}
```

### Environment Variables Security

```bash
# ✅ DO: Use environment variables
export ALPACA_API_KEY="pk_xxx"

# ❌ DON'T: Hardcode in files
# api_key = "pk_xxx"  # NEVER DO THIS

# ✅ DO: Use .gitignore
echo ".env*" >> .gitignore
echo "secrets/" >> .gitignore
```

---

## 7. Project Configurations

### AlphaForge (.mcp.json)

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "alpaca": {
      "command": "uvx",
      "args": ["alpaca-mcp-server", "serve"],
      "env": {
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
        "ALPACA_PAPER_TRADE": "true"
      }
    },
    "polygon": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/polygon-io/mcp_polygon@v0.4.0", "mcp_polygon"],
      "env": { "POLYGON_API_KEY": "${POLYGON_API_KEY}" }
    },
    "snyk": {
      "command": "npx",
      "args": ["-y", "snyk-mcp-server"],
      "env": { "SNYK_TOKEN": "${SNYK_TOKEN}" }
    },
    "context7": {
      "url": "https://mcp.context7.com/sse"
    }
  }
}
```

### State of Witness (.mcp.json)

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "touchdesigner": {
      "command": "npx",
      "args": ["touchdesigner-mcp-server@latest", "--stdio"],
      "env": {
        "TD_HOST": "${TD_HOST:-127.0.0.1}",
        "TD_PORT": "${TD_PORT:-9981}"
      }
    },
    "comfyui": {
      "command": "python",
      "args": ["-m", "mcp_comfyui"],
      "env": {
        "COMFYUI_HOST": "${COMFYUI_HOST:-localhost}",
        "COMFYUI_PORT": "${COMFYUI_PORT:-8188}"
      }
    },
    "mlflow": {
      "command": "mlflow",
      "args": ["mcp"],
      "env": { "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI}" }
    },
    "wandb": {
      "url": "https://mcp.withwandb.com/mcp",
      "apiKey": "${WANDB_API_KEY}"
    },
    "context7": {
      "url": "https://mcp.context7.com/sse"
    }
  }
}
```

---

## 8. Validation & Troubleshooting

### Validation Script

```bash
~/.claude/validate-toolkit.sh
```

### Quick Checks

```bash
# MCP servers
claude mcp list

# Debug MCP connections
claude --mcp-debug

# Test headless mode
claude -p "Hello" --output-format json --max-turns 1

# View hooks
claude /hooks

# Check configuration
claude /config
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| MCP not connecting | Server not running | Check process, verify env vars |
| Tool not approved | Missing from allowedTools | Add to settings.json |
| Hook not triggering | Wrong matcher | Test with `echo $CLAUDE_TOOL_INPUT` |
| Slow performance | Too many MCPs | Reduce to 5-7 servers |
| Memory not loading | Wrong location | Check hierarchy |

### MCP Debug Commands

```bash
# Test Alpaca
uvx alpaca-mcp-server serve

# Test Polygon
uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon

# Test TouchDesigner
npx touchdesigner-mcp-server@latest --stdio
```

---

## Token Budget Summary

| Category | Tokens | Priority |
|----------|--------|----------|
| Core (github, context7, filesystem) | ~4,000 | Critical |
| Trading (alpaca, polygon) | ~6,500 | High |
| Security (snyk) | ~1,500 | Medium |
| Creative (touchdesigner, comfyui) | ~4,500 | High |
| Tracking (mlflow, wandb) | ~3,500 | Medium |

**Optimal Active Budget**: 5-7 servers (~15,000 tokens)

Leave significant context window for actual development work.

---

## Environment Variables Template

```bash
# Core
export ANTHROPIC_API_KEY="your-key"
export GITHUB_TOKEN="your-token"

# Trading
export ALPACA_API_KEY="your-key"
export ALPACA_SECRET_KEY="your-secret"
export ALPACA_PAPER_TRADE="true"  # ALWAYS
export POLYGON_API_KEY="your-key"

# Security
export SNYK_TOKEN="your-token"
export SENTRY_AUTH_TOKEN="your-token"

# Creative/ML
export WANDB_API_KEY="your-key"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export TD_HOST="127.0.0.1"
export TD_PORT="9981"
export COMFYUI_HOST="localhost"
export COMFYUI_PORT="8188"
```

---

## Essential Commands Reference

```bash
# Session
claude                    # Start interactive
claude --resume           # Resume last session
claude --continue         # Continue recent
claude -p "prompt"        # Headless mode

# MCP
claude mcp list           # List servers
claude mcp add <n> -- <cmd> # Add server
claude mcp remove <n>  # Remove
claude --mcp-debug        # Debug

# Memory
/memory                   # Edit memory
/init                     # Bootstrap CLAUDE.md
# prefix                  # Quick add to memory

# Config
/config                   # View settings
/hooks                    # Manage hooks
/allowed-tools            # View permissions
```

---

## Safety Checklist

Before each development session:

- [ ] `ALPACA_PAPER_TRADE=true` verified
- [ ] API keys in environment (not code)
- [ ] `.env` files in `.gitignore`
- [ ] Production hooks blocking
- [ ] Security MCP enabled
- [ ] Backtest before strategy changes

---

*This document consolidates Parts 1-7 of the Ultimate Claude CLI Development Toolkit research series. Use the individual scripts and configurations for production deployment.*
