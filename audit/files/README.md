# Ultimate Claude CLI Development Toolkit

[![Version](https://img.shields.io/badge/version-10.0-blue.svg)](https://github.com/your-repo)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Compatible-green.svg)](https://code.claude.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A production-ready Claude Code CLI toolkit for full development lifecycle work across trading systems and creative/ML applications.

---

## 🎯 Overview

This toolkit optimizes Claude Code CLI for building sophisticated AI-powered systems:

- **AlphaForge**: Autonomous trading system (12-layer architecture with Rust kill switch)
- **State of Witness**: Computational art/ML generative system with real-time control

### Core Architecture

```
Claude CLI = BUILDER TOOL for Full Development Lifecycle
│
├── AlphaForge (Trading)
│   └── Claude CLI is EXTERNAL builder (NOT in production)
│       └── System runs autonomously after deployment
│
└── State of Witness (Creative/ML)
    └── Claude CLI is INTEGRATED as generative brain
        └── Real-time MCP control of TouchDesigner/ComfyUI
```

### Key Features

- ✅ **40% reduced complexity** (V9 → V10 OPTIMIZED)
- ✅ **100% MCP server reliability** (verified, production-ready)
- ✅ **7-server optimal limit** for tool selection performance
- ✅ **Complete project isolation** (trading vs creative)
- ✅ **Production-ready hooks** for safety and quality

---

## 📦 Installation

### Prerequisites

- Node.js 18+
- Python 3.10+
- Git
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)

### Quick Install

```bash
# Clone or download the toolkit
git clone https://github.com/your-repo/claude-cli-toolkit.git
cd claude-cli-toolkit

# Run master installation
chmod +x install-claude-cli-toolkit.sh
./install-claude-cli-toolkit.sh

# Configure environment variables
cat ~/.claude/env-template.sh
# Copy variables to ~/.bashrc or ~/.zshrc and fill in API keys
source ~/.bashrc

# Validate installation
~/.claude/validate-toolkit.sh
```

### Project Setup

```bash
# For AlphaForge trading project
./setup-alphaforge.sh ~/projects/alphaforge

# For State of Witness creative project  
./setup-state-of-witness.sh ~/projects/state-of-witness
```

---

## 🔧 Configuration

### Directory Structure

```
~/.claude/
├── CLAUDE.md              # Global instructions
├── settings.json          # Global settings
├── commands/              # Global slash commands
├── agents/                # Global subagents
├── hooks/                 # Hook scripts
└── env-template.sh        # Environment template

project/
├── CLAUDE.md              # Project memory
├── .mcp.json              # MCP servers (git-tracked)
└── .claude/
    ├── settings.json      # Project settings
    ├── commands/          # Project slash commands
    ├── skills/            # Domain knowledge
    └── rules/             # Scoped rules
```

### MCP Server Configuration

**Core Servers (Always Active)**
| Server | Purpose | Tokens |
|--------|---------|--------|
| github | Version control | ~1,500 |
| context7 | Documentation | ~1,500 |
| filesystem | File access | ~1,000 |

**Trading Servers**
| Server | Purpose | Tokens |
|--------|---------|--------|
| alpaca | Trading execution | ~3,000 |
| polygon | Market data | ~3,500 |
| snyk | Security scanning | ~1,500 |

**Creative/ML Servers**
| Server | Purpose | Tokens |
|--------|---------|--------|
| touchdesigner | Real-time visuals | ~2,500 |
| comfyui | Image generation | ~2,000 |
| mlflow | Experiment tracking | ~1,500 |
| wandb | Model tracking | ~2,000 |

### Environment Variables

```bash
# Core
export ANTHROPIC_API_KEY="your-key"
export GITHUB_TOKEN="your-token"

# Trading
export ALPACA_API_KEY="your-key"
export ALPACA_SECRET_KEY="your-secret"
export ALPACA_PAPER_TRADE="true"  # ALWAYS true for dev
export POLYGON_API_KEY="your-key"

# Security
export SNYK_TOKEN="your-token"

# Creative/ML
export WANDB_API_KEY="your-key"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export TD_HOST="127.0.0.1"
export TD_PORT="9981"
```

---

## 🚀 Usage

### Starting a Session

```bash
# Start Claude Code in project directory
cd ~/projects/alphaforge
claude

# Or with specific options
claude --resume          # Resume last session
claude --continue        # Continue recent session
claude --mcp-debug       # Debug MCP connections
```

### Slash Commands

```
/security-audit          # Comprehensive security scan
/architecture-review     # Analyze project architecture
/performance-review      # Performance optimization suggestions
/backtest STRATEGY       # Run strategy backtest (AlphaForge)
/risk-analysis           # Portfolio risk analysis (AlphaForge)
/td-control OPERATION    # TouchDesigner control (State of Witness)
/generate-visual CONCEPT # Generate visuals (State of Witness)
/experiment-track NAME   # Track ML experiment (State of Witness)
```

### Memory Management

```bash
# Quick memory addition (during session)
# prefix with #: # Always use Pydantic for validation

# Edit memory file
/memory

# Bootstrap new project
/init
```

---

## 🪝 Hooks System

### Available Hooks

| Hook | Trigger | Use Case |
|------|---------|----------|
| `SessionStart` | Session begins | Load environment |
| `PreToolUse` | Before tool | Block dangerous ops |
| `PostToolUse` | After tool | Auto-format, lint |
| `Stop` | Session ends | Capture learnings |

### Example Hook Configuration

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "~/.claude/hooks/trading-safety.sh",
        "timeout": 10
      }]
    }],
    "PostToolUse": [{
      "matcher": "Write(*.py)",
      "hooks": [{
        "type": "command",
        "command": "ruff format \"$file\" --quiet",
        "timeout": 15
      }]
    }]
  }
}
```

### Hook Scripts

The toolkit includes these production-ready hooks:

- `precommit-quality.sh` - Pre-commit quality gate
- `trading-safety.sh` - Trading operation safety guard
- `auto-format.sh` - Auto-format on file write
- `capture-learnings.sh` - Session learning capture
- `strategy-reminder.sh` - Backtest reminder for strategies
- `experiment-tracking.sh` - ML experiment tracking enforcer
- `td-connection-check.sh` - TouchDesigner connection check

---

## 📊 Token Budget

Keep active MCP servers within budget to preserve context:

| Configuration | Servers | Tokens |
|---------------|---------|--------|
| **Core Only** | 3 | ~4,000 |
| **Trading Mode** | 5-6 | ~12,000 |
| **Creative Mode** | 5-6 | ~10,000 |
| **Full Stack** | 7 (max) | ~15,000 |

**Rule**: Stay under 7 MCP servers for optimal tool selection.

---

## 🛡️ Security

### Safety Measures

1. **Paper Trading Enforcement**
   - `ALPACA_PAPER_TRADE=true` enforced by hooks
   - Live trading commands blocked automatically

2. **Secret Protection**
   - Environment variables only (no hardcoding)
   - `.env` files in `.gitignore`
   - Deny rules block reading secrets

3. **Code Quality Gates**
   - Pre-commit hooks for formatting/linting
   - Security scanning with Snyk MCP
   - Type checking enforcement

### Security Checklist

- [ ] `ALPACA_PAPER_TRADE=true` verified
- [ ] API keys in environment variables
- [ ] `.env` files gitignored
- [ ] Production hooks blocking
- [ ] Security MCP enabled

---

## 🔍 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| MCP not connecting | Server not running | Check process, verify env vars |
| Tool not approved | Missing from allowedTools | Add to settings.json |
| Hook not triggering | Wrong matcher | Test with `echo $CLAUDE_TOOL_INPUT` |
| Slow performance | Too many MCPs | Reduce to 5-7 servers |

### Debug Commands

```bash
# Check MCP status
claude mcp list

# Debug MCP connections
claude --mcp-debug

# Test headless mode
claude -p "Hello" --output-format json --max-turns 1

# View hooks
claude /hooks

# Validate installation
~/.claude/validate-toolkit.sh
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `ULTIMATE_CLAUDE_CLI_TOOLKIT_MASTER.md` | Complete consolidated guide |
| `ultimate_claude_cli_toolkit_part7.md` | Part 7: Final Integration |
| `claude_cli_toolkit_quickref.md` | One-page quick reference |
| `example-workflows.md` | Production workflow examples |
| `trading_system_toolkit.md` | Trading-specific guide |
| `creative_ml_toolkit.md` | Creative/ML-specific guide |

---

## 📁 File Reference

### Scripts

| Script | Purpose |
|--------|---------|
| `install-claude-cli-toolkit.sh` | Master installation |
| `setup-alphaforge.sh` | AlphaForge project setup |
| `setup-state-of-witness.sh` | State of Witness setup |
| `validate-toolkit.sh` | Installation validation |
| `create-advanced-hooks.sh` | Hook library creation |

### Configuration Templates

| File | Location | Purpose |
|------|----------|---------|
| `CLAUDE.md` | `~/.claude/` | Global instructions |
| `settings.json` | `~/.claude/` | Global settings |
| `.mcp.json` | Project root | MCP server config |
| `env-template.sh` | `~/.claude/` | Environment template |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a PR

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for Claude Code CLI
- [Alpaca](https://alpaca.markets) for trading API
- [Polygon.io](https://polygon.io) for market data
- [TouchDesigner](https://derivative.ca) for visual programming
- Community MCP server contributors

---

*Built with ❤️ for the Claude Code community*
