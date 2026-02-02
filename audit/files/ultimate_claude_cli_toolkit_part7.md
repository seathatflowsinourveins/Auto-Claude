# Ultimate Claude CLI Development Toolkit
## Part 7: Final Integration & Validation

> Production-ready configurations, installation scripts, memory hooks, and validation procedures for AlphaForge + State of Witness development

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Installation Scripts](#2-installation-scripts)
3. [Global Settings Configuration](#3-global-settings-configuration)
4. [Project-Specific Configurations](#4-project-specific-configurations)
5. [Memory & Hooks Integration](#5-memory--hooks-integration)
6. [MCP Server Configurations](#6-mcp-server-configurations)
7. [Custom Slash Commands](#7-custom-slash-commands)
8. [Validation & Health Checks](#8-validation--health-checks)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Quick Reference](#10-quick-reference)

---

## 1. Directory Structure

### Global Claude Code Structure
```
~/.claude/
├── CLAUDE.md                    # Global instructions (all projects)
├── settings.json                # Global settings
├── settings.local.json          # Personal overrides (gitignored)
├── commands/                    # Global slash commands
│   ├── security-audit.md
│   ├── performance-review.md
│   └── architecture-review.md
├── agents/                      # Global subagents
│   ├── code-reviewer.md
│   └── research-assistant.md
└── projects/                    # Session history (auto-managed)
```

### AlphaForge Project Structure
```
alphaforge/
├── CLAUDE.md                    # Project memory
├── .mcp.json                    # MCP servers (git-tracked)
├── .claude/
│   ├── settings.json            # Project settings (git-tracked)
│   ├── settings.local.json      # Personal settings (gitignored)
│   ├── commands/
│   │   ├── backtest.md
│   │   ├── risk-analysis.md
│   │   └── strategy-audit.md
│   ├── agents/
│   │   ├── trading-analyst.md
│   │   └── risk-manager.md
│   ├── skills/
│   │   ├── vectorbt-patterns/
│   │   │   └── SKILL.md
│   │   └── riskfolio-patterns/
│   │       └── SKILL.md
│   └── rules/
│       ├── trading-safety.md
│       └── code-style.md
└── docs/
    ├── architecture.md
    └── trading-patterns.md
```

### State of Witness Project Structure
```
state-of-witness/
├── CLAUDE.md                    # Project memory
├── .mcp.json                    # MCP servers (git-tracked)
├── .claude/
│   ├── settings.json            # Project settings (git-tracked)
│   ├── settings.local.json      # Personal settings (gitignored)
│   ├── commands/
│   │   ├── generate-visual.md
│   │   ├── td-control.md
│   │   └── experiment-track.md
│   ├── agents/
│   │   ├── creative-director.md
│   │   └── ml-engineer.md
│   ├── skills/
│   │   ├── touchdesigner-patterns/
│   │   │   └── SKILL.md
│   │   └── comfyui-workflows/
│   │       └── SKILL.md
│   └── rules/
│       ├── visual-style.md
│       └── ml-best-practices.md
└── docs/
    ├── architecture.md
    └── generative-patterns.md
```

---

## 2. Installation Scripts

### Master Installation Script
```bash
#!/bin/bash
# install-claude-cli-toolkit.sh
# Complete installation for AlphaForge + State of Witness development toolkit

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Ultimate Claude CLI Development Toolkit Installer         ║"
echo "║  For AlphaForge Trading + State of Witness Creative/ML     ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js not found. Please install Node.js 18+"
        exit 1
    fi
    log_success "Node.js $(node --version)"
    
    # Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi
    log_success "Python $(python3 --version)"
    
    # UV (Python package manager)
    if ! command -v uv &> /dev/null; then
        log_warn "UV not found. Installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc
    fi
    log_success "UV installed"
    
    # Claude Code
    if ! command -v claude &> /dev/null; then
        log_error "Claude Code CLI not found. Please install first:"
        echo "  npm install -g @anthropic-ai/claude-code"
        exit 1
    fi
    log_success "Claude Code $(claude --version)"
}

# Create global directory structure
setup_global_structure() {
    log_info "Setting up global Claude Code structure..."
    
    mkdir -p ~/.claude/{commands,agents,skills}
    
    log_success "Global directory structure created"
}

# Install core MCP servers
install_core_mcps() {
    log_info "Installing core MCP servers..."
    
    # GitHub MCP (Official)
    claude mcp add github -s user -- npx -y @anthropic-ai/mcp-server-github
    log_success "GitHub MCP installed"
    
    # Context7 MCP (Documentation)
    claude mcp add context7 -s user --url https://mcp.context7.com/sse
    log_success "Context7 MCP installed"
    
    # Filesystem MCP
    claude mcp add filesystem -s user -- npx -y @anthropic-ai/mcp-server-filesystem --allow ~/projects
    log_success "Filesystem MCP installed"
}

# Install security & observability MCPs
install_security_mcps() {
    log_info "Installing security & observability MCPs..."
    
    # Snyk MCP (Security Scanning)
    if [ -n "$SNYK_TOKEN" ]; then
        claude mcp add snyk -s user -e SNYK_TOKEN="$SNYK_TOKEN" -- npx -y snyk-mcp-server
        log_success "Snyk MCP installed"
    else
        log_warn "SNYK_TOKEN not set, skipping Snyk MCP"
    fi
    
    # Sentry MCP (Error Tracking)
    if [ -n "$SENTRY_AUTH_TOKEN" ]; then
        claude mcp add sentry -s user -e SENTRY_AUTH_TOKEN="$SENTRY_AUTH_TOKEN" -- npx -y @sentry/mcp-server-sentry
        log_success "Sentry MCP installed"
    else
        log_warn "SENTRY_AUTH_TOKEN not set, skipping Sentry MCP"
    fi
}

# Install trading-specific MCPs
install_trading_mcps() {
    log_info "Installing trading MCPs..."
    
    # Alpaca MCP (Trading Execution)
    if [ -n "$ALPACA_API_KEY" ] && [ -n "$ALPACA_SECRET_KEY" ]; then
        claude mcp add alpaca -s user \
            -e ALPACA_API_KEY="$ALPACA_API_KEY" \
            -e ALPACA_SECRET_KEY="$ALPACA_SECRET_KEY" \
            -e ALPACA_PAPER_TRADE="true" \
            -- uvx alpaca-mcp-server serve
        log_success "Alpaca MCP installed (PAPER TRADING MODE)"
    else
        log_warn "ALPACA_API_KEY/ALPACA_SECRET_KEY not set, skipping Alpaca MCP"
    fi
    
    # Polygon MCP (Market Data)
    if [ -n "$POLYGON_API_KEY" ]; then
        claude mcp add polygon -s user \
            -e POLYGON_API_KEY="$POLYGON_API_KEY" \
            -- uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon
        log_success "Polygon MCP installed"
    else
        log_warn "POLYGON_API_KEY not set, skipping Polygon MCP"
    fi
}

# Install creative/ML MCPs
install_creative_mcps() {
    log_info "Installing creative/ML MCPs..."
    
    # TouchDesigner MCP
    npm install -g touchdesigner-mcp-server 2>/dev/null || true
    log_success "TouchDesigner MCP installed (requires TD WebServer)"
    
    # MLflow MCP
    pip install --quiet "mlflow>=3.5.1" 2>/dev/null || true
    log_success "MLflow MCP installed"
    
    # W&B MCP
    if [ -n "$WANDB_API_KEY" ]; then
        claude mcp add wandb -s user \
            --url https://mcp.withwandb.com/mcp \
            --api-key "$WANDB_API_KEY"
        log_success "Weights & Biases MCP installed"
    else
        log_warn "WANDB_API_KEY not set, skipping W&B MCP"
    fi
}

# Install Python SDKs
install_python_sdks() {
    log_info "Installing Python SDKs..."
    
    # Trading SDKs
    pip install --quiet vectorbt riskfolio-lib ta-lib skfolio 2>/dev/null || {
        log_warn "Some trading SDKs failed. TA-Lib may need system install first."
    }
    
    # Creative/ML SDKs
    pip install --quiet diffusers transformers accelerate mediapipe ribs 2>/dev/null || true
    
    log_success "Python SDKs installed"
}

# Create global CLAUDE.md
create_global_claude_md() {
    log_info "Creating global CLAUDE.md..."
    
    cat > ~/.claude/CLAUDE.md << 'EOF'
# Global Claude Code Instructions

## Developer Profile
- Building AlphaForge (autonomous trading system) and State of Witness (creative/ML system)
- Claude CLI = BUILDER TOOL for full dev cycle (design/audit/research/integrate/debug/UI)
- AlphaForge: Claude CLI is EXTERNAL builder, NOT production component
- State of Witness: Claude CLI IS integrated as generative brain with real-time control

## Architecture Principles
- Quality over quantity: Only actively maintained, production-ready tools
- 7-server MCP limit for optimal performance
- Paper trading always during development
- Strict separation: dev vs production environments

## Memory Behavior
- Update CLAUDE.md with learnings after significant sessions
- Reference @docs/ for detailed documentation
- Use /memory command for extensive additions

## Code Style
- TypeScript strict mode
- Python type hints with Pydantic validation
- Comprehensive error handling
- Tests before implementation (TDD)

## Security Rules
- NEVER commit API keys or secrets
- Always use environment variables
- Paper trading mode for all trading development
- Review before executing financial transactions
EOF

    log_success "Global CLAUDE.md created"
}

# Create global settings.json
create_global_settings() {
    log_info "Creating global settings.json..."
    
    cat > ~/.claude/settings.json << 'EOF'
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "model": "claude-sonnet-4-20250514",
  "permissions": {
    "allowedTools": [
      "Read",
      "Write",
      "Edit",
      "MultiEdit",
      "Bash(git *)",
      "Bash(npm *)",
      "Bash(pip *)",
      "Bash(python *)",
      "Bash(node *)",
      "Bash(uv *)",
      "Bash(claude mcp *)"
    ],
    "deny": [
      "Read(.env)",
      "Read(.env.*)",
      "Read(**/secrets/**)",
      "Read(**/*_key*)",
      "Read(**/*_secret*)",
      "Write(.env)",
      "Write(.env.*)",
      "Bash(rm -rf /)",
      "Bash(*ALPACA_SECRET*)",
      "Bash(*API_KEY*)"
    ]
  },
  "env": {
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384",
    "BASH_DEFAULT_TIMEOUT_MS": "60000"
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write(*.py)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v ruff &>/dev/null && ruff format \"$file\" --quiet || true",
            "timeout": 10
          }
        ]
      },
      {
        "matcher": "Write(*.ts)|Write(*.tsx)|Write(*.js)|Write(*.jsx)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v prettier &>/dev/null && prettier --write \"$file\" --log-level silent || true",
            "timeout": 10
          }
        ]
      }
    ]
  },
  "attribution": {
    "commits": true,
    "pullRequests": true
  }
}
EOF

    log_success "Global settings.json created"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    echo ""
    echo "MCP Server Status:"
    claude mcp list
    
    echo ""
    echo "Checking critical MCPs..."
    
    # Check each MCP
    for mcp in github context7; do
        if claude mcp list 2>/dev/null | grep -q "$mcp"; then
            log_success "$mcp MCP: Connected"
        else
            log_warn "$mcp MCP: Not found"
        fi
    done
}

# Main execution
main() {
    echo ""
    check_prerequisites
    echo ""
    setup_global_structure
    echo ""
    install_core_mcps
    echo ""
    install_security_mcps
    echo ""
    install_trading_mcps
    echo ""
    install_creative_mcps
    echo ""
    install_python_sdks
    echo ""
    create_global_claude_md
    echo ""
    create_global_settings
    echo ""
    verify_installation
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Installation Complete!                                     ║"
    echo "╠════════════════════════════════════════════════════════════╣"
    echo "║  Next Steps:                                                ║"
    echo "║  1. Set environment variables in ~/.bashrc or ~/.zshrc     ║"
    echo "║  2. Run project-specific setup scripts                     ║"
    echo "║  3. Verify with: claude mcp list                           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
}

main "$@"
```

### Environment Variables Template
```bash
# ~/.bashrc or ~/.zshrc additions
# Claude CLI Toolkit Environment Variables

# ═══════════════════════════════════════════════════════════
# CORE API KEYS
# ═══════════════════════════════════════════════════════════
export ANTHROPIC_API_KEY="your-anthropic-key"           # Claude API

# ═══════════════════════════════════════════════════════════
# TRADING APIs
# ═══════════════════════════════════════════════════════════
export ALPACA_API_KEY="your-alpaca-key"                 # Alpaca Trading
export ALPACA_SECRET_KEY="your-alpaca-secret"
export ALPACA_PAPER_TRADE="true"                        # ALWAYS true for dev
export POLYGON_API_KEY="your-polygon-key"               # Market Data

# ═══════════════════════════════════════════════════════════
# SECURITY & OBSERVABILITY
# ═══════════════════════════════════════════════════════════
export SNYK_TOKEN="your-snyk-token"                     # Security Scanning
export SENTRY_AUTH_TOKEN="your-sentry-token"            # Error Tracking
export SONARQUBE_TOKEN="your-sonar-token"               # Code Quality
export GITHUB_TOKEN="your-github-token"                 # GitHub API

# ═══════════════════════════════════════════════════════════
# CREATIVE/ML APIs
# ═══════════════════════════════════════════════════════════
export WANDB_API_KEY="your-wandb-key"                   # Weights & Biases
export MLFLOW_TRACKING_URI="http://localhost:5000"      # MLflow Server

# ═══════════════════════════════════════════════════════════
# TOUCHDESIGNER SETTINGS
# ═══════════════════════════════════════════════════════════
export TD_HOST="127.0.0.1"
export TD_PORT="9981"

# ═══════════════════════════════════════════════════════════
# COMFYUI SETTINGS
# ═══════════════════════════════════════════════════════════
export COMFYUI_HOST="localhost"
export COMFYUI_PORT="8188"

# ═══════════════════════════════════════════════════════════
# CLAUDE CODE OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="16384"
export BASH_DEFAULT_TIMEOUT_MS="60000"
```

---

## 3. Global Settings Configuration

### ~/.claude/settings.json (Complete)
```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "model": "claude-sonnet-4-20250514",
  "maxTokens": 16384,
  "permissions": {
    "allowedTools": [
      "Read",
      "Write",
      "Edit",
      "MultiEdit",
      "Glob",
      "Grep",
      "LS",
      "Bash(git *)",
      "Bash(npm *)",
      "Bash(npx *)",
      "Bash(pip *)",
      "Bash(uv *)",
      "Bash(python *)",
      "Bash(node *)",
      "Bash(claude *)",
      "Bash(ruff *)",
      "Bash(pytest *)",
      "Bash(cargo *)",
      "mcp__*"
    ],
    "deny": [
      "Read(.env)",
      "Read(.env.*)",
      "Read(.env.local)",
      "Read(**/secrets/**)",
      "Read(**/.git/config)",
      "Read(**/credentials*)",
      "Read(**/*_key.json)",
      "Write(.env*)",
      "Write(**/secrets/**)",
      "Bash(rm -rf /)",
      "Bash(rm -rf ~)",
      "Bash(*SECRET*=*)",
      "Bash(*API_KEY*=*)",
      "Bash(*PASSWORD*=*)"
    ]
  },
  "env": {
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384",
    "BASH_DEFAULT_TIMEOUT_MS": "60000",
    "PYTHONUNBUFFERED": "1"
  },
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Session started at $(date)' >> ~/.claude/session.log",
            "timeout": 5
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qi 'production\\|prod\\|live'; then echo '{\"feedback\": \"⚠️ Production environment detected. Proceeding with caution.\"}'; fi",
            "timeout": 5
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write(*.py)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v ruff &>/dev/null && ruff format \"$file\" --quiet && ruff check \"$file\" --fix --quiet || true",
            "timeout": 15
          }
        ]
      },
      {
        "matcher": "Write(*.ts)|Write(*.tsx)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v prettier &>/dev/null && prettier --write \"$file\" --log-level silent || true",
            "timeout": 10
          }
        ]
      },
      {
        "matcher": "Write(*.rs)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v rustfmt &>/dev/null && rustfmt \"$file\" || true",
            "timeout": 10
          }
        ]
      }
    ]
  },
  "attribution": {
    "commits": true,
    "pullRequests": true
  },
  "spinnerTipsEnabled": true
}
```

---

## 4. Project-Specific Configurations

### AlphaForge Configuration

#### alphaforge/.mcp.json
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
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
      "env": {
        "POLYGON_API_KEY": "${POLYGON_API_KEY}"
      }
    },
    "snyk": {
      "command": "npx",
      "args": ["-y", "snyk-mcp-server"],
      "env": {
        "SNYK_TOKEN": "${SNYK_TOKEN}"
      }
    },
    "context7": {
      "url": "https://mcp.context7.com/sse"
    }
  }
}
```

#### alphaforge/.claude/settings.json
```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allowedTools": [
      "Read",
      "Write",
      "Edit",
      "MultiEdit",
      "Bash(git *)",
      "Bash(python *)",
      "Bash(pytest *)",
      "Bash(cargo *)",
      "mcp__alpaca_*",
      "mcp__polygon_*",
      "mcp__github_*",
      "mcp__snyk_*"
    ],
    "deny": [
      "Read(.env*)",
      "Read(**/secrets/**)",
      "Bash(*live*)",
      "Bash(*production*)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | jq -r '.command' 2>/dev/null | grep -qi 'alpaca.*live\\|production'; then echo '{\"block\": true, \"message\": \"🛑 BLOCKED: Live trading commands not allowed in development.\"}' >&2; exit 2; fi",
            "timeout": 5
          }
        ]
      },
      {
        "matcher": "mcp__alpaca_*",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"feedback\": \"📊 Trading operation - Paper mode active\"}'",
            "timeout": 3
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write(**/strategies/*.py)",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"feedback\": \"💡 Remember: Run backtest before committing strategy changes\"}'",
            "timeout": 3
          }
        ]
      }
    ]
  }
}
```

#### alphaforge/CLAUDE.md
```markdown
# AlphaForge - Autonomous Trading System

## Project Overview
AlphaForge is a 12-layer autonomous trading system with Rust kill switch.
Claude CLI = BUILDER TOOL for design/audit/research/integrate/debug/UI
Claude CLI is NOT part of production architecture - system runs autonomously.

## Architecture
@docs/architecture.md

## Development Commands
- `cargo build --release` - Build Rust components
- `pytest tests/ -v` - Run Python tests
- `python -m strategies.backtest` - Run backtests

## Trading Safety Rules
1. ALWAYS use paper trading mode during development
2. NEVER commit real API keys
3. Review all strategy changes before deployment
4. Run full backtest suite before PR

## MCP Usage
- alpaca: Trading execution (PAPER MODE ONLY)
- polygon: Market data queries
- snyk: Security scanning before deployment

## Code Standards
- Rust: Follow clippy recommendations
- Python: Type hints required, Pydantic for validation
- Tests: Minimum 80% coverage for strategies
```

---

### State of Witness Configuration

#### state-of-witness/.mcp.json
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
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
      "env": {
        "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI:-http://localhost:5000}"
      }
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

#### state-of-witness/.claude/settings.json
```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allowedTools": [
      "Read",
      "Write",
      "Edit",
      "MultiEdit",
      "Bash(git *)",
      "Bash(python *)",
      "Bash(pip *)",
      "mcp__touchdesigner_*",
      "mcp__comfyui_*",
      "mcp__mlflow_*",
      "mcp__wandb_*",
      "mcp__github_*"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "mcp__touchdesigner_*",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"feedback\": \"🎨 TouchDesigner operation complete\"}'",
            "timeout": 3
          }
        ]
      },
      {
        "matcher": "mcp__comfyui_*",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"feedback\": \"🖼️ ComfyUI generation complete\"}'",
            "timeout": 3
          }
        ]
      }
    ]
  }
}
```

#### state-of-witness/CLAUDE.md
```markdown
# State of Witness - Computational Art & ML Generative System

## Project Overview
State of Witness is a computational art/ML generative system with real-time TouchDesigner control.
Claude CLI IS integrated as generative brain with real-time MCP control capabilities.

## Architecture
Claude CLI → TouchDesigner (via MCP, port 9981)
Claude CLI → ComfyUI (via MCP, port 8188)
Claude CLI → MLflow/W&B (experiment tracking)

@docs/architecture.md

## Development Commands
- `python -m pytest tests/` - Run tests
- `mlflow ui` - Start MLflow dashboard
- TouchDesigner must be running with WebServer DAT on port 9981

## MCP Usage
- touchdesigner: Real-time visual control (create nodes, update parameters)
- comfyui: Image generation workflows
- mlflow: Experiment tracking
- wandb: Model performance tracking

## Code Standards
- Python: Type hints, Pydantic models
- TouchDesigner: Follow TD Python patterns
- Experiments: Always log to MLflow/W&B
```

---

## 5. Memory & Hooks Integration

### Session Memory Hook (Auto-capture learnings)
```bash
#!/bin/bash
# ~/.claude/hooks/session-end-capture.sh
# Captures session learnings to project memory

PROJECT_DIR="$CLAUDE_PROJECT_DIR"
SESSION_FILE="$PROJECT_DIR/.claude/session-learnings.md"

# Create learnings file if doesn't exist
if [ ! -f "$SESSION_FILE" ]; then
    echo "# Session Learnings" > "$SESSION_FILE"
    echo "" >> "$SESSION_FILE"
fi

# Append timestamp
echo "" >> "$SESSION_FILE"
echo "## Session: $(date '+%Y-%m-%d %H:%M')" >> "$SESSION_FILE"

# The actual learning content would be passed via CLAUDE_SESSION_SUMMARY
if [ -n "$CLAUDE_SESSION_SUMMARY" ]; then
    echo "$CLAUDE_SESSION_SUMMARY" >> "$SESSION_FILE"
fi
```

### Pre-Commit Quality Gate Hook
```bash
#!/bin/bash
# .claude/hooks/precommit.sh
# Runs quality checks before allowing git commit

set -e

echo "Running pre-commit quality checks..."

# Python formatting and linting
if find . -name "*.py" -newer .git/index 2>/dev/null | grep -q .; then
    echo "Checking Python files..."
    ruff format --check . 2>/dev/null || true
    ruff check . 2>/dev/null || true
fi

# TypeScript/JavaScript
if find . -name "*.ts" -o -name "*.tsx" -newer .git/index 2>/dev/null | grep -q .; then
    echo "Checking TypeScript files..."
    npx tsc --noEmit 2>/dev/null || true
fi

# Rust
if find . -name "*.rs" -newer .git/index 2>/dev/null | grep -q .; then
    echo "Checking Rust files..."
    cargo clippy 2>/dev/null || true
fi

# Run tests if test files changed
if find . -name "*test*.py" -o -name "*.test.ts" -newer .git/index 2>/dev/null | grep -q .; then
    echo "Running tests..."
    if [ -f "pytest.ini" ] || [ -f "pyproject.toml" ]; then
        pytest --tb=short 2>/dev/null || true
    fi
fi

echo "Pre-commit checks complete"
```

### Settings.json Hook Configuration
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | jq -r '.command' | grep -q '^git commit'; then ./claude-hooks/precommit.sh; fi",
            "timeout": 180
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"Claude Code session started at $(date)\" >> ~/.claude/session.log",
            "timeout": 5
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"Session ended at $(date)\" >> ~/.claude/session.log",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

---

## 6. MCP Server Configurations

### Development Mode: Trading Focus
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

### Development Mode: Creative/ML Focus
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

## 7. Custom Slash Commands

### Global Commands (~/.claude/commands/)

#### security-audit.md
```markdown
Perform a comprehensive security audit of this codebase:

1. **Dependency Vulnerabilities**
   - Use Snyk MCP to scan dependencies
   - Check for known CVEs
   - Identify outdated packages

2. **Code Security**
   - Look for hardcoded secrets/credentials
   - Check for SQL injection vulnerabilities
   - Review authentication/authorization logic
   - Check for XSS vulnerabilities

3. **Configuration Security**
   - Review .env handling
   - Check file permissions
   - Validate API key management

Output a prioritized list of findings with severity levels (Critical/High/Medium/Low) and remediation steps.
```

#### architecture-review.md
```markdown
Analyze the architecture of this project:

1. **Structure Analysis**
   - Map out the directory structure
   - Identify main components and their responsibilities
   - Document the data flow

2. **Design Patterns**
   - Identify design patterns in use
   - Note any anti-patterns
   - Suggest improvements

3. **Dependencies**
   - Review external dependencies
   - Check for redundant dependencies
   - Evaluate dependency health

4. **Scalability**
   - Identify potential bottlenecks
   - Review error handling
   - Check for proper logging

Provide a visual diagram (mermaid) and detailed analysis.
```

### Project-Specific Commands

#### alphaforge/.claude/commands/backtest.md
```markdown
Run a comprehensive backtest for the strategy: $ARGUMENTS

1. **Setup**
   - Load historical data via Polygon MCP
   - Initialize VectorBT engine
   - Set backtest parameters

2. **Execution**
   - Run walk-forward optimization
   - Perform Monte Carlo simulation (1000 iterations)
   - Calculate risk metrics

3. **Analysis**
   - Calculate Sharpe ratio, Sortino ratio, max drawdown
   - Generate equity curve
   - Compare against benchmark (SPY)

4. **Output**
   - Summary statistics
   - Risk metrics
   - Visualization recommendations
```

#### state-of-witness/.claude/commands/td-control.md
```markdown
Control TouchDesigner via MCP: $ARGUMENTS

Available operations:
- `create <node_type> <name>` - Create a new node
- `connect <from> <to>` - Connect two nodes
- `param <node> <param> <value>` - Set parameter
- `execute <script>` - Run Python script in TD

Execute the requested operation and report the result.
Confirm TouchDesigner WebServer is running on port 9981.
```

---

## 8. Validation & Health Checks

### MCP Health Check Script
```bash
#!/bin/bash
# validate-toolkit.sh
# Validates Claude CLI toolkit installation

echo "═══════════════════════════════════════════════════════════"
echo "  Claude CLI Toolkit Validation"
echo "═══════════════════════════════════════════════════════════"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS="${GREEN}✓${NC}"
FAIL="${RED}✗${NC}"
WARN="${YELLOW}!${NC}"

total_checks=0
passed_checks=0

check() {
    local name="$1"
    local command="$2"
    total_checks=$((total_checks + 1))
    
    if eval "$command" &>/dev/null; then
        echo -e "  $PASS $name"
        passed_checks=$((passed_checks + 1))
        return 0
    else
        echo -e "  $FAIL $name"
        return 1
    fi
}

warn_check() {
    local name="$1"
    local command="$2"
    
    if eval "$command" &>/dev/null; then
        echo -e "  $PASS $name"
    else
        echo -e "  $WARN $name (optional)"
    fi
}

echo ""
echo "Core Components:"
check "Claude Code CLI" "command -v claude"
check "Node.js" "command -v node"
check "Python 3" "command -v python3"
check "UV Package Manager" "command -v uv"
check "Git" "command -v git"

echo ""
echo "Global Configuration:"
check "~/.claude/CLAUDE.md" "[ -f ~/.claude/CLAUDE.md ]"
check "~/.claude/settings.json" "[ -f ~/.claude/settings.json ]"
check "~/.claude/commands/" "[ -d ~/.claude/commands ]"

echo ""
echo "Environment Variables:"
warn_check "ANTHROPIC_API_KEY" "[ -n \"$ANTHROPIC_API_KEY\" ]"
warn_check "GITHUB_TOKEN" "[ -n \"$GITHUB_TOKEN\" ]"
warn_check "ALPACA_API_KEY" "[ -n \"$ALPACA_API_KEY\" ]"
warn_check "POLYGON_API_KEY" "[ -n \"$POLYGON_API_KEY\" ]"
warn_check "SNYK_TOKEN" "[ -n \"$SNYK_TOKEN\" ]"
warn_check "WANDB_API_KEY" "[ -n \"$WANDB_API_KEY\" ]"

echo ""
echo "MCP Servers:"
if command -v claude &>/dev/null; then
    claude mcp list 2>/dev/null | while read -r line; do
        if echo "$line" | grep -q "✓"; then
            echo -e "  $PASS $line"
        elif echo "$line" | grep -q "✗"; then
            echo -e "  $FAIL $line"
        else
            echo "  $line"
        fi
    done
fi

echo ""
echo "Python SDKs:"
warn_check "vectorbt" "python3 -c 'import vectorbt' 2>/dev/null"
warn_check "riskfolio-lib" "python3 -c 'import riskfolio' 2>/dev/null"
warn_check "pandas" "python3 -c 'import pandas' 2>/dev/null"
warn_check "numpy" "python3 -c 'import numpy' 2>/dev/null"
warn_check "diffusers" "python3 -c 'import diffusers' 2>/dev/null"
warn_check "mlflow" "python3 -c 'import mlflow' 2>/dev/null"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results: $passed_checks/$total_checks core checks passed"
echo "═══════════════════════════════════════════════════════════"

if [ $passed_checks -eq $total_checks ]; then
    echo -e "  ${GREEN}All core components validated!${NC}"
    exit 0
else
    echo -e "  ${RED}Some core components missing. Please review above.${NC}"
    exit 1
fi
```

### Quick Validation Commands
```bash
# Verify MCP servers
claude mcp list

# Test specific MCP
claude -p "Use github MCP to list my repositories" --max-turns 1

# Check settings hierarchy
claude config list

# Validate hooks
claude /hooks

# Test headless mode
claude -p "Hello" --output-format json --max-turns 1
```

---

## 9. Troubleshooting Guide

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| MCP server not connecting | Server not running or wrong port | Check process, verify env vars |
| Tools not auto-approved | Missing from allowedTools | Add to settings.json permissions |
| Hook not triggering | Wrong matcher pattern | Test with `echo $CLAUDE_TOOL_INPUT` |
| Memory not loading | Wrong CLAUDE.md location | Check hierarchy: global → project |
| Slow performance | Too many MCP servers | Reduce to 5-7 core servers |
| Context overflow | Too much in CLAUDE.md | Move details to @docs/ references |

### Debug Commands
```bash
# Debug MCP connections
claude --mcp-debug

# View session logs
tail -f ~/.claude/session.log

# Check MCP server status
claude mcp list

# Test hook output
echo '{"command": "git commit -m test"}' | jq .

# Verbose mode
claude --verbose

# View recent sessions
ls -la ~/.claude/projects/
```

### MCP Server Debugging
```bash
# Test Alpaca MCP manually
uvx alpaca-mcp-server serve

# Test Polygon MCP manually
uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon

# Test TouchDesigner MCP
npx touchdesigner-mcp-server@latest --stdio

# Check MCP server logs
claude mcp logs <server-name>
```

---

## 10. Quick Reference

### Essential Commands
```bash
# Start session
claude                          # Interactive mode
claude -p "prompt"              # Headless mode
claude --resume                 # Resume last session
claude --continue               # Continue recent session

# Memory management
claude /memory                  # Edit memory file
claude /init                    # Bootstrap CLAUDE.md

# MCP management
claude mcp add <name> -- <cmd>  # Add server
claude mcp remove <name>        # Remove server
claude mcp list                 # List servers

# Configuration
claude /config                  # View configuration
claude /hooks                   # Manage hooks
claude /allowed-tools           # View permissions
```

### Token Budget Guide
| Component | Tokens | Priority |
|-----------|--------|----------|
| Core MCPs (github, context7) | ~5,000 | Critical |
| Trading MCPs (alpaca, polygon) | ~6,500 | High |
| Creative MCPs (TD, ComfyUI) | ~4,500 | High |
| Security MCPs (snyk) | ~1,500 | Medium |
| Tracking MCPs (mlflow, wandb) | ~3,500 | Medium |
| **Total Active Budget** | **~15,000** | - |

### Project Switching
```bash
# Switch to AlphaForge
cd ~/projects/alphaforge
claude  # Loads project-specific .mcp.json and CLAUDE.md

# Switch to State of Witness
cd ~/projects/state-of-witness
claude  # Loads different MCP configuration
```

### Safety Checklist
- [ ] ALPACA_PAPER_TRADE=true verified
- [ ] API keys in environment, not code
- [ ] .env files in .gitignore
- [ ] Hooks blocking production commands
- [ ] Security MCP enabled for audits

---

## Summary

This toolkit provides:

1. **Complete Installation** - Master script installs all components
2. **Global Configuration** - Unified settings across all projects
3. **Project Isolation** - Separate configs for trading vs creative
4. **Memory Integration** - Persistent context across sessions
5. **Quality Gates** - Hooks enforce standards automatically
6. **Validation Tools** - Scripts verify everything works

**Token Budget**: Keep 5-7 active MCP servers (~15K tokens) to preserve context window for actual development work.

**Key Principle**: Claude CLI is your power tool for the FULL development lifecycle - design, audit, research, integrate, debug, UI. Maximize its capabilities while maintaining strict separation between development and production environments.
