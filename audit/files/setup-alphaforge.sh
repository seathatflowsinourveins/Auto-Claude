#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# AlphaForge Trading System - Project Setup Script
# Claude CLI = EXTERNAL builder, NOT production component
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }

PROJECT_DIR="${1:-$(pwd)}"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   AlphaForge Trading System - Project Setup                   ║"
echo "║   Claude CLI = BUILDER TOOL (External)                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Create directory structure
log_info "Creating project structure in: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"/{.claude/{commands,agents,skills,hooks,rules},docs,src,tests}

# Create .mcp.json
log_info "Creating .mcp.json..."
cat > "$PROJECT_DIR/.mcp.json" << 'EOF'
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
EOF
log_success "Created .mcp.json"

# Create .claude/settings.json
log_info "Creating .claude/settings.json..."
cat > "$PROJECT_DIR/.claude/settings.json" << 'EOF'
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allowedTools": [
      "Read",
      "Write",
      "Edit",
      "MultiEdit",
      "Glob",
      "Grep",
      "Bash(git *)",
      "Bash(python *)",
      "Bash(pytest *)",
      "Bash(cargo *)",
      "Bash(ruff *)",
      "mcp__alpaca_*",
      "mcp__polygon_*",
      "mcp__github_*",
      "mcp__snyk_*",
      "mcp__context7_*"
    ],
    "deny": [
      "Read(.env*)",
      "Read(**/secrets/**)",
      "Bash(*live*)",
      "Bash(*production*)",
      "Bash(*--live*)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | jq -r '.command' 2>/dev/null | grep -qiE 'alpaca.*live|production|--live'; then echo '{\"block\": true, \"message\": \"🛑 BLOCKED: Live/production trading commands not allowed in development.\"}' >&2; exit 2; fi",
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
      },
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
  }
}
EOF
log_success "Created .claude/settings.json"

# Create CLAUDE.md
log_info "Creating CLAUDE.md..."
cat > "$PROJECT_DIR/CLAUDE.md" << 'EOF'
# AlphaForge - Autonomous Trading System

## Project Overview
AlphaForge is a 12-layer autonomous trading system with Rust kill switch.
- Claude CLI = BUILDER TOOL for design/audit/research/integrate/debug/UI
- Claude CLI is NOT part of production architecture
- System runs autonomously after deployment

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    AlphaForge Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 1-3:   Data Ingestion & Preprocessing                │
│  Layer 4-6:   Signal Generation & Strategy                  │
│  Layer 7-9:   Risk Management & Position Sizing             │
│  Layer 10-11: Order Execution & Monitoring                  │
│  Layer 12:    Rust Kill Switch (Safety Critical)            │
└─────────────────────────────────────────────────────────────┘
```

@docs/architecture.md (for detailed architecture)

## Development Commands
```bash
# Build
cargo build --release              # Rust components
pip install -e .                   # Python components

# Test
pytest tests/ -v                   # Python tests
cargo test                         # Rust tests

# Backtest
python -m strategies.backtest      # Run backtests
```

## Trading Safety Rules (CRITICAL)
1. **ALWAYS** use paper trading mode during development
2. **NEVER** commit real API keys or secrets
3. **ALWAYS** review strategy changes before deployment
4. **RUN** full backtest suite before any PR
5. **BLOCK** production commands in hooks

## MCP Usage Guidelines
| Server | Purpose | Notes |
|--------|---------|-------|
| alpaca | Trading execution | PAPER MODE ONLY |
| polygon | Market data | Historical and real-time |
| snyk | Security scanning | Run before deployment |
| github | Version control | PR reviews, issues |
| context7 | Documentation | API references |

## Code Standards
- **Rust**: Follow clippy recommendations, comprehensive tests
- **Python**: Type hints required, Pydantic for validation
- **Testing**: Minimum 80% coverage for strategies
- **Commits**: Conventional commits format

## Risk Management Checklist
- [ ] Position sizing within limits
- [ ] Stop-loss configured
- [ ] Maximum drawdown protection
- [ ] Kill switch tested
- [ ] Backtests passing
EOF
log_success "Created CLAUDE.md"

# Create backtest command
log_info "Creating slash commands..."
cat > "$PROJECT_DIR/.claude/commands/backtest.md" << 'EOF'
Run a comprehensive backtest for the strategy: $ARGUMENTS

## Backtest Procedure

1. **Data Collection**
   - Use Polygon MCP to fetch historical data
   - Validate data quality and completeness
   - Handle splits and dividends

2. **Strategy Execution**
   - Initialize VectorBT engine
   - Configure walk-forward optimization
   - Set realistic slippage and commission

3. **Monte Carlo Simulation**
   - Run 1000 iterations
   - Vary entry/exit timing
   - Assess robustness

4. **Risk Analysis**
   - Calculate Sharpe ratio, Sortino ratio
   - Maximum drawdown analysis
   - Value at Risk (VaR) calculation
   - Expected shortfall

5. **Benchmark Comparison**
   - Compare against SPY buy-and-hold
   - Risk-adjusted return comparison

## Output Requirements
- Summary statistics table
- Equity curve visualization recommendation
- Risk metrics report
- Trade-by-trade analysis
- Optimization suggestions
EOF
log_success "Created backtest.md"

# Create risk-analysis command
cat > "$PROJECT_DIR/.claude/commands/risk-analysis.md" << 'EOF'
Perform comprehensive risk analysis for: $ARGUMENTS

## Risk Assessment

1. **Portfolio Risk Metrics**
   - Value at Risk (VaR) - 95% and 99%
   - Conditional VaR (Expected Shortfall)
   - Maximum Drawdown
   - Calmar Ratio

2. **Position Risk**
   - Individual position sizing
   - Concentration risk
   - Correlation analysis

3. **Market Risk**
   - Beta exposure
   - Sector exposure
   - Factor exposure (momentum, value, etc.)

4. **Stress Testing**
   - 2008 Financial Crisis scenario
   - 2020 COVID crash scenario
   - Interest rate shock scenarios

5. **Recommendations**
   - Position size adjustments
   - Hedging suggestions
   - Diversification improvements

Use riskfolio-lib for calculations where applicable.
EOF
log_success "Created risk-analysis.md"

# Create strategy-audit command
cat > "$PROJECT_DIR/.claude/commands/strategy-audit.md" << 'EOF'
Audit the trading strategy implementation: $ARGUMENTS

## Strategy Audit Checklist

1. **Logic Review**
   - Entry conditions clear and testable
   - Exit conditions properly defined
   - Edge cases handled

2. **Risk Controls**
   - Stop-loss implemented
   - Position sizing correct
   - Maximum positions enforced

3. **Data Handling**
   - Look-ahead bias check
   - Survivorship bias check
   - Data quality validation

4. **Performance**
   - Vectorized operations used
   - Memory efficient
   - Execution speed acceptable

5. **Testing**
   - Unit tests comprehensive
   - Integration tests passing
   - Backtest results consistent

6. **Documentation**
   - Strategy logic documented
   - Parameters explained
   - Risk warnings included

Provide actionable improvements with priority levels.
EOF
log_success "Created strategy-audit.md"

# Create trading-safety rule
cat > "$PROJECT_DIR/.claude/rules/trading-safety.md" << 'EOF'
---
paths:
  - "**/*.py"
  - "**/*.rs"
  - "**/strategies/**"
---

# Trading Safety Rules

## CRITICAL: Paper Trading Only
All trading operations MUST use paper trading mode during development.
The hook system blocks live trading commands automatically.

## API Key Security
- NEVER hardcode API keys
- ALWAYS use environment variables
- Check for accidental key commits

## Position Sizing
- Verify position sizes before execution
- Check against maximum allocation limits
- Validate risk-per-trade calculations

## Kill Switch
- The Rust kill switch must be tested before deployment
- Any changes to kill switch require senior review
- Kill switch failures should halt all trading

## Code Review Requirements
- Strategy changes require full backtest
- Risk parameter changes require risk team approval
- All PRs require at least one reviewer
EOF
log_success "Created trading-safety.md rule"

# Create docs/architecture.md
cat > "$PROJECT_DIR/docs/architecture.md" << 'EOF'
# AlphaForge Architecture

## System Overview

AlphaForge is a 12-layer autonomous trading system designed for safety-critical operations.

## Layer Details

### Layers 1-3: Data Layer
- Market data ingestion (Polygon API)
- Data validation and cleaning
- Feature engineering

### Layers 4-6: Strategy Layer
- Signal generation
- Strategy execution
- Portfolio optimization (Riskfolio-Lib)

### Layers 7-9: Risk Layer
- Position sizing
- Risk calculation
- Exposure monitoring

### Layers 10-11: Execution Layer
- Order management (Alpaca API)
- Execution algorithms
- Fill monitoring

### Layer 12: Safety Layer
- Rust kill switch
- Emergency shutdown
- Position liquidation

## Development Workflow

```
Claude CLI (Builder)          AlphaForge (Production)
      │                              │
      ├── Design ────────────────────┤
      ├── Audit ─────────────────────┤
      ├── Research ──────────────────┤
      ├── Integrate ─────────────────┤
      ├── Debug ─────────────────────┤
      ├── UI Design ─────────────────┤
      │                              │
      └── NOT in production ─────────│──► Runs Autonomously
```

## Safety Architecture

The kill switch (Layer 12) is implemented in Rust for:
- Memory safety guarantees
- Predictable performance
- No garbage collection pauses

## Dependencies

- Python: Strategy, risk, data processing
- Rust: Kill switch, performance-critical paths
- APIs: Alpaca (execution), Polygon (data)
EOF
log_success "Created docs/architecture.md"

# Create .gitignore
cat > "$PROJECT_DIR/.gitignore" << 'EOF'
# Environment
.env
.env.*
.env.local

# Secrets
secrets/
*_key.json
*.pem
*.key

# Claude local settings
.claude/settings.local.json
CLAUDE.local.md

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/

# Rust
target/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Data
data/
*.csv
*.parquet
!tests/fixtures/*.csv
EOF
log_success "Created .gitignore"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   AlphaForge Project Setup Complete!                          ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║   Directory: $PROJECT_DIR${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║   Start development:                                          ║${NC}"
echo -e "${GREEN}║   ${YELLOW}cd $PROJECT_DIR && claude${NC}${GREEN}                          ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║   ⚠️  Remember: ALPACA_PAPER_TRADE=true always!              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
