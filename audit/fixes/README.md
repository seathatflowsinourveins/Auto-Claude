# Claude CLI Toolkit V10 OPTIMIZED - Fix Scripts

> **ONE-COMMAND deployment of all improvements identified in ULTRA_DETAILED_SYNTHESIS_REPORT_V2.md**

---

## Quick Start

```bash
# Preview all changes first (RECOMMENDED)
./master-fix-all.sh --dry-run

# Apply all fixes
./master-fix-all.sh

# Trading project only
./master-fix-all.sh --trading-only

# Creative/ML project only
./master-fix-all.sh --creative-only
```

---

## Script Inventory

| Script | Purpose | Lines | Impact |
|--------|---------|-------|--------|
| `master-fix-all.sh` | **Master orchestrator** - runs all fixes in sequence | ~550 | All systems |
| `fix-memory-system.sh` | Memory system migration | ~280 | Memory layer |
| `fix-trading-safety.sh` | Trading safety hooks | ~500 | Trading safety |
| `create-complete-hooks.sh` | All 15 production hooks | ~700 | Workflow automation |

---

## What Gets Fixed

### 1. Memory System (`fix-memory-system.sh`)

**Problem**: Multiple broken/redundant memory servers causing token waste and errors.

**Removes**:
- `@modelcontextprotocol/server-memory` (GitHub issues #1018, #3074, #3144)
- `graphiti-mcp` (duplicate functionality)
- `qdrant-mcp` (superseded by claude-mem)

**Installs**:
- `claude-mem` (primary cross-session memory)
- `episodic-memory` (backup conversation search)

**Token Savings**: ~4,000 tokens per session

---

### 2. Trading Safety (`fix-trading-safety.sh`)

**Problem**: Incomplete trading safety patterns leaving gaps for accidental live trading.

**Deploys 4 Safety Hooks**:

| Hook | Purpose |
|------|---------|
| `block-live-trading.sh` | Blocks any live/production trading operations |
| `limit-position-size.sh` | Enforces position size limits |
| `verify-paper-trading.sh` | Ensures ALPACA_PAPER_TRADE=true |
| `track-daily-loss.sh` | Tracks cumulative P&L, blocks at limit |

**Pattern Coverage**:
- Live trading keywords (20+ patterns)
- Dangerous API endpoints
- Withdrawal/transfer operations
- Credential exposure prevention

---

### 3. Complete Hooks (`create-complete-hooks.sh`)

**Problem**: Only 7 hooks in original toolkit, missing many workflow automations.

**Deploys 15 Production Hooks**:

#### Code Quality (4 hooks)
| Hook | Trigger | Action |
|------|---------|--------|
| `auto-format-python.sh` | PostToolUse Write(*.py) | ruff format |
| `auto-format-typescript.sh` | PostToolUse Write(*.ts) | prettier |
| `precommit-quality.sh` | PreToolUse git commit | pyright + ruff check |
| `auto-test.sh` | PostToolUse Write(src/*.py) | pytest related tests |

#### Security (3 hooks)
| Hook | Trigger | Action |
|------|---------|--------|
| `detect-secrets.sh` | PreToolUse Write(*) | Scan for API keys, tokens |
| `block-dangerous-git.sh` | PreToolUse Bash(git *) | Block force push, hard reset |
| `security-audit-reminder.sh` | PostToolUse Write(*auth*) | Remind to audit |

#### Trading Safety (2 hooks)
| Hook | Trigger | Action |
|------|---------|--------|
| `trading-safety.sh` | PreToolUse Bash(*trade*) | Block live trading |
| `strategy-reminder.sh` | SessionStart | Show trading limits |

#### Creative/ML (3 hooks)
| Hook | Trigger | Action |
|------|---------|--------|
| `experiment-tracking.sh` | PostToolUse Bash(*train*) | Log to MLflow/W&B |
| `td-connection-check.sh` | SessionStart | Verify TD MCP connection |
| `glsl-version-check.sh` | PreToolUse Write(*.glsl) | TD 2025+ compatibility |

#### Session Management (3 hooks)
| Hook | Trigger | Action |
|------|---------|--------|
| `capture-learnings.sh` | Stop | Save session learnings |
| `doc-sync.sh` | PostToolUse Write(*api*) | Remind to update docs |
| `memory-reminder.sh` | SessionStart | Remind to search memory |

---

## Configuration Created

### .gitignore Security Rules

The scripts create comprehensive .gitignore entries:

```
# Credentials & Secrets
.env
.env.*
*.pem, *.key, *.crt
alpaca*.json, polygon*.json

# Claude Code Sensitive
.claude/settings.local.json
.claude/trading_state/
.claude/learnings/

# Trading Specific
backtest_results/
live_trading_logs/
order_history/
strategies/private/

# Creative/ML Specific
*.ckpt, *.safetensors
mlruns/, wandb/
*.toe.bak
```

### MCP Configuration

Project-type-aware `.mcp.json`:

**Trading Project**:
- github, alpaca, polygon, context7, snyk

**Creative Project**:
- github, touchdesigner, mlflow, wandb, context7

---

## Verification

After running fixes:

```bash
# Check hooks
ls -la .claude/hooks/

# Verify MCP servers
claude mcp list

# Test trading safety (should be blocked)
# In Claude: "Place a live trade for SPY"

# Check environment
echo $ALPACA_PAPER_TRADE  # Should be "true"
```

---

## Rollback

All scripts create timestamped backups before making changes:

```bash
# Find your backup
ls -la ~/.claude/backups/

# Restore global config
cp -r ~/.claude/backups/master_TIMESTAMP/global_claude_config/* ~/.claude/

# Restore project config
cp -r ~/.claude/backups/master_TIMESTAMP/project_claude_config/* .claude/
```

---

## Script Options

All scripts support these common options:

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview changes without applying |
| `--verbose` | Show detailed output |
| `--help` | Show help message |

Additional options for `master-fix-all.sh`:

| Option | Description |
|--------|-------------|
| `--trading-only` | Only apply trading-related fixes |
| `--creative-only` | Only apply creative/ML-related fixes |

---

## Token Budget Impact

| Before | After | Savings |
|--------|-------|---------|
| ~25,000 tokens | ~15,000 tokens | **40%** |

Breakdown:
- Memory system: -4,000 tokens
- Redundant MCPs removed: -3,000 tokens
- Optimized configurations: -3,000 tokens

---

## Related Documentation

- **ULTRA_DETAILED_SYNTHESIS_REPORT_V2.md** - Complete analysis and findings
- **claude_cli_toolkit_quickref.md** - Quick reference card
- **ULTIMATE_CLAUDE_CLI_TOOLKIT_MASTER.md** - Full consolidated guide

---

*Generated: January 2026 | Claude CLI Toolkit V10 OPTIMIZED*
