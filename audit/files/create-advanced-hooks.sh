#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Claude CLI Advanced Hooks Library
# Production-ready hook scripts for quality gates and automation
# ═══════════════════════════════════════════════════════════════════════════════

# This file contains hook scripts to be placed in ~/.claude/hooks/
# Reference them in settings.json

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 1: Pre-Commit Quality Gate
# File: ~/.claude/hooks/precommit-quality.sh
# ═══════════════════════════════════════════════════════════════════════════════
cat > ~/.claude/hooks/precommit-quality.sh << 'HOOK_EOF'
#!/bin/bash
# Pre-commit quality gate - blocks commits until all checks pass
# Usage: Add to PreToolUse hook matching Bash(git commit*)

set -e

echo "🔍 Running pre-commit quality checks..."

ERRORS=0

# Python checks
if find . -name "*.py" -newer .git/index 2>/dev/null | head -1 | grep -q .; then
    echo "  → Checking Python files..."
    
    # Formatting
    if command -v ruff &>/dev/null; then
        if ! ruff format --check . 2>/dev/null; then
            echo "    ❌ Python formatting issues found"
            ERRORS=$((ERRORS + 1))
        else
            echo "    ✓ Python formatting OK"
        fi
        
        # Linting
        if ! ruff check . 2>/dev/null; then
            echo "    ❌ Python linting issues found"
            ERRORS=$((ERRORS + 1))
        else
            echo "    ✓ Python linting OK"
        fi
    fi
    
    # Type checking
    if command -v mypy &>/dev/null; then
        if ! mypy . --ignore-missing-imports 2>/dev/null; then
            echo "    ⚠️ Type hints issues (non-blocking)"
        else
            echo "    ✓ Type hints OK"
        fi
    fi
fi

# TypeScript checks
if find . -name "*.ts" -o -name "*.tsx" -newer .git/index 2>/dev/null | head -1 | grep -q .; then
    echo "  → Checking TypeScript files..."
    
    if command -v tsc &>/dev/null; then
        if ! tsc --noEmit 2>/dev/null; then
            echo "    ❌ TypeScript compilation errors"
            ERRORS=$((ERRORS + 1))
        else
            echo "    ✓ TypeScript OK"
        fi
    fi
fi

# Rust checks
if find . -name "*.rs" -newer .git/index 2>/dev/null | head -1 | grep -q .; then
    echo "  → Checking Rust files..."
    
    if command -v cargo &>/dev/null; then
        if ! cargo clippy --quiet 2>/dev/null; then
            echo "    ❌ Rust clippy warnings"
            ERRORS=$((ERRORS + 1))
        else
            echo "    ✓ Rust clippy OK"
        fi
    fi
fi

# Security scan for secrets
echo "  → Scanning for secrets..."
if grep -rE "(api_key|apikey|secret|password|token)\s*[=:]\s*['\"][^'\"]+['\"]" \
    --include="*.py" --include="*.js" --include="*.ts" --include="*.json" \
    --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=venv . 2>/dev/null | \
    grep -v "example\|placeholder\|your-\|xxx\|CHANGEME" | head -5; then
    echo "    ⚠️ Potential secrets found - please review"
fi

# Run tests if test files changed
if find . -name "*test*.py" -o -name "*.test.ts" -o -name "*.spec.ts" -newer .git/index 2>/dev/null | head -1 | grep -q .; then
    echo "  → Running tests..."
    
    if [ -f "pytest.ini" ] || [ -f "pyproject.toml" ]; then
        if ! pytest --tb=short -q 2>/dev/null; then
            echo "    ❌ Tests failed"
            ERRORS=$((ERRORS + 1))
        else
            echo "    ✓ Tests passed"
        fi
    fi
fi

echo ""
if [ $ERRORS -gt 0 ]; then
    echo "❌ Pre-commit checks failed with $ERRORS error(s)"
    echo '{"block": true, "message": "Pre-commit checks failed. Fix issues before committing."}'
    exit 2
else
    echo "✓ All pre-commit checks passed"
fi
HOOK_EOF
chmod +x ~/.claude/hooks/precommit-quality.sh

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 2: Trading Safety Guard
# File: ~/.claude/hooks/trading-safety.sh
# ═══════════════════════════════════════════════════════════════════════════════
cat > ~/.claude/hooks/trading-safety.sh << 'HOOK_EOF'
#!/bin/bash
# Trading safety guard - blocks dangerous trading operations
# Usage: Add to PreToolUse hook matching mcp__alpaca_* or Bash(*alpaca*)

# Get the tool input
TOOL_INPUT="$CLAUDE_TOOL_INPUT"

# Check for live trading indicators
if echo "$TOOL_INPUT" | grep -qiE 'live|production|real[_-]?money|--live|paper[_-]?trade.*false'; then
    echo '{"block": true, "message": "🛑 BLOCKED: Live trading operations are not allowed in development. Set ALPACA_PAPER_TRADE=true."}'
    exit 2
fi

# Check for large position sizes (configurable threshold)
MAX_SHARES="${MAX_TRADING_SHARES:-1000}"
if echo "$TOOL_INPUT" | grep -oE 'qty["\s:]+([0-9]+)' | grep -oE '[0-9]+' | while read qty; do
    if [ "$qty" -gt "$MAX_SHARES" ]; then
        echo '{"block": true, "message": "🛑 BLOCKED: Position size '$qty' exceeds maximum '$MAX_SHARES' shares."}'
        exit 2
    fi
done; then
    exit 2
fi

# Warn on market orders during volatile hours
HOUR=$(date +%H)
if [ "$HOUR" -lt 10 ] || [ "$HOUR" -gt 15 ]; then
    if echo "$TOOL_INPUT" | grep -qi 'market'; then
        echo '{"feedback": "⚠️ Market order during potentially volatile hours. Consider limit orders."}'
    fi
fi

# Log trading operation
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Trading operation: $(echo "$TOOL_INPUT" | head -c 200)" >> ~/.claude/trading.log

echo '{"feedback": "📊 Trading operation - Paper mode"}'
HOOK_EOF
chmod +x ~/.claude/hooks/trading-safety.sh

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 3: Auto-Format on Write
# File: ~/.claude/hooks/auto-format.sh
# ═══════════════════════════════════════════════════════════════════════════════
cat > ~/.claude/hooks/auto-format.sh << 'HOOK_EOF'
#!/bin/bash
# Auto-format files after write
# Usage: Add to PostToolUse hook matching Write(*)

FILE="$1"
if [ -z "$FILE" ]; then
    FILE=$(echo "$CLAUDE_TOOL_OUTPUT" | jq -r '.file // .path // empty' 2>/dev/null)
fi

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    exit 0
fi

EXT="${FILE##*.}"

case "$EXT" in
    py)
        command -v ruff &>/dev/null && ruff format "$FILE" --quiet 2>/dev/null
        command -v ruff &>/dev/null && ruff check "$FILE" --fix --quiet 2>/dev/null
        ;;
    ts|tsx|js|jsx)
        command -v prettier &>/dev/null && prettier --write "$FILE" --log-level silent 2>/dev/null
        ;;
    rs)
        command -v rustfmt &>/dev/null && rustfmt "$FILE" 2>/dev/null
        ;;
    json)
        command -v jq &>/dev/null && jq '.' "$FILE" > "$FILE.tmp" 2>/dev/null && mv "$FILE.tmp" "$FILE"
        ;;
    yaml|yml)
        command -v yq &>/dev/null && yq -i '.' "$FILE" 2>/dev/null
        ;;
    md)
        command -v prettier &>/dev/null && prettier --write "$FILE" --prose-wrap always --log-level silent 2>/dev/null
        ;;
esac

exit 0
HOOK_EOF
chmod +x ~/.claude/hooks/auto-format.sh

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 4: Session Learning Capture
# File: ~/.claude/hooks/capture-learnings.sh
# ═══════════════════════════════════════════════════════════════════════════════
cat > ~/.claude/hooks/capture-learnings.sh << 'HOOK_EOF'
#!/bin/bash
# Capture session learnings on Stop
# Usage: Add to Stop hook

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
SESSION_LOG="$PROJECT_DIR/.claude/session-learnings.md"
GLOBAL_LOG="$HOME/.claude/learnings.md"

# Create directories if needed
mkdir -p "$(dirname "$SESSION_LOG")" 2>/dev/null
mkdir -p "$(dirname "$GLOBAL_LOG")" 2>/dev/null

# Append session marker
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
{
    echo ""
    echo "---"
    echo "## Session: $TIMESTAMP"
    echo "Project: $PROJECT_DIR"
    echo ""
} >> "$GLOBAL_LOG"

echo "[$(date)] Session ended - learnings logged" >> ~/.claude/session.log
HOOK_EOF
chmod +x ~/.claude/hooks/capture-learnings.sh

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 5: Strategy Change Backtest Reminder
# File: ~/.claude/hooks/strategy-reminder.sh
# ═══════════════════════════════════════════════════════════════════════════════
cat > ~/.claude/hooks/strategy-reminder.sh << 'HOOK_EOF'
#!/bin/bash
# Remind to run backtest when strategy files change
# Usage: Add to PostToolUse hook matching Write(**/strategies/*.py)

FILE="$1"
if [ -z "$FILE" ]; then
    FILE=$(echo "$CLAUDE_TOOL_OUTPUT" | jq -r '.file // .path // empty' 2>/dev/null)
fi

if echo "$FILE" | grep -qE 'strateg|signal|indicator|backtest'; then
    echo '{"feedback": "💡 Strategy file modified. Remember to:\n1. Run backtest before committing\n2. Compare against baseline\n3. Check risk metrics"}'
fi
HOOK_EOF
chmod +x ~/.claude/hooks/strategy-reminder.sh

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 6: Experiment Tracking Enforcer
# File: ~/.claude/hooks/experiment-tracking.sh
# ═══════════════════════════════════════════════════════════════════════════════
cat > ~/.claude/hooks/experiment-tracking.sh << 'HOOK_EOF'
#!/bin/bash
# Enforce experiment tracking for ML files
# Usage: Add to PostToolUse hook matching Write(**/ml/*.py) or Write(**/experiments/*.py)

FILE="$1"
if [ -z "$FILE" ]; then
    FILE=$(echo "$CLAUDE_TOOL_OUTPUT" | jq -r '.file // .path // empty' 2>/dev/null)
fi

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    exit 0
fi

# Check if file uses ML imports but doesn't have mlflow/wandb
if grep -qE 'import torch|import tensorflow|from sklearn|import keras' "$FILE" 2>/dev/null; then
    if ! grep -qE 'import mlflow|import wandb' "$FILE" 2>/dev/null; then
        echo '{"feedback": "⚠️ ML code detected without experiment tracking. Consider adding MLflow or W&B logging."}'
    fi
fi
HOOK_EOF
chmod +x ~/.claude/hooks/experiment-tracking.sh

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 7: TouchDesigner Connection Check
# File: ~/.claude/hooks/td-connection-check.sh
# ═══════════════════════════════════════════════════════════════════════════════
cat > ~/.claude/hooks/td-connection-check.sh << 'HOOK_EOF'
#!/bin/bash
# Check TouchDesigner connection before operations
# Usage: Add to PreToolUse hook matching mcp__touchdesigner_*

TD_HOST="${TD_HOST:-127.0.0.1}"
TD_PORT="${TD_PORT:-9981}"

# Quick connection check
if ! nc -z "$TD_HOST" "$TD_PORT" 2>/dev/null; then
    echo '{"feedback": "⚠️ TouchDesigner WebServer not responding on '$TD_HOST:$TD_PORT'. Ensure TD is running with WebServer DAT enabled."}'
fi
HOOK_EOF
chmod +x ~/.claude/hooks/td-connection-check.sh

echo "✓ Advanced hooks library created in ~/.claude/hooks/"
echo ""
echo "Hooks created:"
ls -la ~/.claude/hooks/*.sh 2>/dev/null
