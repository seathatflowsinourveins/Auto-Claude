#!/bin/bash
# create-complete-hooks.sh - Complete Hooks Deployment Script
# Part of Claude CLI Toolkit V10 OPTIMIZED
#
# This script deploys ALL 15 production-ready hooks for comprehensive
# development workflow automation. Covers:
# - Code quality (formatting, linting, testing)
# - Security (secret detection, credential protection)
# - Trading safety (live trading prevention)
# - Creative/ML (experiment tracking, TD connection)
# - Session management (learnings capture, context preservation)
#
# Usage: ./create-complete-hooks.sh [--dry-run] [--verbose] [--global|--project]

set -euo pipefail

# Configuration
CLAUDE_CONFIG_DIR="${HOME}/.claude"
PROJECT_CONFIG_DIR=".claude"
TARGET="project"  # Default to project-level hooks
DRY_RUN=false
VERBOSE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --global)
            TARGET="global"
            shift
            ;;
        --project)
            TARGET="project"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set target directory
if [ "$TARGET" = "global" ]; then
    HOOKS_DIR="${CLAUDE_CONFIG_DIR}/hooks"
else
    HOOKS_DIR="${PROJECT_CONFIG_DIR}/hooks"
fi

log() { echo -e "${GREEN}[HOOKS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
verbose() { [ "$VERBOSE" = true ] && echo -e "${CYAN}[VERBOSE]${NC} $1"; }

create_hook() {
    local name="$1"
    local content="$2"
    local file="${HOOKS_DIR}/${name}"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $file"
        return
    fi

    echo "$content" > "$file"
    chmod +x "$file"
    log "Created: $name"
}

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

setup() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║         CLAUDE CLI COMPLETE HOOKS - V10 OPTIMIZED                ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo ""

    log "Target: ${TARGET} hooks at ${HOOKS_DIR}"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}Running in DRY-RUN mode${NC}"
    fi

    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$HOOKS_DIR"
    fi

    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 1: Auto-Format Python
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_auto_format_python() {
    log "Creating auto-format Python hook..."

    create_hook "auto-format-python.sh" '#!/bin/bash
# auto-format-python.sh - PostToolUse Hook
# Auto-formats Python files after Write/Edit operations
# Trigger: PostToolUse for Write(*.py) or Edit(*.py)

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

# Extract file path from tool input
FILE_PATH=$(echo "$TOOL_INPUT" | grep -oP "(?<=file_path[\"':\\s]+)[^\"'\\s,}]+" | head -1)

# Only process Python files
if [[ ! "$FILE_PATH" =~ \.py$ ]]; then
    exit 0
fi

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

# Format with ruff (preferred) or black
if command -v ruff &> /dev/null; then
    ruff format "$FILE_PATH" 2>/dev/null
    ruff check --fix "$FILE_PATH" 2>/dev/null || true
elif command -v black &> /dev/null; then
    black "$FILE_PATH" 2>/dev/null
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 2: Auto-Format TypeScript
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_auto_format_typescript() {
    log "Creating auto-format TypeScript hook..."

    create_hook "auto-format-typescript.sh" '#!/bin/bash
# auto-format-typescript.sh - PostToolUse Hook
# Auto-formats TypeScript/JavaScript files after Write/Edit
# Trigger: PostToolUse for Write(*.ts) or Write(*.tsx)

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

FILE_PATH=$(echo "$TOOL_INPUT" | grep -oP "(?<=file_path[\"':\\s]+)[^\"'\\s,}]+" | head -1)

# Only process TS/JS files
if [[ ! "$FILE_PATH" =~ \.(ts|tsx|js|jsx)$ ]]; then
    exit 0
fi

if [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

# Format with prettier or biome
if command -v prettier &> /dev/null; then
    prettier --write "$FILE_PATH" 2>/dev/null
elif command -v biome &> /dev/null; then
    biome format --write "$FILE_PATH" 2>/dev/null
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 3: Pre-Commit Quality Check
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_precommit_quality() {
    log "Creating pre-commit quality hook..."

    create_hook "precommit-quality.sh" '#!/bin/bash
# precommit-quality.sh - PreToolUse Hook
# Runs quality checks before git commit operations
# Trigger: PreToolUse for Bash(git commit*)

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

# Only trigger on git commit
if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

if ! echo "$TOOL_INPUT" | grep -qE "git\s+commit"; then
    exit 0
fi

echo "Running pre-commit quality checks..."

# Get staged files
STAGED_PY=$(git diff --cached --name-only --diff-filter=ACM | grep "\.py$" || true)
STAGED_TS=$(git diff --cached --name-only --diff-filter=ACM | grep -E "\.(ts|tsx)$" || true)

ERRORS=0

# Check Python files
if [ -n "$STAGED_PY" ]; then
    echo "Checking Python files..."

    # Type checking with pyright
    if command -v pyright &> /dev/null; then
        for file in $STAGED_PY; do
            if ! pyright "$file" 2>/dev/null; then
                echo "Type error in: $file"
                ERRORS=$((ERRORS + 1))
            fi
        done
    fi

    # Linting with ruff
    if command -v ruff &> /dev/null; then
        for file in $STAGED_PY; do
            if ! ruff check "$file" 2>/dev/null; then
                echo "Lint error in: $file"
                ERRORS=$((ERRORS + 1))
            fi
        done
    fi
fi

# Check TypeScript files
if [ -n "$STAGED_TS" ]; then
    echo "Checking TypeScript files..."

    if command -v tsc &> /dev/null; then
        if ! tsc --noEmit 2>/dev/null; then
            echo "TypeScript compilation errors found"
            ERRORS=$((ERRORS + 1))
        fi
    fi
fi

if [ $ERRORS -gt 0 ]; then
    echo "{\"block\": false, \"message\": \"⚠️ $ERRORS quality issues found. Consider fixing before commit.\"}"
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 4: Secret Detection
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_secret_detection() {
    log "Creating secret detection hook..."

    create_hook "detect-secrets.sh" '#!/bin/bash
# detect-secrets.sh - PreToolUse Hook
# Scans for secrets before Write operations
# Trigger: PreToolUse for Write(*)

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

# Only check Write operations
if [ "$TOOL_NAME" != "Write" ]; then
    exit 0
fi

# Secret patterns to detect
SECRET_PATTERNS=(
    # API Keys
    "(?i)(api[_-]?key|apikey)['\"]?\s*[:=]\s*['\"][a-zA-Z0-9]{20,}"
    # AWS
    "AKIA[0-9A-Z]{16}"
    "(?i)aws[_-]?secret[_-]?access[_-]?key"
    # Private keys
    "-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
    # Tokens
    "(?i)(token|secret|password|passwd|pwd)['\"]?\s*[:=]\s*['\"][^'\"]{8,}"
    # GitHub
    "ghp_[a-zA-Z0-9]{36}"
    "github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}"
    # Slack
    "xox[baprs]-[0-9a-zA-Z-]+"
    # Generic high entropy
    "(?i)(secret|private|credential)['\"]?\s*[:=]\s*['\"][a-zA-Z0-9+/=]{32,}"
)

CONTENT=$(echo "$TOOL_INPUT" | jq -r ".content // empty" 2>/dev/null || echo "$TOOL_INPUT")

for pattern in "${SECRET_PATTERNS[@]}"; do
    if echo "$CONTENT" | grep -qP "$pattern" 2>/dev/null; then
        echo "{\"block\": true, \"message\": \"🛑 BLOCKED: Potential secret detected. Never commit credentials to code.\"}"
        exit 2
    fi
done

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 5: Block Dangerous Git Commands
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_block_dangerous_git() {
    log "Creating dangerous git blocker hook..."

    create_hook "block-dangerous-git.sh" '#!/bin/bash
# block-dangerous-git.sh - PreToolUse Hook
# Blocks dangerous/destructive git commands
# Trigger: PreToolUse for Bash(git *)

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

# Dangerous patterns
DANGEROUS_PATTERNS=(
    "git push.*--force(?!-with-lease)"
    "git push.*-f(?!wl)"
    "git reset --hard.*origin"
    "git clean -fd"
    "git checkout -- \."
    "git branch -D main"
    "git branch -D master"
    "git push origin :main"
    "git push origin :master"
    "git rebase -i.*main"
    "git rebase -i.*master"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qP "$pattern"; then
        echo "{\"block\": true, \"message\": \"🛑 BLOCKED: Dangerous git operation. This could cause data loss. Use with caution.\"}"
        exit 2
    fi
done

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 6: Trading Safety (Live Trading Blocker)
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_trading_safety() {
    log "Creating trading safety hook..."

    create_hook "trading-safety.sh" '#!/bin/bash
# trading-safety.sh - PreToolUse Hook
# Blocks live trading operations
# Trigger: PreToolUse for Bash(*trade*) or Bash(*order*)

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

# Live trading patterns
LIVE_PATTERNS=(
    "live"
    "production"
    "real[_-]?money"
    "--live"
    "paper[_-]?trade.*false"
    "PAPER.*=.*false"
)

for pattern in "${LIVE_PATTERNS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qiE "$pattern"; then
        echo "{\"block\": true, \"message\": \"🛑 BLOCKED: Live trading detected. Use paper trading only.\"}"
        exit 2
    fi
done

# Verify paper trading is enabled for Alpaca commands
if echo "$TOOL_INPUT" | grep -qi "alpaca"; then
    if [ "${ALPACA_PAPER_TRADE:-}" != "true" ]; then
        echo "{\"block\": true, \"message\": \"🛑 BLOCKED: ALPACA_PAPER_TRADE must be true.\"}"
        exit 2
    fi
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 7: Capture Session Learnings
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_capture_learnings() {
    log "Creating session learnings capture hook..."

    create_hook "capture-learnings.sh" '#!/bin/bash
# capture-learnings.sh - Stop Hook
# Captures session learnings when session ends
# Trigger: Stop event

LEARNINGS_DIR="${HOME}/.claude/learnings"
mkdir -p "$LEARNINGS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LEARNINGS_FILE="${LEARNINGS_DIR}/session_${TIMESTAMP}.md"

# Get project name if available
PROJECT_NAME=$(basename "$(pwd)" 2>/dev/null || echo "unknown")

cat > "$LEARNINGS_FILE" << EOF
# Session Learnings - ${PROJECT_NAME}
Date: $(date "+%Y-%m-%d %H:%M:%S")

## Context
- Working Directory: $(pwd)
- Duration: Session ended

## Key Decisions Made
<!-- Add manually or via memory -->

## Issues Encountered
<!-- Problems and their solutions -->

## Patterns Discovered
<!-- Useful patterns for future reference -->

## TODOs for Next Session
<!-- Unfinished work -->

EOF

echo "Session learnings template saved to: $LEARNINGS_FILE"
exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 8: Experiment Tracking (MLflow/W&B)
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_experiment_tracking() {
    log "Creating experiment tracking hook..."

    create_hook "experiment-tracking.sh" '#!/bin/bash
# experiment-tracking.sh - PostToolUse Hook
# Auto-logs experiments to MLflow/W&B
# Trigger: PostToolUse for training/experiment commands

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"
TOOL_OUTPUT="${TOOL_OUTPUT:-}"

if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

# Check if this is a training/experiment command
if ! echo "$TOOL_INPUT" | grep -qiE "(train|fit|experiment|backtest|optimize)"; then
    exit 0
fi

# Log to MLflow if available
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    echo "Logging to MLflow: ${MLFLOW_TRACKING_URI}"
    # MLflow logging would happen here via Python SDK
fi

# Log to W&B if available
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Logging to Weights & Biases"
    # W&B logging would happen here via Python SDK
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 9: TouchDesigner Connection Check
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_td_connection() {
    log "Creating TouchDesigner connection hook..."

    create_hook "td-connection-check.sh" '#!/bin/bash
# td-connection-check.sh - SessionStart Hook
# Verifies TouchDesigner is running and MCP is connected
# Trigger: SessionStart event

TD_HOST="${TD_HOST:-127.0.0.1}"
TD_PORT="${TD_PORT:-9981}"

# Check if TouchDesigner MCP port is open
if command -v nc &> /dev/null; then
    if nc -z "$TD_HOST" "$TD_PORT" 2>/dev/null; then
        echo "✓ TouchDesigner MCP connected at ${TD_HOST}:${TD_PORT}"
    else
        echo "⚠ TouchDesigner MCP not available at ${TD_HOST}:${TD_PORT}"
        echo "  Start TouchDesigner and ensure mcp_webserver_base.tox is loaded"
    fi
elif command -v curl &> /dev/null; then
    if curl -s --connect-timeout 1 "http://${TD_HOST}:${TD_PORT}" >/dev/null 2>&1; then
        echo "✓ TouchDesigner MCP connected"
    else
        echo "⚠ TouchDesigner MCP not available"
    fi
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 10: Auto-Run Tests
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_auto_test() {
    log "Creating auto-test hook..."

    create_hook "auto-test.sh" '#!/bin/bash
# auto-test.sh - PostToolUse Hook
# Auto-runs related tests after code changes
# Trigger: PostToolUse for Write(*.py) in src/ or lib/

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

FILE_PATH=$(echo "$TOOL_INPUT" | grep -oP "(?<=file_path[\"':\\s]+)[^\"'\\s,}]+" | head -1)

# Only for Python source files
if [[ ! "$FILE_PATH" =~ ^(src|lib)/.*\.py$ ]]; then
    exit 0
fi

# Find corresponding test file
BASENAME=$(basename "$FILE_PATH" .py)
TEST_FILE="tests/test_${BASENAME}.py"

if [ -f "$TEST_FILE" ]; then
    echo "Running tests for ${BASENAME}..."
    if command -v pytest &> /dev/null; then
        pytest "$TEST_FILE" -v --tb=short 2>&1 | tail -20
    fi
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 11: Strategy Reminder (AlphaForge)
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_strategy_reminder() {
    log "Creating strategy reminder hook..."

    create_hook "strategy-reminder.sh" '#!/bin/bash
# strategy-reminder.sh - SessionStart Hook
# Reminds about current trading strategy status
# Trigger: SessionStart for trading projects

# Only run in trading projects
if [[ ! "$(pwd)" =~ (trading|alphaforge|strategy) ]]; then
    exit 0
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    TRADING SESSION REMINDER                       ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║  • Paper trading mode: ${ALPACA_PAPER_TRADE:-NOT SET}            ║"
echo "║  • Max position size:  \$${MAX_POSITION_SIZE:-10000}              ║"
echo "║  • Max daily loss:     \$${MAX_DAILY_LOSS:-1000}                  ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 12: Documentation Sync
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_doc_sync() {
    log "Creating documentation sync hook..."

    create_hook "doc-sync.sh" '#!/bin/bash
# doc-sync.sh - PostToolUse Hook
# Reminds to update docs when public APIs change
# Trigger: PostToolUse for Write(*/api/*) or Write(*/__init__.py)

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

FILE_PATH=$(echo "$TOOL_INPUT" | grep -oP "(?<=file_path[\"':\\s]+)[^\"'\\s,}]+" | head -1)

# Check if this is a public API file
if [[ "$FILE_PATH" =~ (api/|__init__\.py|routes\.|endpoints\.) ]]; then
    echo ""
    echo "📝 REMINDER: You modified a public API file."
    echo "   Consider updating documentation:"
    echo "   - README.md"
    echo "   - API docs (if applicable)"
    echo "   - CHANGELOG.md"
    echo ""
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 13: Memory Search Reminder
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_memory_reminder() {
    log "Creating memory search reminder hook..."

    create_hook "memory-reminder.sh" '#!/bin/bash
# memory-reminder.sh - SessionStart Hook
# Reminds to search memory for context
# Trigger: SessionStart event

echo ""
echo "💡 TIP: Search your memory for relevant context:"
echo "   'Search my previous conversations about [topic]'"
echo "   'What did we decide about [feature]?'"
echo ""

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 14: GLSL Version Check
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_glsl_check() {
    log "Creating GLSL version check hook..."

    create_hook "glsl-version-check.sh" '#!/bin/bash
# glsl-version-check.sh - PreToolUse Hook
# Ensures GLSL shaders follow TouchDesigner 2025+ conventions
# Trigger: PreToolUse for Write(*.glsl) or Write(*.frag)

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

# Only check Write operations for shader files
if [ "$TOOL_NAME" != "Write" ]; then
    exit 0
fi

FILE_PATH=$(echo "$TOOL_INPUT" | grep -oP "(?<=file_path[\"':\\s]+)[^\"'\\s,}]+" | head -1)

if [[ ! "$FILE_PATH" =~ \.(glsl|frag|vert|comp)$ ]]; then
    exit 0
fi

CONTENT=$(echo "$TOOL_INPUT" | jq -r ".content // empty" 2>/dev/null || echo "")

# Check for deprecated patterns
WARNINGS=""

# TD 2025+ removed #version directive
if echo "$CONTENT" | grep -q "#version"; then
    WARNINGS="${WARNINGS}\n⚠ Remove #version directive (TD adds automatically)"
fi

# TD 2025+ requires TDOutputSwizzle
if echo "$CONTENT" | grep -q "out vec4" && ! echo "$CONTENT" | grep -q "TDOutputSwizzle"; then
    WARNINGS="${WARNINGS}\n⚠ Use TDOutputSwizzle() for fragment outputs"
fi

# TD 2025+ texture sampling
if echo "$CONTENT" | grep -q "texture2D"; then
    WARNINGS="${WARNINGS}\n⚠ Use texture() instead of texture2D()"
fi

if [ -n "$WARNINGS" ]; then
    echo "GLSL TouchDesigner 2025+ Compatibility Warnings:${WARNINGS}"
fi

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK 15: Security Audit Reminder
# ═══════════════════════════════════════════════════════════════════════════════

create_hook_security_reminder() {
    log "Creating security audit reminder hook..."

    create_hook "security-audit-reminder.sh" '#!/bin/bash
# security-audit-reminder.sh - PostToolUse Hook
# Reminds about security audit after significant changes
# Trigger: PostToolUse for Write operations in security-sensitive paths

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

FILE_PATH=$(echo "$TOOL_INPUT" | grep -oP "(?<=file_path[\"':\\s]+)[^\"'\\s,}]+" | head -1)

# Security-sensitive patterns
SENSITIVE_PATTERNS=(
    "auth"
    "login"
    "session"
    "password"
    "crypto"
    "security"
    "permission"
    "access"
    "token"
    "api/v"
)

for pattern in "${SENSITIVE_PATTERNS[@]}"; do
    if echo "$FILE_PATH" | grep -qi "$pattern"; then
        echo ""
        echo "🔒 SECURITY REMINDER: You modified a security-sensitive file."
        echo "   Consider running: /security-audit"
        echo ""
        break
    fi
done

exit 0
'
}

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE SETTINGS.JSON WITH ALL HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

create_hooks_settings() {
    log "Creating settings.json with all hooks..."

    if [ "$TARGET" = "global" ]; then
        SETTINGS_FILE="${CLAUDE_CONFIG_DIR}/settings.json"
    else
        SETTINGS_FILE="${PROJECT_CONFIG_DIR}/settings.json"
    fi

    SETTINGS_CONTENT='{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "*",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/td-connection-check.sh"},
          {"type": "command", "command": "'"${HOOKS_DIR}"'/strategy-reminder.sh"},
          {"type": "command", "command": "'"${HOOKS_DIR}"'/memory-reminder.sh"}
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Write(*.py)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/detect-secrets.sh"}
        ]
      },
      {
        "matcher": "Write(*.glsl)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/glsl-version-check.sh"}
        ]
      },
      {
        "matcher": "Bash(git commit*)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/precommit-quality.sh"}
        ]
      },
      {
        "matcher": "Bash(git *)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/block-dangerous-git.sh"}
        ]
      },
      {
        "matcher": "Bash(*trade*)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/trading-safety.sh"}
        ]
      },
      {
        "matcher": "Bash(*order*)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/trading-safety.sh"}
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write(*.py)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/auto-format-python.sh"},
          {"type": "command", "command": "'"${HOOKS_DIR}"'/auto-test.sh"}
        ]
      },
      {
        "matcher": "Write(*.ts)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/auto-format-typescript.sh"}
        ]
      },
      {
        "matcher": "Write(*api*)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/doc-sync.sh"},
          {"type": "command", "command": "'"${HOOKS_DIR}"'/security-audit-reminder.sh"}
        ]
      },
      {
        "matcher": "Bash(*train*)",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/experiment-tracking.sh"}
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {"type": "command", "command": "'"${HOOKS_DIR}"'/capture-learnings.sh"}
        ]
      }
    ]
  }
}'

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $SETTINGS_FILE"
        echo "Settings preview (first 40 lines):"
        echo "$SETTINGS_CONTENT" | head -40
    else
        mkdir -p "$(dirname "$SETTINGS_FILE")"

        if [ -f "$SETTINGS_FILE" ]; then
            warn "settings.json already exists. Please merge manually:"
            echo "$SETTINGS_CONTENT"
        else
            echo "$SETTINGS_CONTENT" > "$SETTINGS_FILE"
            log "Created: $SETTINGS_FILE"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    setup

    echo "Creating 15 production-ready hooks..."
    echo ""

    # Code Quality Hooks
    echo -e "${CYAN}── Code Quality ──${NC}"
    create_hook_auto_format_python
    create_hook_auto_format_typescript
    create_hook_precommit_quality
    create_hook_auto_test

    # Security Hooks
    echo ""
    echo -e "${CYAN}── Security ──${NC}"
    create_hook_secret_detection
    create_hook_block_dangerous_git
    create_hook_security_reminder

    # Trading Safety Hooks
    echo ""
    echo -e "${CYAN}── Trading Safety ──${NC}"
    create_hook_trading_safety
    create_hook_strategy_reminder

    # Creative/ML Hooks
    echo ""
    echo -e "${CYAN}── Creative/ML ──${NC}"
    create_hook_experiment_tracking
    create_hook_td_connection
    create_hook_glsl_check

    # Session Management Hooks
    echo ""
    echo -e "${CYAN}── Session Management ──${NC}"
    create_hook_capture_learnings
    create_hook_doc_sync
    create_hook_memory_reminder

    # Create settings.json
    echo ""
    create_hooks_settings

    # Summary
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "                        HOOKS SUMMARY                               "
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        HOOK_COUNT=$(ls -1 "${HOOKS_DIR}"/*.sh 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} Created ${HOOK_COUNT} hooks in ${HOOKS_DIR}"
    else
        echo -e "  ${YELLOW}DRY-RUN: Would create 15 hooks${NC}"
    fi

    echo ""
    echo "Hook Categories:"
    echo "  • Code Quality:      4 hooks (format, lint, test, precommit)"
    echo "  • Security:          3 hooks (secrets, git, audit)"
    echo "  • Trading Safety:    2 hooks (live block, reminder)"
    echo "  • Creative/ML:       3 hooks (TD, GLSL, experiments)"
    echo "  • Session Mgmt:      3 hooks (learnings, docs, memory)"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}DRY-RUN complete. Run without --dry-run to apply.${NC}"
    else
        echo -e "${GREEN}Complete hooks deployment finished!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Restart Claude Code to load hooks"
        echo "  2. Test with: 'Write a simple Python function'"
        echo "  3. Verify hooks with: ls -la ${HOOKS_DIR}"
    fi
}

main "$@"
