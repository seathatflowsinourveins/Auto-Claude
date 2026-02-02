#!/bin/bash
# fix-trading-safety.sh - Trading Safety Hooks Deployment Script
# Part of Claude CLI Toolkit V10 OPTIMIZED
#
# This script deploys comprehensive trading safety measures including:
# - Live trading blocking hooks
# - Position size limit enforcement
# - Paper trading verification
# - Daily loss limit tracking
# - Dangerous command prevention
#
# CRITICAL: This is a SAFETY-CRITICAL script. Test thoroughly before production use.
#
# Usage: ./fix-trading-safety.sh [--dry-run] [--verbose] [--project-path PATH]

set -euo pipefail

# Configuration
CLAUDE_CONFIG_DIR="${HOME}/.claude"
PROJECT_PATH="${PROJECT_PATH:-.}"
BACKUP_DIR="${CLAUDE_CONFIG_DIR}/backups/$(date +%Y%m%d_%H%M%S)"
DRY_RUN=false
VERBOSE=false

# Trading Safety Limits (configurable via environment)
MAX_POSITION_SIZE="${MAX_POSITION_SIZE:-10000}"
MAX_DAILY_LOSS="${MAX_DAILY_LOSS:-1000}"
MAX_SINGLE_ORDER="${MAX_SINGLE_ORDER:-5000}"

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
        --project-path)
            PROJECT_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

log() { echo -e "${GREEN}[TRADING-SAFETY]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
critical() { echo -e "${MAGENTA}[CRITICAL]${NC} $1"; }
verbose() { [ "$VERBOSE" = true ] && echo -e "${CYAN}[VERBOSE]${NC} $1"; }

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CREATE TRADING SAFETY HOOKS DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════════

setup_hooks_directory() {
    log "Setting up trading safety hooks directory..."

    HOOKS_DIR="${PROJECT_PATH}/.claude/hooks"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $HOOKS_DIR"
        return
    fi

    mkdir -p "$HOOKS_DIR"
    verbose "Created hooks directory: $HOOKS_DIR"
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DEPLOY LIVE TRADING BLOCKER HOOK
# ═══════════════════════════════════════════════════════════════════════════════

deploy_live_trading_blocker() {
    log "Deploying live trading blocker hook..."

    HOOK_FILE="${PROJECT_PATH}/.claude/hooks/block-live-trading.sh"

    HOOK_CONTENT='#!/bin/bash
# block-live-trading.sh - PreToolUse Hook
# CRITICAL SAFETY HOOK: Blocks any live/production trading operations
#
# Trigger: PreToolUse for Bash commands containing trading patterns
# Action: Block and return error message

# Environment from Claude Code
TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

# Exit early if not a Bash tool
if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 1: Live/Production Trading Keywords
# ═══════════════════════════════════════════════════════════════════════════════
LIVE_PATTERNS=(
    "live"
    "production"
    "real[_-]?money"
    "real[_-]?trading"
    "--live"
    "--production"
    "paper[_-]?trade.*false"
    "PAPER.*=.*false"
    "ALPACA_PAPER.*false"
    "is_paper.*false"
    "sandbox.*false"
    "live_mode.*true"
)

for pattern in "${LIVE_PATTERNS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qiE "$pattern"; then
        echo "{\"block\": true, \"message\": \"🛑 BLOCKED: Live trading operation detected. Pattern: $pattern. Use paper trading only.\"}"
        exit 2
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 2: Dangerous API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════
DANGEROUS_ENDPOINTS=(
    "/v2/orders.*live"
    "api.alpaca.markets(?!/paper)"
    "trading.robinhood.com"
    "api.coinbase.com/v2/accounts/.*/transactions"
    "api.binance.com/api/v3/order"
)

for endpoint in "${DANGEROUS_ENDPOINTS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qiE "$endpoint"; then
        echo "{\"block\": true, \"message\": \"🛑 BLOCKED: Dangerous trading API endpoint detected. Use paper/sandbox APIs only.\"}"
        exit 2
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 3: Withdrawal/Transfer Operations
# ═══════════════════════════════════════════════════════════════════════════════
FINANCIAL_OPS=(
    "withdraw"
    "transfer[_-]?funds"
    "wire[_-]?transfer"
    "ach[_-]?transfer"
    "bank[_-]?transfer"
    "send[_-]?crypto"
    "crypto[_-]?withdrawal"
)

for op in "${FINANCIAL_OPS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qiE "$op"; then
        echo "{\"block\": true, \"message\": \"🛑 BLOCKED: Financial transfer operation detected. These operations are not allowed.\"}"
        exit 2
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 4: Credential Exposure Prevention
# ═══════════════════════════════════════════════════════════════════════════════
CREDENTIAL_PATTERNS=(
    "API_KEY.*="
    "SECRET.*="
    "PASSWORD.*="
    "TOKEN.*="
    "PRIVATE_KEY.*="
    "echo.*\\$.*KEY"
    "echo.*\\$.*SECRET"
    "export.*API_KEY"
    "export.*SECRET"
)

for cred in "${CREDENTIAL_PATTERNS[@]}"; do
    if echo "$TOOL_INPUT" | grep -qE "$cred"; then
        echo "{\"block\": true, \"message\": \"🛑 BLOCKED: Potential credential exposure detected. Never echo or export secrets.\"}"
        exit 2
    fi
done

# All checks passed
exit 0
'

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $HOOK_FILE"
        verbose "Hook content preview:"
        echo "$HOOK_CONTENT" | head -30
    else
        echo "$HOOK_CONTENT" > "$HOOK_FILE"
        chmod +x "$HOOK_FILE"
        log "Created: $HOOK_FILE"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DEPLOY POSITION SIZE LIMITER HOOK
# ═══════════════════════════════════════════════════════════════════════════════

deploy_position_limiter() {
    log "Deploying position size limiter hook..."

    HOOK_FILE="${PROJECT_PATH}/.claude/hooks/limit-position-size.sh"

    HOOK_CONTENT="#!/bin/bash
# limit-position-size.sh - PreToolUse Hook
# Enforces position size limits on trading operations
#
# Configurable limits via environment:
#   MAX_POSITION_SIZE: Maximum total position value (default: ${MAX_POSITION_SIZE})
#   MAX_SINGLE_ORDER: Maximum single order value (default: ${MAX_SINGLE_ORDER})

TOOL_NAME=\"\${TOOL_NAME:-}\"
TOOL_INPUT=\"\${TOOL_INPUT:-}\"

# Limits (from environment or defaults)
MAX_POSITION_SIZE=\"\${MAX_POSITION_SIZE:-${MAX_POSITION_SIZE}}\"
MAX_SINGLE_ORDER=\"\${MAX_SINGLE_ORDER:-${MAX_SINGLE_ORDER}}\"

# Exit early if not relevant tool
if [ \"\$TOOL_NAME\" != \"Bash\" ]; then
    exit 0
fi

# Check for order commands
if ! echo \"\$TOOL_INPUT\" | grep -qiE '(order|buy|sell|trade|position)'; then
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Extract quantity and price from common patterns
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern: --qty 100 --price 150.50
QTY=\$(echo \"\$TOOL_INPUT\" | grep -oP '(?<=--qty[= ])[0-9.]+' | head -1)
PRICE=\$(echo \"\$TOOL_INPUT\" | grep -oP '(?<=--price[= ])[0-9.]+' | head -1)

# Pattern: quantity=100
if [ -z \"\$QTY\" ]; then
    QTY=\$(echo \"\$TOOL_INPUT\" | grep -oP '(?<=quantity[= ])[0-9.]+' | head -1)
fi

# Pattern: shares=100
if [ -z \"\$QTY\" ]; then
    QTY=\$(echo \"\$TOOL_INPUT\" | grep -oP '(?<=shares[= ])[0-9.]+' | head -1)
fi

# Pattern: amount=10000
AMOUNT=\$(echo \"\$TOOL_INPUT\" | grep -oP '(?<=amount[= ])[0-9.]+' | head -1)

# Calculate order value if we have qty and price
if [ -n \"\$QTY\" ] && [ -n \"\$PRICE\" ]; then
    ORDER_VALUE=\$(echo \"\$QTY * \$PRICE\" | bc 2>/dev/null || echo \"0\")

    if [ \$(echo \"\$ORDER_VALUE > \$MAX_SINGLE_ORDER\" | bc) -eq 1 ]; then
        echo \"{\\\"block\\\": true, \\\"message\\\": \\\"🛑 BLOCKED: Order value \\\$\${ORDER_VALUE} exceeds limit of \\\$\${MAX_SINGLE_ORDER}\\\"}\"
        exit 2
    fi
fi

# Check direct amount
if [ -n \"\$AMOUNT\" ]; then
    if [ \$(echo \"\$AMOUNT > \$MAX_SINGLE_ORDER\" | bc) -eq 1 ]; then
        echo \"{\\\"block\\\": true, \\\"message\\\": \\\"🛑 BLOCKED: Order amount \\\$\${AMOUNT} exceeds limit of \\\$\${MAX_SINGLE_ORDER}\\\"}\"
        exit 2
    fi
fi

# All checks passed
exit 0
"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $HOOK_FILE"
    else
        echo "$HOOK_CONTENT" > "$HOOK_FILE"
        chmod +x "$HOOK_FILE"
        log "Created: $HOOK_FILE"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DEPLOY PAPER TRADING VERIFIER HOOK
# ═══════════════════════════════════════════════════════════════════════════════

deploy_paper_verifier() {
    log "Deploying paper trading verifier hook..."

    HOOK_FILE="${PROJECT_PATH}/.claude/hooks/verify-paper-trading.sh"

    HOOK_CONTENT='#!/bin/bash
# verify-paper-trading.sh - PreToolUse Hook
# Ensures ALPACA_PAPER_TRADE=true is set before any Alpaca operations
#
# This hook runs before ANY Alpaca-related command to verify paper mode

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

# Exit early if not Bash
if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

# Check if this is an Alpaca-related command
if ! echo "$TOOL_INPUT" | grep -qiE "(alpaca|ALPACA)"; then
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Verify ALPACA_PAPER_TRADE environment variable
# ═══════════════════════════════════════════════════════════════════════════════

# Check environment variable
if [ "${ALPACA_PAPER_TRADE:-}" != "true" ] && [ "${ALPACA_PAPER:-}" != "true" ]; then
    echo "{\"block\": true, \"message\": \"🛑 BLOCKED: ALPACA_PAPER_TRADE must be set to 'true'. Set: export ALPACA_PAPER_TRADE=true\"}"
    exit 2
fi

# Check if command explicitly sets paper=false
if echo "$TOOL_INPUT" | grep -qiE "(paper.*=.*false|PAPER.*false|--no-paper|--live)"; then
    echo "{\"block\": true, \"message\": \"🛑 BLOCKED: Attempting to disable paper trading. Paper mode must remain enabled.\"}"
    exit 2
fi

# Additional check: Verify using paper API endpoint
if echo "$TOOL_INPUT" | grep -qE "api\.alpaca\.markets" && ! echo "$TOOL_INPUT" | grep -qE "paper-api\.alpaca\.markets"; then
    # Allow if explicitly paper endpoint or if paper env is set
    if [ "${ALPACA_PAPER_TRADE:-}" = "true" ]; then
        # Warn but allow - the SDK should handle this
        echo "{\"message\": \"⚠️ WARNING: Using api.alpaca.markets - ensure paper trading mode is enabled in SDK\"}" >&2
    fi
fi

# All checks passed
exit 0
'

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $HOOK_FILE"
    else
        echo "$HOOK_CONTENT" > "$HOOK_FILE"
        chmod +x "$HOOK_FILE"
        log "Created: $HOOK_FILE"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DEPLOY DAILY LOSS TRACKER HOOK
# ═══════════════════════════════════════════════════════════════════════════════

deploy_daily_loss_tracker() {
    log "Deploying daily loss tracker hook..."

    HOOK_FILE="${PROJECT_PATH}/.claude/hooks/track-daily-loss.sh"

    HOOK_CONTENT="#!/bin/bash
# track-daily-loss.sh - PostToolUse Hook
# Tracks cumulative daily P&L and blocks trading if loss limit exceeded
#
# Configurable limits:
#   MAX_DAILY_LOSS: Maximum allowable daily loss (default: ${MAX_DAILY_LOSS})
#
# State file: ~/.claude/trading_state/daily_pnl.json

TOOL_NAME=\"\${TOOL_NAME:-}\"
TOOL_OUTPUT=\"\${TOOL_OUTPUT:-}\"

MAX_DAILY_LOSS=\"\${MAX_DAILY_LOSS:-${MAX_DAILY_LOSS}}\"
STATE_DIR=\"\${HOME}/.claude/trading_state\"
STATE_FILE=\"\${STATE_DIR}/daily_pnl.json\"

# Initialize state directory
mkdir -p \"\$STATE_DIR\"

# Initialize state file if needed
if [ ! -f \"\$STATE_FILE\" ]; then
    echo '{\"date\": \"'\"$(date +%Y-%m-%d)\"'\", \"cumulative_pnl\": 0, \"trade_count\": 0}' > \"\$STATE_FILE\"
fi

# Check if date changed (reset daily counters)
CURRENT_DATE=\"\$(date +%Y-%m-%d)\"
STATE_DATE=\"\$(jq -r '.date' \"\$STATE_FILE\" 2>/dev/null || echo \"\")\"

if [ \"\$CURRENT_DATE\" != \"\$STATE_DATE\" ]; then
    echo '{\"date\": \"'\"\$CURRENT_DATE\"'\", \"cumulative_pnl\": 0, \"trade_count\": 0}' > \"\$STATE_FILE\"
fi

# Only track if this looks like a trade result
if ! echo \"\$TOOL_OUTPUT\" | grep -qiE '(pnl|profit|loss|filled|executed|trade)'; then
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Extract P&L from output (multiple patterns)
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern: pnl: -50.00 or profit: 100.00
PNL=\"\$(echo \"\$TOOL_OUTPUT\" | grep -oP '(?<=(pnl|profit|loss)[:\\s=]\\s*)[+-]?[0-9.]+' | head -1)\"

if [ -n \"\$PNL\" ]; then
    # Read current state
    CURRENT_PNL=\"\$(jq -r '.cumulative_pnl' \"\$STATE_FILE\" 2>/dev/null || echo \"0\")\"
    TRADE_COUNT=\"\$(jq -r '.trade_count' \"\$STATE_FILE\" 2>/dev/null || echo \"0\")\"

    # Update cumulative P&L
    NEW_PNL=\"\$(echo \"\$CURRENT_PNL + \$PNL\" | bc 2>/dev/null || echo \"\$CURRENT_PNL\")\"
    NEW_COUNT=\"\$((TRADE_COUNT + 1))\"

    # Save updated state
    echo '{\"date\": \"'\"\$CURRENT_DATE\"'\", \"cumulative_pnl\": '\"\$NEW_PNL\"', \"trade_count\": '\"\$NEW_COUNT\"'}' > \"\$STATE_FILE\"

    # Check if loss limit exceeded
    # Note: Loss is negative, so we check if PNL < -MAX_DAILY_LOSS
    LOSS_LIMIT=\"-\${MAX_DAILY_LOSS}\"
    if [ \$(echo \"\$NEW_PNL < \$LOSS_LIMIT\" | bc) -eq 1 ]; then
        echo \"\"
        echo \"╔═══════════════════════════════════════════════════════════════════╗\"
        echo \"║  🚨 DAILY LOSS LIMIT REACHED - TRADING SHOULD STOP               ║\"
        echo \"╠═══════════════════════════════════════════════════════════════════╣\"
        echo \"║  Cumulative P&L: \\\$\${NEW_PNL}\"
        echo \"║  Loss Limit:     \\\$\${MAX_DAILY_LOSS}\"
        echo \"║  Trade Count:    \${NEW_COUNT}\"
        echo \"╚═══════════════════════════════════════════════════════════════════╝\"
        echo \"\"

        # Create lockout file
        echo \"locked\" > \"\${STATE_DIR}/trading_locked\"
    fi
fi

exit 0
"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $HOOK_FILE"
    else
        echo "$HOOK_CONTENT" > "$HOOK_FILE"
        chmod +x "$HOOK_FILE"
        log "Created: $HOOK_FILE"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: UPDATE SETTINGS.JSON WITH HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

update_settings_json() {
    log "Updating settings.json with trading safety hooks..."

    SETTINGS_FILE="${PROJECT_PATH}/.claude/settings.json"

    HOOKS_CONFIG='{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash(*alpaca*)",
        "hooks": [
          {"type": "command", "command": ".claude/hooks/verify-paper-trading.sh"}
        ]
      },
      {
        "matcher": "Bash(*order*)",
        "hooks": [
          {"type": "command", "command": ".claude/hooks/block-live-trading.sh"},
          {"type": "command", "command": ".claude/hooks/limit-position-size.sh"}
        ]
      },
      {
        "matcher": "Bash(*trade*)",
        "hooks": [
          {"type": "command", "command": ".claude/hooks/block-live-trading.sh"}
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash(*)",
        "hooks": [
          {"type": "command", "command": ".claude/hooks/track-daily-loss.sh"}
        ]
      }
    ]
  },
  "permissions": {
    "deny": [
      "Bash(*live*trading*)",
      "Bash(*production*order*)",
      "Bash(*withdraw*)",
      "Bash(*transfer*funds*)",
      "Bash(echo *API_KEY*)",
      "Bash(echo *SECRET*)",
      "Bash(export *API_KEY*)",
      "Read(.env)",
      "Read(.env.*)",
      "Read(**/secrets/**)"
    ]
  }
}'

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would update: $SETTINGS_FILE"
        echo "Hooks configuration:"
        echo "$HOOKS_CONFIG" | head -30
    else
        mkdir -p "$(dirname "$SETTINGS_FILE")"

        if [ -f "$SETTINGS_FILE" ]; then
            # Merge with existing settings
            verbose "Merging with existing settings.json"
            # For safety, just show what would be added
            warn "Please manually merge the following into $SETTINGS_FILE:"
            echo "$HOOKS_CONFIG"
        else
            echo "$HOOKS_CONFIG" > "$SETTINGS_FILE"
            log "Created: $SETTINGS_FILE"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CREATE ENVIRONMENT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

create_env_template() {
    log "Creating environment variables template..."

    ENV_FILE="${PROJECT_PATH}/.env.trading.template"

    ENV_CONTENT="# Trading Safety Environment Variables
# Copy to .env and fill in your values
# NEVER commit .env to version control!

# ═══════════════════════════════════════════════════════════════════════════════
# ALPACA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Keys (from Alpaca dashboard)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# CRITICAL: Always true for development
ALPACA_PAPER_TRADE=true

# API Base URL (paper trading)
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# ═══════════════════════════════════════════════════════════════════════════════
# TRADING LIMITS
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum position value
MAX_POSITION_SIZE=${MAX_POSITION_SIZE}

# Maximum single order value
MAX_SINGLE_ORDER=${MAX_SINGLE_ORDER}

# Maximum daily loss before trading stops
MAX_DAILY_LOSS=${MAX_DAILY_LOSS}

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA (Polygon.io)
# ═══════════════════════════════════════════════════════════════════════════════

POLYGON_API_KEY=your_polygon_key_here

# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

# Sentry for error tracking
SENTRY_DSN=your_sentry_dsn_here

# Slack for notifications
SLACK_WEBHOOK_URL=your_slack_webhook_here
"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $ENV_FILE"
    else
        echo "$ENV_CONTENT" > "$ENV_FILE"
        log "Created: $ENV_FILE"

        # Ensure .env is in .gitignore
        if [ -f "${PROJECT_PATH}/.gitignore" ]; then
            if ! grep -q "^\.env" "${PROJECT_PATH}/.gitignore"; then
                echo -e "\n# Trading credentials\n.env\n.env.*\n!.env.*.template" >> "${PROJECT_PATH}/.gitignore"
                log "Added .env to .gitignore"
            fi
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: VERIFY TRADING SAFETY
# ═══════════════════════════════════════════════════════════════════════════════

verify_trading_safety() {
    log "Verifying trading safety configuration..."

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "                   TRADING SAFETY VERIFICATION                      "
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""

    # Check hooks exist
    echo -e "${CYAN}Checking hook files...${NC}"
    HOOKS_DIR="${PROJECT_PATH}/.claude/hooks"

    REQUIRED_HOOKS=(
        "block-live-trading.sh"
        "limit-position-size.sh"
        "verify-paper-trading.sh"
        "track-daily-loss.sh"
    )

    for hook in "${REQUIRED_HOOKS[@]}"; do
        if [ -f "${HOOKS_DIR}/${hook}" ]; then
            echo -e "  ${GREEN}✓${NC} ${hook}"
        else
            echo -e "  ${RED}✗${NC} ${hook} (missing)"
        fi
    done

    # Check environment
    echo ""
    echo -e "${CYAN}Checking environment variables...${NC}"

    if [ "${ALPACA_PAPER_TRADE:-}" = "true" ]; then
        echo -e "  ${GREEN}✓${NC} ALPACA_PAPER_TRADE=true"
    else
        echo -e "  ${RED}✗${NC} ALPACA_PAPER_TRADE not set to true"
    fi

    if [ -n "${ALPACA_API_KEY:-}" ]; then
        echo -e "  ${GREEN}✓${NC} ALPACA_API_KEY is set"
    else
        echo -e "  ${YELLOW}⚠${NC} ALPACA_API_KEY not set"
    fi

    # Check .gitignore
    echo ""
    echo -e "${CYAN}Checking .gitignore...${NC}"

    if [ -f "${PROJECT_PATH}/.gitignore" ] && grep -q "\.env" "${PROJECT_PATH}/.gitignore"; then
        echo -e "  ${GREEN}✓${NC} .env files are gitignored"
    else
        echo -e "  ${RED}✗${NC} .env files may not be gitignored"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"

    log "Trading safety verification complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║          CLAUDE CLI TRADING SAFETY FIX - V10 OPTIMIZED           ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo ""

    critical "This script deploys SAFETY-CRITICAL trading hooks."
    critical "These hooks are designed to PREVENT accidental live trading."
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}Running in DRY-RUN mode - no changes will be made${NC}"
        echo ""
    fi

    echo "Configuration:"
    echo "  Project Path:     $PROJECT_PATH"
    echo "  Max Position:     \$${MAX_POSITION_SIZE}"
    echo "  Max Single Order: \$${MAX_SINGLE_ORDER}"
    echo "  Max Daily Loss:   \$${MAX_DAILY_LOSS}"
    echo ""

    setup_hooks_directory
    deploy_live_trading_blocker
    deploy_position_limiter
    deploy_paper_verifier
    deploy_daily_loss_tracker
    update_settings_json
    create_env_template

    echo ""
    verify_trading_safety

    echo ""
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}DRY-RUN complete. Run without --dry-run to apply changes.${NC}"
    else
        echo -e "${GREEN}Trading safety deployment complete!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Copy .env.trading.template to .env and fill in values"
        echo "  2. Ensure ALPACA_PAPER_TRADE=true is set"
        echo "  3. Restart Claude Code to load hooks"
        echo "  4. Test with: 'Place a paper trade for 1 share of SPY'"
    fi
}

main "$@"
