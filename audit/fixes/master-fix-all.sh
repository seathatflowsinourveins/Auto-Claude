#!/bin/bash
# master-fix-all.sh - Master Fix Orchestrator
# Part of Claude CLI Toolkit V10 OPTIMIZED
#
# This is the ONE-COMMAND solution to deploy all fixes from the
# ULTRA_DETAILED_SYNTHESIS_REPORT_V2.md analysis.
#
# What this script does:
# 1. Backs up existing configuration
# 2. Fixes memory system (removes broken, installs recommended)
# 3. Deploys trading safety hooks
# 4. Deploys complete 15-hook system
# 5. Creates optimized .gitignore
# 6. Updates MCP configuration
# 7. Verifies all changes
#
# Usage: ./master-fix-all.sh [--dry-run] [--verbose] [--trading-only] [--creative-only]
#
# IMPORTANT: Run with --dry-run first to preview all changes!

set -euo pipefail

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
CLAUDE_CONFIG_DIR="${HOME}/.claude"
PROJECT_PATH="${PROJECT_PATH:-$(pwd)}"
BACKUP_DIR="${CLAUDE_CONFIG_DIR}/backups/master_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=false
VERBOSE=false
TRADING_ONLY=false
CREATIVE_ONLY=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
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
        --trading-only)
            TRADING_ONLY=true
            shift
            ;;
        --creative-only)
            CREATIVE_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run       Preview changes without applying them"
            echo "  --verbose       Show detailed output"
            echo "  --trading-only  Only apply trading-related fixes"
            echo "  --creative-only Only apply creative/ML-related fixes"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
log() { echo -e "${GREEN}[MASTER-FIX]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
section() { echo -e "\n${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"; echo -e "${WHITE}  $1${NC}"; echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}\n"; }
verbose() { [ "$VERBOSE" = true ] && echo -e "${CYAN}[VERBOSE]${NC} $1"; }

# ═══════════════════════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════════════════════

show_banner() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                                                                           ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${WHITE}█████╗ █████╗ █████╗ ██╗   ██╗███████╗    █████╗ ██╗     ██╗${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${WHITE}██╔══╝██╔══██╗██╔══██╗██║   ██║██╔════╝   ██╔══██╗██║     ██║${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${WHITE}██║   ██║  ██║██║  ██║██║   ██║█████╗     ███████║██║     ██║${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${WHITE}██║   ██║  ██║██║  ██║██║   ██║██╔══╝     ██╔══██║██║     ██║${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${WHITE}█████╗╚█████╔╝█████╔╝╚██████╔╝███████╗   ██║  ██║███████╗██║${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${WHITE}╚════╝ ╚════╝ ╚════╝  ╚═════╝ ╚══════╝   ╚═╝  ╚═╝╚══════╝╚═╝${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                           ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}              ${GREEN}CLAUDE CLI TOOLKIT V10 OPTIMIZED - MASTER FIX${NC}              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                           ${CYAN}║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}══════════════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}                         DRY-RUN MODE - NO CHANGES WILL BE MADE               ${NC}"
        echo -e "${YELLOW}══════════════════════════════════════════════════════════════════════════════${NC}"
        echo ""
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BACKUP
# ═══════════════════════════════════════════════════════════════════════════════

phase_backup() {
    section "PHASE 1: BACKUP EXISTING CONFIGURATION"

    log "Creating comprehensive backup..."

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create backup at: $BACKUP_DIR"
        return
    fi

    mkdir -p "$BACKUP_DIR"

    # Backup global config
    if [ -d "$CLAUDE_CONFIG_DIR" ]; then
        cp -r "$CLAUDE_CONFIG_DIR" "$BACKUP_DIR/global_claude_config"
        verbose "Backed up global Claude config"
    fi

    # Backup project config
    if [ -d "${PROJECT_PATH}/.claude" ]; then
        cp -r "${PROJECT_PATH}/.claude" "$BACKUP_DIR/project_claude_config"
        verbose "Backed up project Claude config"
    fi

    # Backup .mcp.json
    if [ -f "${PROJECT_PATH}/.mcp.json" ]; then
        cp "${PROJECT_PATH}/.mcp.json" "$BACKUP_DIR/mcp.json.bak"
        verbose "Backed up .mcp.json"
    fi

    # Backup .gitignore
    if [ -f "${PROJECT_PATH}/.gitignore" ]; then
        cp "${PROJECT_PATH}/.gitignore" "$BACKUP_DIR/gitignore.bak"
        verbose "Backed up .gitignore"
    fi

    log "Backup created at: $BACKUP_DIR"
    echo ""
    echo "  To restore: cp -r ${BACKUP_DIR}/* ~/"
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MEMORY SYSTEM FIX
# ═══════════════════════════════════════════════════════════════════════════════

phase_memory_fix() {
    section "PHASE 2: MEMORY SYSTEM FIX"

    log "Applying memory system fixes..."

    if [ -f "${SCRIPT_DIR}/fix-memory-system.sh" ]; then
        if [ "$DRY_RUN" = true ]; then
            bash "${SCRIPT_DIR}/fix-memory-system.sh" --dry-run
        else
            bash "${SCRIPT_DIR}/fix-memory-system.sh"
        fi
    else
        warn "fix-memory-system.sh not found, skipping memory fixes"

        # Inline fix: Remove broken servers
        log "Removing broken memory servers inline..."

        BROKEN_SERVERS=(
            "@modelcontextprotocol/server-memory"
            "graphiti-mcp"
            "qdrant-mcp"
        )

        for server in "${BROKEN_SERVERS[@]}"; do
            if [ "$DRY_RUN" = true ]; then
                echo -e "${BLUE}[DRY-RUN]${NC} Would remove: $server"
            else
                claude mcp remove "$server" 2>/dev/null || true
                verbose "Removed: $server"
            fi
        done
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: TRADING SAFETY
# ═══════════════════════════════════════════════════════════════════════════════

phase_trading_safety() {
    if [ "$CREATIVE_ONLY" = true ]; then
        log "Skipping trading safety (--creative-only specified)"
        return
    fi

    section "PHASE 3: TRADING SAFETY DEPLOYMENT"

    log "Deploying trading safety hooks..."

    if [ -f "${SCRIPT_DIR}/fix-trading-safety.sh" ]; then
        if [ "$DRY_RUN" = true ]; then
            bash "${SCRIPT_DIR}/fix-trading-safety.sh" --dry-run --project-path "$PROJECT_PATH"
        else
            bash "${SCRIPT_DIR}/fix-trading-safety.sh" --project-path "$PROJECT_PATH"
        fi
    else
        warn "fix-trading-safety.sh not found, creating essential trading hooks inline..."

        HOOKS_DIR="${PROJECT_PATH}/.claude/hooks"

        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$HOOKS_DIR"

            # Create minimal trading safety hook
            cat > "${HOOKS_DIR}/trading-safety.sh" << 'HOOK'
#!/bin/bash
TOOL_INPUT="${TOOL_INPUT:-}"
if echo "$TOOL_INPUT" | grep -qiE "(live|production|real[_-]?money)"; then
    echo '{"block": true, "message": "🛑 BLOCKED: Live trading detected"}'
    exit 2
fi
exit 0
HOOK
            chmod +x "${HOOKS_DIR}/trading-safety.sh"
            log "Created essential trading safety hook"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: COMPLETE HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

phase_complete_hooks() {
    section "PHASE 4: COMPLETE HOOKS DEPLOYMENT"

    log "Deploying all 15 production hooks..."

    if [ -f "${SCRIPT_DIR}/create-complete-hooks.sh" ]; then
        if [ "$DRY_RUN" = true ]; then
            bash "${SCRIPT_DIR}/create-complete-hooks.sh" --dry-run --project
        else
            bash "${SCRIPT_DIR}/create-complete-hooks.sh" --project
        fi
    else
        warn "create-complete-hooks.sh not found, creating essential hooks inline..."

        HOOKS_DIR="${PROJECT_PATH}/.claude/hooks"

        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$HOOKS_DIR"

            # Auto-format hook
            cat > "${HOOKS_DIR}/auto-format.sh" << 'HOOK'
#!/bin/bash
FILE_PATH=$(echo "$TOOL_INPUT" | grep -oP '(?<=file_path["\047:\s]+)[^"\047\s,}]+' | head -1)
if [[ "$FILE_PATH" =~ \.py$ ]] && command -v ruff &> /dev/null; then
    ruff format "$FILE_PATH" 2>/dev/null
fi
exit 0
HOOK
            chmod +x "${HOOKS_DIR}/auto-format.sh"
            log "Created auto-format hook"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: GITIGNORE FIX
# ═══════════════════════════════════════════════════════════════════════════════

phase_gitignore_fix() {
    section "PHASE 5: GITIGNORE SECURITY FIX"

    log "Creating comprehensive .gitignore..."

    GITIGNORE_CONTENT='# ═══════════════════════════════════════════════════════════════════
# CLAUDE CLI TOOLKIT V10 OPTIMIZED - SECURITY GITIGNORE
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# CREDENTIALS & SECRETS (CRITICAL)
# ═══════════════════════════════════════════════════════════════════
.env
.env.*
!.env.*.template
!.env.example

# API Keys and Tokens
*.pem
*.key
*.crt
*.p12
*.pfx
*_rsa
*_dsa
*_ecdsa
*_ed25519
*.pub

# AWS
.aws/
credentials
aws-credentials*

# Trading API Keys
alpaca*.json
polygon*.json
*_api_key*
*_secret*

# ═══════════════════════════════════════════════════════════════════
# CLAUDE CODE SENSITIVE
# ═══════════════════════════════════════════════════════════════════
.claude/settings.local.json
.claude/trading_state/
.claude/learnings/
.claude/backups/

# ═══════════════════════════════════════════════════════════════════
# PYTHON
# ═══════════════════════════════════════════════════════════════════
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.venv/
venv/
ENV/
.mypy_cache/
.pytest_cache/
.ruff_cache/

# ═══════════════════════════════════════════════════════════════════
# NODE/TYPESCRIPT
# ═══════════════════════════════════════════════════════════════════
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*
*.tsbuildinfo

# ═══════════════════════════════════════════════════════════════════
# IDE & EDITOR
# ═══════════════════════════════════════════════════════════════════
.idea/
.vscode/settings.json
.vscode/launch.json
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# ═══════════════════════════════════════════════════════════════════
# TRADING SPECIFIC
# ═══════════════════════════════════════════════════════════════════
# Backtest results with sensitive data
backtest_results/
live_trading_logs/
order_history/
position_snapshots/

# Strategy configurations (may contain sensitive parameters)
strategies/private/
config/production/

# ═══════════════════════════════════════════════════════════════════
# CREATIVE/ML SPECIFIC
# ═══════════════════════════════════════════════════════════════════
# Large model files
*.ckpt
*.safetensors
*.bin
models/

# MLflow/W&B artifacts
mlruns/
wandb/
artifacts/

# TouchDesigner backup files
*.toe.bak
*.toe.*.bak
Backup/

# Generated outputs
outputs/
renders/
exports/
'

    GITIGNORE_FILE="${PROJECT_PATH}/.gitignore"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would update: $GITIGNORE_FILE"
        echo "Preview (first 30 lines):"
        echo "$GITIGNORE_CONTENT" | head -30
    else
        if [ -f "$GITIGNORE_FILE" ]; then
            # Merge with existing
            if ! grep -q "CLAUDE CLI TOOLKIT V10" "$GITIGNORE_FILE"; then
                echo "" >> "$GITIGNORE_FILE"
                echo "$GITIGNORE_CONTENT" >> "$GITIGNORE_FILE"
                log "Merged security rules into existing .gitignore"
            else
                verbose ".gitignore already contains toolkit rules"
            fi
        else
            echo "$GITIGNORE_CONTENT" > "$GITIGNORE_FILE"
            log "Created comprehensive .gitignore"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: MCP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

phase_mcp_config() {
    section "PHASE 6: MCP CONFIGURATION OPTIMIZATION"

    log "Optimizing MCP server configuration..."

    # Determine project type
    PROJECT_TYPE="general"
    if [[ "$(pwd)" =~ (trading|alphaforge|strategy) ]]; then
        PROJECT_TYPE="trading"
    elif [[ "$(pwd)" =~ (witness|creative|touchdesigner|td) ]]; then
        PROJECT_TYPE="creative"
    fi

    if [ "$TRADING_ONLY" = true ]; then
        PROJECT_TYPE="trading"
    elif [ "$CREATIVE_ONLY" = true ]; then
        PROJECT_TYPE="creative"
    fi

    log "Detected project type: $PROJECT_TYPE"

    # Create appropriate .mcp.json
    case "$PROJECT_TYPE" in
        trading)
            MCP_CONFIG='{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
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
      "env": {"POLYGON_API_KEY": "${POLYGON_API_KEY}"}
    },
    "context7": {
      "url": "https://mcp.context7.com/sse"
    },
    "snyk": {
      "command": "npx",
      "args": ["-y", "snyk-mcp-server"],
      "env": {"SNYK_TOKEN": "${SNYK_TOKEN}"}
    }
  }
}'
            ;;
        creative)
            MCP_CONFIG='{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
    },
    "touchdesigner": {
      "command": "npx",
      "args": ["touchdesigner-mcp-server@latest", "--stdio"],
      "env": {
        "TD_HOST": "${TD_HOST:-127.0.0.1}",
        "TD_PORT": "${TD_PORT:-9981}"
      }
    },
    "mlflow": {
      "command": "mlflow",
      "args": ["mcp"],
      "env": {"MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI:-http://localhost:5000}"}
    },
    "wandb": {
      "url": "https://mcp.withwandb.com/mcp",
      "apiKey": "${WANDB_API_KEY}"
    },
    "context7": {
      "url": "https://mcp.context7.com/sse"
    }
  }
}'
            ;;
        *)
            MCP_CONFIG='{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-filesystem", "."],
      "env": {}
    },
    "context7": {
      "url": "https://mcp.context7.com/sse"
    }
  }
}'
            ;;
    esac

    MCP_FILE="${PROJECT_PATH}/.mcp.json"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create: $MCP_FILE"
        echo "Configuration for $PROJECT_TYPE project:"
        echo "$MCP_CONFIG" | head -30
    else
        if [ -f "$MCP_FILE" ]; then
            warn ".mcp.json already exists. Please review and merge:"
            echo "$MCP_CONFIG"
        else
            echo "$MCP_CONFIG" > "$MCP_FILE"
            log "Created optimized .mcp.json for $PROJECT_TYPE"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

phase_verification() {
    section "PHASE 7: VERIFICATION"

    log "Verifying all fixes..."

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                     VERIFICATION RESULTS                          ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo ""

    PASS=0
    FAIL=0
    WARN=0

    # Check 1: Hooks directory
    echo -e "${CYAN}Hooks Directory:${NC}"
    if [ -d "${PROJECT_PATH}/.claude/hooks" ]; then
        HOOK_COUNT=$(ls -1 "${PROJECT_PATH}/.claude/hooks"/*.sh 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} Found ${HOOK_COUNT} hooks"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC} Hooks directory not found"
        FAIL=$((FAIL + 1))
    fi

    # Check 2: .gitignore
    echo ""
    echo -e "${CYAN}.gitignore Security:${NC}"
    if [ -f "${PROJECT_PATH}/.gitignore" ]; then
        if grep -q "\.env" "${PROJECT_PATH}/.gitignore"; then
            echo -e "  ${GREEN}✓${NC} .env files protected"
            PASS=$((PASS + 1))
        else
            echo -e "  ${YELLOW}⚠${NC} .env files may not be protected"
            WARN=$((WARN + 1))
        fi
    else
        echo -e "  ${RED}✗${NC} .gitignore not found"
        FAIL=$((FAIL + 1))
    fi

    # Check 3: MCP configuration
    echo ""
    echo -e "${CYAN}MCP Configuration:${NC}"
    if [ -f "${PROJECT_PATH}/.mcp.json" ]; then
        echo -e "  ${GREEN}✓${NC} .mcp.json exists"
        PASS=$((PASS + 1))
    else
        echo -e "  ${YELLOW}⚠${NC} .mcp.json not found (using global config)"
        WARN=$((WARN + 1))
    fi

    # Check 4: Environment variables (trading)
    if [ "$CREATIVE_ONLY" != true ]; then
        echo ""
        echo -e "${CYAN}Trading Environment:${NC}"
        if [ "${ALPACA_PAPER_TRADE:-}" = "true" ]; then
            echo -e "  ${GREEN}✓${NC} ALPACA_PAPER_TRADE=true"
            PASS=$((PASS + 1))
        else
            echo -e "  ${YELLOW}⚠${NC} ALPACA_PAPER_TRADE not set"
            WARN=$((WARN + 1))
        fi
    fi

    # Check 5: Backup created
    echo ""
    echo -e "${CYAN}Backup:${NC}"
    if [ -d "$BACKUP_DIR" ]; then
        echo -e "  ${GREEN}✓${NC} Backup created at ${BACKUP_DIR}"
        PASS=$((PASS + 1))
    else
        if [ "$DRY_RUN" = true ]; then
            echo -e "  ${BLUE}○${NC} Backup would be created (dry-run)"
        else
            echo -e "  ${YELLOW}⚠${NC} No backup found"
            WARN=$((WARN + 1))
        fi
    fi

    # Summary
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    echo -e "  ${GREEN}Passed:${NC}   ${PASS}"
    echo -e "  ${YELLOW}Warnings:${NC} ${WARN}"
    echo -e "  ${RED}Failed:${NC}   ${FAIL}"
    echo ""

    if [ $FAIL -eq 0 ]; then
        echo -e "  ${GREEN}Overall: VERIFICATION PASSED${NC}"
    else
        echo -e "  ${RED}Overall: VERIFICATION FAILED${NC}"
        echo "  Review failed checks above and re-run."
    fi
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    show_banner

    echo "Configuration:"
    echo "  Project Path: $PROJECT_PATH"
    echo "  Config Dir:   $CLAUDE_CONFIG_DIR"
    echo "  Backup Dir:   $BACKUP_DIR"
    echo ""

    if [ "$TRADING_ONLY" = true ]; then
        echo -e "  Mode: ${CYAN}Trading Only${NC}"
    elif [ "$CREATIVE_ONLY" = true ]; then
        echo -e "  Mode: ${CYAN}Creative/ML Only${NC}"
    else
        echo -e "  Mode: ${CYAN}Full (Trading + Creative)${NC}"
    fi
    echo ""

    # Execute phases
    phase_backup
    phase_memory_fix
    phase_trading_safety
    phase_complete_hooks
    phase_gitignore_fix
    phase_mcp_config
    phase_verification

    # Final summary
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                     MASTER FIX COMPLETE                           ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}DRY-RUN complete. No changes were made.${NC}"
        echo ""
        echo "To apply all changes, run:"
        echo "  ./master-fix-all.sh"
        echo ""
    else
        echo -e "${GREEN}All fixes have been applied!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Restart Claude Code to load new configuration"
        echo "  2. Verify with: claude mcp list"
        echo "  3. Test hooks with a simple operation"
        echo ""
        echo "Backup location: $BACKUP_DIR"
        echo "To restore: cp -r ${BACKUP_DIR}/* ~/"
    fi
}

main "$@"
