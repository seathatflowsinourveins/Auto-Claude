#!/bin/bash
# fix-memory-system.sh - Memory System Migration Script
# Part of Claude CLI Toolkit V10 OPTIMIZED
#
# This script migrates from broken/redundant memory systems to the recommended
# 2-layer architecture: claude-mem (primary) + episodic-memory (backup)
#
# CRITICAL ISSUES FIXED:
# - Removes @modelcontextprotocol/server-memory (GitHub issues #1018, #3074, #3144)
# - Removes graphiti-mcp (duplicate functionality)
# - Removes qdrant-mcp (superseded by claude-mem's vector store)
# - Configures proper memory hierarchy
#
# Usage: ./fix-memory-system.sh [--dry-run] [--verbose]

set -euo pipefail

# Configuration
CLAUDE_CONFIG_DIR="${HOME}/.claude"
BACKUP_DIR="${CLAUDE_CONFIG_DIR}/backups/$(date +%Y%m%d_%H%M%S)"
DRY_RUN=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

log() {
    echo -e "${GREEN}[FIX-MEMORY]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}[VERBOSE]${NC} $1"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BACKUP EXISTING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

backup_existing_config() {
    log "Creating backup of existing configuration..."

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would create backup at: $BACKUP_DIR"
        return
    fi

    mkdir -p "$BACKUP_DIR"

    # Backup settings.json if exists
    if [ -f "${CLAUDE_CONFIG_DIR}/settings.json" ]; then
        cp "${CLAUDE_CONFIG_DIR}/settings.json" "${BACKUP_DIR}/settings.json.bak"
        verbose "Backed up settings.json"
    fi

    # Backup any .mcp.json in current directory
    if [ -f ".mcp.json" ]; then
        cp ".mcp.json" "${BACKUP_DIR}/mcp.json.bak"
        verbose "Backed up .mcp.json"
    fi

    log "Backup created at: $BACKUP_DIR"
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: REMOVE BROKEN MEMORY SERVERS
# ═══════════════════════════════════════════════════════════════════════════════

remove_broken_servers() {
    log "Removing broken/redundant memory MCP servers..."

    # List of servers to remove
    BROKEN_SERVERS=(
        "@modelcontextprotocol/server-memory"  # GitHub issues #1018, #3074, #3144
        "graphiti-mcp"                          # Duplicate functionality
        "qdrant-mcp"                            # Superseded by claude-mem
        "server-memory"                         # Alias for broken server
    )

    for server in "${BROKEN_SERVERS[@]}"; do
        if [ "$DRY_RUN" = true ]; then
            echo -e "${BLUE}[DRY-RUN]${NC} Would remove MCP server: $server"
        else
            verbose "Attempting to remove: $server"
            claude mcp remove "$server" 2>/dev/null || true
        fi
    done

    log "Broken servers removal complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: INSTALL RECOMMENDED MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

install_recommended_memory() {
    log "Installing recommended 2-layer memory system..."

    # Layer 1: claude-mem (primary - cross-session memory)
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would install claude-mem plugin"
    else
        verbose "Installing claude-mem plugin..."
        # claude-mem is typically a plugin, not MCP server
        # Check if already installed via plugins
        if ! claude plugins list 2>/dev/null | grep -q "claude-mem"; then
            warn "claude-mem plugin should be installed via Claude Code plugins"
            echo "  Run: claude plugins add claude-mem"
        else
            log "claude-mem already installed"
        fi
    fi

    # Layer 2: episodic-memory (backup - conversation search)
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would configure episodic-memory"
    else
        verbose "Configuring episodic-memory..."
        # episodic-memory is also typically a plugin
        if ! claude plugins list 2>/dev/null | grep -q "episodic-memory"; then
            warn "episodic-memory plugin should be installed via Claude Code plugins"
            echo "  Run: claude plugins add episodic-memory"
        else
            log "episodic-memory already installed"
        fi
    fi

    log "Memory system installation complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CREATE OPTIMIZED MEMORY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

create_memory_config() {
    log "Creating optimized memory configuration..."

    # Memory hierarchy documentation for CLAUDE.md
    MEMORY_DOCS="
## Memory System Configuration (V10 OPTIMIZED)

### 2-Layer Architecture
\`\`\`
Layer 1: claude-mem (Primary)
├── Cross-session memory persistence
├── Semantic search via pgvector embeddings
├── Multi-project isolation
└── ~1,500 tokens overhead

Layer 2: episodic-memory (Backup)
├── Conversation history search
├── Vector + text search modes
├── Date filtering support
└── ~1,000 tokens overhead
\`\`\`

### Removed Systems (Do NOT Re-enable)
- @modelcontextprotocol/server-memory (BROKEN - Issues #1018, #3074, #3144)
- graphiti-mcp (Redundant)
- qdrant-mcp (Superseded)

### Memory Usage Patterns
- Use \`mcp__plugin_claude-mem_mcp-search__search\` for semantic queries
- Use \`mcp__plugin_episodic-memory_episodic-memory__search\` for conversation history
- Always search memory BEFORE starting complex tasks
"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would append memory docs to CLAUDE.md"
        echo "$MEMORY_DOCS"
    else
        # Append to global CLAUDE.md if not already present
        if [ -f "${CLAUDE_CONFIG_DIR}/CLAUDE.md" ]; then
            if ! grep -q "Memory System Configuration (V10 OPTIMIZED)" "${CLAUDE_CONFIG_DIR}/CLAUDE.md"; then
                echo "$MEMORY_DOCS" >> "${CLAUDE_CONFIG_DIR}/CLAUDE.md"
                log "Added memory documentation to CLAUDE.md"
            else
                verbose "Memory documentation already exists in CLAUDE.md"
            fi
        else
            warn "CLAUDE.md not found at ${CLAUDE_CONFIG_DIR}/CLAUDE.md"
        fi
    fi

    log "Memory configuration complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VERIFY MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

verify_memory_system() {
    log "Verifying memory system configuration..."

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "                    MEMORY SYSTEM VERIFICATION                      "
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""

    # Check for broken servers
    echo -e "${CYAN}Checking for broken servers...${NC}"
    BROKEN_FOUND=false

    if claude mcp list 2>/dev/null | grep -qE "(server-memory|graphiti|qdrant-mcp)"; then
        echo -e "${RED}✗ Broken memory servers still present${NC}"
        BROKEN_FOUND=true
    else
        echo -e "${GREEN}✓ No broken memory servers found${NC}"
    fi

    # Check for recommended plugins
    echo ""
    echo -e "${CYAN}Checking for recommended plugins...${NC}"

    if claude plugins list 2>/dev/null | grep -q "claude-mem"; then
        echo -e "${GREEN}✓ claude-mem plugin installed${NC}"
    else
        echo -e "${YELLOW}⚠ claude-mem plugin not found${NC}"
    fi

    if claude plugins list 2>/dev/null | grep -q "episodic-memory"; then
        echo -e "${GREEN}✓ episodic-memory plugin installed${NC}"
    else
        echo -e "${YELLOW}⚠ episodic-memory plugin not found${NC}"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"

    if [ "$BROKEN_FOUND" = true ]; then
        warn "Some broken servers still present. Run without --dry-run to fix."
        return 1
    fi

    log "Memory system verification complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║           CLAUDE CLI MEMORY SYSTEM FIX - V10 OPTIMIZED           ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}Running in DRY-RUN mode - no changes will be made${NC}"
        echo ""
    fi

    backup_existing_config
    echo ""

    remove_broken_servers
    echo ""

    install_recommended_memory
    echo ""

    create_memory_config
    echo ""

    verify_memory_system
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}DRY-RUN complete. Run without --dry-run to apply changes.${NC}"
    else
        echo -e "${GREEN}Memory system fix complete!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Restart Claude Code to apply changes"
        echo "  2. Test memory with: 'Search my previous conversations about X'"
        echo "  3. Verify with: claude mcp list"
    fi
}

main "$@"
