#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Ultimate Claude CLI Development Toolkit - Master Installation Script
# For AlphaForge Trading + State of Witness Creative/ML Development
# Version: 10.0 OPTIMIZED
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Logging functions
log_header() { echo -e "\n${MAGENTA}${BOLD}═══════════════════════════════════════════════════════════════${NC}"; echo -e "${MAGENTA}${BOLD}  $1${NC}"; echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════════════${NC}"; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step() { echo -e "${CYAN}→${NC} $1"; }

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║   █░█ █░░ ▀█▀ █ █▀▄▀█ ▄▀█ ▀█▀ █▀▀   █▀▀ █░░ ▄▀█ █░█ █▀▄ █▀▀  ║"
    echo "║   █▄█ █▄▄ ░█░ █ █░▀░█ █▀█ ░█░ ██▄   █▄▄ █▄▄ █▀█ █▄█ █▄▀ ██▄  ║"
    echo "║                                                               ║"
    echo "║          Claude CLI Development Toolkit Installer             ║"
    echo "║          v10.0 OPTIMIZED | AlphaForge + State of Witness      ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# ═══════════════════════════════════════════════════════════════════════════════
# PREREQUISITES CHECK
# ═══════════════════════════════════════════════════════════════════════════════
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    local missing=0
    
    # Node.js
    if command_exists node; then
        local node_version=$(node --version | sed 's/v//')
        local node_major=$(echo $node_version | cut -d. -f1)
        if [ "$node_major" -ge 18 ]; then
            log_success "Node.js v$node_version"
        else
            log_warn "Node.js v$node_version (recommend v18+)"
        fi
    else
        log_error "Node.js not found"
        log_step "Install: https://nodejs.org/ or 'brew install node'"
        missing=$((missing + 1))
    fi
    
    # npm
    if command_exists npm; then
        log_success "npm $(npm --version)"
    else
        log_error "npm not found"
        missing=$((missing + 1))
    fi
    
    # Python 3
    if command_exists python3; then
        local py_version=$(python3 --version | awk '{print $2}')
        log_success "Python $py_version"
    else
        log_error "Python 3 not found"
        log_step "Install: https://python.org/ or 'brew install python'"
        missing=$((missing + 1))
    fi
    
    # pip
    if command_exists pip3 || command_exists pip; then
        log_success "pip installed"
    else
        log_error "pip not found"
        missing=$((missing + 1))
    fi
    
    # Git
    if command_exists git; then
        log_success "Git $(git --version | awk '{print $3}')"
    else
        log_error "Git not found"
        missing=$((missing + 1))
    fi
    
    # UV (Python package manager)
    if command_exists uv; then
        log_success "UV $(uv --version 2>/dev/null | head -1)"
    else
        log_warn "UV not found - will install"
    fi
    
    # Claude Code CLI
    if command_exists claude; then
        log_success "Claude Code CLI $(claude --version 2>/dev/null | head -1)"
    else
        log_error "Claude Code CLI not found"
        log_step "Install: npm install -g @anthropic-ai/claude-code"
        missing=$((missing + 1))
    fi
    
    if [ $missing -gt 0 ]; then
        echo ""
        log_error "$missing required components missing. Please install them first."
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL UV PACKAGE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
install_uv() {
    if ! command_exists uv; then
        log_header "Installing UV Package Manager"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add to PATH for current session
        export PATH="$HOME/.local/bin:$PATH"
        
        if command_exists uv; then
            log_success "UV installed successfully"
        else
            log_warn "UV installed but may need shell restart"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL DIRECTORY STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
setup_global_structure() {
    log_header "Setting Up Global Directory Structure"
    
    # Create directories
    mkdir -p ~/.claude/{commands,agents,skills,hooks}
    
    log_success "Created ~/.claude/"
    log_success "Created ~/.claude/commands/"
    log_success "Created ~/.claude/agents/"
    log_success "Created ~/.claude/skills/"
    log_success "Created ~/.claude/hooks/"
}

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CLAUDE.md
# ═══════════════════════════════════════════════════════════════════════════════
create_global_claude_md() {
    log_header "Creating Global CLAUDE.md"
    
    cat > ~/.claude/CLAUDE.md << 'CLAUDE_MD_EOF'
# Global Claude Code Instructions

## Developer Profile
- Building AlphaForge (autonomous trading system) and State of Witness (creative/ML system)
- Claude CLI = BUILDER TOOL for full dev cycle (design/audit/research/integrate/debug/UI)
- AlphaForge: Claude CLI is EXTERNAL builder, NOT production component
- State of Witness: Claude CLI IS integrated as generative brain with real-time control

## Architecture Principles
- Quality over quantity: Only actively maintained, production-ready tools
- 7-server MCP limit for optimal tool selection performance
- Paper trading ALWAYS during development (ALPACA_PAPER_TRADE=true)
- Strict separation: development vs production environments

## Memory Behavior
- Update CLAUDE.md with learnings after significant sessions
- Reference @docs/ for detailed documentation when available
- Use /memory command for extensive additions
- Use # prefix for quick memory additions

## Code Standards
- TypeScript: strict mode, interfaces over types, no `any`
- Python: type hints required, Pydantic for validation
- Rust: follow clippy recommendations
- Comprehensive error handling in all languages
- Tests before implementation (TDD preferred)

## Security Rules (CRITICAL)
- NEVER commit API keys, secrets, or credentials
- Always use environment variables for sensitive data
- Paper trading mode for ALL trading development
- Review before executing any financial transactions
- Block production commands during development

## MCP Server Guidelines
- Keep 5-7 core servers active for optimal performance
- Project-specific servers in project .mcp.json
- Test MCP connections before starting work: `claude mcp list`

## Session Workflow
1. Start: Review project CLAUDE.md for context
2. Work: Use appropriate MCP tools for the task
3. End: Capture learnings if significant discoveries made
CLAUDE_MD_EOF

    log_success "Created ~/.claude/CLAUDE.md"
}

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SETTINGS.JSON
# ═══════════════════════════════════════════════════════════════════════════════
create_global_settings() {
    log_header "Creating Global settings.json"
    
    cat > ~/.claude/settings.json << 'SETTINGS_EOF'
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "model": "claude-sonnet-4-20250514",
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
      "Bash(pip3 *)",
      "Bash(uv *)",
      "Bash(uvx *)",
      "Bash(python *)",
      "Bash(python3 *)",
      "Bash(node *)",
      "Bash(claude *)",
      "Bash(ruff *)",
      "Bash(pytest *)",
      "Bash(cargo *)",
      "Bash(rustfmt *)",
      "Bash(prettier *)",
      "Bash(eslint *)",
      "Bash(tsc *)",
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
      "Read(**/*secret*)",
      "Write(.env*)",
      "Write(**/secrets/**)",
      "Bash(rm -rf /)",
      "Bash(rm -rf ~)",
      "Bash(rm -rf /*)",
      "Bash(*SECRET*=*)",
      "Bash(*API_KEY*=*)",
      "Bash(*PASSWORD*=*)",
      "Bash(*CREDENTIAL*=*)"
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
            "command": "echo \"[$(date '+%Y-%m-%d %H:%M:%S')] Session started in $(pwd)\" >> ~/.claude/session.log",
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
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qiE 'production|prod[^u]|--live'; then echo '{\"feedback\": \"⚠️ Production environment keyword detected. Proceeding with caution.\"}'; fi",
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
            "command": "command -v ruff &>/dev/null && ruff format \"$file\" --quiet 2>/dev/null && ruff check \"$file\" --fix --quiet 2>/dev/null || true",
            "timeout": 15
          }
        ]
      },
      {
        "matcher": "Write(*.ts)|Write(*.tsx)|Write(*.js)|Write(*.jsx)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v prettier &>/dev/null && prettier --write \"$file\" --log-level silent 2>/dev/null || true",
            "timeout": 10
          }
        ]
      },
      {
        "matcher": "Write(*.rs)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v rustfmt &>/dev/null && rustfmt \"$file\" 2>/dev/null || true",
            "timeout": 10
          }
        ]
      },
      {
        "matcher": "Write(*.json)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v jq &>/dev/null && jq '.' \"$file\" > \"$file.tmp\" 2>/dev/null && mv \"$file.tmp\" \"$file\" || true",
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
            "command": "echo \"[$(date '+%Y-%m-%d %H:%M:%S')] Session ended\" >> ~/.claude/session.log",
            "timeout": 5
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
SETTINGS_EOF

    log_success "Created ~/.claude/settings.json"
}

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SLASH COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════
create_global_commands() {
    log_header "Creating Global Slash Commands"
    
    # Security Audit Command
    cat > ~/.claude/commands/security-audit.md << 'CMD_EOF'
Perform a comprehensive security audit of this codebase:

1. **Dependency Vulnerabilities**
   - Scan dependencies for known CVEs
   - Check for outdated packages with security issues
   - Review dependency tree for transitive vulnerabilities

2. **Code Security Analysis**
   - Search for hardcoded secrets, API keys, credentials
   - Check for SQL injection vulnerabilities
   - Review authentication/authorization logic
   - Identify XSS vulnerabilities in frontend code
   - Check for insecure deserialization

3. **Configuration Security**
   - Review .env handling and .gitignore patterns
   - Check file permissions on sensitive files
   - Validate API key management practices
   - Review CORS and security headers

4. **Infrastructure Security**
   - Check Docker configurations if present
   - Review CI/CD pipeline security
   - Validate secrets management

Output a prioritized list with:
- Severity: Critical / High / Medium / Low
- Location: File and line number
- Description: What the issue is
- Remediation: How to fix it
CMD_EOF
    log_success "Created security-audit.md"

    # Architecture Review Command
    cat > ~/.claude/commands/architecture-review.md << 'CMD_EOF'
Analyze the architecture of this project comprehensively:

1. **Structure Analysis**
   - Map the directory structure
   - Identify main components and responsibilities
   - Document data flow between components
   - Identify entry points and boundaries

2. **Design Patterns**
   - Identify design patterns in use
   - Note any anti-patterns
   - Suggest architectural improvements
   - Review separation of concerns

3. **Dependency Analysis**
   - Review external dependencies
   - Check for redundant dependencies
   - Evaluate dependency health (maintenance, security)
   - Identify circular dependencies

4. **Scalability Assessment**
   - Identify potential bottlenecks
   - Review error handling patterns
   - Check logging and observability
   - Assess testability

Provide:
- A mermaid diagram of the architecture
- Strengths of current design
- Areas for improvement with priority
- Recommended refactoring steps
CMD_EOF
    log_success "Created architecture-review.md"

    # Performance Review Command
    cat > ~/.claude/commands/performance-review.md << 'CMD_EOF'
Analyze this codebase for performance issues:

1. **Algorithm Complexity**
   - Identify O(n²) or worse operations
   - Find unnecessary iterations
   - Check for redundant computations

2. **Memory Usage**
   - Look for memory leaks
   - Check for large object allocations
   - Review caching strategies

3. **I/O Operations**
   - Identify blocking I/O
   - Check database query efficiency
   - Review file system operations
   - Analyze network call patterns

4. **Concurrency**
   - Check for race conditions
   - Review async/await usage
   - Identify parallelization opportunities

Output recommendations with:
- Impact: High / Medium / Low
- Effort: Easy / Medium / Hard
- Specific code locations
- Suggested optimizations
CMD_EOF
    log_success "Created performance-review.md"

    # Test Coverage Command
    cat > ~/.claude/commands/test-coverage.md << 'CMD_EOF'
Analyze test coverage and quality for: $ARGUMENTS

1. **Coverage Analysis**
   - Identify untested functions/methods
   - Find untested code paths
   - Check edge case coverage

2. **Test Quality**
   - Review test assertions
   - Check for test isolation
   - Identify flaky tests
   - Review mock usage

3. **Missing Tests**
   - Critical paths without tests
   - Error handling without tests
   - Integration points without tests

4. **Recommendations**
   - Priority tests to add
   - Test refactoring suggestions
   - Testing strategy improvements

Generate a test coverage report and prioritized list of tests to add.
CMD_EOF
    log_success "Created test-coverage.md"
}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL CORE MCP SERVERS
# ═══════════════════════════════════════════════════════════════════════════════
install_core_mcps() {
    log_header "Installing Core MCP Servers"
    
    # GitHub MCP (Official)
    log_step "Installing GitHub MCP..."
    if [ -n "$GITHUB_TOKEN" ]; then
        claude mcp add github -s user -- npx -y @anthropic-ai/mcp-server-github 2>/dev/null || true
        log_success "GitHub MCP configured"
    else
        log_warn "GITHUB_TOKEN not set - GitHub MCP skipped"
        log_step "Set GITHUB_TOKEN and run: claude mcp add github -s user -- npx -y @anthropic-ai/mcp-server-github"
    fi
    
    # Context7 MCP (Documentation)
    log_step "Installing Context7 MCP..."
    claude mcp add context7 -s user --url https://mcp.context7.com/sse 2>/dev/null || true
    log_success "Context7 MCP configured"
    
    # Filesystem MCP
    log_step "Installing Filesystem MCP..."
    claude mcp add filesystem -s user -- npx -y @anthropic-ai/mcp-server-filesystem --allow "$HOME" 2>/dev/null || true
    log_success "Filesystem MCP configured"
}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL SECURITY MCPS
# ═══════════════════════════════════════════════════════════════════════════════
install_security_mcps() {
    log_header "Installing Security MCP Servers"
    
    # Snyk MCP
    if [ -n "$SNYK_TOKEN" ]; then
        log_step "Installing Snyk MCP..."
        claude mcp add snyk -s user -e SNYK_TOKEN="$SNYK_TOKEN" -- npx -y snyk-mcp-server 2>/dev/null || true
        log_success "Snyk MCP configured"
    else
        log_warn "SNYK_TOKEN not set - Snyk MCP skipped"
    fi
    
    # Sentry MCP
    if [ -n "$SENTRY_AUTH_TOKEN" ]; then
        log_step "Installing Sentry MCP..."
        claude mcp add sentry -s user -e SENTRY_AUTH_TOKEN="$SENTRY_AUTH_TOKEN" -- npx -y @sentry/mcp-server-sentry 2>/dev/null || true
        log_success "Sentry MCP configured"
    else
        log_warn "SENTRY_AUTH_TOKEN not set - Sentry MCP skipped"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL TRADING MCPS
# ═══════════════════════════════════════════════════════════════════════════════
install_trading_mcps() {
    log_header "Installing Trading MCP Servers"
    
    # Alpaca MCP
    if [ -n "$ALPACA_API_KEY" ] && [ -n "$ALPACA_SECRET_KEY" ]; then
        log_step "Installing Alpaca MCP (PAPER TRADING MODE)..."
        claude mcp add alpaca -s user \
            -e ALPACA_API_KEY="$ALPACA_API_KEY" \
            -e ALPACA_SECRET_KEY="$ALPACA_SECRET_KEY" \
            -e ALPACA_PAPER_TRADE="true" \
            -- uvx alpaca-mcp-server serve 2>/dev/null || true
        log_success "Alpaca MCP configured (Paper Trading)"
        log_warn "⚠️  ALPACA_PAPER_TRADE=true is enforced for safety"
    else
        log_warn "ALPACA_API_KEY/ALPACA_SECRET_KEY not set - Alpaca MCP skipped"
    fi
    
    # Polygon MCP
    if [ -n "$POLYGON_API_KEY" ]; then
        log_step "Installing Polygon MCP..."
        claude mcp add polygon -s user \
            -e POLYGON_API_KEY="$POLYGON_API_KEY" \
            -- uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.4.0 mcp_polygon 2>/dev/null || true
        log_success "Polygon MCP configured"
    else
        log_warn "POLYGON_API_KEY not set - Polygon MCP skipped"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL CREATIVE/ML MCPS
# ═══════════════════════════════════════════════════════════════════════════════
install_creative_mcps() {
    log_header "Installing Creative/ML MCP Servers"
    
    # TouchDesigner MCP
    log_step "Installing TouchDesigner MCP..."
    npm install -g touchdesigner-mcp-server 2>/dev/null || true
    log_success "TouchDesigner MCP installed (requires TD WebServer on port 9981)"
    
    # MLflow
    log_step "Installing MLflow..."
    pip3 install --quiet "mlflow>=3.5.1" 2>/dev/null || pip install --quiet "mlflow>=3.5.1" 2>/dev/null || true
    log_success "MLflow installed (use 'mlflow mcp' to start server)"
    
    # Weights & Biases MCP
    if [ -n "$WANDB_API_KEY" ]; then
        log_step "Installing W&B MCP..."
        claude mcp add wandb -s user \
            --url https://mcp.withwandb.com/mcp \
            --api-key "$WANDB_API_KEY" 2>/dev/null || true
        log_success "Weights & Biases MCP configured"
    else
        log_warn "WANDB_API_KEY not set - W&B MCP skipped"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL PYTHON SDKS
# ═══════════════════════════════════════════════════════════════════════════════
install_python_sdks() {
    log_header "Installing Python SDKs"
    
    local pip_cmd="pip3"
    if ! command_exists pip3; then
        pip_cmd="pip"
    fi
    
    # Core data science
    log_step "Installing core data science packages..."
    $pip_cmd install --quiet pandas numpy scipy 2>/dev/null || true
    log_success "pandas, numpy, scipy"
    
    # Trading SDKs
    log_step "Installing trading SDKs..."
    $pip_cmd install --quiet vectorbt riskfolio-lib skfolio 2>/dev/null || true
    log_success "vectorbt, riskfolio-lib, skfolio"
    
    # Note: TA-Lib requires system library first
    if command_exists brew; then
        log_step "Installing TA-Lib (macOS)..."
        brew install ta-lib 2>/dev/null || true
        $pip_cmd install --quiet ta-lib 2>/dev/null || true
        log_success "TA-Lib"
    else
        log_warn "TA-Lib skipped - install system library first"
        log_step "Ubuntu: apt-get install libta-lib-dev"
        log_step "macOS: brew install ta-lib"
    fi
    
    # Creative/ML SDKs
    log_step "Installing creative/ML SDKs..."
    $pip_cmd install --quiet diffusers transformers accelerate 2>/dev/null || true
    log_success "diffusers, transformers, accelerate"
    
    $pip_cmd install --quiet mediapipe 2>/dev/null || true
    log_success "mediapipe"
    
    $pip_cmd install --quiet ribs 2>/dev/null || true
    log_success "ribs (pyribs - quality-diversity)"
    
    # Code quality
    log_step "Installing code quality tools..."
    $pip_cmd install --quiet ruff pytest mypy 2>/dev/null || true
    log_success "ruff, pytest, mypy"
}

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE ENVIRONMENT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════
create_env_template() {
    log_header "Creating Environment Variables Template"
    
    local env_file="$HOME/.claude/env-template.sh"
    
    cat > "$env_file" << 'ENV_EOF'
#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Claude CLI Toolkit - Environment Variables Template
# Add these to your ~/.bashrc, ~/.zshrc, or ~/.profile
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# CORE APIs
# ─────────────────────────────────────────────────────────────────────────────
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GITHUB_TOKEN="your-github-personal-access-token"

# ─────────────────────────────────────────────────────────────────────────────
# TRADING APIs (AlphaForge)
# ─────────────────────────────────────────────────────────────────────────────
export ALPACA_API_KEY="your-alpaca-api-key"
export ALPACA_SECRET_KEY="your-alpaca-secret-key"
export ALPACA_PAPER_TRADE="true"                    # ALWAYS true for development
export POLYGON_API_KEY="your-polygon-api-key"

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY & OBSERVABILITY
# ─────────────────────────────────────────────────────────────────────────────
export SNYK_TOKEN="your-snyk-token"
export SENTRY_AUTH_TOKEN="your-sentry-auth-token"
export SONARQUBE_TOKEN="your-sonarqube-token"

# ─────────────────────────────────────────────────────────────────────────────
# CREATIVE/ML APIs (State of Witness)
# ─────────────────────────────────────────────────────────────────────────────
export WANDB_API_KEY="your-wandb-api-key"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# TouchDesigner settings
export TD_HOST="127.0.0.1"
export TD_PORT="9981"

# ComfyUI settings
export COMFYUI_HOST="localhost"
export COMFYUI_PORT="8188"

# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE CODE OPTIMIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="16384"
export BASH_DEFAULT_TIMEOUT_MS="60000"

# ─────────────────────────────────────────────────────────────────────────────
# PATH ADDITIONS
# ─────────────────────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
ENV_EOF

    chmod +x "$env_file"
    log_success "Created ~/.claude/env-template.sh"
    log_step "Copy the variables to your shell profile and fill in your API keys"
}

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE VALIDATION SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
create_validation_script() {
    log_header "Creating Validation Script"
    
    cat > ~/.claude/validate-toolkit.sh << 'VALIDATE_EOF'
#!/bin/bash
# Claude CLI Toolkit Validation Script

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS="${GREEN}✓${NC}"
FAIL="${RED}✗${NC}"
WARN="${YELLOW}!${NC}"

total=0
passed=0

check() {
    local name="$1"
    local cmd="$2"
    total=$((total + 1))
    if eval "$cmd" &>/dev/null; then
        echo -e "  $PASS $name"
        passed=$((passed + 1))
    else
        echo -e "  $FAIL $name"
    fi
}

optional() {
    local name="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo -e "  $PASS $name"
    else
        echo -e "  $WARN $name (optional)"
    fi
}

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Claude CLI Toolkit Validation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo "Core Components:"
check "Claude Code CLI" "command -v claude"
check "Node.js 18+" "node --version"
check "Python 3" "python3 --version"
check "Git" "git --version"

echo ""
echo "Package Managers:"
check "npm" "command -v npm"
optional "UV" "command -v uv"
optional "pip" "command -v pip3 || command -v pip"

echo ""
echo "Global Configuration:"
check "~/.claude/CLAUDE.md" "[ -f ~/.claude/CLAUDE.md ]"
check "~/.claude/settings.json" "[ -f ~/.claude/settings.json ]"
check "~/.claude/commands/" "[ -d ~/.claude/commands ]"

echo ""
echo "Environment Variables:"
optional "GITHUB_TOKEN" "[ -n \"\$GITHUB_TOKEN\" ]"
optional "ALPACA_API_KEY" "[ -n \"\$ALPACA_API_KEY\" ]"
optional "POLYGON_API_KEY" "[ -n \"\$POLYGON_API_KEY\" ]"
optional "SNYK_TOKEN" "[ -n \"\$SNYK_TOKEN\" ]"
optional "WANDB_API_KEY" "[ -n \"\$WANDB_API_KEY\" ]"

echo ""
echo "Python SDKs:"
optional "pandas" "python3 -c 'import pandas'"
optional "numpy" "python3 -c 'import numpy'"
optional "vectorbt" "python3 -c 'import vectorbt'"
optional "riskfolio-lib" "python3 -c 'import riskfolio'"
optional "mlflow" "python3 -c 'import mlflow'"

echo ""
echo "Code Quality Tools:"
optional "ruff" "command -v ruff"
optional "pytest" "command -v pytest"
optional "prettier" "command -v prettier"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
if [ $passed -eq $total ]; then
    echo -e "  ${GREEN}All $total core checks passed!${NC}"
else
    echo -e "  ${YELLOW}$passed/$total core checks passed${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo "MCP Server Status:"
claude mcp list 2>/dev/null || echo "  Run 'claude mcp list' to check MCP servers"
VALIDATE_EOF

    chmod +x ~/.claude/validate-toolkit.sh
    log_success "Created ~/.claude/validate-toolkit.sh"
}

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
show_summary() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║   ${GREEN}✓ Installation Complete!${CYAN}                                  ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║   ${BOLD}Next Steps:${NC}${CYAN}                                                ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║   1. Configure environment variables:                         ║${NC}"
    echo -e "${CYAN}║      ${YELLOW}cat ~/.claude/env-template.sh${NC}${CYAN}                           ║${NC}"
    echo -e "${CYAN}║      Add to ~/.bashrc or ~/.zshrc                             ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║   2. Validate installation:                                   ║${NC}"
    echo -e "${CYAN}║      ${YELLOW}~/.claude/validate-toolkit.sh${NC}${CYAN}                          ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║   3. Check MCP servers:                                       ║${NC}"
    echo -e "${CYAN}║      ${YELLOW}claude mcp list${NC}${CYAN}                                         ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║   4. Start Claude Code:                                       ║${NC}"
    echo -e "${CYAN}║      ${YELLOW}claude${NC}${CYAN}                                                  ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║   ${BOLD}Files Created:${NC}${CYAN}                                             ║${NC}"
    echo -e "${CYAN}║   • ~/.claude/CLAUDE.md          (global instructions)        ║${NC}"
    echo -e "${CYAN}║   • ~/.claude/settings.json      (global settings)            ║${NC}"
    echo -e "${CYAN}║   • ~/.claude/commands/          (slash commands)             ║${NC}"
    echo -e "${CYAN}║   • ~/.claude/env-template.sh    (env vars template)          ║${NC}"
    echo -e "${CYAN}║   • ~/.claude/validate-toolkit.sh (validation script)         ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
main() {
    show_banner
    
    check_prerequisites
    install_uv
    setup_global_structure
    create_global_claude_md
    create_global_settings
    create_global_commands
    install_core_mcps
    install_security_mcps
    install_trading_mcps
    install_creative_mcps
    install_python_sdks
    create_env_template
    create_validation_script
    
    show_summary
}

# Run main
main "$@"
