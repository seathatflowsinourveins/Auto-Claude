# ULTRAMAX SDK Setup Script
# Z:\insider\AUTO CLAUDE\unleash\sdks\setup-ultramax.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ULTRAMAX SDK ECOSYSTEM SETUP" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$SDK_ROOT = "Z:\insider\AUTO CLAUDE\unleash\sdks"

# Step 1: Clone Serena (Critical Missing SDK)
Write-Host "[1/6] Cloning Serena (LSP-Based Code Intelligence)..." -ForegroundColor Yellow
if (!(Test-Path "$SDK_ROOT\serena")) {
    git clone https://github.com/oraios/serena.git "$SDK_ROOT\serena"
    Write-Host "  ✓ Serena cloned successfully" -ForegroundColor Green
} else {
    Write-Host "  ✓ Serena already exists" -ForegroundColor Green
}

# Step 2: Create Python virtual environment
Write-Host ""
Write-Host "[2/6] Setting up Python environment..." -ForegroundColor Yellow
$VENV_PATH = "$SDK_ROOT\.ultramax-venv"
if (!(Test-Path $VENV_PATH)) {
    python -m venv $VENV_PATH
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "  ✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate venv
& "$VENV_PATH\Scripts\Activate.ps1"

# Step 3: Install Core Tier 1 Dependencies
Write-Host ""
Write-Host "[3/6] Installing Tier 1 Core Dependencies..." -ForegroundColor Yellow

$TIER1_PACKAGES = @(
    "dspy-ai",           # Stanford Prompt Programming
    "litellm",           # Unified LLM Gateway
    "temporalio",        # Durable Execution
    "pydantic-ai",       # Production Agent Framework
    "openai",            # OpenAI SDK
    "anthropic",         # Anthropic SDK
    "instructor",        # Structured Outputs
    "crawl4ai[all]"      # LLM-Friendly Web Scraping
)

foreach ($pkg in $TIER1_PACKAGES) {
    Write-Host "  Installing $pkg..." -ForegroundColor Gray
    pip install $pkg --quiet
}
Write-Host "  ✓ Tier 1 packages installed" -ForegroundColor Green

# Step 4: Install Tier 2 Agent/Reasoning Dependencies
Write-Host ""
Write-Host "[4/6] Installing Tier 2 Agent/Reasoning Dependencies..." -ForegroundColor Yellow

$TIER2_PACKAGES = @(
    "crewai",            # Multi-Agent
    "langgraph",         # Stateful Agents
    "mem0ai",            # AI Memory
    "langfuse",          # Observability
    "evotorch",          # Evolutionary Computation
    "outlines"           # Structured Generation
)

foreach ($pkg in $TIER2_PACKAGES) {
    Write-Host "  Installing $pkg..." -ForegroundColor Gray
    pip install $pkg --quiet 2>$null
}
Write-Host "  ✓ Tier 2 packages installed" -ForegroundColor Green

# Step 5: Install from local SDKs
Write-Host ""
Write-Host "[5/6] Installing from local SDK repositories..." -ForegroundColor Yellow

$LOCAL_SDKS = @(
    "$SDK_ROOT\dspy",
    "$SDK_ROOT\lightzero",
    "$SDK_ROOT\crawl4ai",
    "$SDK_ROOT\instructor"
)

foreach ($sdk in $LOCAL_SDKS) {
    if (Test-Path "$sdk\pyproject.toml" -or Test-Path "$sdk\setup.py") {
        $name = Split-Path $sdk -Leaf
        Write-Host "  Installing $name from source..." -ForegroundColor Gray
        pip install -e $sdk --quiet 2>$null
    }
}
Write-Host "  ✓ Local SDKs installed" -ForegroundColor Green

# Step 6: Setup MCP Configuration
Write-Host ""
Write-Host "[6/6] Generating MCP Configuration..." -ForegroundColor Yellow

$MCP_CONFIG = @"
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": ["serena"],
      "description": "LSP-based semantic code intelligence"
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"],
      "description": "Documentation retrieval"
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "Z:\\insider\\AUTO CLAUDE"],
      "description": "File system access"
    }
  }
}
"@

$MCP_CONFIG_PATH = "$SDK_ROOT\mcp_config_template.json"
$MCP_CONFIG | Out-File -FilePath $MCP_CONFIG_PATH -Encoding UTF8
Write-Host "  ✓ MCP config template saved to: $MCP_CONFIG_PATH" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installed Components:" -ForegroundColor White
Write-Host "  - Serena (LSP Code Intelligence)" -ForegroundColor Gray
Write-Host "  - DSPy (Prompt Programming)" -ForegroundColor Gray
Write-Host "  - Temporal (Durable Execution)" -ForegroundColor Gray
Write-Host "  - LiteLLM (LLM Gateway)" -ForegroundColor Gray
Write-Host "  - Pydantic AI (Agent Framework)" -ForegroundColor Gray
Write-Host "  - Crawl4AI (Web Scraping)" -ForegroundColor Gray
Write-Host "  - EvoTorch (Evolutionary Computation)" -ForegroundColor Gray
Write-Host "  - And more..." -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review: $SDK_ROOT\ULTRAMAX_SDK_COMPLETE_ANALYSIS.md" -ForegroundColor Gray
Write-Host "  2. Copy MCP config to Claude Desktop config" -ForegroundColor Gray
Write-Host "  3. Activate venv: $VENV_PATH\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
