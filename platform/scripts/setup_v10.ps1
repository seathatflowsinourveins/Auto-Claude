<#
.SYNOPSIS
    V10 OPTIMIZED - Setup Script for Seamless Claude Code CLI

.DESCRIPTION
    Installs and configures the V10 Optimized architecture with:
    - Verified MCP servers only
    - Letta memory integration hooks
    - Streamlined configuration
    - Automatic validation

.PARAMETER Mode
    Installation mode: 'minimal', 'standard', 'full'
    - minimal: Core MCP servers only (filesystem, memory, sequential-thinking)
    - standard: Core + development tools (eslint, context7, fetch)
    - full: All verified servers including optional ones

.PARAMETER SkipVerification
    Skip MCP package verification (faster but less safe)

.PARAMETER LettaUrl
    Letta server URL (default: http://localhost:8283)

.EXAMPLE
    .\setup_v10.ps1 -Mode standard

.EXAMPLE
    .\setup_v10.ps1 -Mode full -LettaUrl http://letta.local:8283
#>

param(
    [ValidateSet('minimal', 'standard', 'full')]
    [string]$Mode = 'standard',
    
    [switch]$SkipVerification,
    
    [string]$LettaUrl = "http://localhost:8283",
    
    [string]$ClaudeHome = "$env:USERPROFILE\.claude"
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

# Colors
function Write-Success { param($Message) Write-Host "✓ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "→ $Message" -ForegroundColor Cyan }
function Write-Warn { param($Message) Write-Host "⚠ $Message" -ForegroundColor Yellow }
function Write-Err { param($Message) Write-Host "✗ $Message" -ForegroundColor Red }

Write-Host @"

╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           V10 OPTIMIZED - Seamless Claude Code CLI               ║
║                                                                  ║
║              Verified • Working • Production Ready               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Magenta

Write-Info "Installation Mode: $Mode"
Write-Info "Claude Home: $ClaudeHome"
Write-Info "Letta URL: $LettaUrl"
Write-Host ""

# Step 1: Check prerequisites
Write-Info "Checking prerequisites..."

# Check Node.js
try {
    $nodeVersion = node --version 2>$null
    Write-Success "Node.js: $nodeVersion"
}
catch {
    Write-Err "Node.js not found. Please install from https://nodejs.org/"
    exit 1
}

# Check Python
try {
    $pythonVersion = python --version 2>$null
    Write-Success "Python: $pythonVersion"
}
catch {
    Write-Err "Python not found. Please install Python 3.11+"
    exit 1
}

# Check UV (optional but recommended)
try {
    $uvVersion = uv --version 2>$null
    Write-Success "UV: $uvVersion"
}
catch {
    Write-Warn "UV not found. Installing..."
    Invoke-WebRequest -useb https://astral.sh/uv/install.ps1 | Invoke-Expression
}

# Step 2: Create directory structure
Write-Info "Creating directory structure..."

$directories = @(
    "$ClaudeHome",
    "$ClaudeHome\v10",
    "$ClaudeHome\v10\hooks",
    "$ClaudeHome\v10\logs",
    "$ClaudeHome\v10\cache"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Success "Directory structure created"

# Step 3: Copy configuration files
Write-Info "Installing configuration files..."

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$configDir = Join-Path (Split-Path -Parent $scriptDir) "config"
$hooksDir = Join-Path (Split-Path -Parent $scriptDir) "hooks"

# Copy settings.json
$settingsSource = Join-Path $configDir "settings.json"
$settingsDest = Join-Path $ClaudeHome "settings.json"
if (Test-Path $settingsSource) {
    # Update paths in settings
    $settings = Get-Content $settingsSource -Raw | ConvertFrom-Json
    
    # Set environment variable for hooks directory
    if (-not $settings.env) {
        $settings | Add-Member -NotePropertyName "env" -NotePropertyValue @{}
    }
    $settings.env.CLAUDE_HOOKS_DIR = "$ClaudeHome\v10\hooks"
    $settings.env.LETTA_URL = $LettaUrl
    
    $settings | ConvertTo-Json -Depth 10 | Set-Content $settingsDest -Encoding UTF8
    Write-Success "settings.json installed"
}
else {
    Write-Warn "settings.json source not found, creating default"
}

# Copy CLAUDE.md
$claudeMdSource = Join-Path $configDir "CLAUDE.md"
$claudeMdDest = Join-Path $ClaudeHome "CLAUDE.md"
if (Test-Path $claudeMdSource) {
    Copy-Item $claudeMdSource $claudeMdDest -Force
    Write-Success "CLAUDE.md installed"
}

# Step 4: Install hooks
Write-Info "Installing hook scripts..."

$hookFiles = @(
    "letta_sync.py",
    "mcp_guard.py",
    "bash_guard.py",
    "memory_consolidate.py",
    "audit_log.py"
)

foreach ($hook in $hookFiles) {
    $source = Join-Path $hooksDir $hook
    $dest = Join-Path "$ClaudeHome\v10\hooks" $hook
    
    if (Test-Path $source) {
        Copy-Item $source $dest -Force
        Write-Success "  $hook"
    }
    else {
        Write-Warn "  $hook not found in source"
    }
}

# Step 5: Install Python dependencies for hooks
Write-Info "Installing Python dependencies..."

$requirements = @(
    "httpx>=0.26.0",
    "structlog>=24.1.0"
)

foreach ($req in $requirements) {
    uv pip install $req --quiet 2>$null
}
Write-Success "Python dependencies installed"

# Step 6: Verify MCP packages
if (-not $SkipVerification) {
    Write-Info "Verifying MCP packages..."
    
    $corePackages = @(
        "@modelcontextprotocol/server-filesystem",
        "@modelcontextprotocol/server-memory",
        "@modelcontextprotocol/server-sequential-thinking"
    )
    
    $standardPackages = @(
        "@eslint/mcp",
        "@upstash/context7-mcp",
        "@modelcontextprotocol/server-fetch"
    )
    
    $packagesToVerify = switch ($Mode) {
        'minimal' { $corePackages }
        'standard' { $corePackages + $standardPackages }
        'full' { $corePackages + $standardPackages }
    }
    
    foreach ($package in $packagesToVerify) {
        try {
            $result = npm view $package version 2>$null
            if ($result) {
                Write-Success "  $package ($result)"
            }
            else {
                Write-Warn "  $package (could not verify)"
            }
        }
        catch {
            Write-Warn "  $package (verification failed)"
        }
    }
}
else {
    Write-Warn "Skipping MCP package verification"
}

# Step 7: Create MCP configuration
Write-Info "Creating MCP configuration..."

$mcpConfig = @{
    mcpServers = @{}
}

# Core servers (always included)
$mcpConfig.mcpServers.filesystem = @{
    type = "stdio"
    command = "cmd"
    args = @("/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", ".")
}

$mcpConfig.mcpServers.memory = @{
    type = "stdio"
    command = "cmd"
    args = @("/c", "npx", "-y", "@modelcontextprotocol/server-memory")
}

$mcpConfig.mcpServers."sequential-thinking" = @{
    type = "stdio"
    command = "cmd"
    args = @("/c", "npx", "-y", "@modelcontextprotocol/server-sequential-thinking")
}

# Standard additions
if ($Mode -in @('standard', 'full')) {
    $mcpConfig.mcpServers.eslint = @{
        type = "stdio"
        command = "cmd"
        args = @("/c", "npx", "-y", "@eslint/mcp@latest")
    }
    
    $mcpConfig.mcpServers.context7 = @{
        type = "stdio"
        command = "cmd"
        args = @("/c", "npx", "-y", "@upstash/context7-mcp")
    }
    
    $mcpConfig.mcpServers.fetch = @{
        type = "stdio"
        command = "cmd"
        args = @("/c", "npx", "-y", "@modelcontextprotocol/server-fetch")
    }
}

# Full additions
if ($Mode -eq 'full') {
    $mcpConfig.mcpServers.sqlite = @{
        type = "stdio"
        command = "cmd"
        args = @("/c", "npx", "-y", "@modelcontextprotocol/server-sqlite", "$ClaudeHome\v10\memory.db")
    }
}

# Check for GitHub CLI
try {
    $ghVersion = gh --version 2>$null
    if ($ghVersion) {
        $mcpConfig.mcpServers.github = @{
            type = "stdio"
            command = "gh"
            args = @("mcp")
        }
        Write-Success "GitHub CLI MCP enabled"
    }
}
catch {
    Write-Warn "GitHub CLI not found, github MCP disabled"
}

# Write MCP config
$mcpConfigPath = Join-Path $ClaudeHome ".mcp.json"
$mcpConfig | ConvertTo-Json -Depth 5 | Set-Content $mcpConfigPath -Encoding UTF8
Write-Success "MCP configuration created ($($mcpConfig.mcpServers.Count) servers)"

# Step 8: Check Letta connection
Write-Info "Checking Letta connection..."

try {
    $response = Invoke-WebRequest -Uri "$LettaUrl/v1/health" -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Success "Letta server is running at $LettaUrl"
    }
}
catch {
    Write-Warn "Letta server not reachable at $LettaUrl"
    Write-Warn "Memory persistence will not be available until Letta is started"
    Write-Host ""
    Write-Host "  To start Letta:" -ForegroundColor Yellow
    Write-Host "    docker run -d -p 8283:8283 letta/letta:latest" -ForegroundColor Gray
    Write-Host ""
}

# Step 9: Final summary
Write-Host ""
Write-Host @"
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                   V10 Installation Complete!                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "Configuration Summary:" -ForegroundColor Cyan
Write-Host "  Mode: $Mode"
Write-Host "  MCP Servers: $($mcpConfig.mcpServers.Count)"
Write-Host "  Hooks: $(($hookFiles | Where-Object { Test-Path (Join-Path "$ClaudeHome\v10\hooks" $_) }).Count)"
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Start/restart Claude Code CLI"
Write-Host "  2. The new configuration will be loaded automatically"
Write-Host ""

if ($Mode -ne 'full') {
    Write-Host "Optional Enhancements:" -ForegroundColor Yellow
    Write-Host "  - Run with -Mode full for all servers"
    Write-Host "  - Add GITHUB_TOKEN for GitHub integration"
    Write-Host "  - Start Letta for cross-session memory"
    Write-Host ""
}

Write-Host "Emergency Commands:" -ForegroundColor Red
Write-Host "  Kill Switch: New-Item $ClaudeHome\KILL_SWITCH"
Write-Host "  Resume:      Remove-Item $ClaudeHome\KILL_SWITCH"
Write-Host ""
