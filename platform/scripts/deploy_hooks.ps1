<#
.SYNOPSIS
    Deploy V10.1 hooks to the Claude Code hooks directory.

.DESCRIPTION
    This script deploys hook files from v10_optimized/hooks/ to ~/.claude/v10/hooks/
    and creates the necessary directory structure for the V10.1 system.

    Based on Claude Code Hooks v2.0.10+ specification:
    https://code.claude.com/docs/en/hooks

.PARAMETER Force
    Overwrite existing hook files without prompting.

.PARAMETER Verify
    Run verification after deployment to confirm hooks are properly installed.

.EXAMPLE
    .\deploy_hooks.ps1
    # Interactive deployment with prompts

.EXAMPLE
    .\deploy_hooks.ps1 -Force -Verify
    # Force overwrite and verify after deployment

.NOTES
    Version: 1.0.0
    Date: January 17, 2026
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$Verify
)

$ErrorActionPreference = "Stop"

# Configuration
$SourceDir = Split-Path -Parent $PSScriptRoot
$HooksSource = Join-Path $SourceDir "hooks"
$V10Dir = Join-Path $env:USERPROFILE ".claude\v10"
$HooksTarget = Join-Path $V10Dir "hooks"
$LogsDir = Join-Path $V10Dir "logs"
$CacheDir = Join-Path $V10Dir "cache"

# Hook files to deploy
$HookFiles = @(
    "hook_utils.py",
    "letta_sync_v2.py",
    "mcp_guard_v2.py",
    "letta_sync.py",
    "mcp_guard.py",
    "bash_guard.py",
    "memory_consolidate.py",
    "audit_log.py"
)

function Write-Status {
    param(
        [string]$Message,
        [ValidateSet("Info", "Success", "Warning", "Error")]
        [string]$Type = "Info"
    )

    $prefix = switch ($Type) {
        "Info"    { "[*]" }
        "Success" { "[OK]" }
        "Warning" { "[!]" }
        "Error"   { "[X]" }
    }

    $color = switch ($Type) {
        "Info"    { "Cyan" }
        "Success" { "Green" }
        "Warning" { "Yellow" }
        "Error"   { "Red" }
    }

    Write-Host "$prefix $Message" -ForegroundColor $color
}

function Test-PythonAvailable {
    try {
        $null = python --version 2>&1
        return $true
    } catch {
        return $false
    }
}

function Test-UvAvailable {
    try {
        $null = uv --version 2>&1
        return $true
    } catch {
        return $false
    }
}

# Main deployment logic
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  V10.1 Hooks Deployment Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify source directory exists
Write-Status "Checking source directory..." "Info"
if (-not (Test-Path $HooksSource)) {
    Write-Status "Source hooks directory not found: $HooksSource" "Error"
    exit 1
}
Write-Status "Source directory: $HooksSource" "Success"

# Step 2: Create target directories
Write-Status "Creating target directories..." "Info"

$directories = @($V10Dir, $HooksTarget, $LogsDir, $CacheDir)
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Status "Created: $dir" "Success"
    } else {
        Write-Status "Exists: $dir" "Info"
    }
}

# Step 3: Deploy hook files
Write-Host ""
Write-Status "Deploying hook files..." "Info"

$deployedCount = 0
$skippedCount = 0
$errorCount = 0

foreach ($file in $HookFiles) {
    $source = Join-Path $HooksSource $file
    $target = Join-Path $HooksTarget $file

    if (-not (Test-Path $source)) {
        Write-Status "Source file not found: $file" "Warning"
        $skippedCount++
        continue
    }

    $shouldCopy = $true
    if ((Test-Path $target) -and (-not $Force)) {
        $sourceHash = (Get-FileHash $source -Algorithm MD5).Hash
        $targetHash = (Get-FileHash $target -Algorithm MD5).Hash

        if ($sourceHash -eq $targetHash) {
            Write-Status "$file - Already up to date" "Info"
            $shouldCopy = $false
            $skippedCount++
        } else {
            $response = Read-Host "  $file exists and differs. Overwrite? [y/N]"
            if ($response -notmatch '^[Yy]') {
                Write-Status "$file - Skipped (user choice)" "Warning"
                $shouldCopy = $false
                $skippedCount++
            }
        }
    }

    if ($shouldCopy) {
        try {
            Copy-Item -Path $source -Destination $target -Force
            Write-Status "$file - Deployed" "Success"
            $deployedCount++
        } catch {
            Write-Status "$file - Failed: $_" "Error"
            $errorCount++
        }
    }
}

# Step 4: Create settings.json symlink or copy
Write-Host ""
Write-Status "Checking configuration..." "Info"

$settingsSource = Join-Path $SourceDir "config\settings.json"
$settingsTarget = Join-Path $V10Dir "settings.json"

if (Test-Path $settingsSource) {
    if (-not (Test-Path $settingsTarget) -or $Force) {
        Copy-Item -Path $settingsSource -Destination $settingsTarget -Force
        Write-Status "Copied settings.json" "Success"
    } else {
        Write-Status "settings.json already exists" "Info"
    }
}

# Step 5: Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Deployment Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Status "Deployed: $deployedCount files" "Success"
if ($skippedCount -gt 0) {
    Write-Status "Skipped: $skippedCount files" "Warning"
}
if ($errorCount -gt 0) {
    Write-Status "Errors: $errorCount files" "Error"
}
Write-Host ""
Write-Status "Target directory: $HooksTarget" "Info"

# Step 6: Verification (optional)
if ($Verify) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Verification" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    # Check Python syntax of deployed files
    if (Test-PythonAvailable) {
        Write-Status "Checking Python syntax..." "Info"

        $syntaxErrors = 0
        foreach ($file in $HookFiles) {
            $target = Join-Path $HooksTarget $file
            if (Test-Path $target) {
                try {
                    $result = python -m py_compile $target 2>&1
                    if ($LASTEXITCODE -eq 0) {
                        Write-Status "$file - Syntax OK" "Success"
                    } else {
                        Write-Status "$file - Syntax error: $result" "Error"
                        $syntaxErrors++
                    }
                } catch {
                    Write-Status "$file - Check failed: $_" "Warning"
                }
            }
        }

        if ($syntaxErrors -eq 0) {
            Write-Status "All deployed files have valid Python syntax" "Success"
        }
    } else {
        Write-Status "Python not available for syntax verification" "Warning"
    }

    # Run health check if uv is available
    $healthCheck = Join-Path $PSScriptRoot "health_check.py"
    if ((Test-Path $healthCheck) -and (Test-UvAvailable)) {
        Write-Host ""
        Write-Status "Running health check..." "Info"
        Push-Location $PSScriptRoot
        try {
            uv run health_check.py --quick
        } catch {
            Write-Status "Health check failed: $_" "Warning"
        }
        Pop-Location
    }
}

# Step 7: Next steps
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Next Steps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Configure hooks in Claude Code settings:" -ForegroundColor White
Write-Host "   Add hook configurations to ~/.claude/settings.json" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start Letta server (optional, for memory persistence):" -ForegroundColor White
Write-Host "   docker run -d -p 8283:8283 letta/letta:latest" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Run full health check:" -ForegroundColor White
Write-Host "   uv run health_check.py" -ForegroundColor Gray
Write-Host ""

if ($errorCount -gt 0) {
    exit 1
}
exit 0
