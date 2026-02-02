# UNLEASH Code Intelligence Environment Setup
# ============================================
# Source this file to set up required environment variables
#
# Usage:
#   . .\platform\scripts\env.ps1
#
# NOTE: Replace placeholder values with your actual API keys

# Voyage AI (for code embeddings)
# Get your key at: https://dash.voyageai.com/
$env:VOYAGE_API_KEY = "YOUR_VOYAGE_API_KEY_HERE"

# Opik (for observability, optional)
# Get your key at: https://www.comet.com/opik
$env:OPIK_API_KEY = "YOUR_OPIK_API_KEY_HERE"
$env:OPIK_PROJECT_NAME = "unleash-code-intelligence"

# Qdrant (local by default)
$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_COLLECTION = "unleash_code"

# narsil-mcp cache
$env:NARSIL_CACHE_DIR = "Z:/insider/AUTO CLAUDE/unleash/.narsil-cache"

# Add Go bin to PATH (for mcp-language-server)
$gobin = "$env:USERPROFILE\go\bin"
if (Test-Path $gobin) {
    $env:PATH = "$gobin;$env:PATH"
    Write-Host "[+] Added Go bin to PATH: $gobin" -ForegroundColor Green
}

# Add Cargo bin to PATH (for narsil-mcp)
$cargobin = "$env:USERPROFILE\.cargo\bin"
if (Test-Path $cargobin) {
    $env:PATH = "$cargobin;$env:PATH"
    Write-Host "[+] Added Cargo bin to PATH: $cargobin" -ForegroundColor Green
}

# Verify API keys are set
if ($env:VOYAGE_API_KEY -eq "YOUR_VOYAGE_API_KEY_HERE") {
    Write-Host "[!] WARNING: VOYAGE_API_KEY not configured - embeddings will fail" -ForegroundColor Yellow
    Write-Host "    Get your key at: https://dash.voyageai.com/" -ForegroundColor Yellow
} else {
    Write-Host "[+] VOYAGE_API_KEY configured" -ForegroundColor Green
}

# Verify Qdrant is running
try {
    $response = Invoke-WebRequest -Uri "$env:QDRANT_URL/healthz" -TimeoutSec 2 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "[+] Qdrant is running at $env:QDRANT_URL" -ForegroundColor Green
    }
} catch {
    Write-Host "[!] WARNING: Qdrant not responding at $env:QDRANT_URL" -ForegroundColor Yellow
    Write-Host "    Start with: docker run -d -p 6333:6333 qdrant/qdrant" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Environment configured for UNLEASH Code Intelligence" -ForegroundColor Cyan
Write-Host "Run 'python -m platform.core.embedding_pipeline --help' to get started" -ForegroundColor Cyan
