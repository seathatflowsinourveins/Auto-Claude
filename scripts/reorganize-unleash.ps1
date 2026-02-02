# UNLEASH SDK Reorganization Script
# Version: 30.0
# Purpose: Reorganize 170+ repositories into tiered structure

$UNLEASH_ROOT = "Z:\insider\AUTO CLAUDE\unleash"
$STACK_DIR = "$UNLEASH_ROOT\stack"
$SKILLS_DIR = "$UNLEASH_ROOT\skills"
$ARCHIVE_DIR = "$UNLEASH_ROOT\archive"

Write-Host "=== UNLEASH SDK Reorganization Script ===" -ForegroundColor Cyan
Write-Host "Root: $UNLEASH_ROOT" -ForegroundColor Gray

# Create new directory structure
$tiers = @(
    "tier-0-critical",
    "tier-1-orchestration",
    "tier-2-memory",
    "tier-3-reasoning",
    "tier-4-evolution",
    "tier-5-safety",
    "tier-6-evaluation",
    "tier-7-code",
    "tier-8-document",
    "tier-9-protocols"
)

Write-Host "`nCreating directory structure..." -ForegroundColor Yellow
foreach ($tier in $tiers) {
    $path = "$STACK_DIR\$tier"
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "  Created: $tier" -ForegroundColor Green
    }
}

# Skills directories
$skillDirs = @("superpowers", "everything-claude-code", "workflow-skills", "anthropic-official", "plugins")
foreach ($dir in $skillDirs) {
    $path = "$SKILLS_DIR\$dir"
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# Define SDK mappings (source -> tier)
$tier0Critical = @{
    # Critical 14 SDKs
    "temporal-python" = "temporal"
    "langgraph" = "langgraph"
    "fastmcp" = "fastmcp"
    "instructor" = "instructor"
    "litellm" = "litellm"
    "langfuse" = "langfuse"
    "pydantic-ai" = "pydantic-ai"
    "crewai" = "crewai"
    "zep" = "zep"
    "deepeval" = "deepeval"
    "pyribs" = "pyribs"
    "ast-grep" = "ast-grep"
    "nemo-guardrails" = "nemo-guardrails"
    "dspy" = "dspy"
}

$tier1Orchestration = @{
    "autogen" = "autogen"
    "smolagents" = "smolagents"
    "agent-squad" = "agent-squad"
    "strands-agents" = "strands-agents"
    "openai-agents" = "openai-agents"
    "fast-agent" = "fast-agent"
    "kagent" = "kagent"
    "hive-agents" = "hive-agents"
    "camel-ai" = "camel-ai"
    "deer-flow" = "deer-flow"
}

$tier2Memory = @{
    "graphiti" = "graphiti"
    "letta" = "letta"
    "mem0" = "mem0"
    "mem0-full" = "mem0-full"
    "hindsight" = "hindsight"
    "memgpt" = "memgpt"
}

$tier3Reasoning = @{
    "llm-reasoners" = "llm-reasoners"
    "tree-of-thoughts" = "tree-of-thoughts"
    "reflexion" = "reflexion"
    "thinking-claude" = "thinking-claude"
    "sketch-of-thought" = "sketch-of-thought"
}

$tier4Evolution = @{
    "EvoAgentX" = "evoagentx"
    "evotorch" = "evotorch"
    "textgrad" = "textgrad"
    "qdax" = "qdax"
    "tensorneat" = "tensorneat"
}

$tier5Safety = @{
    "llm-guard" = "llm-guard"
    "guardrails-ai" = "guardrails-ai"
    "purplellama" = "purplellama"
    "rebuff" = "rebuff"
    "any-guardrail" = "any-guardrail"
}

$tier6Evaluation = @{
    "ragas" = "ragas"
    "opik" = "opik"
    "opik-full" = "opik-full"
    "promptfoo" = "promptfoo"
    "swe-bench" = "swe-bench"
    "swe-agent" = "swe-agent"
    "tau-bench" = "tau-bench"
    "letta-evals" = "letta-evals"
    "braintrust" = "braintrust"
    "agentops" = "agentops"
}

$tier7Code = @{
    "aider" = "aider"
    "serena" = "serena"
    "cline" = "cline"
    "continue" = "continue"
    "code-reasoning" = "code-reasoning"
}

$tier8Document = @{
    "docling" = "docling"
    "unstructured" = "unstructured"
    "chonkie" = "chonkie"
    "crawl4ai" = "crawl4ai"
    "firecrawl" = "firecrawl"
    "firecrawl-sdk" = "firecrawl-sdk"
}

$tier9Protocols = @{
    "mcp" = "mcp"
    "mcp-python-sdk" = "mcp-python-sdk"
    "mcp-typescript-sdk" = "mcp-typescript-sdk"
    "mcp-servers" = "mcp-servers"
    "mcp-agent" = "mcp-agent"
    "a2a-protocol" = "a2a-protocol"
    "acp-sdk" = "acp-sdk"
    "agent-rpc" = "agent-rpc"
}

# Function to create symlink
function Create-SDKLink {
    param(
        [string]$Source,
        [string]$Target
    )

    if (Test-Path $Source) {
        if (-not (Test-Path $Target)) {
            # Create junction (symlink for directories)
            cmd /c mklink /J "$Target" "$Source" 2>$null
            if ($?) {
                Write-Host "  Linked: $Target" -ForegroundColor Green
            } else {
                Write-Host "  Failed: $Target" -ForegroundColor Red
            }
        } else {
            Write-Host "  Exists: $Target" -ForegroundColor Gray
        }
    } else {
        Write-Host "  Missing source: $Source" -ForegroundColor Yellow
    }
}

# Process each tier
Write-Host "`nProcessing Tier 0 (Critical)..." -ForegroundColor Cyan
foreach ($sdk in $tier0Critical.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-0-critical\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

Write-Host "`nProcessing Tier 1 (Orchestration)..." -ForegroundColor Cyan
foreach ($sdk in $tier1Orchestration.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-1-orchestration\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

# Add Claude-Flow from advanced
$cfSource = "$UNLEASH_ROOT\github-advanced-unleash\claude-flow-v3"
$cfTarget = "$STACK_DIR\tier-1-orchestration\claude-flow-v3"
Create-SDKLink -Source $cfSource -Target $cfTarget

Write-Host "`nProcessing Tier 2 (Memory)..." -ForegroundColor Cyan
foreach ($sdk in $tier2Memory.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-2-memory\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

Write-Host "`nProcessing Tier 3 (Reasoning)..." -ForegroundColor Cyan
foreach ($sdk in $tier3Reasoning.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-3-reasoning\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

Write-Host "`nProcessing Tier 4 (Evolution)..." -ForegroundColor Cyan
foreach ($sdk in $tier4Evolution.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-4-evolution\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

# Add EvoAgentX from advanced
$evoSource = "$UNLEASH_ROOT\github-advanced-unleash\evoagentx-evolution"
$evoTarget = "$STACK_DIR\tier-4-evolution\evoagentx-advanced"
Create-SDKLink -Source $evoSource -Target $evoTarget

Write-Host "`nProcessing Tier 5 (Safety)..." -ForegroundColor Cyan
foreach ($sdk in $tier5Safety.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-5-safety\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

Write-Host "`nProcessing Tier 6 (Evaluation)..." -ForegroundColor Cyan
foreach ($sdk in $tier6Evaluation.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-6-evaluation\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

Write-Host "`nProcessing Tier 7 (Code)..." -ForegroundColor Cyan
foreach ($sdk in $tier7Code.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-7-code\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

Write-Host "`nProcessing Tier 8 (Document)..." -ForegroundColor Cyan
foreach ($sdk in $tier8Document.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-8-document\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

Write-Host "`nProcessing Tier 9 (Protocols)..." -ForegroundColor Cyan
foreach ($sdk in $tier9Protocols.GetEnumerator()) {
    $source = "$UNLEASH_ROOT\sdks\$($sdk.Key)"
    $target = "$STACK_DIR\tier-9-protocols\$($sdk.Value)"
    Create-SDKLink -Source $source -Target $target
}

# Process Skills
Write-Host "`nProcessing Skills..." -ForegroundColor Cyan

# Superpowers
$spSource = "$UNLEASH_ROOT\github-best-claude-code\obra-superpowers-33k"
$spTarget = "$SKILLS_DIR\superpowers"
if (Test-Path $spSource) {
    Create-SDKLink -Source $spSource -Target "$spTarget\obra-superpowers"
}

# Everything Claude Code
$eccSource = "$UNLEASH_ROOT\everything-claude-code-full"
$eccTarget = "$SKILLS_DIR\everything-claude-code"
if (Test-Path $eccSource) {
    Create-SDKLink -Source $eccSource -Target "$eccTarget\full"
}

# Workflow Skills
$wfSource = "$UNLEASH_ROOT\github-best-claude-code\claude-code-skills-workflow"
$wfTarget = "$SKILLS_DIR\workflow-skills"
if (Test-Path $wfSource) {
    Create-SDKLink -Source $wfSource -Target "$wfTarget\workflow"
}

# Anthropic Official
$aoSource = "$UNLEASH_ROOT\github-best-claude-code\anthropics-skills-official"
$aoTarget = "$SKILLS_DIR\anthropic-official"
if (Test-Path $aoSource) {
    Create-SDKLink -Source $aoSource -Target "$aoTarget\official"
}

Write-Host "`n=== Reorganization Complete ===" -ForegroundColor Cyan
Write-Host "New structure created at: $STACK_DIR" -ForegroundColor Green

# Summary
Write-Host "`nSummary:" -ForegroundColor Yellow
$tierCount = (Get-ChildItem -Path $STACK_DIR -Directory).Count
Write-Host "  Tiers created: $tierCount" -ForegroundColor White

foreach ($tier in $tiers) {
    $count = (Get-ChildItem -Path "$STACK_DIR\$tier" -Directory -ErrorAction SilentlyContinue).Count
    Write-Host "    $tier : $count SDKs" -ForegroundColor Gray
}
