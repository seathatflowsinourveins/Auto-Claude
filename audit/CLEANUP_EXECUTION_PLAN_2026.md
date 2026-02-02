# CLEANUP EXECUTION PLAN
## Step-by-Step Cleanup Commands (2026-01-24)

**IMPORTANT**: This document contains the actual commands to execute cleanup.
Run ONLY after reviewing SDK_CLEANUP_GROUPS_2026.md and confirming each group.

---

## PRE-CLEANUP CHECKLIST

- [ ] Review SDK_CLEANUP_GROUPS_2026.md
- [ ] Review SDK_KEEP_ARCHITECTURE_2026.md
- [ ] Review ULTIMATE_ARCHITECTURE_2026.md
- [ ] Backup current state (optional but recommended)
- [ ] Confirm understanding of what will be deleted

---

## PHASE 1: CRITICAL DELETE (6 items)

These are safe to delete immediately - confirmed deprecated/merged/inactive.

### 1.1 Delete Merged/Deprecated SDKs

```powershell
# CRITICAL DELETE - memgpt (merged into Letta)
Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\sdks\memgpt"

# CRITICAL DELETE - memgpt duplicate in stack
Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\stack\tier-2-memory\memgpt"

# CRITICAL DELETE - infinite-agentic-loop (>12 months inactive)
Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\sdks\infinite-agentic-loop"

# CRITICAL DELETE - snarktank-ralph (superseded by V12)
Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\sdks\snarktank-ralph"

# CRITICAL DELETE - firecrawl-sdk (duplicate of firecrawl)
Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\sdks\firecrawl-sdk"

# CRITICAL DELETE - mem0-full (duplicate of mem0)
Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\sdks\mem0-full"
```

### 1.2 Verification
```powershell
# Verify deletions
$deleted = @(
    "Z:\insider\AUTO CLAUDE\unleash\sdks\memgpt",
    "Z:\insider\AUTO CLAUDE\unleash\stack\tier-2-memory\memgpt",
    "Z:\insider\AUTO CLAUDE\unleash\sdks\infinite-agentic-loop",
    "Z:\insider\AUTO CLAUDE\unleash\sdks\snarktank-ralph",
    "Z:\insider\AUTO CLAUDE\unleash\sdks\firecrawl-sdk",
    "Z:\insider\AUTO CLAUDE\unleash\sdks\mem0-full"
)

foreach ($path in $deleted) {
    if (Test-Path $path) {
        Write-Host "WARNING: $path still exists" -ForegroundColor Red
    } else {
        Write-Host "DELETED: $path" -ForegroundColor Green
    }
}
```

---

## PHASE 2: MOVE DOCUMENTATION FILES

Move .md and script files from sdks/ to proper locations.

```powershell
# Create target directories
New-Item -ItemType Directory -Force -Path "Z:\insider\AUTO CLAUDE\unleash\docs\sdk-research"

# Move documentation files
$docsToMove = @(
    "BACKBONE_ARCHITECTURE_DEEP_RESEARCH.md",
    "NEW_SDK_INTEGRATIONS.md",
    "SDK_INDEX.md",
    "SDK_INTEGRATION_PATTERNS_V30.md",
    "SDK_QUICK_REFERENCE.md",
    "SELF_IMPROVEMENT_RESEARCH_2026.md",
    "ULTRAMAX_SDK_COMPLETE_ANALYSIS.md"
)

foreach ($doc in $docsToMove) {
    $source = "Z:\insider\AUTO CLAUDE\unleash\sdks\$doc"
    $dest = "Z:\insider\AUTO CLAUDE\unleash\docs\sdk-research\$doc"
    if (Test-Path $source) {
        Move-Item -Path $source -Destination $dest -Force
        Write-Host "Moved: $doc" -ForegroundColor Cyan
    }
}

# Move script file
Move-Item -Path "Z:\insider\AUTO CLAUDE\unleash\sdks\setup-ultramax.ps1" `
          -Destination "Z:\insider\AUTO CLAUDE\unleash\scripts\setup-ultramax.ps1" -Force
```

---

## PHASE 3: HIGH DELETE (Provider-Locked & Duplicates)

**REQUIRES YOUR CONFIRMATION** before proceeding.

### 3.1 Provider-Locked SDKs (Optional - Your Decision)

```powershell
# OPTIONAL DELETE - openai-agents (OpenAI-only, use langgraph+litellm)
# Uncomment if you want to remove:
# Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\sdks\openai-agents"

# OPTIONAL DELETE - google-adk (Gemini-only, use litellm)
# Uncomment if you want to remove:
# Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\sdks\google-adk"
```

### 3.2 Stack Tier Duplicates

```powershell
# DELETE stack duplicates (keep sdks/ versions)
$stackDuplicates = @(
    "Z:\insider\AUTO CLAUDE\unleash\stack\tier-1-orchestration\openai-agents",
    "Z:\insider\AUTO CLAUDE\unleash\stack\tier-4-evolution\evoagentx",
    "Z:\insider\AUTO CLAUDE\unleash\stack\tier-4-evolution\evoagentx-advanced"
)

foreach ($dup in $stackDuplicates) {
    if (Test-Path $dup) {
        Remove-Item -Recurse -Force $dup
        Write-Host "Removed duplicate: $dup" -ForegroundColor Yellow
    }
}
```

### 3.3 Delete Empty/Placeholder Directory

```powershell
# Delete nul (empty placeholder)
if (Test-Path "Z:\insider\AUTO CLAUDE\unleash\sdks\nul") {
    Remove-Item -Recurse -Force "Z:\insider\AUTO CLAUDE\unleash\sdks\nul"
}
```

---

## PHASE 4: CONSOLIDATE opik-full

If opik and opik-full have different contents, merge them.

```powershell
# Check if both exist
$opik = "Z:\insider\AUTO CLAUDE\unleash\sdks\opik"
$opikFull = "Z:\insider\AUTO CLAUDE\unleash\sdks\opik-full"

if ((Test-Path $opik) -and (Test-Path $opikFull)) {
    # opik-full likely has more content - keep it as opik
    # First backup opik
    Move-Item -Path $opik -Destination "${opik}-backup" -Force

    # Rename opik-full to opik
    Move-Item -Path $opikFull -Destination $opik -Force

    Write-Host "Consolidated opik-full into opik" -ForegroundColor Green
}
```

---

## PHASE 5: UPDATE STACK SYMLINKS

Create symlinks in stack/ pointing to sdks/ for organized view.

```powershell
# Create tier directories if needed
$tiers = @(
    "tier-0-protocol",
    "tier-1-orchestration",
    "tier-2-memory",
    "tier-3-structured",
    "tier-4-reasoning",
    "tier-5-observability",
    "tier-6-safety",
    "tier-7-processing",
    "tier-8-knowledge"
)

foreach ($tier in $tiers) {
    $tierPath = "Z:\insider\AUTO CLAUDE\unleash\stack\$tier"
    if (-not (Test-Path $tierPath)) {
        New-Item -ItemType Directory -Force -Path $tierPath
    }
}

# Note: Creating symlinks in Windows requires admin privileges
# Alternative: Update stack/ to contain copies or junction points
```

---

## PHASE 6: CLEAN SERENA MEMORIES

Update Serena memories to remove references to deleted SDKs.

```powershell
# List memories that may reference deleted SDKs
Get-ChildItem "Z:\insider\AUTO CLAUDE\unleash\.serena\memories" |
    Select-String -Pattern "memgpt|snarktank-ralph|infinite-agentic-loop" |
    Select-Object Path, LineNumber, Line
```

Then manually update relevant memory files via Serena tools.

---

## PHASE 7: VERIFICATION

### 7.1 Count Remaining SDKs

```powershell
$sdkCount = (Get-ChildItem -Directory "Z:\insider\AUTO CLAUDE\unleash\sdks" |
             Where-Object { $_.Name -notmatch "^\." }).Count
Write-Host "Remaining SDKs: $sdkCount" -ForegroundColor Cyan
# Target: ~40-45 after Phase 1+2
```

### 7.2 Check for Broken References

```powershell
# Search for references to deleted SDKs in code
$deletedNames = @("memgpt", "snarktank-ralph", "infinite-agentic-loop")

foreach ($name in $deletedNames) {
    $refs = Get-ChildItem -Recurse -File "Z:\insider\AUTO CLAUDE\unleash" -Include "*.py","*.md","*.yaml" |
            Select-String -Pattern $name
    if ($refs) {
        Write-Host "References to deleted SDK '$name':" -ForegroundColor Yellow
        $refs | ForEach-Object { Write-Host $_.Path }
    }
}
```

### 7.3 Disk Space Recovered

```powershell
# Compare before/after (run this before cleanup too)
$size = (Get-ChildItem -Recurse "Z:\insider\AUTO CLAUDE\unleash\sdks" |
         Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "SDK directory size: $([math]::Round($size, 2)) GB" -ForegroundColor Cyan
```

---

## POST-CLEANUP TASKS

### Update CLAUDE.md
After cleanup, update the main CLAUDE.md to reflect:
- New 35-SDK architecture
- Updated project configurations
- Removed deprecated references

### Update Serena Memories
Run via Serena:
```
- Update cross_session_bootstrap memory
- Update project_overview memory
- Remove deprecated SDK references
```

### Commit Changes
```bash
git add -A
git commit -m "Cleanup: Remove 6 CRITICAL deprecated SDKs, reorganize documentation

- Removed memgpt (merged into Letta)
- Removed infinite-agentic-loop (>12mo inactive)
- Removed snarktank-ralph (superseded by V12)
- Removed duplicate SDKs (firecrawl-sdk, mem0-full)
- Moved documentation to docs/sdk-research/
- Updated stack tier structure

Part of SDK portfolio audit 2026-01-24"
```

---

## EXECUTION SUMMARY

| Phase | Items | Status | Size Recovered |
|-------|-------|--------|----------------|
| Phase 1: Critical | 6 SDKs | Pending Approval | ~500MB |
| Phase 2: Docs Move | 8 files | Pending | ~50MB |
| Phase 3: High Delete | Optional | Your Decision | ~2GB |
| Phase 4: Consolidate | 1 merge | Pending | ~100MB |
| Phase 5: Symlinks | 9 tiers | Pending | None |
| Phase 6: Memories | Variable | Pending | None |

---

## ROLLBACK PLAN

If needed, the deleted SDKs can be re-cloned from GitHub:

```powershell
# Restore memgpt (if needed - but use Letta instead)
# git clone https://github.com/cpacker/MemGPT sdks/memgpt

# Restore infinite-agentic-loop
# git clone https://github.com/disler/infinite-agentic-loop sdks/infinite-agentic-loop

# Restore snarktank-ralph
# git clone https://github.com/snarktank/ralph sdks/snarktank-ralph
```

---

**Document Version**: 1.0
**Generated**: 2026-01-24
**Status**: AWAITING USER APPROVAL BEFORE EXECUTION
