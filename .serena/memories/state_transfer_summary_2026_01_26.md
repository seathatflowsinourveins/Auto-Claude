# State Transfer Summary - 2026-01-26

> **Session Focus**: Permanent fix for Serena project detection and CLAUDE.md configuration

## Critical Fixes Applied

### 1. Serena Plugin Configuration (PERMANENT FIX)

**Problem**: Serena plugin was using `uvx --from git+https://github.com/oraios/serena` which doesn't use our custom project detection, causing it to connect to WITNESS instead of UNLEASH.

**Solution**: Modified ALL Serena plugin .mcp.json files to use `serena-dynamic.py`:

**Files Modified**:
- `C:/Users/42/.claude/plugins/marketplaces/claude-plugins-official/external_plugins/serena/.mcp.json`
- `C:/Users/42/.claude/plugins/cache/claude-plugins-official/serena/e30768372b41/.mcp.json`
- `C:/Users/42/.claude/plugins/cache/claude-plugins-official/serena/96276205880a/.mcp.json`

**New Configuration**:
```json
{
  "serena": {
    "command": "python",
    "args": [
      "C:/Users/42/.claude/scripts/serena-dynamic.py",
      "--enable-web-dashboard", "true",
      "--open-web-dashboard", "true"
    ],
    "env": {
      "SERENA_PROJECT": "${SERENA_PROJECT}"
    },
    "description": "Serena with DYNAMIC project detection. Default: UNLEASH."
  }
}
```

### 2. UNLEASH CLAUDE.md Created

**Created**: `Z:/insider/AUTO CLAUDE/unleash/CLAUDE.md`

Contains:
- 8-layer SDK architecture diagram
- P0/P1/P2 SDK classifications
- Key directory structure
- Essential Serena memories list
- Quick command reference

### 3. Memory Documentation Fixed

**Updated**: Serena memory `serena_dynamic_project_switching`
- Changed "Default fallback: State of Witness" → "Default fallback: UNLEASH"

## Project Detection Logic (How It Works)

```
serena-dynamic.py detection priority:
1. SERENA_PROJECT env var (explicit override)
2. CLAUDE_CWD env var (set by Claude Code)
3. Current working directory matching PROJECT_REGISTRY
4. Default fallback: Z:/insider/AUTO CLAUDE/unleash
```

## Verification Checklist for Future Sessions

```powershell
# 1. Check which Serena tools are available
#    Should see both mcp__serena__* AND mcp__plugin_serena_serena__*

# 2. Test Serena project detection
mcp__plugin_serena_serena__list_dir "." 
#    Should return UNLEASH structure (sdks/, platform/, etc.)

# 3. Verify project detection wrapper
python "C:\Users\42\.claude\scripts\serena-dynamic.py" --help 2>&1
#    Look for "[serena-dynamic] Detected project: Z:/insider/AUTO CLAUDE/unleash"
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `C:/Users/42/.claude/scripts/serena-dynamic.py` | Project detection wrapper |
| `C:/Users/42/.claude/mcp_servers_OPTIMAL.json` | MCP server config |
| `C:/Users/42/.claude/settings.json` | Main Claude Code settings |
| `C:/Users/42/.claude/hooks/project-bootstrap.py` | Bootstrap hook |
| `Z:/insider/AUTO CLAUDE/unleash/CLAUDE.md` | UNLEASH project context |

## Dual Serena Instances (Expected Behavior)

After this fix, both Serena instances should connect to UNLEASH:
- `mcp__serena__*` - From Serena plugin (now uses serena-dynamic.py)
- `mcp__plugin_serena_serena__*` - From MCP server (always used serena-dynamic.py)

## Restart Required

**IMPORTANT**: Claude Code must be restarted for changes to take effect.
The Serena MCP server starts once at session initialization and cannot be
hot-swapped mid-session.

---

*Generated: 2026-01-26 | Status: COMPLETE*
