# Migration Guide: V9 APEX → V10 OPTIMIZED

> **Purpose**: Step-by-step guide to migrate from the overcomplicated V9 configuration to the streamlined V10 architecture.

---

## Why Migrate?

### V9 Problems
- **40+ MCP servers configured**, only ~15 actually work
- **Complex settings.json** with theoretical features
- **Non-existent packages** causing errors (e.g., `@letta-ai/mcp-server`)
- **Hooks reference missing files**
- **No validation** of configuration

### V10 Solutions
- **8 core verified servers** that actually work
- **Minimal settings.json** with proven configurations
- **All packages verified** on npm/pypi
- **Working hook implementations** included
- **Automatic validation** during setup

---

## Migration Steps

### Step 1: Backup Current Configuration

```powershell
# Create backup directory
$backupDir = "$env:USERPROFILE\.claude\backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $backupDir -Force

# Backup current config
Copy-Item "$env:USERPROFILE\.claude\settings.json" "$backupDir\" -ErrorAction SilentlyContinue
Copy-Item "$env:USERPROFILE\.claude\CLAUDE.md" "$backupDir\" -ErrorAction SilentlyContinue
Copy-Item "$env:USERPROFILE\.claude\.mcp.json" "$backupDir\" -ErrorAction SilentlyContinue

Write-Host "Backed up to: $backupDir"
```

### Step 2: Run V10 Setup

```powershell
# Navigate to V10 directory
cd "Z:\insider\AUTO CLAUDE\unleash\v10_optimized\scripts"

# Run setup (choose mode based on needs)
.\setup_v10.ps1 -Mode standard
```

**Mode Options:**
- `minimal`: Core servers only (filesystem, memory, sequential-thinking)
- `standard`: Core + development tools (recommended)
- `full`: All verified servers

### Step 3: Verify MCP Packages

```powershell
# Run verification to confirm packages work
cd "Z:\insider\AUTO CLAUDE\unleash\v10_optimized\scripts"
uv run verify_mcp.py

# If issues found, generate fixed config
uv run verify_mcp.py --fix
```

### Step 4: Start Letta (Optional but Recommended)

```powershell
# Start Letta for cross-session memory
docker run -d --name letta -p 8283:8283 letta/letta:latest

# Verify it's running
Invoke-WebRequest -Uri "http://localhost:8283/v1/health" -UseBasicParsing
```

### Step 5: Test the Configuration

```powershell
# Start Claude Code CLI
claude

# Test MCP servers
/mcp status

# Test a simple file operation
# (This should trigger the mcp_guard hook)
```

### Step 6: Migrate Project-Specific Settings

If you have project-specific `CLAUDE.md` files, they can remain unchanged. V10 adds `CLAUDE.local.md` which is auto-generated from Letta memory.

**Project hierarchy (unchanged):**
1. `~/.claude/CLAUDE.md` - Global (from V10)
2. `./CLAUDE.md` - Project-specific (keep your existing)
3. `./CLAUDE.local.md` - Auto-generated from Letta

---

## Configuration Comparison

### Settings.json Diff

**V9 (Complex - 355 lines):**
```json
{
  "model": {...},
  "mcp": {
    "pools": {
      "primary": {...},
      "failover": {...},
      "trading": {...},
      "creative": {...}
    }
  },
  "hooks": {
    "architecture": "event_sourced",
    "eventStore": {...},
    "pipeline": {...}
  },
  "letta": {
    "memory": {
      "tiers": {
        "L0_ultra": {...},
        "L1_hot": {...},
        ...
      }
    }
  },
  "safety": {
    "layers": 18,
    ...
  },
  "skills": {
    "architecture": "dag_graph",
    "graphs": {...}
  }
}
```

**V10 (Minimal - 128 lines):**
```json
{
  "model": "sonnet",
  "permissions": {
    "allow": [...],
    "deny": [...]
  },
  "hooks": {
    "SessionStart": [...],
    "SessionEnd": [...],
    "PreToolUse": [...],
    "PostToolUse": [...]
  }
}
```

### MCP Configuration Diff

**V9 (40+ servers, many broken):**
```json
{
  "mcpServers": {
    "letta": {...},          // ❌ Package doesn't exist
    "mem0": {...},           // ❌ Package doesn't exist
    "langfuse": {...},       // ❌ Package doesn't exist
    "slack": {...},          // ❌ Package doesn't exist
    "polygon": {...},        // ❌ Wrong package name
    "alpaca": {...},         // ❌ Wrong package name
    ...
  }
}
```

**V10 (8 servers, all verified):**
```json
{
  "mcpServers": {
    "filesystem": {...},          // ✅ Verified working
    "memory": {...},              // ✅ Verified working
    "sequential-thinking": {...}, // ✅ Verified working
    "context7": {...},            // ✅ Verified working
    "eslint": {...},              // ✅ Verified working
    "fetch": {...},               // ✅ Verified working
    "sqlite": {...},              // ✅ Verified working
    "github": {...}               // ✅ Verified working (gh CLI)
  }
}
```

---

## What Happens to Your Data

### Preserved
- Project-specific `CLAUDE.md` files
- Git repositories
- Source code
- Any custom scripts

### Replaced
- Global `~/.claude/settings.json`
- Global `~/.claude/CLAUDE.md`
- Global `~/.claude/.mcp.json`

### New
- `~/.claude/v10/` directory with:
  - `hooks/` - Working hook scripts
  - `logs/` - Audit logs
  - `memory.db` - Local SQLite cache
- `CLAUDE.local.md` in projects (auto-generated from Letta)

---

## Rollback Procedure

If something goes wrong:

```powershell
# Find your backup
$backupDir = Get-ChildItem "$env:USERPROFILE\.claude\backup_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Restore settings
Copy-Item "$backupDir\settings.json" "$env:USERPROFILE\.claude\" -Force
Copy-Item "$backupDir\CLAUDE.md" "$env:USERPROFILE\.claude\" -Force
Copy-Item "$backupDir\.mcp.json" "$env:USERPROFILE\.claude\" -Force

Write-Host "Restored from: $backupDir"
```

---

## Feature Mapping

| V9 Feature | V10 Equivalent | Notes |
|------------|----------------|-------|
| 18-layer safety | Simplified guards | `mcp_guard.py`, `bash_guard.py` |
| RL model router | Model selection | Use `model` setting or extended thinking |
| 6-tier memory cache | Letta memory | Automatic with hooks |
| Event-sourced hooks | Standard hooks | Simpler, working implementation |
| Skill DAG graphs | Skills directory | Use standard Claude Code skills |
| MCP pools with failover | Single pool | Verified servers only |
| Trading/Creative modes | Project CLAUDE.md | Per-project configuration |

---

## Post-Migration Checklist

- [ ] V10 setup completed successfully
- [ ] MCP verification passed
- [ ] Claude Code CLI starts without errors
- [ ] Basic file operations work
- [ ] Letta memory persists across sessions (if using Letta)
- [ ] Hooks execute without errors
- [ ] Kill switch works (`New-Item ~/.claude/KILL_SWITCH`)

---

## Common Issues

### "npm ERR! code E404"
**Cause**: Package doesn't exist on npm
**Solution**: Run `verify_mcp.py --fix` to remove broken packages

### "Letta connection refused"
**Cause**: Letta server not running
**Solution**: Start with `docker run -d -p 8283:8283 letta/letta:latest`

### "Hook script not found"
**Cause**: Hooks not installed
**Solution**: Re-run `setup_v10.ps1`

### "Permission denied on file write"
**Cause**: Path not in allowed list
**Solution**: Check `permissions.allow` in settings.json

---

## Getting Help

1. **Check logs**: `~/.claude/v10/logs/`
2. **Run verification**: `uv run verify_mcp.py`
3. **Test hooks manually**: `echo '{}' | python ~/.claude/v10/hooks/mcp_guard.py`

---

*V10 Optimized - Simpler. Verified. Working.*
