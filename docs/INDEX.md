# INDEX - Unleash Directory Quick Reference

> **Updated**: January 17, 2026 | **Latest**: V10.1 OPTIMIZED

---

## ⚡ RECOMMENDED: V10.1 OPTIMIZED

**The verified, SDK-powered, production-ready architecture.**

```powershell
# Quick install
cd "Z:\insider\AUTO CLAUDE\unleash\v10_optimized\scripts"
.\setup_v10.ps1 -Mode standard

# Deploy hooks to ~/.claude/v10 (NEW in V10.1)
.\deploy_hooks.ps1 -Force -Verify

# Run health check (NEW in V10.1)
uv run health_check.py
```

| V10.1 Benefits | Details |
|----------------|---------|
| ✅ All MCP servers verified | 8/8 working (vs 15/40 in V9) |
| ✅ Complete hook implementations | 8 production-ready scripts (V2 + original) |
| ✅ Official Letta SDK | `letta_client` with `SleeptimeManagerUpdate` |
| ✅ Input modification | PreToolUse MODIFY support (v2.0.10+) |
| ✅ Health checking | Comprehensive `health_check.py --fix` |
| ✅ Minimal configuration | 128 lines vs 355 in V9 |

**[→ V10.1 Documentation](v10_optimized/V10_ARCHITECTURE_V2.md)** (NEW)
**[→ V10.0 Documentation](v10_optimized/V10_ARCHITECTURE.md)**

---

## DIRECTORY STRUCTURE

### `/v10_optimized` - **RECOMMENDED** ⭐
| Path | Purpose |
|------|---------|
| `README.md` | Quick start guide |
| `V10_ARCHITECTURE_V2.md` | V10.1 architecture documentation (NEW) |
| `V10_ARCHITECTURE.md` | V10.0 architecture documentation |
| `MIGRATION_GUIDE.md` | V9 → V10 migration steps |
| `config/settings.json` | Minimal verified configuration |
| `config/CLAUDE.md` | Streamlined global instructions |
| `hooks/hook_utils.py` | Shared utilities library (NEW) |
| `hooks/letta_sync_v2.py` | SDK-based Letta sync (NEW) |
| `hooks/mcp_guard_v2.py` | MODIFY support guard (NEW) |
| `hooks/*.py` | Original hook implementations |
| `scripts/setup_v10.ps1` | Installation script |
| `scripts/verify_mcp.py` | Package verification tool |
| `scripts/health_check.py` | System health verification (NEW) |
| `scripts/deploy_hooks.ps1` | Hook deployment script (NEW) |

### `/active` - V9 APEX (Previous)
| Path | Purpose |
|------|---------|
| `implementations/model_router_rl.py` | RL model router |
| `implementations/safety_fortress_v9.py` | 18-layer safety |
| `config/settings.json` | Complex configuration |

### `/docs/essential` - Documentation
| File | Content |
|------|---------|
| `ECOSYSTEM_STATUS.md` | System status dashboard |
| `HONEST_AUDIT.md` | What works vs needs setup |
| `UNLEASHED_PATTERNS.md` | Production patterns |
| `V9_APEX_README.md` | V9 documentation |

### `/docs/reference` - Research
| File | Content |
|------|---------|
| `COMPASS_SYNTHESIS.md` | 8 compass artifacts synthesized |

### `/archive` - Historical
| File | Notes |
|------|-------|
| `LETTA_ULTRAMAX_V8_ULTIMATE.md` | Previous version (170KB) |
| `compass_artifact_*.md` (8) | Original research |

---

## QUICK DECISIONS

| I want to... | Go to |
|--------------|-------|
| **Start fresh with working setup** | `v10_optimized/` ⭐ |
| Deploy V9 (complex) | `active/implementations/` |
| See what actually works | `docs/essential/HONEST_AUDIT.md` |
| Research MCP/Skills | `docs/reference/COMPASS_SYNTHESIS.md` |
| Migrate V9 → V10 | `v10_optimized/MIGRATION_GUIDE.md` |
| Emergency shutdown | `New-Item ~/.claude/KILL_SWITCH` |

---

## VERSION COMPARISON

| Feature | V10.1 OPTIMIZED | V10.0 | V9 APEX |
|---------|-----------------|-------|---------|
| MCP Servers | 8 (all verified) | 8 (all verified) | 40+ (15 working) |
| Hook Implementations | 8 (V2 + original) | 5 complete | ❌ Missing |
| Letta SDK | Official `letta_client` | httpx raw | ⚠️ Theoretical |
| Input Modification | ✅ v2.0.10+ | ❌ | ❌ |
| Health Check | ✅ Comprehensive | Manual | None |
| Configuration Lines | 128 | 128 | 355 |
| Non-existent Packages | 0 | 0 | 8+ |

---

## EMERGENCY COMMANDS

```powershell
# KILL SWITCH (blocks ALL operations)
New-Item -Path ~/.claude/KILL_SWITCH -ItemType File

# RESUME
Remove-Item ~/.claude/KILL_SWITCH
```

---

## SYSTEM STATS

| Component | V10.1 | V10.0 | V9 |
|-----------|-------|-------|-----|
| MCP Working | 8/8 (100%) | 8/8 (100%) | 15/40 (37%) |
| Hooks Working | 8/8 | 5/5 | 0/4 |
| Skills | 32 | 32 | 32 |
| Lines of Config | ~900 | ~900 | 1,500+ |
| Letta SDK | Official | httpx | None |
| Health Check | ✅ | Manual | None |

---

## V10.1 NEW FILES

| File | Purpose |
|------|---------|
| `hooks/hook_utils.py` | Shared utilities (HookInput, HookResponse, HookConfig) |
| `hooks/letta_sync_v2.py` | Official Letta SDK with fallback |
| `hooks/mcp_guard_v2.py` | PreToolUse MODIFY support (v2.0.10+) |
| `scripts/health_check.py` | Comprehensive system verification |
| `V10_ARCHITECTURE_V2.md` | Updated documentation |

---

*Use V10.1 for production. Reference V10.0/V9/archive for historical context.*
