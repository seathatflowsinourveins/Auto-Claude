# V10 OPTIMIZED - Seamless Claude Code CLI

> **The Verified, Working, Production-Ready Architecture**

---

## 🚀 Quick Start

```powershell
# 1. Run setup
cd scripts
.\setup_v10.ps1 -Mode standard

# 2. Verify packages work
uv run verify_mcp.py

# 3. Start Letta (optional, for memory persistence)
docker run -d -p 8283:8283 letta/letta:latest

# 4. Start Claude Code CLI
claude
```

---

## 📁 Directory Structure

```
v10_optimized/
├── README.md                  # This file
├── V10_ARCHITECTURE.md        # Full architecture guide
├── MIGRATION_GUIDE.md         # V9 → V10 migration steps
│
├── config/
│   ├── settings.json          # Claude Code settings (128 lines)
│   ├── CLAUDE.md              # Global instructions (143 lines)
│   └── mcp_servers.json       # Verified MCP config
│
├── hooks/
│   ├── letta_sync.py          # Session start/end memory sync
│   ├── mcp_guard.py           # MCP tool validation
│   ├── bash_guard.py          # Bash command validation
│   ├── memory_consolidate.py  # Sleeptime trigger
│   └── audit_log.py           # File change logging
│
└── scripts/
    ├── setup_v10.ps1          # Installation script
    └── verify_mcp.py          # Package verification
```

---

## ✅ Verified Components

### MCP Servers (8 total, all working)

| Server | Package | Status |
|--------|---------|--------|
| filesystem | `@modelcontextprotocol/server-filesystem` | ✅ |
| memory | `@modelcontextprotocol/server-memory` | ✅ |
| sequential-thinking | `@modelcontextprotocol/server-sequential-thinking` | ✅ |
| context7 | `@upstash/context7-mcp` | ✅ |
| eslint | `@eslint/mcp` | ✅ |
| fetch | `@modelcontextprotocol/server-fetch` | ✅ |
| sqlite | `@modelcontextprotocol/server-sqlite` | ✅ |
| github | `gh mcp` (CLI) | ✅ |

### Hooks (5 total, all implemented)

| Hook | Purpose | Status |
|------|---------|--------|
| letta_sync.py | Memory sync with Letta | ✅ |
| mcp_guard.py | MCP security validation | ✅ |
| bash_guard.py | Command security validation | ✅ |
| memory_consolidate.py | Sleeptime triggers | ✅ |
| audit_log.py | File change audit trail | ✅ |

---

## 📊 V10 vs V9 Comparison

| Metric | V9 APEX | V10 OPTIMIZED |
|--------|---------|---------------|
| MCP Servers Working | 15/40 (37%) | 8/8 (100%) |
| Settings.json Lines | 355 | 128 |
| Hook Files Implemented | 0/4 | 5/5 |
| Non-Existent Packages | 8+ | 0 |
| Setup Complexity | High | Low |

---

## 🔧 Installation Modes

```powershell
# Minimal - Core servers only
.\setup_v10.ps1 -Mode minimal

# Standard - Core + development tools (recommended)
.\setup_v10.ps1 -Mode standard

# Full - All verified servers
.\setup_v10.ps1 -Mode full
```

---

## 🛡️ Emergency Commands

```powershell
# Activate kill switch (blocks ALL operations)
New-Item -Path ~/.claude/KILL_SWITCH -ItemType File

# Deactivate kill switch
Remove-Item ~/.claude/KILL_SWITCH
```

---

## 📚 Documentation

- [V10 Architecture](V10_ARCHITECTURE.md) - Full system documentation
- [Migration Guide](MIGRATION_GUIDE.md) - V9 → V10 migration
- [Claude Code Docs](https://code.claude.com/docs) - Official documentation
- [Letta API](https://docs.letta.com/api/) - Memory system API

---

## 💡 Key Principles

1. **Verified**: Every package confirmed on npm/pypi
2. **Minimal**: Only what's actually needed
3. **Seamless**: Everything works together automatically
4. **Recoverable**: Easy backup and rollback

---

*V10 OPTIMIZED - January 2026*
