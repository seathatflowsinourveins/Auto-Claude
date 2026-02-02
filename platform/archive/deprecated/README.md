# Archived/Deprecated Files

> This directory contains files that have been superseded by improved versions.
> Files here are kept for reference but should NOT be used in production.

## Archive Log

### 2026-01-30

| File | Reason | Replacement |
|------|--------|-------------|
| `mcp_guard_v1_archived.py` | Superseded by V2 with audit logging, input modification, kill switch | `../hooks/mcp_guard_v2.py` |

## Pattern: Why Archive Instead of Delete?

1. **Reference**: Old patterns may be useful for understanding evolution
2. **Rollback**: If V2 has issues, V1 provides quick fallback
3. **Learning**: Documents what patterns were deprecated and why

## Note on Verified Patterns (2026-01-30 - Iteration 2)

Files in the main codebase have been verified against official docs:

### Adapters (VERIFIED)
- **LangGraph Adapter**: Uses `add_conditional_edges(source, fn, path_map)` ✅
- **DSPy Adapter**: Uses verified optimizer classes (BootstrapFewShot, GEPA, etc.) ✅
- **Letta Voyage Adapter**: Uses context retrieval patterns (no direct messaging) ✅

### Hooks (VERIFIED)
- **MCP Guard V2**: Implements official Claude Code hooks spec v2.0.10+ ✅
- **Bash Guard**: Uses correct `hookSpecificOutput` format ✅
- **Letta Sync**: Uses correct `client.agents.messages.create()` pattern ✅
- **Memory Consolidate**: Uses correct Letta API pattern ✅
- **Audit Log**: PostToolUse logging, no decision control ✅

### Source Verification
- Context7: `/modelcontextprotocol/python-sdk`
- everything-claude-code: `github.com/affaan-m/everything-claude-code`
- Official Claude Code SDK: `hookify` plugin reference
- Official Anthropic docs and claude-cookbooks
