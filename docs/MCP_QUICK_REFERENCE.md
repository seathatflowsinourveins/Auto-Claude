# MCP 2026 Quick Reference

**Last Updated**: 2026-02-05

---

## Transport Comparison

| Transport | Status | Use Case | Latency | Notes |
|-----------|--------|----------|---------|-------|
| **stdio** | ✅ Recommended | Local servers | <1ms | Best for CLI/local tools |
| **Streamable HTTP** | ✅ Current | Remote servers | 3-5ms | Replaces SSE |
| **SSE** | ❌ Deprecated | None | N/A | Removed in SDK v2.0 (Q1 2026) |

---

## UNLEASH Platform Status

**11 Active MCP Servers**:
- 10 use **stdio** (91%) - ✅ No action required
- 1 uses **HTTP** (9%) - ⚠️ Verify protocol (GitHub)
- 0 use **SSE** (0%) - ✅ No legacy servers

---

## Streamable HTTP Essentials

### Session Headers

```http
# Initialization request
POST /mcp HTTP/1.1
MCP-Protocol-Version: 2025-11-25
Accept: application/json, text/event-stream
Content-Type: application/json

# Response includes session ID
HTTP/1.1 200 OK
Mcp-Session-Id: 550e8400-e29b-41d4-a716-446655440000

# Subsequent requests
POST /mcp HTTP/1.1
Mcp-Session-Id: 550e8400-e29b-41d4-a716-446655440000
```

### Protocol Detection

```python
# Test if server supports Streamable HTTP
transport = await detect_mcp_transport(server_url)
# Returns: "streamable-http", "legacy-sse", or "unknown"
```

---

## Testing Commands

### GitHub Server Protocol Test

```bash
cd /c/Users/42 && uv run --no-project --with pytest,pytest-asyncio,httpx python -m pytest \
  "Z:/insider/AUTO CLAUDE/unleash/platform/tests/test_github_mcp_protocol.py" -v
```

### Manual Protocol Detection

```python
import asyncio
import httpx
import os

async def test():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.githubcopilot.com/mcp",
            headers={
                "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
                "MCP-Protocol-Version": "2025-11-25"
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25", "capabilities": {}}
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Session ID: {response.headers.get('Mcp-Session-Id')}")

asyncio.run(test())
```

---

## Key Decisions

1. **stdio servers**: No migration (recommended transport)
2. **GitHub server**: Verify protocol version (1-2 days)
3. **Timeline**: Complete before SDK v2.0 (Q1 2026)

---

## Resources

- **Migration Guide**: `docs/MCP_SSE_TO_HTTP_MIGRATION.md`
- **Summary**: `docs/MCP_2026_MIGRATION_SUMMARY.md`
- **Test File**: `platform/tests/test_github_mcp_protocol.py`
- **Spec**: https://modelcontextprotocol.io/specification/2025-11-25

---

**Priority**: LOW (91% compliant)  
**Risk**: LOW  
**Effort**: 1-2 days (GitHub verification only)
