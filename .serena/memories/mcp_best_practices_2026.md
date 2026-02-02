# MCP Best Practices 2026

## Overview
Model Context Protocol has reached 8M+ downloads and 5,800+ community servers.
Now under Linux Foundation governance for enterprise adoption.

## Core Architectural Principles

### 1. Single Responsibility Principle
Each MCP server should do ONE thing well:
- **BAD**: Single server for files, database, and API
- **GOOD**: Separate servers: filesystem, postgres, rest-api

### 2. Defense in Depth Security Model
Layer security at multiple levels:
- Transport: TLS 1.3 required
- Authentication: OAuth 2.0 / API keys
- Authorization: Fine-grained tool permissions
- Validation: Input sanitization at every boundary
- Audit: Full request/response logging

### 3. Fail-Safe Design Patterns
```python
# Circuit breaker pattern
class MCPCircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_time=30):
        self.failures = 0
        self.threshold = failure_threshold
        self.recovery_time = recovery_time
        self.state = "CLOSED"  # CLOSED -> OPEN -> HALF_OPEN

    async def call(self, func, *args):
        if self.state == "OPEN":
            if time_since_open > self.recovery_time:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError()

        try:
            result = await func(*args)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
```

## Performance Targets

| Metric | Target | Critical |
|--------|--------|----------|
| Throughput | >1000 req/s | >500 req/s |
| P95 Latency | <100ms | <250ms |
| P99 Latency | <250ms | <500ms |
| Error Rate | <0.1% | <1% |
| Availability | 99.9% | 99% |

## Tool Design Best Practices

### 1. Explicit Descriptions
```json
{
  "name": "read_file",
  "description": "Reads UTF-8 text file content. Returns error for binary files. Max 10MB.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Absolute file path. Symlinks resolved."
      }
    },
    "required": ["path"]
  }
}
```

### 2. Idempotent Operations
Design tools to be safely retryable:
- GET operations: naturally idempotent
- POST/PUT: use idempotency keys
- DELETE: succeed even if already deleted

### 3. Pagination for Large Results
```python
@mcp.tool()
async def list_files(directory: str, cursor: str = None, limit: int = 100):
    """List files with cursor-based pagination."""
    files = await get_files(directory, after=cursor, limit=limit+1)
    has_more = len(files) > limit
    return {
        "files": files[:limit],
        "next_cursor": files[limit-1].id if has_more else None,
        "has_more": has_more
    }
```

## Resource Management

### 1. Connection Pooling
```python
# Shared connection pool for database MCP server
pool = asyncpg.create_pool(
    min_size=5,
    max_size=20,
    max_inactive_connection_lifetime=300
)
```

### 2. Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_client_id)

@mcp.tool()
@limiter.limit("100/minute")
async def expensive_operation():
    ...
```

### 3. Graceful Degradation
When downstream services fail, provide useful fallbacks:
- Cached responses
- Reduced functionality with clear messaging
- Queue for later processing

## Unleash Integration Pattern

```python
from core.tools import MCPToolRegistry

# Auto-discover and register MCP servers
registry = MCPToolRegistry()
await registry.discover_servers([
    "touchdesigner-creative",
    "filesystem",
    "memory",
])

# Unified tool access with circuit breakers
tool = registry.get_tool("touchdesigner-creative", "create_node")
result = await tool.execute(type="noiseTOP", name="noise1")
```

## References
- MCP Specification: https://spec.modelcontextprotocol.io
- Best Practices: https://modelcontextprotocol.info/best-practices
- Server Registry: https://mcp.so/servers

---
*Created: 2026-01-25*
*Source: Exa research + Unleash integration patterns*
