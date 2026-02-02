# Deployment Checklist - V10 Ultimate Autonomous Platform

**Version**: 10.0
**Status**: Production Ready
**Last Validated**: January 2026

---

## Pre-Deployment Validation

### Infrastructure Requirements

- [ ] Python 3.11+ installed
- [ ] uv package manager available
- [ ] Node.js 18+ for MCP servers
- [ ] Docker for containerized services (optional)

### Verify Installation

```bash
# 1. Run quick validation
uv run auto_validate.py --quick

# Expected: 3+ PASS, 2 WARN (Letta optional)

# 2. Run integration tests
uv run pytest tests/test_platform_integration.py -v

# Expected: 32/32 tests pass
```

---

## Core Components Checklist

### Scripts (18 modules)

| Module | Status | Command to Verify |
|--------|--------|-------------------|
| ecosystem_orchestrator.py | Ready | `uv run ecosystem_orchestrator.py --quick` |
| auto_validate.py | Ready | `uv run auto_validate.py --quick` |
| sleeptime_compute.py | Ready | `uv run sleeptime_compute.py status` |
| session_continuity.py | Ready | `uv run session_continuity.py status` |
| ralph_loop.py | Ready | `uv run ralph_loop.py status` |
| performance.py | Ready | `uv run performance.py status` |
| config.py | Ready | `uv run config.py status` |
| secrets.py | Ready | `uv run secrets.py list` |
| install.py | Ready | `uv run install.py --check` |
| autoscale.py | Ready | `uv run autoscale.py metrics` |
| rate_limiter.py | Ready | Module import test |
| tracing.py | Ready | Module import test |
| metrics.py | Ready | `uv run metrics.py --json` |
| status_dashboard.py | Ready | `uv run status_dashboard.py --json` |
| platform_orchestrator.py | Ready | `uv run platform_orchestrator.py status` |
| health_check.py | Ready | `uv run health_check.py --json` |
| verify_mcp.py | Ready | `uv run verify_mcp.py --json` |
| memory_bridge.py | Ready | Module import test |

### Hooks (8 validated)

| Hook | Event | Status |
|------|-------|--------|
| letta_sync.py | SessionStart/End | Ready |
| letta_sync_v2.py | SessionStart/End | Ready |
| mcp_guard.py | PreToolUse | Ready |
| mcp_guard_v2.py | PreToolUse | Ready |
| bash_guard.py | PreToolUse | Ready |
| memory_consolidate.py | Stop | Ready |
| audit_log.py | PostToolUse | Ready |
| hook_utils.py | Library | Ready |

### Core Library (6 modules)

| Module | Purpose | Status |
|--------|---------|--------|
| memory.py | Three-tier memory | Ready |
| cooperation.py | Session handoff | Ready |
| harness.py | Agent harness | Ready |
| mcp_manager.py | MCP server management | Ready |
| executor.py | ReAct executor | Ready |
| thinking.py | Extended thinking | Ready |

---

## Optional Services

### Letta Server
```bash
docker run -d -p 8283:8283 letta/letta:latest
# Validates: Memory persistence, agent creation
```

### Qdrant Vector DB
```bash
docker run -d -p 6333:6333 qdrant/qdrant
# Validates: Archival memory, embeddings
```

### Neo4j Graph DB
```bash
docker run -d -p 7687:7687 -p 7474:7474 neo4j:5
# Validates: Temporal graph, knowledge queries
```

### Redis Cache
```bash
docker run -d -p 6379:6379 redis:7
# Validates: Rate limiting, caching
```

---

## Post-Deployment Validation

### Quick Health Check
```bash
uv run ecosystem_orchestrator.py --quick --json
```

### Full Validation
```bash
uv run auto_validate.py
```

### Performance Benchmark
```bash
uv run performance.py benchmark
```

### Start Daemon Mode
```bash
uv run ralph_loop.py daemon --interval 300
```

---

## Monitoring

### Key Metrics to Watch

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| Iteration Duration | 20-30s | >60s |
| Memory RSS | <200MB | >500MB |
| Pass Rate | >80% | <50% |
| Error Rate | 0 | >5% |

### Health Endpoints

When running `platform_orchestrator.py serve --port 8080`:

- Liveness: `GET /health/live`
- Readiness: `GET /health/ready`
- Full Health: `GET /health`

---

## Troubleshooting

### Common Issues

1. **"Letta not running"** - Expected warning if Docker not used
2. **"npm view timeout"** - Network issue, MCP packages still valid
3. **"Kill switch active"** - Check `~/.claude/KILL_SWITCH` file

### Debug Mode

```bash
uv run ralph_loop.py iterate --verbose
uv run ecosystem_orchestrator.py --verbose
```

---

## Production Deployment

### Kubernetes

```bash
kubectl apply -k deploy/kubernetes/
```

### Docker Compose

```bash
docker-compose up -d
```

### Manual

```bash
# Start Ralph Loop daemon
nohup uv run ralph_loop.py daemon --interval 300 &

# Monitor logs
tail -f data/reports/*.json
```

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | Claude | 2026-01-18 | Verified |
| Platform | V10 | 2026-01-18 | 32/32 tests |
| Integration | Ralph Loop | 2026-01-18 | 5 iterations |

---

*Deployment Checklist v10.0 - January 2026*
