# V10 Optimized Scripts

**Ultimate Autonomous Platform - Script Modules**

All scripts use PEP 723 inline metadata for zero-config execution:
```bash
uv run <script>.py [command] [options]
```

---

## Quick Start

```bash
# 1. Check platform health
uv run ecosystem_orchestrator.py --quick

# 2. Run validation
uv run auto_validate.py

# 3. Start Ralph Loop iteration
uv run ralph_loop.py iterate

# 4. View performance metrics
uv run performance.py status
```

---

## Module Index

### Orchestration Layer
| Script | Purpose | Command |
|--------|---------|---------|
| `ralph_loop.py` | Main iteration loop | `iterate`, `daemon`, `status` |
| `ecosystem_orchestrator.py` | Health dashboard | `--quick`, `--watch` |
| `platform_orchestrator.py` | Unified entry | `status`, `health`, `serve` |

### Validation Layer
| Script | Purpose | Command |
|--------|---------|---------|
| `auto_validate.py` | 6-step validation | `--quick`, `--json` |
| `verify_mcp.py` | MCP verification | `--json` |
| `health_check.py` | Component health | `--json` |

### Memory Layer
| Script | Purpose | Command |
|--------|---------|---------|
| `sleeptime_compute.py` | Background processing | `consolidate`, `warmstart` |
| `session_continuity.py` | Session management | `init`, `export`, `trinity` |
| `memory_bridge.py` | Memory sync | `sync`, `status` |

### Performance Layer
| Script | Purpose | Command |
|--------|---------|---------|
| `performance.py` | Profiling/benchmarks | `benchmark`, `optimize` |
| `metrics.py` | Prometheus metrics | `--port 9090` |
| `tracing.py` | OpenTelemetry | `--endpoint` |
| `status_dashboard.py` | Status display | `--watch` |

### Infrastructure Layer
| Script | Purpose | Command |
|--------|---------|---------|
| `config.py` | Configuration | `get`, `set`, `export` |
| `secrets.py` | Secrets management | `set`, `get`, `rotate` |
| `install.py` | Platform installer | `--check`, `--quick` |
| `autoscale.py` | HPA metrics | `metrics`, `recommend` |
| `rate_limiter.py` | Rate limiting | Various algorithms |

---

## Common Flags

All scripts support:
- `--json`, `-j` - Machine-readable JSON output
- `--verbose`, `-v` - Detailed output
- `--quick`, `-q` - Skip slow operations
- `--help`, `-h` - Show usage

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Warning (degraded) |
| 2 | Failure |

---

## Dependencies

Scripts auto-install dependencies via `uv run`. Common deps:
- `structlog>=24.1.0` - Structured logging
- `pydantic>=2.0.0` - Data validation
- `httpx>=0.26.0` - HTTP client
- `psutil>=5.9.0` - System info

---

*V10 Optimized - January 2026*
