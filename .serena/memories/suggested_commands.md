# Suggested Commands for Unleash Platform

## Development Commands

### Python Development
- `uv run pytest tests/ -v` - Run all tests
- `uv run pytest tests/ -m "not slow"` - Run fast tests only
- `uv run python -m py_compile <file>` - Syntax check
- `pyright <file>` - Type checking

### Project Management
- `uv sync` - Sync dependencies
- `uv pip install -e .` - Install in editable mode

### SDK Commands
- `python platform/core/sdk_integrations.py` - Check SDK status
- `python platform/core/research_engine.py` - Run research engine

### Health Checks
- `python scripts/health_check.py` - Platform health check
- `python platform/cli/main.py status` - CLI status

## MCP Server Commands
- Serena Dashboard: http://127.0.0.1:24282
- TouchDesigner MCP: Port 9981
- Qdrant: Port 6333
- PostgreSQL: Port 5432

## Windows-Specific Commands
- `tasklist | findstr "python"` - List Python processes
- `netstat -ano | findstr "<port>"` - Check port usage
- `powershell Stop-Process -Name "<name>" -Force` - Kill process

## On Task Completion
1. Run type checking: `pyright <changed_files>`
2. Run tests: `uv run pytest tests/ -v --tb=short`
3. Format code: `uv run ruff format <files>`
4. Commit if all pass
