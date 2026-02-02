#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
#     "pydantic>=2.0.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Ecosystem Orchestrator - Ultimate Autonomous Platform

Unified health check dashboard integrating:
1. V10 Hooks Validation - Verify all hook scripts exist and are valid
2. MCP Server Availability - Check MCP server connectivity and tools
3. Letta Memory Connectivity - Verify Letta server and agent status
4. Claude-Flow Patterns - Check Claude-Flow accessibility and components
5. Unified Health Dashboard - JSON output for automation

Usage:
    uv run ecosystem_orchestrator.py                    # Full ecosystem check
    uv run ecosystem_orchestrator.py --json             # JSON output only
    uv run ecosystem_orchestrator.py --quick            # Quick check (skip slow tests)
    uv run ecosystem_orchestrator.py --component hooks  # Check specific component
    uv run ecosystem_orchestrator.py --fix              # Attempt to fix issues
    uv run ecosystem_orchestrator.py --watch            # Continuous monitoring

Platform: Windows 11 + Python 3.11+
Architecture: V10 Optimized (Verified, Minimal, Seamless)

Reference:
- Claude Code Hooks: https://code.claude.com/docs/en/hooks
- MCP Protocol: https://modelcontextprotocol.io/
- Letta API: https://docs.letta.com/api/
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import httpx
import structlog
from pydantic import BaseModel, Field

# =============================================================================
# Windows Unicode Compatibility
# =============================================================================

if sys.platform == "win32":
    # Set UTF-8 encoding for subprocess outputs
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Enable UTF-8 mode in Windows console
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11),
            7  # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        )
    except Exception:
        pass


# =============================================================================
# Structured Logging Configuration
# =============================================================================

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")


class ComponentStatus(str, Enum):
    """Health status for a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


class ComponentType(str, Enum):
    """Type of ecosystem component."""
    HOOKS = "hooks"
    MCP = "mcp"
    LETTA = "letta"
    CLAUDE_FLOW = "claude_flow"
    INFRASTRUCTURE = "infrastructure"


# =============================================================================
# Data Models
# =============================================================================

class ComponentCheck(BaseModel):
    """Result of checking a single component."""
    name: str
    component_type: ComponentType
    status: ComponentStatus
    message: str
    latency_ms: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)
    fix_available: bool = False
    fix_command: Optional[str] = None


class EcosystemHealth(BaseModel):
    """Complete ecosystem health report."""
    timestamp: str
    platform: str
    python_version: str
    overall_status: ComponentStatus
    components: List[ComponentCheck] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


@dataclass
class CheckContext:
    """Context passed to check functions."""
    claude_home: Path
    v10_dir: Path
    hooks_dir: Path
    quick_mode: bool = False
    verbose: bool = False


# =============================================================================
# Check Result Aggregator
# =============================================================================

class HealthAggregator:
    """Aggregates health check results."""

    def __init__(self):
        self.checks: List[ComponentCheck] = []
        self.start_time = time.perf_counter()

    def add(self, check: ComponentCheck) -> None:
        """Add a check result."""
        self.checks.append(check)

    def get_summary(self) -> Dict[str, int]:
        """Get count by status."""
        summary = {status.value: 0 for status in ComponentStatus}
        for check in self.checks:
            summary[check.status.value] += 1
        return summary

    def get_overall_status(self) -> ComponentStatus:
        """Determine overall ecosystem health."""
        summary = self.get_summary()
        if summary.get("unavailable", 0) > 2:
            return ComponentStatus.UNAVAILABLE
        if summary.get("degraded", 0) > 1 or summary.get("unavailable", 0) > 0:
            return ComponentStatus.DEGRADED
        if summary.get("healthy", 0) > 0:
            return ComponentStatus.HEALTHY
        return ComponentStatus.UNKNOWN

    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on check results."""
        recommendations = []

        # Check for missing critical components
        hooks_status = [c for c in self.checks if c.component_type == ComponentType.HOOKS]
        mcp_status = [c for c in self.checks if c.component_type == ComponentType.MCP]
        letta_status = [c for c in self.checks if c.component_type == ComponentType.LETTA]

        if any(c.status == ComponentStatus.UNAVAILABLE for c in hooks_status):
            recommendations.append("Run setup_v10.ps1 to deploy missing hooks")

        if any(c.status == ComponentStatus.UNAVAILABLE for c in mcp_status):
            recommendations.append("Install missing MCP servers: npm install -g @modelcontextprotocol/server-<name>")

        if any(c.status == ComponentStatus.UNAVAILABLE for c in letta_status):
            recommendations.append("Start Letta server: docker run -d -p 8283:8283 letta/letta:latest")

        # Check for degraded components
        if any(c.status == ComponentStatus.DEGRADED for c in self.checks):
            recommendations.append("Review degraded components for potential issues")

        return recommendations

    def to_report(self) -> EcosystemHealth:
        """Generate final health report."""
        elapsed = (time.perf_counter() - self.start_time) * 1000

        return EcosystemHealth(
            timestamp=datetime.now(timezone.utc).isoformat(),
            platform=f"{sys.platform} (Python {sys.version_info.major}.{sys.version_info.minor})",
            python_version=sys.version,
            overall_status=self.get_overall_status(),
            components=self.checks,
            summary=self.get_summary(),
            recommendations=self.get_recommendations(),
        )


# =============================================================================
# Hook Validation
# =============================================================================

async def check_hooks(ctx: CheckContext, aggregator: HealthAggregator) -> None:
    """Validate V10 hook scripts."""

    required_hooks = [
        ("letta_sync.py", "Session start/end memory sync"),
        ("mcp_guard.py", "MCP tool validation"),
        ("bash_guard.py", "Bash command validation"),
        ("memory_consolidate.py", "Sleeptime trigger"),
        ("audit_log.py", "File change logging"),
    ]

    optional_hooks = [
        ("hook_utils.py", "Shared utilities"),
        ("letta_sync_v2.py", "Enhanced Letta sync"),
        ("mcp_guard_v2.py", "Enhanced MCP guard"),
    ]

    for hook_file, description in required_hooks:
        hook_path = ctx.hooks_dir / hook_file
        check = await _validate_hook_file(hook_path, hook_file, description, required=True)
        aggregator.add(check)

    if not ctx.quick_mode:
        for hook_file, description in optional_hooks:
            hook_path = ctx.hooks_dir / hook_file
            if hook_path.exists():
                check = await _validate_hook_file(hook_path, hook_file, description, required=False)
                aggregator.add(check)


async def _validate_hook_file(
    path: Path,
    name: str,
    description: str,
    required: bool
) -> ComponentCheck:
    """Validate a single hook file."""
    start = time.perf_counter()

    if not path.exists():
        status = ComponentStatus.UNAVAILABLE if required else ComponentStatus.SKIPPED
        return ComponentCheck(
            name=f"Hook: {name}",
            component_type=ComponentType.HOOKS,
            status=status,
            message=f"Missing: {description}",
            details={"path": str(path), "required": required},
            fix_available=required,
            fix_command=f'Deploy from v10_optimized/hooks/{name}' if required else None,
        )

    # Verify Python syntax
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        latency = (time.perf_counter() - start) * 1000

        if result.returncode == 0:
            # Check for inline script metadata (PEP 723)
            content = path.read_text(encoding="utf-8")
            has_deps = "# /// script" in content

            return ComponentCheck(
                name=f"Hook: {name}",
                component_type=ComponentType.HOOKS,
                status=ComponentStatus.HEALTHY,
                message=f"{description} (PEP 723: {'yes' if has_deps else 'no'})",
                latency_ms=latency,
                details={
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "has_inline_deps": has_deps,
                },
            )
        else:
            return ComponentCheck(
                name=f"Hook: {name}",
                component_type=ComponentType.HOOKS,
                status=ComponentStatus.DEGRADED,
                message=f"Syntax error: {result.stderr[:100]}",
                latency_ms=latency,
                details={"path": str(path), "error": result.stderr},
            )

    except subprocess.TimeoutExpired:
        return ComponentCheck(
            name=f"Hook: {name}",
            component_type=ComponentType.HOOKS,
            status=ComponentStatus.DEGRADED,
            message="Syntax check timed out",
            details={"path": str(path)},
        )
    except Exception as e:
        return ComponentCheck(
            name=f"Hook: {name}",
            component_type=ComponentType.HOOKS,
            status=ComponentStatus.UNKNOWN,
            message=f"Check failed: {e}",
            details={"path": str(path), "error": str(e)},
        )


# =============================================================================
# MCP Server Availability
# =============================================================================

async def check_mcp_servers(ctx: CheckContext, aggregator: HealthAggregator) -> None:
    """Check MCP server availability."""

    # Core servers (verified working in V10)
    core_servers = [
        ("@modelcontextprotocol/server-filesystem", "filesystem", "File operations"),
        ("@modelcontextprotocol/server-memory", "memory", "Key-value store"),
        ("@modelcontextprotocol/server-sequential-thinking", "sequential-thinking", "Extended reasoning"),
    ]

    # Standard servers
    standard_servers = [
        ("@upstash/context7-mcp", "context7", "Library documentation"),
        ("@eslint/mcp", "eslint", "Code linting"),
    ]

    # Check core servers
    for package, name, description in core_servers:
        check = await _check_npm_package(package, name, description)
        aggregator.add(check)

    # Check standard servers (skip in quick mode)
    if not ctx.quick_mode:
        for package, name, description in standard_servers:
            check = await _check_npm_package(package, name, description)
            aggregator.add(check)

    # Check for local MCP config
    mcp_config = ctx.claude_home / ".mcp.json"
    if mcp_config.exists():
        try:
            config = json.loads(mcp_config.read_text(encoding="utf-8"))
            server_count = len(config.get("mcpServers", {}))
            aggregator.add(ComponentCheck(
                name="MCP: Configuration",
                component_type=ComponentType.MCP,
                status=ComponentStatus.HEALTHY,
                message=f"{server_count} servers configured",
                details={"path": str(mcp_config), "servers": list(config.get("mcpServers", {}).keys())},
            ))
        except json.JSONDecodeError as e:
            aggregator.add(ComponentCheck(
                name="MCP: Configuration",
                component_type=ComponentType.MCP,
                status=ComponentStatus.DEGRADED,
                message=f"Invalid JSON: {e}",
                details={"path": str(mcp_config)},
            ))
    else:
        aggregator.add(ComponentCheck(
            name="MCP: Configuration",
            component_type=ComponentType.MCP,
            status=ComponentStatus.DEGRADED,
            message="No .mcp.json found",
            details={"expected_path": str(mcp_config)},
            fix_available=True,
            fix_command="Create .mcp.json with server definitions",
        ))


async def _check_npm_package(package: str, name: str, description: str) -> ComponentCheck:
    """Check if npm package exists in registry."""
    start = time.perf_counter()
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    try:
        result = subprocess.run(
            [npm_cmd, "view", package, "version"],
            capture_output=True,
            text=True,
            timeout=30,
            shell=(sys.platform == "win32"),
        )
        latency = (time.perf_counter() - start) * 1000

        if result.returncode == 0:
            version = result.stdout.strip()
            return ComponentCheck(
                name=f"MCP: {name}",
                component_type=ComponentType.MCP,
                status=ComponentStatus.HEALTHY,
                message=f"{description} (v{version})",
                latency_ms=latency,
                details={"package": package, "version": version},
            )
        else:
            return ComponentCheck(
                name=f"MCP: {name}",
                component_type=ComponentType.MCP,
                status=ComponentStatus.UNAVAILABLE,
                message=f"Package not found: {package}",
                latency_ms=latency,
                fix_available=True,
                fix_command=f"npm install -g {package}",
            )

    except subprocess.TimeoutExpired:
        return ComponentCheck(
            name=f"MCP: {name}",
            component_type=ComponentType.MCP,
            status=ComponentStatus.DEGRADED,
            message="Timeout checking package",
        )
    except FileNotFoundError:
        return ComponentCheck(
            name=f"MCP: {name}",
            component_type=ComponentType.MCP,
            status=ComponentStatus.UNAVAILABLE,
            message="npm not found - Node.js required",
            fix_available=True,
            fix_command="Install Node.js from https://nodejs.org",
        )
    except Exception as e:
        return ComponentCheck(
            name=f"MCP: {name}",
            component_type=ComponentType.MCP,
            status=ComponentStatus.UNKNOWN,
            message=f"Check failed: {e}",
        )


# =============================================================================
# Letta Memory Connectivity
# =============================================================================

async def check_letta(ctx: CheckContext, aggregator: HealthAggregator) -> None:
    """Check Letta memory server connectivity."""
    letta_url = os.environ.get("LETTA_URL", "http://localhost:8500")

    # Check server health
    server_check = await _check_letta_health(letta_url)
    aggregator.add(server_check)

    if server_check.status != ComponentStatus.HEALTHY:
        return  # Skip further checks if server is down

    # Check agents (if not quick mode)
    if not ctx.quick_mode:
        agent_check = await _check_letta_agents(letta_url)
        aggregator.add(agent_check)


async def _check_letta_health(url: str) -> ComponentCheck:
    """Check Letta server health endpoint."""
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try multiple health endpoints (letta-sovereign uses /health)
            for endpoint in ["/health", "/v1/health"]:
                try:
                    response = await client.get(f"{url}{endpoint}")
                    latency = (time.perf_counter() - start) * 1000

                    if response.status_code == 200:
                        return ComponentCheck(
                            name="Letta: Server",
                            component_type=ComponentType.LETTA,
                            status=ComponentStatus.HEALTHY,
                            message=f"Running at {url}",
                            latency_ms=latency,
                            details={"url": url, "endpoint": endpoint},
                        )
                except Exception:
                    pass

            # Fallback: try root endpoint
            try:
                response = await client.get(f"{url}/")
                latency = (time.perf_counter() - start) * 1000

                if response.status_code == 200:
                    # Check if it looks like Letta
                    is_letta = "letta" in response.text.lower() or "memory" in response.text.lower()
                    return ComponentCheck(
                        name="Letta: Server",
                        component_type=ComponentType.LETTA,
                        status=ComponentStatus.HEALTHY if is_letta else ComponentStatus.DEGRADED,
                        message=f"Running at {url}" if is_letta else "Server responding but may not be Letta",
                        latency_ms=latency,
                        details={"url": url, "endpoint": "/"},
                    )
            except Exception:
                pass

            return ComponentCheck(
                name="Letta: Server",
                component_type=ComponentType.LETTA,
                status=ComponentStatus.UNAVAILABLE,
                message=f"No valid response from {url}",
                fix_available=True,
                fix_command="docker run -d -p 8500:8283 --name letta-sovereign letta/letta:latest",
            )

    except httpx.ConnectError:
        return ComponentCheck(
            name="Letta: Server",
            component_type=ComponentType.LETTA,
            status=ComponentStatus.UNAVAILABLE,
            message=f"Cannot connect to {url}",
            details={"url": url, "note": "Memory persistence disabled without Letta"},
            fix_available=True,
            fix_command="docker run -d -p 8500:8283 --name letta-sovereign letta/letta:latest",
        )
    except Exception as e:
        return ComponentCheck(
            name="Letta: Server",
            component_type=ComponentType.LETTA,
            status=ComponentStatus.UNKNOWN,
            message=f"Check failed: {e}",
            details={"url": url, "error": str(e)},
        )


async def _check_letta_agents(url: str) -> ComponentCheck:
    """Check Letta agents."""
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{url}/v1/agents")
            latency = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                agents = response.json()
                agent_count = len(agents) if isinstance(agents, list) else 0
                claude_agents = [
                    a for a in (agents if isinstance(agents, list) else [])
                    if a.get("name", "").startswith("claude-code-")
                ]

                return ComponentCheck(
                    name="Letta: Agents",
                    component_type=ComponentType.LETTA,
                    status=ComponentStatus.HEALTHY,
                    message=f"{agent_count} agents ({len(claude_agents)} Claude Code)",
                    latency_ms=latency,
                    details={
                        "total_agents": agent_count,
                        "claude_agents": len(claude_agents),
                        "agent_names": [a.get("name") for a in claude_agents[:5]],
                    },
                )
            else:
                return ComponentCheck(
                    name="Letta: Agents",
                    component_type=ComponentType.LETTA,
                    status=ComponentStatus.DEGRADED,
                    message=f"Unexpected status: {response.status_code}",
                    latency_ms=latency,
                )

    except Exception as e:
        return ComponentCheck(
            name="Letta: Agents",
            component_type=ComponentType.LETTA,
            status=ComponentStatus.DEGRADED,
            message=f"Agent check failed: {e}",
        )


# =============================================================================
# Claude-Flow Patterns
# =============================================================================

async def check_claude_flow(ctx: CheckContext, aggregator: HealthAggregator) -> None:
    """Check Claude-Flow pattern accessibility."""

    # Define Claude-Flow locations to check
    claude_flow_paths = [
        Path("Z:/insider/AUTO CLAUDE/unleash/ruvnet-claude-flow/v2"),
        Path("Z:/insider/AUTO CLAUDE/unleash/ruvnet-claude-flow/v3"),
    ]

    # Check for Claude-Flow installation
    for cf_path in claude_flow_paths:
        if cf_path.exists():
            check = await _check_claude_flow_version(cf_path)
            aggregator.add(check)

    # Check for Claude-Flow commands in user's .claude directory
    commands_dir = ctx.claude_home / "commands"
    if commands_dir.exists():
        cf_commands = list(commands_dir.glob("claude-flow*.md"))
        if cf_commands:
            aggregator.add(ComponentCheck(
                name="Claude-Flow: Commands",
                component_type=ComponentType.CLAUDE_FLOW,
                status=ComponentStatus.HEALTHY,
                message=f"{len(cf_commands)} commands available",
                details={"commands": [c.stem for c in cf_commands]},
            ))

    # Check V10 core orchestrator patterns
    core_orchestrator = ctx.v10_dir.parent / "core" / "orchestrator.py"
    if core_orchestrator.exists():
        aggregator.add(ComponentCheck(
            name="Claude-Flow: Orchestrator",
            component_type=ComponentType.CLAUDE_FLOW,
            status=ComponentStatus.HEALTHY,
            message="Agent orchestration patterns available",
            details={"path": str(core_orchestrator)},
        ))
    else:
        aggregator.add(ComponentCheck(
            name="Claude-Flow: Orchestrator",
            component_type=ComponentType.CLAUDE_FLOW,
            status=ComponentStatus.DEGRADED,
            message="Core orchestrator not found",
            details={"expected_path": str(core_orchestrator)},
        ))


async def _check_claude_flow_version(path: Path) -> ComponentCheck:
    """Check a specific Claude-Flow version."""
    version = path.name  # e.g., "v2" or "v3"

    # V2 and V3 have different structures
    if version == "v3":
        # V3 structure: agent-lifecycle, coordination, infrastructure, mcp, memory
        components = {
            "agent-lifecycle": path / "src" / "agent-lifecycle",
            "coordination": path / "src" / "coordination",
            "mcp": path / "src" / "mcp",
        }
    else:
        # V2 structure: bin/cli, src/swarm, src/mcp
        components = {
            "cli": path / "bin" / "claude-flow.js",
            "mcp": path / "src" / "mcp",
            "swarm": path / "src" / "swarm",
        }

    available = {k: v.exists() for k, v in components.items()}
    healthy = sum(available.values()) >= 2

    return ComponentCheck(
        name=f"Claude-Flow: {version}",
        component_type=ComponentType.CLAUDE_FLOW,
        status=ComponentStatus.HEALTHY if healthy else ComponentStatus.DEGRADED,
        message=f"{sum(available.values())}/{len(components)} components found",
        details={"path": str(path), "components": available},
    )


# =============================================================================
# Infrastructure Checks
# =============================================================================

async def check_infrastructure(ctx: CheckContext, aggregator: HealthAggregator) -> None:
    """Check infrastructure components."""

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    aggregator.add(ComponentCheck(
        name="Infra: Python",
        component_type=ComponentType.INFRASTRUCTURE,
        status=ComponentStatus.HEALTHY if py_ok else ComponentStatus.DEGRADED,
        message=f"Python {py_version}" + (" (3.11+ required)" if not py_ok else ""),
    ))

    # Check uv availability
    uv_check = await _check_command("uv", "--version", "Python package manager")
    aggregator.add(uv_check)

    # Check Node.js
    if not ctx.quick_mode:
        node_check = await _check_command("node", "--version", "JavaScript runtime")
        aggregator.add(node_check)

    # Check kill switch
    kill_switch = ctx.claude_home / "KILL_SWITCH"
    if kill_switch.exists():
        aggregator.add(ComponentCheck(
            name="Infra: Kill Switch",
            component_type=ComponentType.INFRASTRUCTURE,
            status=ComponentStatus.DEGRADED,
            message="ACTIVE - Operations blocked!",
            fix_available=True,
            fix_command=f'Remove-Item "{kill_switch}"',
        ))
    else:
        aggregator.add(ComponentCheck(
            name="Infra: Kill Switch",
            component_type=ComponentType.INFRASTRUCTURE,
            status=ComponentStatus.HEALTHY,
            message="Inactive (normal operation)",
        ))

    # Check settings.json
    settings = ctx.claude_home / "settings.json"
    if settings.exists():
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
            aggregator.add(ComponentCheck(
                name="Infra: Settings",
                component_type=ComponentType.INFRASTRUCTURE,
                status=ComponentStatus.HEALTHY,
                message=f"Valid ({len(json.dumps(data))} bytes)",
                details={"path": str(settings)},
            ))
        except json.JSONDecodeError as e:
            aggregator.add(ComponentCheck(
                name="Infra: Settings",
                component_type=ComponentType.INFRASTRUCTURE,
                status=ComponentStatus.DEGRADED,
                message=f"Invalid JSON: {e}",
            ))
    else:
        aggregator.add(ComponentCheck(
            name="Infra: Settings",
            component_type=ComponentType.INFRASTRUCTURE,
            status=ComponentStatus.DEGRADED,
            message="settings.json not found",
        ))


async def _check_command(cmd: str, version_flag: str, description: str) -> ComponentCheck:
    """Check if a command is available."""
    start = time.perf_counter()

    try:
        # Handle Windows .cmd extensions
        if sys.platform == "win32" and not cmd.endswith(".exe"):
            cmd_options = [cmd, f"{cmd}.cmd", f"{cmd}.exe"]
        else:
            cmd_options = [cmd]

        for cmd_try in cmd_options:
            try:
                result = subprocess.run(
                    [cmd_try, version_flag],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=(sys.platform == "win32"),
                )
                latency = (time.perf_counter() - start) * 1000

                if result.returncode == 0:
                    version = result.stdout.strip().split("\n")[0]
                    return ComponentCheck(
                        name=f"Infra: {cmd}",
                        component_type=ComponentType.INFRASTRUCTURE,
                        status=ComponentStatus.HEALTHY,
                        message=f"{description} ({version})",
                        latency_ms=latency,
                    )
            except FileNotFoundError:
                continue

        return ComponentCheck(
            name=f"Infra: {cmd}",
            component_type=ComponentType.INFRASTRUCTURE,
            status=ComponentStatus.UNAVAILABLE,
            message=f"{description} not found",
        )

    except Exception as e:
        return ComponentCheck(
            name=f"Infra: {cmd}",
            component_type=ComponentType.INFRASTRUCTURE,
            status=ComponentStatus.UNKNOWN,
            message=f"Check failed: {e}",
        )


# =============================================================================
# Main Orchestrator
# =============================================================================

class EcosystemOrchestrator:
    """
    Unified Ecosystem Orchestrator for the Ultimate Autonomous Platform.

    Coordinates health checks across all platform components and provides
    a unified dashboard for automation and monitoring.
    """

    def __init__(
        self,
        claude_home: Optional[Path] = None,
        quick_mode: bool = False,
        verbose: bool = False,
    ):
        self.claude_home = claude_home or Path.home() / ".claude"
        self.v10_dir = self.claude_home / "v10"
        self.hooks_dir = self.v10_dir / "hooks"
        self.quick_mode = quick_mode
        self.verbose = verbose

        self.ctx = CheckContext(
            claude_home=self.claude_home,
            v10_dir=self.v10_dir,
            hooks_dir=self.hooks_dir,
            quick_mode=quick_mode,
            verbose=verbose,
        )

    async def check_all(self, component: Optional[str] = None) -> EcosystemHealth:
        """Run all ecosystem health checks."""
        aggregator = HealthAggregator()

        check_functions = {
            "hooks": check_hooks,
            "mcp": check_mcp_servers,
            "letta": check_letta,
            "claude_flow": check_claude_flow,
            "infrastructure": check_infrastructure,
        }

        if component and component in check_functions:
            # Run single component check
            await check_functions[component](self.ctx, aggregator)
        else:
            # Run all checks concurrently
            await asyncio.gather(
                check_hooks(self.ctx, aggregator),
                check_mcp_servers(self.ctx, aggregator),
                check_letta(self.ctx, aggregator),
                check_claude_flow(self.ctx, aggregator),
                check_infrastructure(self.ctx, aggregator),
            )

        return aggregator.to_report()

    async def watch(self, interval: int = 30) -> None:
        """Continuously monitor ecosystem health."""
        print(f"[WATCH] Monitoring ecosystem health every {interval}s (Ctrl+C to stop)")
        print("-" * 60)

        try:
            while True:
                report = await self.check_all()
                timestamp = datetime.now().strftime("%H:%M:%S")
                status_str = report.overall_status.value.upper()

                # Simple ASCII status line
                print(f"[{timestamp}] Status: {status_str} | " +
                      f"OK: {report.summary.get('healthy', 0)} | " +
                      f"WARN: {report.summary.get('degraded', 0)} | " +
                      f"ERR: {report.summary.get('unavailable', 0)}")

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\n[WATCH] Stopped")


# =============================================================================
# Output Formatting
# =============================================================================

def print_report(report: EcosystemHealth, verbose: bool = False) -> None:
    """Print health report to console."""
    # Header
    print("=" * 60)
    print("ECOSYSTEM HEALTH DASHBOARD")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Platform:  {report.platform}")
    print(f"Overall:   [{report.overall_status.value.upper()}]")
    print()

    # Summary
    print(f"Summary: OK={report.summary.get('healthy', 0)} | " +
          f"WARN={report.summary.get('degraded', 0)} | " +
          f"ERR={report.summary.get('unavailable', 0)} | " +
          f"SKIP={report.summary.get('skipped', 0)}")
    print()

    # Components by type
    print("-" * 60)
    print(f"{'Component':<35} {'Status':<12} {'Message'}")
    print("-" * 60)

    status_icons = {
        ComponentStatus.HEALTHY: "[OK]",
        ComponentStatus.DEGRADED: "[WARN]",
        ComponentStatus.UNAVAILABLE: "[ERR]",
        ComponentStatus.UNKNOWN: "[?]",
        ComponentStatus.SKIPPED: "[--]",
    }

    # Group by component type
    by_type: Dict[ComponentType, List[ComponentCheck]] = {}
    for check in report.components:
        by_type.setdefault(check.component_type, []).append(check)

    for ctype in ComponentType:
        checks = by_type.get(ctype, [])
        if checks:
            for check in checks:
                icon = status_icons.get(check.status, "[?]")
                msg = check.message[:30] + "..." if len(check.message) > 30 else check.message
                print(f"{check.name:<35} {icon:<12} {msg}")

    # Recommendations
    if report.recommendations:
        print()
        print("-" * 60)
        print("RECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    # Verbose details
    if verbose:
        print()
        print("-" * 60)
        print("DETAILED STATUS:")
        for check in report.components:
            if check.details:
                print(f"\n{check.name}:")
                for k, v in check.details.items():
                    print(f"  {k}: {v}")

    print()
    print("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ecosystem Orchestrator - Ultimate Autonomous Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run ecosystem_orchestrator.py              # Full health check
    uv run ecosystem_orchestrator.py --json       # JSON output for automation
    uv run ecosystem_orchestrator.py --quick      # Quick check
    uv run ecosystem_orchestrator.py --watch      # Continuous monitoring
    uv run ecosystem_orchestrator.py --component hooks  # Check only hooks
        """,
    )

    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON for automation",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick check (skip slow tests)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with details",
    )
    parser.add_argument(
        "--component", "-c",
        choices=["hooks", "mcp", "letta", "claude_flow", "infrastructure"],
        help="Check specific component only",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuous monitoring mode",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Watch interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues (not implemented)",
    )

    args = parser.parse_args()

    orchestrator = EcosystemOrchestrator(
        quick_mode=args.quick,
        verbose=args.verbose,
    )

    if args.watch:
        await orchestrator.watch(interval=args.interval)
        return

    report = await orchestrator.check_all(component=args.component)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2, default=str))
    else:
        print_report(report, verbose=args.verbose)

    # Exit code based on status
    if report.overall_status == ComponentStatus.UNAVAILABLE:
        sys.exit(2)
    elif report.overall_status == ComponentStatus.DEGRADED:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
