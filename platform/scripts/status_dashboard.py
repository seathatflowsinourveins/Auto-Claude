#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "rich>=13.7.0",
# ]
# ///
"""
Unified Status Dashboard - Ultimate Autonomous Platform

Comprehensive monitoring dashboard for all platform components:
- Docker containers (Letta, Qdrant, Neo4j, etc.)
- MCP servers configuration
- Hook implementations
- Memory systems
- Swarm coordination

Usage:
    uv run status_dashboard.py
    python status_dashboard.py --json  # JSON output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try rich for beautiful output, fallback to plain text
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class ComponentStatus:
    """Status of a platform component."""
    name: str
    status: str  # "ok", "warning", "error", "unknown"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None


@dataclass
class PlatformStatus:
    """Overall platform status."""
    timestamp: str
    components: List[ComponentStatus]
    summary: Dict[str, int]  # counts by status


class StatusChecker:
    """Checks status of all platform components."""

    def __init__(self):
        self.v10_dir = Path.home() / ".claude" / "v10"
        self.hooks_dir = self.v10_dir / "hooks"
        self.project_dir = Path(os.environ.get(
            "CLAUDE_PROJECT_DIR",
            r"Z:\insider\AUTO CLAUDE\unleash"
        ))

    async def check_all(self) -> PlatformStatus:
        """Check all components and return status."""
        components = []

        # Run checks concurrently where possible
        checks = [
            self._check_docker_containers(),
            self._check_letta_server(),
            self._check_qdrant_server(),
            self._check_hooks(),
            self._check_mcp_config(),
            self._check_ralph_loop(),
            self._check_memory_systems(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                components.append(ComponentStatus(
                    name="unknown",
                    status="error",
                    message=str(result)
                ))
            elif isinstance(result, list):
                components.extend(result)
            else:
                components.append(result)

        # Calculate summary
        summary = {"ok": 0, "warning": 0, "error": 0, "unknown": 0}
        for comp in components:
            summary[comp.status] = summary.get(comp.status, 0) + 1

        return PlatformStatus(
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=components,
            summary=summary
        )

    async def _check_docker_containers(self) -> List[ComponentStatus]:
        """Check Docker container status."""
        results = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "ps", "--format", "{{.Names}}|{{.Status}}|{{.Ports}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return [ComponentStatus(
                    name="Docker",
                    status="error",
                    message="Docker not running or not accessible"
                )]

            # Parse container info
            expected_containers = {
                "letta": {"port": "8283", "critical": True},  # Main Letta server
                "letta-sovereign": {"port": "8500", "critical": False},  # Sovereign instance
                "qdrant": {"port": "6333", "critical": True},
                "neo4j": {"port": "7687", "critical": False},
                "redis": {"port": "6379", "critical": False},
                "postgres": {"port": "5432", "critical": False},
                "questdb": {"port": "9000", "critical": False},
                "grafana": {"port": "3000", "critical": False},
                "prometheus": {"port": "9090", "critical": False},
            }

            running_containers = {}
            for line in stdout.decode().strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    name, status = parts[0], parts[1]
                    running_containers[name.lower()] = {
                        "status": status,
                        "ports": parts[2] if len(parts) > 2 else ""
                    }

            for container, config in expected_containers.items():
                # Check various name formats
                found = None
                for key in running_containers:
                    if container in key.lower():
                        found = running_containers[key]
                        break

                if found and "Up" in found["status"]:
                    results.append(ComponentStatus(
                        name=f"Docker: {container}",
                        status="ok",
                        message=found["status"],
                        details={"ports": found["ports"]}
                    ))
                else:
                    status = "error" if config["critical"] else "warning"
                    results.append(ComponentStatus(
                        name=f"Docker: {container}",
                        status=status,
                        message="Not running"
                    ))

        except FileNotFoundError:
            results.append(ComponentStatus(
                name="Docker",
                status="error",
                message="Docker not installed"
            ))

        return results

    async def _check_letta_server(self) -> ComponentStatus:
        """Check Letta server health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                start = datetime.now()
                # Note: Letta API requires trailing slash on health endpoint
                resp = await client.get("http://localhost:8283/v1/health/")
                latency = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = {}
                    if resp.headers.get("content-type", "").startswith("application/json"):
                        data = resp.json()
                    version = data.get("version", "unknown")
                    return ComponentStatus(
                        name="Letta Server",
                        status="ok",
                        message=f"v{version}",
                        latency_ms=latency,
                        details=data
                    )
                else:
                    return ComponentStatus(
                        name="Letta Server",
                        status="warning",
                        message=f"HTTP {resp.status_code}"
                    )
        except Exception as e:
            return ComponentStatus(
                name="Letta Server",
                status="error",
                message=str(e)[:50]
            )

    async def _check_qdrant_server(self) -> ComponentStatus:
        """Check Qdrant vector database."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                start = datetime.now()
                resp = await client.get("http://localhost:6333/collections")
                latency = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    collections = data.get("result", {}).get("collections", [])
                    return ComponentStatus(
                        name="Qdrant",
                        status="ok",
                        message=f"{len(collections)} collections",
                        latency_ms=latency,
                        details={"collections": [c.get("name") for c in collections]}
                    )
                else:
                    return ComponentStatus(
                        name="Qdrant",
                        status="warning",
                        message=f"HTTP {resp.status_code}"
                    )
        except Exception as e:
            return ComponentStatus(
                name="Qdrant",
                status="error",
                message=str(e)[:50]
            )

    async def _check_hooks(self) -> List[ComponentStatus]:
        """Check hook implementations."""
        results = []

        hooks = [
            ("bash_guard.py", "PreToolUse - Bash security"),
            ("mcp_guard_v2.py", "PreToolUse - MCP security"),
            ("letta_sync_v2.py", "SessionStart/End - Memory sync"),
            ("audit_log.py", "PostToolUse - Audit logging"),
            ("memory_consolidate.py", "Stop - Sleeptime trigger"),
        ]

        for hook_file, description in hooks:
            hook_path = self.hooks_dir / hook_file
            if hook_path.exists():
                # Check if it's valid Python
                try:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "py_compile", str(hook_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    _, stderr = await proc.communicate()

                    if proc.returncode == 0:
                        results.append(ComponentStatus(
                            name=f"Hook: {hook_file}",
                            status="ok",
                            message=description
                        ))
                    else:
                        results.append(ComponentStatus(
                            name=f"Hook: {hook_file}",
                            status="error",
                            message=f"Syntax error: {stderr.decode()[:50]}"
                        ))
                except Exception as e:
                    results.append(ComponentStatus(
                        name=f"Hook: {hook_file}",
                        status="warning",
                        message=str(e)[:50]
                    ))
            else:
                results.append(ComponentStatus(
                    name=f"Hook: {hook_file}",
                    status="warning",
                    message="Not found"
                ))

        return results

    async def _check_mcp_config(self) -> ComponentStatus:
        """Check MCP configuration."""
        mcp_paths = [
            self.project_dir / ".mcp.json",
            Path.home() / ".claude" / ".mcp.json",
            Path.home() / ".mcp.json",
        ]

        for mcp_path in mcp_paths:
            if mcp_path.exists():
                try:
                    with open(mcp_path) as f:
                        config = json.load(f)

                    servers = config.get("mcpServers", {})
                    return ComponentStatus(
                        name="MCP Config",
                        status="ok",
                        message=f"{len(servers)} servers configured",
                        details={
                            "path": str(mcp_path),
                            "servers": list(servers.keys())[:10]  # First 10
                        }
                    )
                except json.JSONDecodeError:
                    return ComponentStatus(
                        name="MCP Config",
                        status="error",
                        message=f"Invalid JSON: {mcp_path}"
                    )

        return ComponentStatus(
            name="MCP Config",
            status="warning",
            message="No .mcp.json found"
        )

    async def _check_ralph_loop(self) -> ComponentStatus:
        """Check Ralph Loop status."""
        ralph_file = Path.home() / ".claude" / "ralph-loop.local.md"

        if not ralph_file.exists():
            return ComponentStatus(
                name="Ralph Loop",
                status="unknown",
                message="Not configured"
            )

        try:
            content = ralph_file.read_text()

            # Parse YAML frontmatter
            if content.startswith("---"):
                _, frontmatter, _ = content.split("---", 2)
                import re
                active = re.search(r"active:\s*(\w+)", frontmatter)
                iteration = re.search(r"iteration:\s*(\d+)", frontmatter)
                max_iter = re.search(r"max_iterations:\s*(\d+)", frontmatter)

                is_active = active and active.group(1).lower() == "true"
                current = int(iteration.group(1)) if iteration else 0
                maximum = int(max_iter.group(1)) if max_iter else 0

                if is_active:
                    return ComponentStatus(
                        name="Ralph Loop",
                        status="ok",
                        message=f"Active: {current}/{maximum}",
                        details={
                            "iteration": current,
                            "max_iterations": maximum
                        }
                    )
                else:
                    return ComponentStatus(
                        name="Ralph Loop",
                        status="warning",
                        message="Inactive"
                    )

        except Exception as e:
            return ComponentStatus(
                name="Ralph Loop",
                status="error",
                message=str(e)[:50]
            )

        return ComponentStatus(
            name="Ralph Loop",
            status="unknown",
            message="Could not parse"
        )

    async def _check_memory_systems(self) -> List[ComponentStatus]:
        """Check memory system directories."""
        results = []

        memory_paths = [
            (self.v10_dir / "logs", "Security Logs"),
            (self.v10_dir / "memory", "Memory Store"),
            (Path.home() / ".serena", "Serena Memory"),
        ]

        for path, name in memory_paths:
            if path.exists():
                # Count files
                try:
                    file_count = len(list(path.glob("*")))
                    results.append(ComponentStatus(
                        name=f"Memory: {name}",
                        status="ok",
                        message=f"{file_count} files",
                        details={"path": str(path)}
                    ))
                except Exception:
                    results.append(ComponentStatus(
                        name=f"Memory: {name}",
                        status="warning",
                        message="Cannot access"
                    ))
            else:
                results.append(ComponentStatus(
                    name=f"Memory: {name}",
                    status="unknown",
                    message="Not created"
                ))

        return results


def render_plain(status: PlatformStatus) -> str:
    """Render status as plain text."""
    lines = []
    lines.append("=" * 60)
    lines.append("ULTIMATE AUTONOMOUS PLATFORM - STATUS DASHBOARD")
    lines.append(f"Timestamp: {status.timestamp}")
    lines.append("=" * 60)

    # Summary
    lines.append(f"\nSummary: {status.summary['ok']} OK, "
                 f"{status.summary['warning']} Warnings, "
                 f"{status.summary['error']} Errors")
    lines.append("-" * 60)

    # Components by status
    for stat in ["error", "warning", "ok", "unknown"]:
        comps = [c for c in status.components if c.status == stat]
        if comps:
            lines.append(f"\n[{stat.upper()}]")
            for comp in comps:
                latency = f" ({comp.latency_ms:.0f}ms)" if comp.latency_ms else ""
                lines.append(f"  {comp.name}: {comp.message}{latency}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def render_rich(status: PlatformStatus, console: Console) -> None:
    """Render status using rich."""
    # Header
    console.print(Panel.fit(
        f"[bold cyan]ULTIMATE AUTONOMOUS PLATFORM[/bold cyan]\n"
        f"Status Dashboard - {status.timestamp}",
        border_style="cyan"
    ))

    # Summary
    summary_text = (
        f"[green]{status.summary['ok']} OK[/green] | "
        f"[yellow]{status.summary['warning']} Warnings[/yellow] | "
        f"[red]{status.summary['error']} Errors[/red] | "
        f"[dim]{status.summary['unknown']} Unknown[/dim]"
    )
    console.print(f"\n{summary_text}\n")

    # Components table
    table = Table(title="Components", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Message")
    table.add_column("Latency", justify="right")

    status_colors = {
        "ok": "[green]OK[/green]",
        "warning": "[yellow]WARN[/yellow]",
        "error": "[red]ERROR[/red]",
        "unknown": "[dim]?[/dim]"
    }

    # Sort by status (errors first)
    sorted_comps = sorted(
        status.components,
        key=lambda c: {"error": 0, "warning": 1, "unknown": 2, "ok": 3}.get(c.status, 4)
    )

    for comp in sorted_comps:
        latency = f"{comp.latency_ms:.0f}ms" if comp.latency_ms else "-"
        table.add_row(
            comp.name,
            status_colors.get(comp.status, comp.status),
            comp.message,
            latency
        )

    console.print(table)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Platform Status Dashboard")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", action="store_true", help="Watch mode (refresh every 5s)")
    args = parser.parse_args()

    checker = StatusChecker()

    if args.watch and RICH_AVAILABLE:
        console = Console()
        with Live(console=console, refresh_per_second=0.2) as live:
            while True:
                status = await checker.check_all()
                render_rich(status, console)
                await asyncio.sleep(5)
    else:
        status = await checker.check_all()

        if args.json:
            # JSON output
            output = {
                "timestamp": status.timestamp,
                "summary": status.summary,
                "components": [
                    {
                        "name": c.name,
                        "status": c.status,
                        "message": c.message,
                        "latency_ms": c.latency_ms,
                        "details": c.details
                    }
                    for c in status.components
                ]
            }
            print(json.dumps(output, indent=2))
        elif RICH_AVAILABLE:
            console = Console()
            render_rich(status, console)
        else:
            print(render_plain(status))


if __name__ == "__main__":
    asyncio.run(main())
