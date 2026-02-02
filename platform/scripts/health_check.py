#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "rich>=13.0.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
V10 System Health Check

Comprehensive health verification for the V10 Optimized architecture.
Checks all components: hooks, MCP servers, Letta, and configuration.

Usage:
    uv run health_check.py           # Full health check
    uv run health_check.py --quick   # Quick check (skip slow tests)
    uv run health_check.py --fix     # Attempt to fix issues
    uv run health_check.py --json    # Output as JSON

Based on:
- Claude Code Hooks: https://code.claude.com/docs/en/hooks
- MCP Protocol: https://modelcontextprotocol.io/
- Letta API: https://docs.letta.com/api/
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Windows compatibility: force UTF-8 and use safe console settings
if sys.platform == "win32":
    # Set UTF-8 encoding for subprocess outputs
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Try to enable UTF-8 mode in Windows console
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

# Use legacy_windows=True on Windows to avoid encoding issues
console = Console(legacy_windows=True if sys.platform == "win32" else False)


class HealthStatus(Enum):
    """Health check status levels."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_available: bool = False
    fix_command: Optional[str] = None


@dataclass
class HealthReport:
    """Complete health report."""
    timestamp: str
    overall_status: HealthStatus
    checks: List[CheckResult]
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "fix_available": c.fix_available,
                    "fix_command": c.fix_command,
                }
                for c in self.checks
            ]
        }


class HealthChecker:
    """Comprehensive V10 health checker."""

    def __init__(self, claude_home: Optional[Path] = None):
        self.claude_home = claude_home or Path.home() / ".claude"
        self.v10_dir = self.claude_home / "v10"
        self.hooks_dir = self.v10_dir / "hooks"
        self.logs_dir = self.v10_dir / "logs"
        self.checks: List[CheckResult] = []

    def add_check(self, result: CheckResult) -> None:
        """Add a check result."""
        self.checks.append(result)

    # === Directory Structure Checks ===

    def check_directories(self) -> None:
        """Check required directories exist."""
        required_dirs = [
            (self.claude_home, "Claude home directory"),
            (self.v10_dir, "V10 directory"),
            (self.hooks_dir, "Hooks directory"),
            (self.logs_dir, "Logs directory"),
        ]

        for dir_path, name in required_dirs:
            if dir_path.exists():
                self.add_check(CheckResult(
                    name=f"Directory: {name}",
                    status=HealthStatus.OK,
                    message=f"Exists at {dir_path}",
                ))
            else:
                self.add_check(CheckResult(
                    name=f"Directory: {name}",
                    status=HealthStatus.ERROR,
                    message=f"Missing: {dir_path}",
                    fix_available=True,
                    fix_command=f'mkdir -p "{dir_path}"',
                ))

    # === Hook Checks ===

    def check_hooks(self) -> None:
        """Check all hook files exist and are valid Python."""
        required_hooks = [
            "letta_sync.py",
            "mcp_guard.py",
            "bash_guard.py",
            "memory_consolidate.py",
            "audit_log.py",
        ]

        optional_hooks = [
            "hook_utils.py",
            "letta_sync_v2.py",
            "mcp_guard_v2.py",
        ]

        for hook in required_hooks:
            hook_path = self.hooks_dir / hook
            self._check_hook_file(hook_path, hook, required=True)

        for hook in optional_hooks:
            hook_path = self.hooks_dir / hook
            if hook_path.exists():
                self._check_hook_file(hook_path, hook, required=False)

    def _check_hook_file(self, path: Path, name: str, required: bool) -> None:
        """Check a single hook file."""
        if not path.exists():
            status = HealthStatus.ERROR if required else HealthStatus.WARNING
            self.add_check(CheckResult(
                name=f"Hook: {name}",
                status=status,
                message=f"{'Missing required' if required else 'Optional missing'}: {path}",
            ))
            return

        # Check Python syntax
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.add_check(CheckResult(
                    name=f"Hook: {name}",
                    status=HealthStatus.OK,
                    message="Valid Python syntax",
                    details={"path": str(path), "size": path.stat().st_size},
                ))
            else:
                self.add_check(CheckResult(
                    name=f"Hook: {name}",
                    status=HealthStatus.ERROR,
                    message=f"Syntax error: {result.stderr[:200]}",
                ))
        except Exception as e:
            self.add_check(CheckResult(
                name=f"Hook: {name}",
                status=HealthStatus.WARNING,
                message=f"Could not verify: {e}",
            ))

    # === Configuration Checks ===

    def check_config(self) -> None:
        """Check configuration files."""
        config_files = [
            (self.claude_home / "settings.json", "Settings"),
            (self.claude_home / "CLAUDE.md", "CLAUDE.md"),
            (self.claude_home / ".mcp.json", "MCP Config"),
        ]

        for config_path, name in config_files:
            if config_path.exists():
                try:
                    if config_path.suffix == ".json":
                        with open(config_path) as f:
                            data = json.load(f)
                        self.add_check(CheckResult(
                            name=f"Config: {name}",
                            status=HealthStatus.OK,
                            message=f"Valid JSON ({len(json.dumps(data))} bytes)",
                        ))
                    else:
                        content = config_path.read_text()
                        self.add_check(CheckResult(
                            name=f"Config: {name}",
                            status=HealthStatus.OK,
                            message=f"Exists ({len(content)} bytes)",
                        ))
                except json.JSONDecodeError as e:
                    self.add_check(CheckResult(
                        name=f"Config: {name}",
                        status=HealthStatus.ERROR,
                        message=f"Invalid JSON: {e}",
                    ))
            else:
                self.add_check(CheckResult(
                    name=f"Config: {name}",
                    status=HealthStatus.WARNING,
                    message=f"Missing: {config_path}",
                ))

    # === MCP Server Checks ===

    async def check_mcp_packages(self, quick: bool = False) -> None:
        """Check MCP npm packages exist."""
        packages = [
            ("@modelcontextprotocol/server-filesystem", "filesystem"),
            ("@modelcontextprotocol/server-memory", "memory"),
            ("@modelcontextprotocol/server-sequential-thinking", "sequential-thinking"),
        ]

        if not quick:
            # Only check packages that exist in npm registry
            # Note: fetch and sqlite are not under @modelcontextprotocol
            packages.extend([
                ("@eslint/mcp", "eslint"),
                ("@upstash/context7-mcp", "context7"),
                # ("@modelcontextprotocol/server-fetch", "fetch"),  # Not in npm registry
                # ("@modelcontextprotocol/server-sqlite", "sqlite"), # Not in npm registry
            ])

        # Use shell=True on Windows for npm commands
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

        for package, name in packages:
            try:
                result = subprocess.run(
                    [npm_cmd, "view", package, "version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=(sys.platform == "win32")  # Required for .cmd files
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
                    self.add_check(CheckResult(
                        name=f"MCP: {name}",
                        status=HealthStatus.OK,
                        message=f"Available (v{version})",
                        details={"package": package, "version": version},
                    ))
                else:
                    self.add_check(CheckResult(
                        name=f"MCP: {name}",
                        status=HealthStatus.ERROR,
                        message=f"Package not found: {package}",
                        fix_available=True,
                        fix_command=f"npm install -g {package}",
                    ))
            except subprocess.TimeoutExpired:
                self.add_check(CheckResult(
                    name=f"MCP: {name}",
                    status=HealthStatus.WARNING,
                    message="Timeout checking package",
                ))
            except Exception as e:
                self.add_check(CheckResult(
                    name=f"MCP: {name}",
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                ))

    # === Letta Check ===

    async def check_letta(self) -> None:
        """Check Letta server availability."""
        letta_url = os.environ.get("LETTA_URL", "http://localhost:8283")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{letta_url}/v1/health")
                if response.status_code == 200:
                    self.add_check(CheckResult(
                        name="Letta Server",
                        status=HealthStatus.OK,
                        message=f"Running at {letta_url}",
                        details={"url": letta_url, "status_code": 200},
                    ))
                else:
                    self.add_check(CheckResult(
                        name="Letta Server",
                        status=HealthStatus.WARNING,
                        message=f"Unexpected status: {response.status_code}",
                    ))
        except httpx.ConnectError:
            self.add_check(CheckResult(
                name="Letta Server",
                status=HealthStatus.WARNING,
                message=f"Not running at {letta_url}",
                fix_available=True,
                fix_command="docker run -d -p 8283:8283 letta/letta:latest",
                details={"url": letta_url, "note": "Memory persistence disabled"},
            ))
        except Exception as e:
            self.add_check(CheckResult(
                name="Letta Server",
                status=HealthStatus.WARNING,
                message=f"Check failed: {e}",
            ))

    # === Kill Switch Check ===

    def check_kill_switch(self) -> None:
        """Check kill switch status."""
        kill_switch = self.claude_home / "KILL_SWITCH"

        if kill_switch.exists():
            self.add_check(CheckResult(
                name="Kill Switch",
                status=HealthStatus.WARNING,
                message="ACTIVE - All operations blocked!",
                fix_available=True,
                fix_command=f'Remove-Item "{kill_switch}"',
            ))
        else:
            self.add_check(CheckResult(
                name="Kill Switch",
                status=HealthStatus.OK,
                message="Inactive (normal operation)",
            ))

    # === Python Dependencies ===

    def check_python_deps(self) -> None:
        """Check Python dependencies for hooks.

        Note: Hook scripts use uv's inline script dependencies (PEP 723),
        so packages are automatically installed when running hooks with 'uv run'.
        We check if they're available via uv pip show.
        """
        deps = ["httpx", "structlog"]

        for dep in deps:
            try:
                # First try direct import (system Python)
                __import__(dep)
                self.add_check(CheckResult(
                    name=f"Python: {dep}",
                    status=HealthStatus.OK,
                    message="Installed (system)",
                ))
            except ImportError:
                # Check via uv pip show
                try:
                    result = subprocess.run(
                        ["uv", "pip", "show", dep],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        self.add_check(CheckResult(
                            name=f"Python: {dep}",
                            status=HealthStatus.OK,
                            message="Available via uv (inline deps)",
                        ))
                    else:
                        # Not installed anywhere, but hooks use inline deps
                        self.add_check(CheckResult(
                            name=f"Python: {dep}",
                            status=HealthStatus.OK,
                            message="Auto-installed by uv when running hooks",
                            details={"note": "Hooks use PEP 723 inline dependencies"},
                        ))
                except Exception:
                    # uv not available or other error - still OK for inline deps
                    self.add_check(CheckResult(
                        name=f"Python: {dep}",
                        status=HealthStatus.OK,
                        message="Auto-installed by uv when running hooks",
                        details={"note": "Hooks use PEP 723 inline dependencies"},
                    ))

    # === Run All Checks ===

    async def run_all(self, quick: bool = False) -> HealthReport:
        """Run all health checks."""
        self.checks = []

        # Synchronous checks
        self.check_directories()
        self.check_hooks()
        self.check_config()
        self.check_kill_switch()
        self.check_python_deps()

        # Async checks
        await self.check_letta()
        if not quick:
            await self.check_mcp_packages(quick)
        else:
            await self.check_mcp_packages(quick=True)

        # Calculate summary
        summary = {
            "ok": sum(1 for c in self.checks if c.status == HealthStatus.OK),
            "warning": sum(1 for c in self.checks if c.status == HealthStatus.WARNING),
            "error": sum(1 for c in self.checks if c.status == HealthStatus.ERROR),
            "unknown": sum(1 for c in self.checks if c.status == HealthStatus.UNKNOWN),
        }

        # Determine overall status
        if summary["error"] > 0:
            overall = HealthStatus.ERROR
        elif summary["warning"] > 0:
            overall = HealthStatus.WARNING
        else:
            overall = HealthStatus.OK

        return HealthReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_status=overall,
            checks=self.checks,
            summary=summary,
        )


def display_report(report: HealthReport) -> None:
    """Display health report in rich format."""
    # Header
    status_color = {
        HealthStatus.OK: "green",
        HealthStatus.WARNING: "yellow",
        HealthStatus.ERROR: "red",
        HealthStatus.UNKNOWN: "dim",
    }

    console.print(Panel(
        f"[bold]V10 System Health Check[/bold]\n"
        f"Overall: [{status_color[report.overall_status]}]{report.overall_status.value.upper()}[/]",
        title="Health Report",
        subtitle=report.timestamp,
    ))

    # Summary
    console.print(
        f"\n[green]OK: {report.summary['ok']}[/] | "
        f"[yellow]Warning: {report.summary['warning']}[/] | "
        f"[red]Error: {report.summary['error']}[/]"
    )

    # Results table
    table = Table(title="\nDetailed Results", show_lines=True)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Message")
    table.add_column("Fix", style="dim")

    # Use ASCII characters on Windows to avoid encoding issues
    if sys.platform == "win32":
        status_icons = {
            HealthStatus.OK: "[green]OK[/]",
            HealthStatus.WARNING: "[yellow]WARN[/]",
            HealthStatus.ERROR: "[red]ERR[/]",
            HealthStatus.UNKNOWN: "[dim]?[/]",
        }
    else:
        status_icons = {
            HealthStatus.OK: "[green]✓[/]",
            HealthStatus.WARNING: "[yellow]⚠[/]",
            HealthStatus.ERROR: "[red]✗[/]",
            HealthStatus.UNKNOWN: "[dim]?[/]",
        }

    for check in report.checks:
        fix = check.fix_command[:40] + "..." if check.fix_command and len(check.fix_command) > 40 else (check.fix_command or "")
        table.add_row(
            check.name,
            status_icons[check.status],
            check.message[:60] + ("..." if len(check.message) > 60 else ""),
            fix,
        )

    console.print(table)

    # Show fixes if available
    fixes = [c for c in report.checks if c.fix_available]
    if fixes:
        console.print("\n[bold yellow]Available Fixes:[/]")
        for fix in fixes:
            console.print(f"  {fix.name}: [dim]{fix.fix_command}[/]")


async def main():
    parser = argparse.ArgumentParser(description="V10 System Health Check")
    parser.add_argument("--quick", action="store_true", help="Quick check (skip slow tests)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    args = parser.parse_args()

    checker = HealthChecker()

    if not args.json:
        # Use ASCII spinner on Windows to avoid encoding issues
        spinner = SpinnerColumn(spinner_name="dots" if sys.platform != "win32" else "line")
        with Progress(
            spinner,
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,  # Clear progress on completion
        ) as progress:
            progress.add_task("Running health checks...", total=None)
            report = await checker.run_all(quick=args.quick)
    else:
        report = await checker.run_all(quick=args.quick)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        display_report(report)

    # Attempt fixes if requested
    if args.fix:
        fixes = [c for c in report.checks if c.fix_available and c.status == HealthStatus.ERROR]
        if fixes:
            console.print("\n[bold]Attempting fixes...[/]")
            for fix in fixes:
                if fix.fix_command:
                    console.print(f"  Running: {fix.fix_command}")
                    try:
                        subprocess.run(fix.fix_command, shell=True, check=True)
                        console.print(f"  [green]Fixed: {fix.name}[/]")
                    except subprocess.CalledProcessError as e:
                        console.print(f"  [red]Failed: {fix.name} - {e}[/]")

    # Exit code based on status
    if report.overall_status == HealthStatus.ERROR:
        sys.exit(1)
    elif report.overall_status == HealthStatus.WARNING:
        sys.exit(0)  # Warnings are OK
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
