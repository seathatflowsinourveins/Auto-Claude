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
MCP Package Verifier - V10 Optimized

Verifies which MCP packages actually exist on npm/pypi and work.
Run this before configuring MCP servers to avoid 404 errors.

Usage:
    uv run verify_mcp.py           # Check all known packages
    uv run verify_mcp.py --quick   # Check core packages only
    uv run verify_mcp.py --fix     # Generate fixed config
"""

import argparse
import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from rich.console import Console
from rich.table import Table

# Force ASCII mode for Windows compatibility
console = Console(force_terminal=True, legacy_windows=False, no_color=False)


class PackageSource(Enum):
    NPM = "npm"
    PYPI = "pypi"
    UVX = "uvx"
    CLI = "cli"


class PackageStatus(Enum):
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    DEPRECATED = "deprecated"
    UNKNOWN = "unknown"


@dataclass
class MCPPackage:
    name: str
    package: str
    source: PackageSource
    description: str
    status: PackageStatus = PackageStatus.UNKNOWN
    version: Optional[str] = None
    notes: Optional[str] = None


# Known MCP packages with their status
KNOWN_PACKAGES = [
    # Official MCP packages (npm)
    MCPPackage("filesystem", "@modelcontextprotocol/server-filesystem", PackageSource.NPM, "File system operations"),
    MCPPackage("memory", "@modelcontextprotocol/server-memory", PackageSource.NPM, "Key-value memory"),
    MCPPackage("sequential-thinking", "@modelcontextprotocol/server-sequential-thinking", PackageSource.NPM, "Extended reasoning"),
    MCPPackage("fetch", "@modelcontextprotocol/server-fetch", PackageSource.NPM, "HTTP fetching"),
    MCPPackage("sqlite", "@modelcontextprotocol/server-sqlite", PackageSource.NPM, "SQLite database"),
    MCPPackage("brave-search", "@modelcontextprotocol/server-brave-search", PackageSource.NPM, "Web search"),
    MCPPackage("everything", "@modelcontextprotocol/server-everything", PackageSource.NPM, "Test/demo server"),
    
    # Official vendor packages
    MCPPackage("eslint", "@eslint/mcp", PackageSource.NPM, "ESLint integration"),
    MCPPackage("context7", "@upstash/context7-mcp", PackageSource.NPM, "Library documentation"),
    MCPPackage("playwright", "@anthropic-ai/mcp-server-playwright", PackageSource.NPM, "Browser automation"),
    MCPPackage("puppeteer", "@anthropic-ai/mcp-server-puppeteer", PackageSource.NPM, "Headless browser"),
    
    # CLI-based
    MCPPackage("github", "gh mcp", PackageSource.CLI, "GitHub operations"),
    MCPPackage("semgrep", "semgrep mcp", PackageSource.CLI, "Security scanning"),
    
    # UVX packages
    MCPPackage("qdrant", "mcp-server-qdrant", PackageSource.UVX, "Vector search"),
    MCPPackage("alpaca", "alpaca-mcp-server", PackageSource.UVX, "Trading (paper)"),
    
    # Known non-existent (for warning)
    MCPPackage("letta", "@letta-ai/mcp-server", PackageSource.NPM, "DOES NOT EXIST", PackageStatus.NOT_FOUND, notes="npm 404"),
    MCPPackage("mem0", "mem0-mcp", PackageSource.NPM, "DOES NOT EXIST", PackageStatus.NOT_FOUND, notes="npm 404"),
    MCPPackage("langfuse", "@langfuse/mcp-server", PackageSource.NPM, "DOES NOT EXIST", PackageStatus.NOT_FOUND, notes="npm 404"),
    MCPPackage("slack-mcp", "@anthropic-ai/mcp-server-slack", PackageSource.NPM, "DOES NOT EXIST", PackageStatus.NOT_FOUND, notes="npm 404"),
    
    # Deprecated
    MCPPackage("github-old", "@modelcontextprotocol/server-github", PackageSource.NPM, "Deprecated", PackageStatus.DEPRECATED, notes="Use gh mcp"),
    MCPPackage("postgres-old", "@modelcontextprotocol/server-postgres", PackageSource.NPM, "Deprecated", PackageStatus.DEPRECATED),
]


async def check_npm_package(package: str) -> tuple[PackageStatus, Optional[str]]:
    """Check if an npm package exists and get its version."""
    try:
        # Fix Windows PATH: include nodejs directory explicitly
        import os
        env = os.environ.copy()
        nodejs_path = r"C:\Program Files\nodejs"
        if nodejs_path not in env.get("PATH", ""):
            env["PATH"] = nodejs_path + os.pathsep + env.get("PATH", "")

        result = subprocess.run(
            ["npm", "view", package, "version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            shell=True  # Required for Windows to find npm.cmd
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return PackageStatus.VERIFIED, version
        else:
            return PackageStatus.NOT_FOUND, None
    except Exception as e:
        return PackageStatus.UNKNOWN, str(e)


async def check_pypi_package(package: str) -> tuple[PackageStatus, Optional[str]]:
    """Check if a PyPI/uvx package exists."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://pypi.org/pypi/{package}/json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                version = data.get("info", {}).get("version")
                return PackageStatus.VERIFIED, version
            else:
                return PackageStatus.NOT_FOUND, None
    except Exception as e:
        return PackageStatus.UNKNOWN, str(e)


async def check_cli_tool(command: str) -> tuple[PackageStatus, Optional[str]]:
    """Check if a CLI tool is available."""
    try:
        tool = command.split()[0]
        result = subprocess.run(
            [tool, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            return PackageStatus.VERIFIED, version
        else:
            return PackageStatus.NOT_FOUND, None
    except FileNotFoundError:
        return PackageStatus.NOT_FOUND, None
    except Exception as e:
        return PackageStatus.UNKNOWN, str(e)


async def verify_package(pkg: MCPPackage) -> MCPPackage:
    """Verify a single package."""
    # Skip already-known bad packages
    if pkg.status in (PackageStatus.NOT_FOUND, PackageStatus.DEPRECATED):
        return pkg
    
    if pkg.source == PackageSource.NPM:
        status, version = await check_npm_package(pkg.package)
    elif pkg.source in (PackageSource.PYPI, PackageSource.UVX):
        # Extract package name for pypi
        pypi_name = pkg.package.replace("-", "_") if "-" in pkg.package else pkg.package
        status, version = await check_pypi_package(pypi_name)
    elif pkg.source == PackageSource.CLI:
        status, version = await check_cli_tool(pkg.package)
    else:
        status, version = PackageStatus.UNKNOWN, None
    
    pkg.status = status
    pkg.version = version
    return pkg


async def verify_all_packages(packages: List[MCPPackage], quick: bool = False) -> List[MCPPackage]:
    """Verify all packages."""
    if quick:
        # Only check core packages
        core_names = {"filesystem", "memory", "sequential-thinking", "eslint", "context7", "github"}
        packages = [p for p in packages if p.name in core_names]
    
    tasks = [verify_package(pkg) for pkg in packages]
    return await asyncio.gather(*tasks)


def display_results(packages: List[MCPPackage]):
    """Display verification results in a table."""
    table = Table(title="MCP Package Verification Results")
    
    table.add_column("Name", style="cyan")
    table.add_column("Package", style="dim")
    table.add_column("Source", style="blue")
    table.add_column("Status", style="bold")
    table.add_column("Version", style="green")
    table.add_column("Notes", style="yellow")
    
    for pkg in sorted(packages, key=lambda x: (x.status.value, x.name)):
        # Use ASCII-safe status indicators for Windows compatibility
        status_style = {
            PackageStatus.VERIFIED: "[green][OK] Verified[/green]",
            PackageStatus.NOT_FOUND: "[red][ERR] Not Found[/red]",
            PackageStatus.DEPRECATED: "[yellow][WARN] Deprecated[/yellow]",
            PackageStatus.UNKNOWN: "[dim][?] Unknown[/dim]",
        }.get(pkg.status, pkg.status.value)
        
        table.add_row(
            pkg.name,
            pkg.package[:40] + "..." if len(pkg.package) > 40 else pkg.package,
            pkg.source.value,
            status_style,
            pkg.version or "-",
            pkg.notes or ""
        )
    
    console.print(table)
    
    # Summary
    verified = sum(1 for p in packages if p.status == PackageStatus.VERIFIED)
    not_found = sum(1 for p in packages if p.status == PackageStatus.NOT_FOUND)
    deprecated = sum(1 for p in packages if p.status == PackageStatus.DEPRECATED)
    
    console.print(f"\n[bold]Summary:[/bold] {verified} verified, {not_found} not found, {deprecated} deprecated")


def generate_fixed_config(packages: List[MCPPackage]) -> dict:
    """Generate a fixed MCP configuration with only verified packages."""
    config = {"mcpServers": {}}
    
    for pkg in packages:
        if pkg.status != PackageStatus.VERIFIED:
            continue
        
        if pkg.source == PackageSource.NPM:
            config["mcpServers"][pkg.name] = {
                "type": "stdio",
                "command": "cmd",
                "args": ["/c", "npx", "-y", pkg.package]
            }
        elif pkg.source == PackageSource.UVX:
            config["mcpServers"][pkg.name] = {
                "type": "stdio",
                "command": "uvx",
                "args": [pkg.package]
            }
        elif pkg.source == PackageSource.CLI:
            parts = pkg.package.split()
            config["mcpServers"][pkg.name] = {
                "type": "stdio",
                "command": parts[0],
                "args": parts[1:]
            }
    
    return config


async def main():
    parser = argparse.ArgumentParser(description="Verify MCP packages")
    parser.add_argument("--quick", action="store_true", help="Check core packages only")
    parser.add_argument("--fix", action="store_true", help="Generate fixed configuration")
    parser.add_argument("--output", help="Output file for fixed config")
    args = parser.parse_args()
    
    print("MCP Package Verifier - V10 Optimized\n")
    print("Verifying packages...")
    packages = await verify_all_packages(KNOWN_PACKAGES.copy(), args.quick)

    # Simple ASCII output for Windows compatibility
    print("\n" + "=" * 70)
    print("MCP Package Verification Results")
    print("=" * 70)
    print(f"{'Name':<15} {'Package':<30} {'Status':<12} {'Version'}")
    print("-" * 70)

    for pkg in sorted(packages, key=lambda x: (x.status.value, x.name)):
        status_str = {
            PackageStatus.VERIFIED: "[OK]",
            PackageStatus.NOT_FOUND: "[ERR]",
            PackageStatus.DEPRECATED: "[WARN]",
            PackageStatus.UNKNOWN: "[?]",
        }.get(pkg.status, "?")
        pkg_name = pkg.package[:28] + ".." if len(pkg.package) > 30 else pkg.package
        print(f"{pkg.name:<15} {pkg_name:<30} {status_str:<12} {pkg.version or '-'}")

    # Summary
    verified = sum(1 for p in packages if p.status == PackageStatus.VERIFIED)
    not_found = sum(1 for p in packages if p.status == PackageStatus.NOT_FOUND)
    deprecated = sum(1 for p in packages if p.status == PackageStatus.DEPRECATED)
    print("-" * 70)
    print(f"Summary: {verified} verified, {not_found} not found, {deprecated} deprecated")
    
    if args.fix:
        config = generate_fixed_config(packages)
        output_file = args.output or ".mcp.verified.json"
        
        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)
        
        console.print(f"\n[green]Fixed configuration written to {output_file}[/green]")
        console.print(f"Contains {len(config['mcpServers'])} verified servers")


if __name__ == "__main__":
    asyncio.run(main())
